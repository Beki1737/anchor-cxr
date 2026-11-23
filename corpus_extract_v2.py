#!/usr/bin/env python3
"""ANCHOR E1 v2 extractor (D43). Fixes the error classes found on the 200-row dev sheet (r22 audit):
  LAT_SIDE 53   constrained decoding defaulted to the first enum value "right" when no side was stated
                -> enum reordered ("none" first), explicit rule, and a deterministic laterality resolver
  MISS_* 35     recall gaps on normal-structure statements, chronic/incidental findings and devices
                -> coverage rules + a second, sentence-focused pass on every sentence that came back empty
  FAKE/NAME/HEDGE 15 -> stricter wording rules
v1 code stays untouched (corpus_extract.py) so the v1 corpus remains reproducible.

    python corpus_extract_v2.py devcheck                 # v2 on the dev-sheet reports -> e1_recheck_v2.csv (local)
    python corpus_extract_v2.py full --split train       # full corpus -> ~/anchor/work/corpus_v2/<split>/
    python corpus_extract_v2.py sheet --n 200 --seed 1   # fresh held-out gate sheet (excludes dev-sheet sentences)
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import corpus_extract as v1  # noqa: E402

HOME = v1.HOME
REGION_VOCAB = v1.REGION_VOCAB
LAT = ["none", "right", "left", "bilateral"]
SCHEMA = json.loads(json.dumps(v1.SCHEMA))
_f = SCHEMA["properties"]["sentences"]["items"]["properties"]["findings"]["items"]["properties"]
_f["laterality"]["enum"] = LAT
_f["polarity"]["enum"] = ["absent", "present", "uncertain"]

RULES = """Rules:
- One entry per distinct finding in a sentence. Only sentences that are purely technique, view list, comparison date,
  dictation or recommendation get an empty list. Every sentence that describes anatomy, a device, prior surgery,
  a chronic or incidental change, or says that something is normal MUST produce at least one entry.
- Normal statements become the finding they rule out, polarity absent:
  "no pleural effusion" -> pleural effusion, absent; "lungs are clear" -> airspace opacity, absent, right lung + left lung;
  "heart size is normal" -> cardiomegaly, absent, heart; "mediastinal contours are normal" or "cardiomediastinal silhouette
  is normal/stable" -> cardiomegaly, absent, heart AND mediastinal widening, absent, mediastinum;
  "no acute osseous abnormality" or "bones are intact" -> fracture, absent; "no pneumothorax" -> pneumothorax, absent.
- Chronic or incidental findings are mapped as present when noted: spine degenerative changes, calcified granuloma, aortic arch calcification or tortuosity, spine scoliosis, osteopenia, prior fracture, hiatal hernia,
  calcification or tortuosity (aortic arch), scoliosis (spine), osteopenia, old or healed fracture, hiatal hernia,
  postsurgical change (sternotomy wires, surgical clips), elevated hemidiaphragm.
- Devices: "support device: <which>" (endotracheal tube, central line, pacemaker, chest tube, nasogastric tube, ...),
  present; "removed" -> absent.
- "stable" or "unchanged" finding -> present with change stable. Hedged words (possible, may represent, cannot exclude,
  likely, suggestive of) -> polarity uncertain. Never add a finding the sentence does not state.
- Use short canonical finding names: pleural effusion, pneumothorax, consolidation, airspace opacity, atelectasis,
  pulmonary edema, cardiomegaly, mediastinal widening, nodule, mass, interstitial opacity, hyperinflation, fracture,
  degenerative changes, calcified granuloma, aortic calcification, scoliosis, postsurgical change, support device: <which>.
- laterality: "none" unless the sentence itself says right, left, bilateral or both for that finding.
- regions only from the allowed list. Mapping hints: hilum -> mid zone of that side; costophrenic angle or pleural effusion
  -> base of that side; lingula -> left upper lobe; bilateral findings list both sides; whole-lung findings -> right lung
  and/or left lung; unknown -> unspecified.
- change: new / worse / better / stable when the sentence compares with a prior, else unspecified.
Allowed regions: """ + ", ".join(REGION_VOCAB)

RECHECK = ("These sentences were previously returned with NO findings. Re-read each one. If it describes anatomy, a device, "
           "prior surgery, a chronic or incidental change, or states that something is normal, extract it with the rules. "
           "Return an empty list only for pure technique, view, comparison-date, dictation or recommendation sentences.")
NOFIND_RE = re.compile(r"^(comparison|technique|views?|pa and lateral|ap |portable|frontal and lateral|dictated|"
                       r"recommend|correlat|findings? (were|was) discussed|history|indication|exam)", re.I)


def lat_resolve(f, sentence):
    """Deterministic laterality: regions decide when lateralised; otherwise keep a side only if the sentence names it."""
    regs = f.get("regions", [])
    r = any(x.startswith("R_") for x in regs); l = any(x.startswith("L_") for x in regs)
    if r and l:
        return "bilateral"
    if r or l:
        return "right" if r else "left"
    s = sentence.lower()
    lat = f.get("laterality") or "none"
    if lat == "bilateral" and re.search(r"\b(bilateral|both|bibasilar|bibasal)\b", s):
        return "bilateral"
    if lat in ("right", "left") and re.search(rf"\b{lat}\b", s):
        return lat
    return "none"


def build_messages(sents, recheck=False):
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sents))
    head = RULES + ("\n\n" + RECHECK if recheck else "")
    return [{"role": "system", "content": v1.SYSTEM},
            {"role": "user", "content": f"{head}\n\nSentences:\n{numbered}\n\nReturn {{\"sentences\": [...]}} with one item per sentence index."}]


def parse(raw, n):
    res, ok = v1.parse_output(raw, n)
    return res, ok


def make_vllm(model, gpu_util, max_len, quant=None):
    from vllm import LLM, SamplingParams
    kw = {"quantization": quant} if quant else {}
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_len,
              enable_prefix_caching=True, **kw)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0, max_tokens=1536, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    except Exception:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0, max_tokens=1536, guided_decoding=GuidedDecodingParams(json=SCHEMA))

    def generate(batch_messages):
        outs = llm.chat(batch_messages, sp, use_tqdm=False)
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]
    return generate


def extract_reports(part, generate):
    """part: DataFrame key, findings, impression -> sentence rows (pass 1 + recheck pass on empty sentences)."""
    jobs = []
    for r in part.itertuples(index=False):
        sents = [("findings", s) for s in v1.split_sentences(r.findings)] + [("impression", s) for s in v1.split_sentences(r.impression)]
        if sents:
            jobs.append((r.key, sents))
    outs = generate([build_messages([s for _, s in sents]) for _, sents in jobs]) if jobs else []
    results, re_jobs = [], []
    for (key, sents), (raw, fin) in zip(jobs, outs):
        parsed, ok = parse(raw, len(sents))
        results.append([key, sents, parsed, ok, fin])
        empty = [i for i, fl in enumerate(parsed) if not fl and not NOFIND_RE.match(sents[i][1].strip())]
        if empty:
            re_jobs.append((len(results) - 1, empty))
    n_rechecked = n_recovered = 0
    if re_jobs:
        outs2 = generate([build_messages([results[j][1][i][1] for i in idx], recheck=True) for j, idx in re_jobs])
        for (j, idx), (raw, fin) in zip(re_jobs, outs2):
            p2, _ = parse(raw, len(idx))
            for i, fl in zip(idx, p2):
                n_rechecked += 1
                if fl:
                    results[j][2][i] = fl; n_recovered += 1
    rows = []
    for key, sents, parsed, ok, fin in results:
        for i, ((section, text), fl) in enumerate(zip(sents, parsed)):
            for f in fl:
                f["laterality_raw"] = f.get("laterality")
                f["laterality"] = lat_resolve(f, text)
            rows.append({"key": key, "section": section, "sid": i, "sentence": text, "findings_json": json.dumps(fl),
                         "n_findings": len(fl), "report_valid": ok, "truncated": fin == "length"})
    return pd.DataFrame(rows), n_rechecked, n_recovered


def show(js):
    return " || ".join(f"{f['finding']} | {f['polarity']} | {','.join(f['regions']) or '-'} | {f['laterality']} | {f['change']}"
                       for f in json.loads(js)) or "(no findings)"


def dev_pick():
    """Rebuild the exact dev-sheet sample (v1.label_sheet logic, seed 0) to recover each row's report key."""
    s = pd.concat([pd.read_parquet(p) for p in sorted((HOME / "work/corpus/train_pilot2000").glob("shard_*.parquet"))],
                  ignore_index=True)
    n = 200
    with_f = s[s.n_findings > 0].sample(n=min(int(n * 0.7), (s.n_findings > 0).sum()), random_state=0)
    without = s[s.n_findings == 0].sample(n=min(n - len(with_f), (s.n_findings == 0).sum()), random_state=0)
    return pd.concat([with_f, without]).sample(frac=1, random_state=0).reset_index(drop=True)


def get_generate(a):
    g = v1.pick_gpu()
    if g is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(g); print(f"using GPU {g}", flush=True)
    return make_vllm(a.model, a.gpu_util, a.max_len, a.quant or None)


def cmd_devcheck(a, generate=None):
    sheet = pd.read_csv(HOME / "work/e1_label_200.csv")
    pick = dev_pick()
    match = float((pick.sentence.values == sheet.sentence.values).mean())
    print(f"dev sheet rows re-identified: {match:.3f}")
    assert match > 0.99, "dev sheet could not be re-identified; stop"
    man = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key")
    keys = sorted(set(pick.key))
    generate = generate or get_generate(a)
    rows, nre, nrec = extract_reports(man.loc[keys, ["findings", "impression"]].reset_index(), generate)
    v2 = rows.set_index(["key", "section", "sid"])
    ext2 = [show(v2.loc[(k, sec, sid), "findings_json"]) if (k, sec, sid) in v2.index else "(missing)"
            for k, sec, sid in zip(pick.key, pick.section, pick.sid)]
    for f in ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]:
        sheet[f] = pd.to_numeric(sheet[f], errors="coerce")
    err = (sheet.finding_and_polarity_ok == 0) | (sheet.regions_ok == 0) | (sheet.laterality_ok == 0) | (sheet.missed_finding == 1)
    rng = np.random.default_rng(0)
    bad, good = sheet.index[err].to_numpy(), sheet.index[~err].to_numpy()
    bad = rng.choice(bad, size=min(40, len(bad)), replace=False)          # small blind recheck: 40 errors + 20 controls
    ctrl = rng.choice(good, size=min(20, len(good)), replace=False)
    sel = np.concatenate([bad, ctrl])
    pd.DataFrame({"row": sel, "is_control": np.isin(sel, ctrl)}).to_csv(HOME / "work/e1_recheck_v2_key.csv", index=False)
    rc = pd.DataFrame({"row": sel, "sentence": sheet.sentence.values[sel], "extraction_v2": np.array(ext2)[sel],
                       "finding_and_polarity_ok": "", "regions_ok": "", "laterality_ok": "", "missed_finding": "", "note": ""})
    rc = rc.sample(frac=1, random_state=1).reset_index(drop=True)
    rc.to_csv(HOME / "work/e1_recheck_v2.csv", index=False)
    changed = float((np.array(ext2) != sheet.extraction.values).mean())
    nf2 = np.array([e == "(no findings)" for e in ext2])
    print(f"v2 on {len(keys)} dev reports: rechecked {nre} empty sentences, recovered {nrec}")
    print(f"extraction changed vs v1 on {changed:.3f} of dev rows | '(no findings)' rows v1 {int((sheet.extraction == '(no findings)').sum())} -> v2 {int(nf2.sum())}")
    print(f"recheck sheet: {len(rc)} rows ({len(bad)} of {int(err.sum())} v1-error rows + {len(ctrl)} v1-correct controls, shuffled, blind) -> work/e1_recheck_v2.csv")
    print("DEVCHECK_DONE")


def cmd_full(a, generate=None):
    man = pd.read_parquet(HOME / f"work/manifest/rex_{a.split}.parquet")
    man = man[man.eligible].sort_values("key").reset_index(drop=True)
    out = HOME / "work" / a.corpus_name / a.split; out.mkdir(parents=True, exist_ok=True)
    generate = generate or get_generate(a)
    t0, done, tot_re, tot_rec = time.time(), 0, 0, 0
    for s0 in range(0, len(man), a.shard):
        sp = out / f"shard_{s0 // a.shard:05d}.parquet"
        part = man.iloc[s0:s0 + a.shard]
        if not sp.exists():
            rows, nre, nrec = extract_reports(part[["key", "findings", "impression"]], generate)
            rows.to_parquet(sp, index=False); tot_re += nre; tot_rec += nrec
        done += len(part)
        print(time.strftime("[%H:%M:%S] ") + f"{sp.name}: {done}/{len(man)} reports, {done / max(time.time() - t0, 1e-6):.2f} reports/s"
              f" | recheck recovered {tot_rec}/{tot_re}", flush=True)
    print(f"CORPUS_V2_DONE {a.split}", flush=True)


def cmd_abtest(a, generate=None):
    """Model selection on a THIRD sentence pool (not dev, not held-out): current v2 extraction vs candidate model,
    shown blind as extraction_1 / extraction_2 in random order. Grade only finding/polarity and missed finding."""
    v2 = pd.concat([pd.read_parquet(p) for p in sorted((HOME / "work/corpus_v2/train").glob("shard_*.parquet"))], ignore_index=True)
    held = pd.read_csv(HOME / "work/e1_label_200_v2_heldout.csv").sentence
    excl = set(dev_pick().sentence) | set(held)
    pool = v2[~v2.sentence.isin(excl)]
    with_f = pool[pool.n_findings > 0].sample(n=int(a.ab_n * 0.7), random_state=3)
    without = pool[pool.n_findings == 0].sample(n=a.ab_n - len(with_f), random_state=3)
    pick = pd.concat([with_f, without]).sample(frac=1, random_state=3).reset_index(drop=True)
    man = pd.read_parquet(HOME / "work/manifest/rex_train.parquet").set_index("key")
    keys = sorted(set(pick.key))
    generate = generate or get_generate(a)
    rows, nre, nrec = extract_reports(man.loc[keys, ["findings", "impression"]].reset_index(), generate)
    cand = rows.set_index(["key", "section", "sid"])
    ext_b = [show(cand.loc[(k, sec, sid), "findings_json"]) if (k, sec, sid) in cand.index else "(missing)"
             for k, sec, sid in zip(pick.key, pick.section, pick.sid)]
    ext_a = pick.findings_json.map(show).tolist()
    flip = np.random.default_rng(4).random(len(pick)) < 0.5            # True: candidate shown as extraction_1
    e1 = [b if f else x for x, b, f in zip(ext_a, ext_b, flip)]
    e2 = [x if f else b for x, b, f in zip(ext_a, ext_b, flip)]
    sheet = pd.DataFrame({"row": range(len(pick)), "sentence": pick.sentence, "extraction_1": e1, "extraction_2": e2,
                          "ok_1": "", "ok_2": "", "missed_1": "", "missed_2": "", "note": ""})
    sheet.to_csv(HOME / "work/e1_abtest.csv", index=False)
    pd.DataFrame({"row": range(len(pick)), "candidate_is_1": flip, "identical": np.array(e1) == np.array(e2)}).to_csv(
        HOME / "work/e1_abtest_key.csv", index=False)
    print(f"A/B sheet: {len(sheet)} rows, identical extractions {float((np.array(e1) == np.array(e2)).mean()):.3f}, "
          f"candidate recheck recovered {nrec}/{nre} -> work/e1_abtest.csv (grade ok_1, ok_2, missed_1, missed_2)")
    print("ABTEST_DONE")


def cmd_abscore(a):
    s = pd.read_csv(HOME / "work/e1_abtest.csv").merge(pd.read_csv(HOME / "work/e1_abtest_key.csv"), on="row")
    for c in ["ok_1", "ok_2", "missed_1", "missed_2"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["ok_1", "ok_2"])
    cand_ok = np.where(s.candidate_is_1, s.ok_1, s.ok_2); v2_ok = np.where(s.candidate_is_1, s.ok_2, s.ok_1)
    cand_m = np.where(s.candidate_is_1, s.missed_1, s.missed_2); v2_m = np.where(s.candidate_is_1, s.missed_2, s.missed_1)
    b = int(((cand_ok == 1) & (v2_ok == 0)).sum()); c = int(((cand_ok == 0) & (v2_ok == 1)).sum())
    from math import comb
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
    print(f"ABSCORE n={len(s)} finding/polarity ok: v2={np.nanmean(v2_ok):.3f} candidate={np.nanmean(cand_ok):.3f} | "
          f"discordant: candidate better {b}, v2 better {c}, exact McNemar p={p:.4f} | "
          f"missed: v2={np.nanmean(v2_m):.3f} candidate={np.nanmean(cand_m):.3f}")


def cmd_gentext(a, generate=None):
    """Extract claims from GENERATED reports (same v2 extractor as the corpus) -> work/corpus_gen/<gen stem>/shard_*.parquet."""
    g = pd.read_parquet(HOME / "work/gen" / a.gen)
    df = pd.DataFrame({"key": g.key, "findings": g.generated.fillna(""), "impression": ""})
    out = HOME / "work/corpus_gen" / pathlib.Path(a.gen).stem; out.mkdir(parents=True, exist_ok=True)
    generate = generate or get_generate(a)
    t0 = time.time()
    for s0 in range(0, len(df), a.shard):
        sp = out / f"shard_{s0 // a.shard:05d}.parquet"
        if not sp.exists():
            extract_reports(df.iloc[s0:s0 + a.shard], generate)[0].to_parquet(sp, index=False)
    print(f"GENTEXT_DONE {a.gen}: {len(df)} reports in {time.time() - t0:.0f}s", flush=True)


def cmd_sheet(a):
    s = pd.concat([pd.read_parquet(p) for p in sorted((HOME / "work/corpus_v2/train").glob("shard_*.parquet"))], ignore_index=True)
    dev = set(dev_pick().sentence)
    s = s[~s.sentence.isin(dev)]
    with_f = s[s.n_findings > 0].sample(n=int(a.n * 0.7), random_state=a.seed)
    without = s[s.n_findings == 0].sample(n=a.n - len(with_f), random_state=a.seed)
    pick = pd.concat([with_f, without]).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    sheet = pd.DataFrame({"row": range(len(pick)), "sentence": pick.sentence, "extraction": pick.findings_json.map(show),
                          "finding_and_polarity_ok": "", "regions_ok": "", "laterality_ok": "", "missed_finding": "",
                          "note": "", "err_code": ""})
    p = HOME / f"work/e1_label_{a.n}_v2_heldout.csv"
    sheet.to_csv(p, index=False)
    print(f"held-out gate sheet: {len(sheet)} rows -> {p.name} (dev-sheet sentences excluded)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["devcheck", "full", "sheet", "abtest", "abscore", "gentext"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="./models/Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quant", default="", help="e.g. fp8 for 32B/72B on a shared GPU")
    ap.add_argument("--corpus-name", default="corpus_v2")
    ap.add_argument("--ab-n", type=int, default=100)
    ap.add_argument("--gen", default="", help="gentext: generations parquet name in work/gen")
    a = ap.parse_args(argv)
    {"devcheck": cmd_devcheck, "full": cmd_full, "sheet": cmd_sheet, "abtest": cmd_abtest, "abscore": cmd_abscore, "gentext": cmd_gentext}[a.stage](a)


if __name__ == "__main__":
    main()
