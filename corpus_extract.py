#!/usr/bin/env python3
"""ANCHOR E1: structure the retrieval corpus. Split Findings/Impression into sentences, then a local LLM
(Qwen2.5-14B-Instruct via vLLM, JSON-constrained) extracts per sentence:
finding, polarity, ANCHOR regions, laterality, severity, change.

    python corpus_extract.py --split train --limit 2000 --label 200     # pilot + 200-sentence labelling sheet
    python corpus_extract.py --split train                               # full corpus (resumable shards)

Outputs (local only, contain report text): ~/anchor/work/corpus/<tag>/shard_*.parquet, label sheet CSV.
Prints aggregates only.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
REGION_VOCAB = {  # words the LLM may use -> ANCHOR region id (see region_cache.REGIONS)
    "right apex": "R_apical", "right upper zone": "R_upper", "right mid zone": "R_mid", "right base": "R_base",
    "left apex": "L_apical", "left upper zone": "L_upper", "left mid zone": "L_mid", "left base": "L_base",
    "right upper lobe": "R_upper_lobe", "right middle lobe": "R_middle_lobe", "right lower lobe": "R_lower_lobe",
    "left upper lobe": "L_upper_lobe", "left lower lobe": "L_lower_lobe",
    "right lung": "R_lung", "left lung": "L_lung", "heart": "heart", "upper mediastinum": "upper_mediastinum",
    "mediastinum": "cardiomediastinum", "trachea": "trachea", "aortic arch": "aortic_arch",
    "right hemidiaphragm": "R_hemidiaphragm", "left hemidiaphragm": "L_hemidiaphragm",
    "spine": "spine", "clavicles": "clavicles", "unspecified": None,
}
POL = ["present", "absent", "uncertain"]
LAT = ["right", "left", "bilateral", "none"]
SEV = ["none", "mild", "moderate", "severe", "unspecified"]
CHG = ["new", "worse", "better", "stable", "unspecified"]
SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "findings": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string", "maxLength": 60},
                    "polarity": {"enum": POL},
                    "regions": {"type": "array", "items": {"enum": list(REGION_VOCAB)}, "maxItems": 6},
                    "laterality": {"enum": LAT},
                    "severity": {"enum": SEV},
                    "change": {"enum": CHG}},
                "required": ["finding", "polarity", "regions", "laterality", "severity", "change"]}}},
        "required": ["i", "findings"]}}},
    "required": ["sentences"]}
SYSTEM = "You convert chest X-ray report sentences into structured findings. Reply with JSON only."
RULES = """Rules:
- One entry per distinct finding in a sentence. Sentences about technique, comparison only, or recommendations get an empty list.
- Normal statements become the finding they rule out, polarity absent: "no pleural effusion" -> pleural effusion, absent;
  "lungs are clear" -> airspace opacity, absent, regions right lung + left lung; "heart size is normal" -> cardiomegaly, absent, heart.
- Hedged words (possible, may represent, cannot exclude, likely) -> polarity uncertain.
- Use short canonical finding names: pleural effusion, pneumothorax, consolidation, airspace opacity, atelectasis, pulmonary edema,
  cardiomegaly, nodule, mass, interstitial opacity, hyperinflation, fracture, support device (say which, e.g. support device: endotracheal tube).
- regions only from the allowed list. Mapping hints: hilum -> mid zone of that side; costophrenic angle or pleural effusion -> base of that side;
  lingula -> left upper lobe; bilateral findings list both sides; whole-lung findings -> right lung and/or left lung; unknown -> unspecified.
- change: new / worse / better / stable when the sentence compares with a prior, else unspecified.
Allowed regions: """ + ", ".join(REGION_VOCAB)

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_sentences(text):
    if text is None or (isinstance(text, float) and np.isnan(text)):  # parquet turns missing text into NaN
        return []
    t = re.sub(r"\s+", " ", str(text)).strip()
    if not t:
        return []
    out = []
    for p in SENT_RE.split(t):
        p = p.strip()
        if not p:
            continue
        if out and (re.fullmatch(r"\d+\.", out[-1]) or len(out[-1]) < 4):
            out[-1] = (out[-1] + " " + p).strip()
        else:
            out.append(p)
    return out


def build_messages(sents):
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sents))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{RULES}\n\nSentences:\n{numbered}\n\nReturn {{\"sentences\": [...]}} with one item per sentence index."}]


def parse_output(raw, n):
    """Return list (len n) of findings lists, plus a validity flag."""
    try:
        obj = json.loads(raw)
        items = {int(x["i"]): x.get("findings", []) for x in obj.get("sentences", [])}
        res = []
        for i in range(n):
            fl = []
            for f in items.get(i, []):
                regs = [REGION_VOCAB[r] for r in f.get("regions", []) if REGION_VOCAB.get(r)]
                fl.append({"finding": str(f.get("finding", "")).strip().lower(), "polarity": f.get("polarity"),
                           "regions": sorted(set(regs)), "laterality": f.get("laterality"),
                           "severity": f.get("severity"), "change": f.get("change")})
            res.append(fl)
        return res, all(i in items for i in range(n))
    except Exception:
        return [[] for _ in range(n)], False


def pick_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):          # chain already chose a GPU with enough free memory
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", os.environ.get("ANCHOR_GPUS", "0,1"), "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return min((tuple(int(v) for v in l.split(",")) for l in out.strip().splitlines()), key=lambda r: r[1])[0]
    except Exception:
        return None


def make_vllm(model, gpu_util, max_len):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=max_len,
              enable_prefix_caching=True)
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


def run(df, generate, out_dir, shard, log_every=1):
    """df: key, findings, impression. Writes sentence-level parquet shards; returns stats."""
    t0, n_done, stats = time.time(), 0, {"reports": 0, "sentences": 0, "valid_json": 0, "truncated": 0}
    for s0 in range(0, len(df), shard):
        sp = out_dir / f"shard_{s0 // shard:05d}.parquet"
        if sp.exists():
            continue
        part = df.iloc[s0:s0 + shard]
        jobs = []
        for r in part.itertuples(index=False):
            sents = [("findings", s) for s in split_sentences(r.findings)] + [("impression", s) for s in split_sentences(r.impression)]
            jobs.append((r.key, sents))
        msgs = [build_messages([s for _, s in sents]) for _, sents in jobs if sents]
        outs = iter(generate(msgs)) if msgs else iter([])
        rows = []
        for key, sents in jobs:
            stats["reports"] += 1
            if not sents:
                continue
            raw, finish = next(outs)
            parsed, ok = parse_output(raw, len(sents))
            stats["valid_json"] += int(ok); stats["truncated"] += int(finish == "length")
            for i, ((section, text), fl) in enumerate(zip(sents, parsed)):
                rows.append({"key": key, "section": section, "sid": i, "sentence": text,
                             "findings_json": json.dumps(fl), "n_findings": len(fl), "report_valid": ok})
            stats["sentences"] += len(sents)
        pd.DataFrame(rows).to_parquet(sp, index=False)
        n_done += len(part)
        print(time.strftime("[%H:%M:%S] ") + f"{sp.name}: {n_done}/{len(df)} reports, "
              f"{n_done / max(time.time() - t0, 1e-6):.2f} reports/s", flush=True)
    return stats


def label_sheet(out_dir, path, n=200, seed=0):
    s = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("shard_*.parquet"))], ignore_index=True)
    with_f = s[s.n_findings > 0].sample(n=min(int(n * 0.7), (s.n_findings > 0).sum()), random_state=seed)
    without = s[s.n_findings == 0].sample(n=min(n - len(with_f), (s.n_findings == 0).sum()), random_state=seed)
    pick = pd.concat([with_f, without]).sample(frac=1, random_state=seed).reset_index(drop=True)

    def show(js):
        return " || ".join(f"{f['finding']} | {f['polarity']} | {','.join(f['regions']) or '-'} | {f['laterality']} | {f['change']}"
                           for f in json.loads(js)) or "(no findings)"
    sheet = pd.DataFrame({"row": range(len(pick)), "sentence": pick.sentence, "extraction": pick.findings_json.map(show),
                          "finding_and_polarity_ok": "", "regions_ok": "", "laterality_ok": "", "missed_finding": "", "note": ""})
    sheet.to_csv(path, index=False)
    return len(sheet)


def main(argv=None, generate=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="./models/Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--manifest", default=str(HOME / "work/manifest"))
    ap.add_argument("--out", default=str(HOME / "work/corpus"))
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--gpu-util", type=float, default=0.45)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--label", type=int, default=0, help="write an N-sentence labelling sheet after the run")
    a = ap.parse_args(argv)

    man = pd.read_parquet(pathlib.Path(a.manifest) / f"rex_{a.split}.parquet")
    man = man[man.eligible].sort_values("key").reset_index(drop=True)
    if a.limit:
        man = man.sample(n=min(a.limit, len(man)), random_state=1).sort_values("key").reset_index(drop=True)
    tag = f"{a.split}_pilot{a.limit}" if a.limit else a.split
    out = pathlib.Path(a.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{tag}: {len(man)} reports", flush=True)
    if generate is None:
        g = pick_gpu()
        if g is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(g); print(f"using GPU {g}", flush=True)
        assert pathlib.Path(a.model).exists(), f"model not found: {a.model}"
        generate = make_vllm(a.model, a.gpu_util, a.max_len)
    t0 = time.time()
    stats = run(man[["key", "findings", "impression"]], generate, out, a.shard)
    el = time.time() - t0
    s = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("shard_*.parquet"))], ignore_index=True)
    fl = [f for js in s.findings_json for f in json.loads(js)]
    summ = {"reports_run_now": stats["reports"], "reports_per_s": round(stats["reports"] / max(el, 1e-6), 2),
            "sentences_total": len(s), "report_valid_json_rate": round(float(s.groupby("key").report_valid.first().mean()), 4),
            "truncated_now": stats["truncated"], "sentences_with_findings": round(float((s.n_findings > 0).mean()), 3),
            "findings_total": len(fl),
            "polarity": pd.Series([f["polarity"] for f in fl]).value_counts(normalize=True).round(3).to_dict(),
            "with_region": round(float(np.mean([bool(f["regions"]) for f in fl])) if fl else 0, 3),
            "top_findings": pd.Series([f["finding"] for f in fl]).value_counts().head(15).to_dict(),
            "region_counts": pd.Series([r for f in fl for r in f["regions"]]).value_counts().to_dict()}
    if stats["reports"]:
        summ["est_hours_full_train_107637"] = round(107637 / max(summ["reports_per_s"], 1e-6) / 3600, 1)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    if a.label:
        path = HOME / f"work/e1_label_{a.label}.csv"
        k = label_sheet(out, path, a.label)
        print(f"labelling sheet: {k} sentences -> {path} (local only)")
    print("CORPUS_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
