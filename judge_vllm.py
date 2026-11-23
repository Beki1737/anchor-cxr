#!/usr/bin/env python3
"""D34: batched vLLM port of the two LLM-judge metrics (GREEN, CRIMSON), bit-faithful to radeval.

Three stages so that radeval's own prompt builders, tokenizer setup and parsers are reused verbatim,
and only the slow generate() call moves to vLLM:

  pairs   (anchor_eval)  reference/candidate pairs for every condition          -> pairs.parquet
  prep    (anchor_eval)  radeval prompt + chat template -> token ids             -> {judge}_req.parquet, {judge}_cfg.json
  gen     (anchor venv)  vLLM generate from token ids (no re-tokenisation)       -> {judge}_out.parquet
  score   (anchor_eval)  radeval parser on the decoded text                      -> {judge}_scores.parquet, summary
  refcheck(anchor_eval)  original radeval scorer on a small subset, per sample   -> {judge}_radeval_ref.parquet
  compare (any)          port vs radeval: aggregate deltas and per-sample agreement

All files stay in ~/anchor/work/judge/<tag>/ (they contain report text; never upload). Only aggregates print.
"""
import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
TEMPLATE = "The lungs are clear. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal."
REPO = {"green": "StanfordAIMI/GREEN-radllama2-7b", "crimson": "rajpurkarlab/medgemma-4b-it-crimson"}
GREEN_MAX_LEN = 2048                      # radeval GREEN: max_length=2048 (prompt + completion)
CRIMSON_SYSTEM = ("You are an expert radiology evaluator that assesses "
                  "the accuracy of radiology reports.")
CRIMSON_MAX_NEW = 8192
GREEN_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['from'] == 'human' %}\n"
    "{{ '<|user|>\\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'system' %}\n"
    "{{ '<|system|>\\n' + message['value'] + eos_token }}\n"
    "{% elif message['from'] == 'gpt' %}\n"
    "{{ '<|assistant|>\\n'  + message['value'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def jdir(tag):
    d = HOME / "work/judge" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def alias_radeval():
    import sys as _s
    import radeval
    _s.modules.setdefault("RadEval", radeval)


def model_path(repo):
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return snapshot_download(repo)


# ----------------------------------------------------------------------------------------------- pairs
def build_pairs(gen_files, manifest, conds=("ceiling_self", "floor_shuffled", "normal_template")):
    """Same construction as eval_smoke.py: refs in generation-file order, floor = refs shuffled with Random(0)."""
    man = manifest
    first = pd.read_parquet(gen_files[0])
    keys = first.key.tolist()
    refs = man.loc[keys, "findings"].fillna("").tolist()
    sh = refs[:]; random.Random(0).shuffle(sh)
    rows = []
    for cond, hyps in [("ceiling_self", refs), ("floor_shuffled", sh), ("normal_template", [TEMPLATE] * len(refs))]:
        if cond not in conds:
            continue
        rows += [{"cond": cond, "key": k, "ref": r, "hyp": h} for k, r, h in zip(keys, refs, hyps)]
    for f in gen_files:
        g = pd.read_parquet(f)
        cond = "_".join(pathlib.Path(f).stem.split("_")[:-1])  # b0_medgemma4b_dev200 -> b0_medgemma4b
        r = man.loc[g.key, "findings"].fillna("").tolist()
        rows += [{"cond": cond, "key": k, "ref": rr, "hyp": h}
                 for k, rr, h in zip(g.key, r, g.generated.fillna("").tolist())]
    df = pd.DataFrame(rows)
    df.insert(0, "pid", df.cond + "|" + df.key.astype(str))
    assert df.pid.is_unique
    return df


def cmd_pairs(a):
    man = pd.read_parquet(HOME / f"work/manifest/rex_{a.split}.parquet").set_index("key")
    files = [str(HOME / "work/gen" / f) if "/" not in f else f for f in a.gens.split(",")]
    df = build_pairs(files, man, conds=tuple(a.ref_conds.split(",")))
    df.to_parquet(jdir(a.tag) / "pairs.parquet", index=False)
    print("pairs:", df.groupby("cond", sort=False).size().to_dict())


# ----------------------------------------------------------------------------------------------- prep
def green_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, add_eos_token=True, use_fast=True, trust_remote_code=True,
                                        padding_side="left")
    tok.chat_template = GREEN_CHAT_TEMPLATE
    tok.pad_token = tok.eos_token
    tok.clean_up_tokenization_spaces = True
    return tok


def _ids(x):
    if hasattr(x, "keys") and "input_ids" in x.keys():
        x = x["input_ids"]
    x = list(x)
    if x and isinstance(x[0], (list, tuple)):
        x = list(x[0])
    return [int(t) for t in x]


def cmd_prep(a):
    alias_radeval()
    d = jdir(a.tag)
    pairs = pd.read_parquet(d / "pairs.parquet")
    if a.only_conds:
        pairs = pairs[pairs.cond.isin(a.only_conds.split(","))].reset_index(drop=True)
    path = model_path(REPO[a.judge])
    cfg = {"judge": a.judge, "repo": REPO[a.judge], "model_path": path}
    if a.judge == "green":
        from radeval.metrics.green_score.utils import make_prompt
        tok = green_tokenizer(path)
        ids = []
        for r, h in zip(pairs.ref, pairs.hyp):
            text = tok.apply_chat_template([{"from": "human", "value": make_prompt(r, h)}, {"from": "gpt", "value": ""}],
                                           tokenize=False, add_generation_prompt=True)
            ids.append(_ids(tok(text, truncation=True, max_length=GREEN_MAX_LEN)))
        max_new = [max(1, GREEN_MAX_LEN - len(x)) for x in ids]
        cfg.update(dtype="float16", max_model_len=GREEN_MAX_LEN + 8, sampling={"temperature": 0.0})
    else:
        from radeval.metrics.crimson.prompt_parts import build_prompt
        from transformers import AutoTokenizer, GenerationConfig
        tok = AutoTokenizer.from_pretrained(path)
        ids = []
        for r, h in zip(pairs.ref, pairs.hyp):
            p = build_prompt(r, h, patient_context=None, include_significance_examples=False,
                             include_attribute_guidelines=False, include_context_guidelines=False)
            msgs = [{"role": "system", "content": CRIMSON_SYSTEM}, {"role": "user", "content": p}]
            ids.append(_ids(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)))
        try:
            gc = GenerationConfig.from_pretrained(path)
            has_gc = getattr(gc, "_from_model_config", None) is False
        except Exception:
            gc, has_gc = None, False
        samp = {"repetition_penalty": 1.1, "seed": 42}
        if has_gc and a.crimson_decode == "radeval":
            samp["temperature"] = float(gc.temperature or 1.0) if gc.do_sample else 0.0
            if gc.do_sample:
                if gc.top_p is not None: samp["top_p"] = float(gc.top_p)
                if gc.top_k: samp["top_k"] = int(gc.top_k)
            mn = gc.max_new_tokens or CRIMSON_MAX_NEW
        else:
            samp["temperature"] = 0.0
            mn = CRIMSON_MAX_NEW
        longest = max(len(x) for x in ids)
        cfg.update(dtype="bfloat16", max_model_len=int(longest + mn + 16), sampling=samp,
                   radeval_uses_model_generation_config=bool(has_gc), crimson_decode=a.crimson_decode,
                   generation_config={k: getattr(gc, k, None) for k in
                                      ["do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "max_length"]}
                   if gc is not None else None)
        max_new = [int(mn)] * len(ids)
    req = pd.DataFrame({"pid": pairs.pid, "prompt_ids": ids, "max_new": max_new})
    req.to_parquet(d / f"{a.judge}_req.parquet", index=False)
    json.dump(cfg, open(d / f"{a.judge}_cfg.json", "w"), indent=1, default=str)
    L = np.array([len(x) for x in ids])
    print(f"[{a.judge}] {len(req)} requests | prompt tokens p50={int(np.median(L))} max={L.max()} | "
          f"truncated at {GREEN_MAX_LEN}: {int((L >= GREEN_MAX_LEN).sum()) if a.judge == 'green' else 'n/a'}")
    print(f"[{a.judge}] cfg:", json.dumps({k: v for k, v in cfg.items() if k not in ('model_path',)}, default=str))


# ----------------------------------------------------------------------------------------------- gen
def cmd_gen(a):
    from vllm import LLM, SamplingParams
    d = jdir(a.tag)
    cfg = json.load(open(d / f"{a.judge}_cfg.json"))
    req = pd.read_parquet(d / f"{a.judge}_req.parquet")
    arch = json.load(open(pathlib.Path(cfg["model_path"]) / "config.json")).get("architectures", [""])[0]
    hf = json.load(open(pathlib.Path(cfg["model_path"]) / "config.json"))
    kw = dict(model=cfg["model_path"], dtype=cfg["dtype"], gpu_memory_utilization=a.gpu_util,
              max_model_len=cfg["max_model_len"], seed=42)
    s = dict(cfg["sampling"])
    ids_only = a.judge == "green" and not a.green_remote
    if ids_only:
        # GREEN's repo ships custom tokenizer code (needs matplotlib). We already hold radeval's exact token ids,
        # so vLLM never needs a tokenizer: skip it, stop on the model's EOS id, decode later in anchor_eval.
        eos = hf.get("eos_token_id", 2)
        kw.update(skip_tokenizer_init=True, trust_remote_code=False)
        s.update(stop_token_ids=eos if isinstance(eos, list) else [int(eos)], detokenize=False)
    elif a.judge == "green":
        kw["trust_remote_code"] = True
    if "ConditionalGeneration" in arch:
        kw["limit_mm_per_prompt"] = {"image": 0}
    print(f"[{a.judge}] vLLM {arch} dtype={cfg['dtype']} max_model_len={cfg['max_model_len']} ids_only={ids_only}", flush=True)
    llm = LLM(**kw)
    sps = [SamplingParams(max_tokens=int(m), **s) for m in req.max_new]
    prompts = [{"prompt_token_ids": [int(t) for t in x]} for x in req.prompt_ids]
    t0 = time.time()
    outs = llm.generate(prompts, sps, use_tqdm=False)
    dt = time.time() - t0
    rows = [{"pid": p, "out_ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text or "",
             "finish_reason": o.outputs[0].finish_reason} for p, o in zip(req.pid, outs)]
    df = pd.DataFrame(rows)
    df.to_parquet(d / f"{a.judge}_out.parquet", index=False)
    n = len(df)
    print(f"[{a.judge}] {n} generations in {dt:.0f}s = {n / dt:.2f} pairs/s | "
          f"truncated={int((df.finish_reason == 'length').sum())} | out tokens p50={int(df.out_ids.map(len).median())}")
    print("JUDGE_GEN_DONE", flush=True)


# ----------------------------------------------------------------------------------------------- score
def green_parser():
    from radeval.metrics.green_score.green import GREEN
    return GREEN(model_name=None, cpu=True)


def crimson_parse(text):
    from radeval.metrics.crimson.crimson import CRIMSONScore
    sc = object.__new__(CRIMSONScore)
    try:
        res = sc._parse_response(text)
        return res["crimson_score"], res["error_counts"]
    except Exception:
        return float("nan"), {}


def cmd_score(a):
    alias_radeval()
    d = jdir(a.tag)
    pairs = pd.read_parquet(d / "pairs.parquet").set_index("pid")
    req = pd.read_parquet(d / f"{a.judge}_req.parquet").set_index("pid")
    out = pd.read_parquet(d / f"{a.judge}_out.parquet").set_index("pid")
    cfg = json.load(open(d / f"{a.judge}_cfg.json"))
    rows = []
    if a.judge == "green":
        from radeval.metrics.green_score.utils import clean_responses
        tok = green_tokenizer(cfg["model_path"])
        g = green_parser()
        for pid in out.index:
            full = tok.decode([int(t) for t in req.at[pid, "prompt_ids"]] + [int(t) for t in out.at[pid, "out_ids"]],
                              skip_special_tokens=True)
            resp = clean_responses(full)
            sc = g.compute_green(resp)
            cnt = g.compute_error_count(resp)
            rows.append({"pid": pid, "score": np.nan if sc is None else float(sc), "response": resp,
                         **{f"n_{i}": int(c) for i, c in enumerate(cnt)}})
    else:
        for pid in out.index:
            sc, cnt = crimson_parse(out.at[pid, "text"])
            rows.append({"pid": pid, "score": float(sc), "response": out.at[pid, "text"],
                         **{f"n_{k}": int(v) for k, v in cnt.items()}})
    df = pd.DataFrame(rows).set_index("pid").join(pairs[["cond", "key"]])
    df.reset_index().to_parquet(d / f"{a.judge}_scores.parquet", index=False)
    summ = {}
    print(f"\n=== {a.judge.upper()} (vLLM port) per condition ===")
    for c, s in df.groupby("cond", sort=False).score:
        v = s.dropna().to_numpy()
        boots = [np.mean(np.random.default_rng(i).choice(v, len(v))) for i in range(1000)] if len(v) else [np.nan]
        summ[c] = {"mean": round(float(np.mean(v)), 4) if len(v) else None, "n": int(len(s)), "n_failed": int(s.isna().sum()),
                   "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}
        print(f"{c:<18} mean={summ[c]['mean']}  95%CI={summ[c]['ci95']}  n={summ[c]['n']}  parse_failed={summ[c]['n_failed']}")
    json.dump(summ, open(d / f"{a.judge}_summary.json", "w"), indent=1)
    print("JUDGE_SCORE_DONE", flush=True)


# ----------------------------------------------------------------------------------------------- refcheck
def cmd_refcheck(a):
    """Original radeval scorer on the first k pairs of each smoke condition, kept per sample."""
    alias_radeval()
    d = jdir(a.tag)
    pairs = pd.read_parquet(d / "pairs.parquet")
    conds = ["ceiling_self", "floor_shuffled", "normal_template", "b0_medgemma4b"]
    sub = pd.concat([pairs[pairs.cond == c].head(a.k) for c in conds])
    refs, hyps = sub.ref.tolist(), sub.hyp.tolist()
    t0 = time.time()
    extra = {}
    if a.judge == "green":
        from radeval.metrics.green_score.green import GREEN
        g = GREEN(REPO["green"], output_dir=".")
        _, _, scores, _ = g(refs, hyps)
        extra["response"] = list(g.completions)
    else:
        from radeval.metrics.crimson.crimson import CRIMSON
        c = CRIMSON(provider="hf")
        _, _, scores, _ = c(refs, hyps)
    dt = time.time() - t0
    out = pd.DataFrame({"pid": sub.pid.tolist(), "score_radeval": [np.nan if s is None else float(s) for s in scores], **extra})
    out.to_parquet(d / f"{a.judge}_radeval_ref.parquet", index=False)
    print(f"[{a.judge}] radeval reference on {len(out)} pairs in {dt:.0f}s ({dt / len(out):.2f} s/pair)")
    print("JUDGE_REFCHECK_DONE", flush=True)


# ----------------------------------------------------------------------------------------------- compare
def load_smoke(judge):
    """Aggregates from the earlier radeval smoke runs (eval_smoke.py json files)."""
    found = {}
    for f in sorted((HOME / "work/eval").glob("smoke_b0_medgemma4b_dev200*.json")):
        for k, v in json.load(open(f)).items():
            m, c = k.split("|", 1)
            if m == judge and isinstance(v, dict) and judge in v:
                found["b0_medgemma4b" if c == "b0" else c] = v[judge]
    return found


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_compare(a):
    d = jdir(a.tag)
    port = pd.read_parquet(d / f"{a.judge}_scores.parquet").set_index("pid")
    summ = json.load(open(d / f"{a.judge}_summary.json"))
    smoke = load_smoke(a.judge)
    if a.ref_agg:   # radeval aggregates recorded from an earlier log, when the smoke json was overwritten
        smoke = {k: float(v) for k, v in (x.split("=") for x in a.ref_agg.split(","))}
    tol_agg = a.tol_agg
    ok = True
    print(f"\n=== {a.judge.upper()}: vLLM port vs radeval smoke aggregates (n=200 each) ===")
    for c, ref in smoke.items():
        p = summ.get(c, {}).get("mean")
        if p is None:
            continue
        delta = p - ref
        flag = "OK" if abs(delta) <= tol_agg else "DIFF"
        ok &= flag == "OK"
        print(f"{c:<18} radeval={ref:.4f}  port={p:.4f}  delta={delta:+.4f}  {flag}")
    if not smoke:
        print("(no radeval smoke json found for this judge)")
    rf = d / f"{a.judge}_radeval_ref.parquet"
    res = {"aggregate_ok": bool(ok and smoke)}
    if rf.exists():
        rr = pd.read_parquet(rf).set_index("pid")
        r = rr[["score_radeval"]].join(port[["score", "cond"]], how="inner")
        both = r.dropna(subset=["score_radeval", "score"])
        diff = (both.score - both.score_radeval).abs()
        res.update(n=int(len(r)), n_both_valid=int(len(both)), mae=round(float(diff.mean()), 4),
                   identical=round(float((diff < 1e-6).mean()), 4), spearman=round(spearman(both.score, both.score_radeval), 4))
        if "response" in rr.columns:                      # GREEN: radeval keeps the judge text, compare verbatim
            same = rr.response.str.strip() == port.loc[rr.index, "response"].str.strip()
            res["identical_judge_text"] = round(float(same.mean()), 4)
        print(f"\n=== {a.judge.upper()}: per-sample agreement with radeval on {res['n']} pairs ===")
        print(json.dumps(res))
        for c, s in both.groupby("cond"):
            print(f"  {c:<18} radeval={s.score_radeval.mean():.4f} port={s.score.mean():.4f} n={len(s)}")
        res["per_sample_ok"] = bool(res["mae"] <= a.tol_mae)
    verdict = "PORT_VALID" if res.get("aggregate_ok") and res.get("per_sample_ok", True) else "PORT_CHECK_FAILED"
    res["verdict"] = verdict
    json.dump(res, open(d / f"{a.judge}_compare.json", "w"), indent=1)
    print(f"[{a.judge}] {verdict}")


def load_scores(judge, tags):
    """Per-sample scores from several tags (e.g. B0 + template in 'valid', E3 systems in 'valid_e3')."""
    parts = [pd.read_parquet(HOME / "work/judge" / t / f"{judge}_scores.parquet") for t in tags
             if (HOME / "work/judge" / t / f"{judge}_scores.parquet").exists()]
    sc = pd.concat(parts, ignore_index=True)
    return sc.drop_duplicates(["cond", "key"], keep="first")


def cmd_paired(a):
    """Paired comparison of two conditions on the same studies (per-sample scores), bootstrap over studies."""
    d = jdir(a.tag)
    sc = load_scores(a.judge, [a.tag] + [t for t in a.extra_tags.split(",") if t])
    w = sc.pivot_table(index="key", columns="cond", values="score")
    if a.cond_a not in w or a.cond_b not in w:
        print("conditions missing:", a.cond_a, a.cond_b); return
    x = w[[a.cond_a, a.cond_b]].dropna()
    diff = (x[a.cond_a] - x[a.cond_b]).to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = {"judge": a.judge, "a": a.cond_a, "b": a.cond_b, "n": int(len(diff)), "mean_a": round(float(x[a.cond_a].mean()), 4),
           "mean_b": round(float(x[a.cond_b].mean()), 4), "diff": round(float(diff.mean()), 4),
           "ci95": [round(float(lo), 4), round(float(hi), 4)], "p_boot_two_sided": round(float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))), 4),
           "win": round(float((diff > 1e-9).mean()), 4), "tie": round(float((np.abs(diff) <= 1e-9).mean()), 4),
           "loss": round(float((diff < -1e-9).mean()), 4)}
    json.dump(res, open(d / f"{a.judge}_paired_{a.cond_a}_vs_{a.cond_b}.json", "w"), indent=1)
    print("PAIRED", json.dumps(res))


def cmd_diag(a):
    """Judge disagreement on the same pairs: GREEN vs CRIMSON per-sample rank correlation per condition,
    and what the CRIMSON ties between cond_a and cond_b consist of. Aggregates only."""
    d = jdir(a.tag)
    g = pd.read_parquet(d / "green_scores.parquet").set_index("pid")
    c = pd.read_parquet(d / "crimson_scores.parquet").set_index("pid")
    x = g[["score", "cond"]].rename(columns={"score": "green"}).join(c[["score"]].rename(columns={"score": "crimson"}), how="inner")
    x = x[x.cond != "ceiling_self"].dropna()
    out = {"spearman_all_noceiling": round(spearman(x.green, x.crimson), 4), "n": int(len(x))}
    out["spearman_by_cond"] = {k: round(spearman(v.green, v.crimson), 4) for k, v in x.groupby("cond")}
    sys_means = x.groupby("cond")[["green", "crimson"]].mean()
    out["system_rank_green"] = sys_means.green.sort_values(ascending=False).index.tolist()
    out["system_rank_crimson"] = sys_means.crimson.sort_values(ascending=False).index.tolist()
    w = c.join(c.index.to_series().str.split("|", n=1, expand=True).rename(columns={0: "c", 1: "k"}))
    w = w.pivot_table(index="k", columns="c", values="score")
    if a.cond_a in w and a.cond_b in w:
        t = w[[a.cond_a, a.cond_b]].dropna()
        tie = (t[a.cond_a] - t[a.cond_b]).abs() <= 1e-9
        tv = t[a.cond_a][tie]
        out["crimson_ties"] = {"tie_rate": round(float(tie.mean()), 4), "of_ties_both_1": round(float((tv >= 0.9999).mean()), 4),
                               "of_ties_both_0": round(float((tv.abs() <= 1e-9).mean()), 4),
                               "of_ties_other": round(float(((tv < 0.9999) & (tv.abs() > 1e-9)).mean()), 4)}
        rest = t[~tie]
        out["crimson_nonties"] = {"n": int(len(rest)), "mean_a": round(float(rest[a.cond_a].mean()), 4),
                                  "mean_b": round(float(rest[a.cond_b].mean()), 4)}
    json.dump(out, open(d / "judge_diag.json", "w"), indent=1)
    print("DIAG", json.dumps(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pairs", "prep", "gen", "score", "refcheck", "compare", "paired", "diag"])
    ap.add_argument("--judge", choices=["green", "crimson"], default="green")
    ap.add_argument("--tag", default="dev200")
    ap.add_argument("--gens", default="b0_medgemma4b_dev200.parquet,b0_lingshu7b_dev200.parquet,b0_qwen25vl7b_dev200.parquet")
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--crimson-decode", choices=["radeval", "greedy"], default="radeval")
    ap.add_argument("--split", default="train", help="manifest split the generations belong to")
    ap.add_argument("--ref-conds", default="ceiling_self,floor_shuffled,normal_template")
    ap.add_argument("--green-remote", action="store_true", help="fallback: let vLLM load GREEN's own tokenizer code")
    ap.add_argument("--cond-a", default="b0_medgemma4b")
    ap.add_argument("--cond-b", default="normal_template")
    ap.add_argument("--ref-agg", default="", help="cond=value,... radeval aggregates to compare against")
    ap.add_argument("--only-conds", default="", help="prep: restrict to these conditions")
    ap.add_argument("--extra-tags", default="", help="paired: also read scores from these tags")
    ap.add_argument("--k", type=int, default=16, help="refcheck pairs per condition")
    ap.add_argument("--tol-agg", type=float, default=0.02)
    ap.add_argument("--tol-mae", type=float, default=0.03)
    a = ap.parse_args(argv)
    {"pairs": cmd_pairs, "prep": cmd_prep, "gen": cmd_gen, "score": cmd_score,
     "refcheck": cmd_refcheck, "compare": cmd_compare, "paired": cmd_paired, "diag": cmd_diag}[a.stage](a)


if __name__ == "__main__":
    main()
