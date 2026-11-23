#!/usr/bin/env python3
"""D40 test: is the base VLM's gain over the normal template concentrated in normal studies?
Stratifies valid studies by the E1-extracted reference findings (no new model calls) and reruns the paired
MedGemma-B0 vs normal-template comparison per stratum, for GREEN and CRIMSON. Aggregates only.

Strata (from the reference report's extracted findings):
  normal        no finding with polarity present/uncertain (support devices ignored)
  abnormal      at least one present/uncertain finding that is not a support device
  abnormal_key  abnormal AND at least one of the key thoracic findings below
"""
import json
import pathlib

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
KEY = ["pleural effusion", "pneumothorax", "consolidation", "airspace opacity", "pulmonary edema", "cardiomegaly",
       "nodule", "mass", "atelectasis", "fracture", "interstitial opacity"]


def study_strata(corpus_dir):
    s = pd.concat([pd.read_parquet(p, columns=["key", "findings_json"]) for p in sorted(corpus_dir.glob("shard_*.parquet"))])
    rows = {}
    for key, js in zip(s.key, s.findings_json):
        r = rows.setdefault(key, {"abn": False, "key_abn": False})
        for f in json.loads(js):
            name = str(f.get("finding", ""))
            if f.get("polarity") in ("present", "uncertain") and not name.startswith("support device"):
                r["abn"] = True
                if any(k in name for k in KEY):
                    r["key_abn"] = True
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["stratum"] = np.where(~df.abn, "normal", "abnormal")
    return df


def paired(x, a, b, seed=0):
    d = (x[a] - x[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    tie = np.abs(d) <= 1e-9
    out = {"n": int(len(d)), "mean_sys": round(float(x[a].mean()), 4), "mean_template": round(float(x[b].mean()), 4),
           "diff": round(float(d.mean()), 4), "ci95": [round(float(v), 4) for v in np.percentile(boots, [2.5, 97.5])],
           "win": round(float((d > 1e-9).mean()), 4), "tie": round(float(tie.mean()), 4), "loss": round(float((d < -1e-9).mean()), 4)}
    tv = x[a][tie]
    if len(tv):
        out["tie_both_1"] = round(float((tv >= 0.9999).mean() * tie.mean()), 4)
        out["tie_both_0"] = round(float((tv.abs() <= 1e-9).mean() * tie.mean()), 4)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="valid", help="judge tags to merge, e.g. valid,valid_e3")
    ap.add_argument("--out", default="strata_paired.json")
    ap.add_argument("--corpus", default="corpus", help="corpus (v1) or corpus_v2")
    a = ap.parse_args(argv)
    cdir = HOME / "work" / a.corpus / "valid"
    if not list(cdir.glob("shard_*.parquet")):
        print("valid corpus not extracted yet"); return
    st = study_strata(cdir)
    print("strata (valid primary cohort keys with extraction):",
          {"normal": int((st.stratum == "normal").sum()), "abnormal": int((st.stratum == "abnormal").sum()),
           "abnormal_key": int(st.key_abn.sum())})
    res = {}
    for judge in ["green", "crimson"]:
        parts = [pd.read_parquet(HOME / f"work/judge/{t}/{judge}_scores.parquet") for t in a.tags.split(",")
                 if (HOME / f"work/judge/{t}/{judge}_scores.parquet").exists()]
        if not parts:
            continue
        sc = pd.concat(parts).drop_duplicates(["cond", "key"])
        W = sc.pivot_table(index="key", columns="cond", values="score")
        if "normal_template" not in W:
            continue
        for cond in [c for c in W.columns if c not in ("normal_template", "floor_shuffled", "ceiling_self")]:
            w = W[[cond, "normal_template"]].dropna().join(st, how="inner")
            res.setdefault(judge, {})[cond] = {}
            for name in ["all", "normal", "abnormal", "abnormal_key"]:
                x = w if name == "all" else (w[w.key_abn] if name == "abnormal_key" else w[w.stratum == name])
                if len(x) >= 30:
                    r = paired(x.rename(columns={cond: "sys"}), "sys", "normal_template")
                    res[judge][cond][name] = r
                    print(f"{judge:<8} {cond:<22} {name:<13} {json.dumps(r)}")
    json.dump(res, open(HOME / "work/judge/valid" / a.out, "w"), indent=1)
    print("STRATA_DONE")


if __name__ == "__main__":
    main()
