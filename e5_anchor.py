#!/usr/bin/env python3
"""E5: full ANCHOR pipeline on the valid primary cohort (~/venvs/anchor). Propose broadly, admit selectively, with a
conformal risk control (CRC) guarantee on false claims per report.

  build    candidates for ALL valid studies (E4b builder: VLM claims U top-8 region U top-8 global; neighbour votes)
           -> then run e4c_probe.py score --dir e5 for the trained region-probe scores
  run      patients split by hash into 4 folds: fold 0 fits the claim scorer (logistic on votes + probe),
           fold 1 calibrates CRC thresholds, folds 2-3 are the evaluation set (never used for fitting or calibration)
           CRC: lambda_hat = most permissive threshold with n/(n+1) * R_hat + 1/(n+1) <= alpha, for losses
             any : 1[report contains >= 1 false claim]            (alpha = target probability)
             fp  : min(#false claims, 3) / 3                       (alpha = target mean capped false claims / 3)
           writes generated reports (deterministic realizer: admitted claims + normal statements for uncovered
           structures; with no admitted claim the report equals the normal template) for:
             e5anchor_any15 / e5anchor_any30 / e5anchor_fp15 / e5anchor_fp30  (operating points)
             e5b0real   (MedGemma B0 claims through the same realizer: isolates realizer from admission)
             e5oracle   (only true candidates admitted: upper bound of the pool)
           and prints claim-level recall / precision / false claims per report + realized risk vs alpha on eval.
Aggregates only. Reports stay local.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e4b_verify as E4B  # noqa: E402

OUT = HOME / "work/e5"
LUNG = {"airspace opacity", "consolidation", "pulmonary edema", "interstitial opacity", "atelectasis", "nodule", "mass"}
PHR = {"pleural effusion": "{lat} pleural effusion", "pneumothorax": "{lat} pneumothorax", "consolidation": "{lat} consolidation",
       "airspace opacity": "{lat} airspace opacity", "atelectasis": "{lat} atelectasis", "pulmonary edema": "pulmonary edema",
       "cardiomegaly": "cardiomegaly", "nodule": "{lat} pulmonary nodule", "mass": "{lat} mass",
       "interstitial opacity": "{lat} interstitial opacity", "hyperinflation": "hyperinflation of the lungs",
       "fracture": "{lat} fracture", "degenerative changes": "degenerative changes of the spine",
       "calcified granuloma": "{lat} calcified granuloma", "aortic calcification": "aortic calcification",
       "scoliosis": "scoliosis", "postsurgical change": "postsurgical change", "support device": "a support device in place",
       "mediastinal widening": "widening of the mediastinum", "elevated hemidiaphragm": "elevation of the {lat} hemidiaphragm",
       "hiatal hernia": "a hiatal hernia", "osteopenia": "osteopenia"}
FEATS = ["from_vlm", "inv_region_rank", "inv_global_rank", "votes_global", "votes_region_max", "votes_region_mean",
         "probe_region", "probe_global"]


def realize(claims):
    """claims: list of (finding, lat). Deterministic, faithful by construction."""
    sents, names = [], set()
    for f, lat in sorted(set(claims)):
        names.add(f)
        side = {"right": "right", "left": "left", "bilateral": "bilateral"}.get(lat, "")
        txt = PHR.get(f, "{lat} " + f).format(lat=side)
        txt = " ".join(txt.split()).replace("elevation of the  hemidiaphragm", "elevation of a hemidiaphragm")
        txt = txt.replace("of the hemidiaphragm", "of a hemidiaphragm")
        sents.append(f"There is {txt}.")
    if not names & LUNG:
        sents.append("The lungs are clear.")
    eff, ptx = "pleural effusion" in names, "pneumothorax" in names
    if not eff and not ptx:
        sents.append("No pleural effusion or pneumothorax.")
    elif not ptx:
        sents.append("No pneumothorax.")
    elif not eff:
        sents.append("No pleural effusion.")
    if not names & {"cardiomegaly", "mediastinal widening"}:
        sents.append("The cardiomediastinal silhouette is normal.")
    return " ".join(sents)


def cmd_build(a):
    E4B.OUT = OUT; OUT.mkdir(parents=True, exist_ok=True)
    E4B.cmd_build(argparse.Namespace(n=10 ** 9, k=a.k, m=a.m, vlm=a.vlm))


def crc_threshold(cal, losses_fn, alpha, grid):
    """Most permissive threshold whose adjusted calibration risk stays <= alpha for it and every stricter threshold."""
    n = cal.key.nunique()
    best = np.inf
    for t in grid:                                   # descending: stricter -> more permissive
        r = losses_fn(cal, t)
        if n / (n + 1) * r + 1 / (n + 1) <= alpha:
            best = t
        else:
            break
    return best


def fp_per_study(df, t, keys):
    adm = df[df.score >= t]
    fp = adm[~adm.label].groupby("key").size()
    return fp.reindex(keys, fill_value=0).to_numpy()


def cmd_run(a):
    df = pd.read_parquet(OUT / "candidates.parquet")
    pr = pd.read_parquet(OUT / "verify_probe.parquet")
    df["probe_region"] = pr.probe_region.to_numpy(); df["probe_global"] = pr.probe_global.to_numpy()
    df[["probe_region", "probe_global"]] = df[["probe_region", "probe_global"]].fillna(0.5)
    df["inv_region_rank"] = 1 / df.region_rank; df["inv_global_rank"] = 1 / df.global_rank
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    fold = np.array([int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 4 for p in df.patient])
    df["fold"] = fold
    X = df[FEATS].astype(float).to_numpy(); y = df.label.astype(float).to_numpy()
    tr = fold == 0
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    w, b = E4B.logreg_fit((X[tr] - mu) / sd, y[tr])
    df["score"] = 1 / (1 + np.exp(-(((X - mu) / sd) @ w + b)))
    cal, ev = df[df.fold == 1], df[df.fold >= 2]
    cal_keys, ev_keys = sorted(cal.key.unique()), sorted(ev.key.unique())
    print(f"E5: {df.key.nunique()} studies | scorer fit {df[tr].key.nunique()} | CRC calibration {len(cal_keys)} | "
          f"evaluation {len(ev_keys)} | eval AUROC {E4.auroc(ev.score, ev.label):.3f}")
    grid = np.quantile(df.score, np.linspace(1, 0, 400))
    loss = {"any": lambda d, t: float((fp_per_study(d, t, sorted(d.key.unique())) >= 1).mean()),
            "fp": lambda d, t: float((np.minimum(fp_per_study(d, t, sorted(d.key.unique())), 3) / 3).mean())}
    points = {"any15": ("any", 0.15), "any30": ("any", 0.30), "fp15": ("fp", 0.15 / 3), "fp30": ("fp", 0.30 / 3)}
    systems = {}
    res = {"eval_studies": len(ev_keys), "crc": {}}
    for name, (kind, alpha) in points.items():
        t = crc_threshold(cal, loss[kind], alpha, grid)
        realized = loss[kind](ev, t)
        res["crc"][name] = {"loss": kind, "alpha": round(alpha, 4), "threshold": float(t), "eval_risk": round(realized, 4),
                            "guarantee_met": bool(realized <= alpha)}
        systems[f"e5anchor_{name}"] = ev[ev.score >= t]
    systems["e5b0real"] = ev[ev.from_vlm]
    systems["e5oracle"] = ev[ev.label]
    print(f"\n{'system':<20}{'recall':>8}{'precision':>11}{'claims/rpt':>12}{'false/rpt':>11}{'P(any false)':>14}   CRC")
    nref = sum(len(ref.get(k, {})) for k in ev_keys)
    gen_dir = HOME / "work/gen"; gen_dir.mkdir(parents=True, exist_ok=True)
    for name, adm in systems.items():
        cov = E4B.covered(adm, ref) if len(adm) else 0
        fp = fp_per_study(adm.assign(score=1.0), 0.5, ev_keys)
        row = {"recall": round(cov / max(nref, 1), 4), "precision": round(float(adm.label.mean()) if len(adm) else float("nan"), 4),
               "claims_per_report": round(len(adm) / len(ev_keys), 3), "false_per_report": round(float(fp.mean()), 3),
               "p_any_false": round(float((fp >= 1).mean()), 4)}
        res[name] = row
        tag = name.replace("e5anchor_", "")
        crc = res["crc"].get(tag)
        crc_s = f"alpha={crc['alpha']:.3f} eval={crc['eval_risk']:.3f} {'OK' if crc['guarantee_met'] else 'VIOLATED'}" if crc else ""
        print(f"{name:<20}{row['recall']:>8.3f}{row['precision']:>11.3f}{row['claims_per_report']:>12.2f}"
              f"{row['false_per_report']:>11.3f}{row['p_any_false']:>14.3f}   {crc_s}")
        by = {k: list(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
        gen = pd.DataFrame({"key": ev_keys, "generated": [realize(by.get(k, [])) for k in ev_keys]})
        gen.to_parquet(gen_dir / f"{name}_valid.parquet", index=False)
    pd.DataFrame({"key": ev_keys}).to_parquet(OUT / "eval_keys.parquet", index=False)
    print(f"\nscorer weights (standardised): " + ", ".join(f"{f}={v:+.2f}" for f, v in zip(FEATS, w)))
    json.dump(res, open(OUT / "e5_results.json", "w"), indent=1)
    print(f"template check: realize([]) == template -> {realize([]) == 'The lungs are clear. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is normal.'}")
    print("E5_RUN_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "run"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--vlm", default="b0_medgemma4b_valid")
    a = ap.parse_args(argv)
    {"build": cmd_build, "run": cmd_run}[a.stage](a)


if __name__ == "__main__":
    main()
