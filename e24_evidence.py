#!/usr/bin/env python3
"""E24 (round 48): WHAT KIND OF IMAGE EVIDENCE VERIFIES A CLAIM? (Prof. Jang, point 2.) Validation only; --test GATED.
~/venvs/anchor, small GPU for the probe (side resolution).

Same candidate pool (frozen ANCHOR pool), same recipe for every variant: logistic scorer fitted on fold 0, focal offset
(frozen delta), CRC threshold on fold 1 calibrated on STATED claims (E23), evaluation on folds 2-3 (3,330 studies).
Only the admission evidence changes:
  PRIOR      anatomical/population prior only: log training frequency of the finding (no image)
  GLOBAL-P   prior + whole-image probe head (CLS token)
  REGION-P   prior + region probe head (max over visible anatomical regions)
  BOTH-P     prior + global + region probe heads
  GLOBAL-N   prior + whole-image retrieval evidence (votes / rank among global neighbours)
  REGION-N   prior + per-region retrieval evidence (votes / rank among anatomically matched neighbours)
  ANCHOR     frozen ANCHOR scorer (all evidence + draft flag)
WHERE (laterality) on lateralised findings, for ANCHOR's admitted claims: region resolver vs keeping the proposer's side
vs the population-majority side of the finding (strict precision and strict recall).
PRE-STATED (before running): H24a REGION-P > GLOBAL-P in recall at matched certified risk (patient-cluster CI > 0);
H24b REGION-N > GLOBAL-N likewise; H24c PRIOR alone is far below every image variant; H24d the region resolver beats
both side baselines in strict precision. Disproof: CI of H24a or H24b includes 0 or is negative.
Writes realised reports for GLOBAL-P, REGION-P, BOTH-P (work/gen/e24*_valid.parquet) for CRIMSON. Aggregates only."""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd

import e2x_common as X

E4, A5, F, D = X.E4, X.A5, X.F, X.D
VARS = {"prior": ["logprior"], "globalP": ["logprior", "probe_global"], "regionP": ["logprior", "probe_region"],
        "bothP": ["logprior", "probe_global", "probe_region"],
        "globalN": ["logprior", "votes_global", "inv_global_rank"],
        "regionN": ["logprior", "votes_region_max", "votes_region_mean", "inv_region_rank"]}


def train_prior():
    corp = E4.load_claims(X.HOME / "work/corpus_v2/train")
    n = len(corp); fc, side = Counter(), {}
    for k, cl in corp.items():
        seen = set()
        for (f, l) in cl:
            if f not in seen:
                fc[f] += 1; seen.add(f)
            side.setdefault(f, Counter())[l] += 1
    prior = {f: np.log((c + 1) / (n + 2)) for f, c in fc.items()}
    major = {f: max((s for s in c if s in ("left", "right", "bilateral")), key=lambda s: c[s], default="none") for f, c in side.items()}
    return prior, float(np.log(1 / (n + 2))), major


def run_split(split, C, fz, models, prior_floor, keys_eval, ref, probe, thresholds, tops):
    out, adm_all = {}, {}
    for nm in list(VARS) + ["anchor"]:
        m = fz["scorer_full"] if nm == "anchor" else models[nm]
        c = C.assign(score=F.apply_scorer(C, m, fz["delta"]))
        fin = X.final_claims(c, probe, fz["tau"], ref)
        if nm not in thresholds:
            grid = np.quantile(c.score, np.linspace(1, 0, 600))
            thresholds[nm] = X.calibrate(fin[fin.fold == 1], grid, "valid", sorted(C.key.unique()), tops)[0]
        e = fin[fin.key.isin(set(keys_eval))]
        auc = E4.auroc(e.score, e.label)
        adm = e[e.score >= thresholds[nm]]
        r = X.claim_summary(f"{nm} (claim AUROC {auc:.3f})", adm, ref, keys_eval, split, tops)
        r["claim_auroc"] = round(auc, 4); r["t"] = float(thresholds[nm]); out[nm] = r; adm_all[nm] = adm
    return out, adm_all


def side_analysis(C, fz, probe, ref, keys_eval, major):
    c = C[C.key.isin(set(keys_eval))].assign(score=F.apply_scorer(C[C.key.isin(set(keys_eval))], fz["scorer_full"], fz["delta"]))
    adm = c[(c.score >= fz["t_full"]) & c.finding.isin(D.LATF)].drop_duplicates(["key", "finding", "lat"])
    res = {}
    ref = {k: {r: v for r, v in d.items() if r[0] in D.LATF} for k, d in ref.items()}
    for nm, fn in [("region resolver", lambda k, f, l: D.resolve(probe, k, f, l, fz["tau"])), ("proposer side", lambda k, f, l: l),
                   ("population-majority side", lambda k, f, l: major.get(f, "none"))]:
        a = adm.assign(lat=[fn(k, f, l) for k, f, l in zip(adm.key, adm.finding, adm.lat)]).drop_duplicates(["key", "finding", "lat"])
        ms = D.metrics(a, ref, keys_eval, strict=True)
        res[nm] = {"strict_precision": ms["precision"], "strict_recall": ms["recall"]}
        print(f"  {nm:<26} lateralised items: strict precision {ms['precision']:.3f}, strict recall {ms['recall']:.3f}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true", help="GATED: post hoc test look")
    a = ap.parse_args(argv)
    fz = json.load(open(F.OUT / "freeze.json"))
    prior, floor, major = train_prior()
    C = F.load_cands("valid"); C["logprior"] = C.finding.map(prior).fillna(floor)
    keys_all = sorted(C.key.unique()); ek = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key)
    ref = E4.load_claims(X.HOME / "work/corpus_v2/valid"); probe = F.probe_arrays("valid", keys_all)
    models = {nm: F.fit(C, feats) for nm, feats in VARS.items()}
    print("=== E24 VALID folds 2-3: admission evidence at matched certified risk (alpha 0.10, stated claims) ===")
    thresholds = {}
    res, adm = run_split("valid", C, fz, models, floor, ek, ref, probe, thresholds, fz["group_tops"])
    pat = X.patients("valid", ek)

    def cov(adm_):
        by = {k: list(zip(g.finding, g.lat)) for k, g in adm_.groupby("key")}
        c = np.array([sum(E4.matches(r, by.get(k, [])) for r in ref.get(k, {})) for k in ek], float)
        return c
    n = np.array([len(ref.get(k, {})) for k in ek], float)
    covs = {nm: cov(v) for nm, v in adm.items()}
    print("\npaired recall differences (patient-cluster 95% CI):")
    res["contrasts"] = {}
    for a_, b_ in [("globalP", "regionP"), ("globalP", "bothP"), ("globalN", "regionN"), ("prior", "regionP"), ("bothP", "anchor")]:
        lo, hi = X.cluster_ratio_diff(covs[a_], covs[b_], n, n, pat)
        d = covs[b_].sum() / n.sum() - covs[a_].sum() / n.sum()
        res["contrasts"][f"{b_}-{a_}"] = [d, lo, hi]
        print(f"  {b_} minus {a_}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("\nWHERE: laterality of ANCHOR's admitted lateralised claims (folds 2-3)")
    res["side"] = side_analysis(C, fz, probe, ref, ek, major)
    for nm in ["globalP", "regionP", "bothP", "prior"]:
        X.write_reports(f"e24{nm.lower()}", "valid", adm[nm], ek)
    json.dump({"thresholds": thresholds, "models": models}, open(X.HOME / "work/e8/e24_models.json", "w"), indent=1, default=float)
    if a.test:
        print("\n=== TEST (GATED, post hoc; thresholds from valid) ===")
        Ct = F.load_cands("test"); Ct["logprior"] = Ct.finding.map(prior).fillna(floor)
        kt = sorted(Ct.key.unique()); reft = E4.load_claims(X.HOME / "work/corpus_v2/test"); pt = F.probe_arrays("test", kt)
        res["test"], admt = run_split("test", Ct, fz, models, floor, kt, reft, pt, thresholds, fz["group_tops"])
        for nm in ["globalP", "regionP", "bothP"]:
            X.write_reports(f"e24{nm.lower()}", "test", admt[nm], kt)
    json.dump(res, open(X.HOME / "work/e8/e24_results.json", "w"), indent=1, default=float)
    print("E24_DONE")


if __name__ == "__main__":
    main()
