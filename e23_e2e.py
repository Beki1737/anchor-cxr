#!/usr/bin/env python3
"""E23 (round 48): END-TO-END CERTIFICATE (answers Prof. Jang, point 1: the guarantee must cover the final admitted
findings, including laterality). ~/venvs/anchor, small GPU for the probe. Validation by default; --test is GATED.

Observation behind it: the side resolver maps each candidate to a stated claim using only the image (region probe),
never the threshold. Hence the stated set at threshold t is A(t) = {rho(c): s(c) >= t}, nested in t, and equal to
{a: s~(a) >= t} with s~(a) = max over candidates resolved to a. The number of FALSE STATED findings is therefore
monotone in t, and CRC calibrated on stated-claim labels certifies exactly what the report says.
This script:
  1. builds the stated-claim table (resolve all candidates, relabel, collapse) for ANCHOR and FUS on valid;
  2. re-calibrates the marginal and multi-group thresholds on fold 1 with stated-claim labels;
  3. compares them with the frozen thresholds (calibrated on candidate labels) and reports how many admitted claims
     change label through resolution;
  4. evaluates both on folds 2-3 (realised risk on stated claims, recall, precision, worst subgroup).
PRE-STATED: if the end-to-end thresholds equal the frozen ones, the frozen test results are already certified end to
end and no new test look is needed. If they differ, the end-to-end variant needs ONE labelled test look (--test, gated).
Aggregates only."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import e2x_common as X

E4, A5, F = X.E4, X.A5, X.F


def label_flips(c_res, c_raw_labels):
    return int((c_res & ~c_raw_labels).sum()), int((~c_res & c_raw_labels).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--test", action="store_true", help="GATED: one labelled test look")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(X.CODE)); import e16_fusion as FU
    fz = json.load(open(F.OUT / "freeze.json")); fz2 = json.load(open(FU.OUT / "freeze2.json"))
    print(f"freeze.json {X.md5(F.OUT / 'freeze.json')} | freeze2.json {X.md5(FU.OUT / 'freeze2.json')}")
    C = F.load_cands("valid"); C["score"] = F.apply_scorer(C, fz["scorer_full"], fz["delta"])
    ref = E4.load_claims(X.HOME / "work/corpus_v2/valid"); keys_all = sorted(C.key.unique())
    probe = F.probe_arrays("valid", keys_all)
    ek = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key)
    res = {}
    # candidate-label view of the same candidates (what the frozen calibration used)
    raw_lab = np.array([E4.matches((f, l), set(ref.get(k, {}))) for k, f, l in zip(C.key, C.finding, C.lat)])
    fin = X.final_claims(C, probe, fz["tau"], ref)
    n_side = int((fin.lat != fin.lat0).sum())
    print(f"\nANCHOR valid: {len(C)} candidates -> {len(fin)} stated claims; side changed by the resolver for {n_side} claims")
    cal_raw = C[C.fold == 1].assign(label=raw_lab[(C.fold == 1).to_numpy()])
    adm_raw = cal_raw[cal_raw.score >= fz["t_full"]]
    adm_fin = fin[(fin.fold == 1) & (fin.score >= fz["t_full"])]
    print(f"fold 1 at the frozen threshold: candidate-label FP {int((~adm_raw.label.astype(bool)).sum())} vs stated-claim FP "
          f"{int((~adm_fin.label).sum())} (claims {len(adm_raw)} vs {len(adm_fin)})")
    grid = np.quantile(C.score, np.linspace(1, 0, 600))
    # monotonicity check on fold 1 (stated claims)
    fpv = [int((~fin[(fin.fold == 1) & (fin.score >= t)].label).sum()) for t in grid[::20]]
    print(f"monotonicity of stated-claim FP over the threshold grid (fold 1): {'OK' if all(np.diff(fpv) >= 0) else 'VIOLATED'}")
    t_e, g_e = X.calibrate(fin[fin.fold == 1], grid, "valid", keys_all, fz["group_tops"])
    same_t = abs(t_e - fz["t_full"]) < 1e-12
    diffg = {k: (fz["group_thresholds"].get(k), v) for k, v in g_e.items() if abs(v - fz["group_thresholds"].get(k, np.nan)) > 1e-12}
    print(f"marginal threshold: frozen {fz['t_full']:.6f} | end-to-end {t_e:.6f} -> {'IDENTICAL' if same_t else 'DIFFERENT'}")
    print(f"group thresholds: {len(g_e) - len(diffg)}/{len(g_e)} identical" + (f"; differ: {diffg}" if diffg else ""))
    print("\n=== VALID folds 2-3 (stated claims, after side resolution) ===")
    th_f = X.multi_thresholds("valid", ek, fz["group_thresholds"], fz["t_full"], fz["group_tops"])
    th_e = X.multi_thresholds("valid", ek, g_e, t_e, fz["group_tops"])
    res["anchor"] = {"frozen": X.claim_summary("ANCHOR frozen", X.admit(fin, ek, fz["t_full"]), ref, ek, "valid", fz["group_tops"]),
                     "e2e": X.claim_summary("ANCHOR end-to-end", X.admit(fin, ek, t_e), ref, ek, "valid", fz["group_tops"]),
                     "multi_frozen": X.claim_summary("ANCHOR-multi frozen", X.admit(fin, ek, th_f), ref, ek, "valid", fz["group_tops"]),
                     "multi_e2e": X.claim_summary("ANCHOR-multi end-to-end", X.admit(fin, ek, th_e), ref, ek, "valid", fz["group_tops"]),
                     "t_frozen": fz["t_full"], "t_e2e": t_e, "identical": bool(same_t and not diffg), "group_e2e": g_e}
    # FUS
    _, fu = FU.build("valid", C, probe, ref, fz2["tau"])
    fu["score"] = F.apply_scorer(fu, fz2["scorer_fus"], fz2["delta"])
    finf = X.final_claims(fu, probe, fz2["tau"], ref)
    gridf = np.quantile(fu.score, np.linspace(1, 0, 600))
    tf_e, gf_e = X.calibrate(finf[finf.fold == 1], gridf, "valid", keys_all, fz2["group_tops"])
    dff = {k: (fz2["group_thresholds_fus"].get(k), v) for k, v in gf_e.items() if abs(v - fz2["group_thresholds_fus"].get(k, np.nan)) > 1e-12}
    print(f"\nFUS: marginal frozen {fz2['t_fus']:.6f} | end-to-end {tf_e:.6f}; group thresholds identical {len(gf_e) - len(dff)}/{len(gf_e)}"
          + (f"; differ: {dff}" if dff else ""))
    thf_f = X.multi_thresholds("valid", ek, fz2["group_thresholds_fus"], fz2["t_fus"], fz2["group_tops"])
    thf_e = X.multi_thresholds("valid", ek, gf_e, tf_e, fz2["group_tops"])
    res["fus"] = {"multi_frozen": X.claim_summary("FUS-multi frozen", X.admit(finf, ek, thf_f), ref, ek, "valid", fz2["group_tops"]),
                  "multi_e2e": X.claim_summary("FUS-multi end-to-end", X.admit(finf, ek, thf_e), ref, ek, "valid", fz2["group_tops"]),
                  "t_frozen": fz2["t_fus"], "t_e2e": tf_e, "identical": bool(abs(tf_e - fz2["t_fus"]) < 1e-12 and not dff), "group_e2e": gf_e}
    verdict = res["anchor"]["identical"] and res["fus"]["identical"]
    print(f"\nVERDICT: {'frozen thresholds ARE the end-to-end thresholds; existing test results are certified end to end' if verdict else 'thresholds differ: end-to-end variant needs one labelled test look (--test, gated)'}")
    z = {"anchor_t": t_e, "anchor_groups": g_e, "fus_t": tf_e, "fus_groups": gf_e}
    json.dump(z, open(X.HOME / "work/e8/freeze_e2e.json", "w"), indent=1)
    print(f"freeze_e2e.json md5 {X.md5(X.HOME / 'work/e8/freeze_e2e.json')} (valid only)")
    if a.test:
        print("\n=== TEST (GATED; post-selection look at the end-to-end thresholds) ===")
        Ct = F.load_cands("test"); Ct["score"] = F.apply_scorer(Ct, fz["scorer_full"], fz["delta"])
        reft = E4.load_claims(X.HOME / "work/corpus_v2/test"); kt = sorted(Ct.key.unique())
        pt = F.probe_arrays("test", kt); fint = X.final_claims(Ct, pt, fz["tau"], reft)
        tops = fz["group_tops"]
        res["test"] = {"anchor_frozen": X.claim_summary("ANCHOR frozen", X.admit(fint, kt, fz["t_full"]), reft, kt, "test", tops),
                       "anchor_e2e": X.claim_summary("ANCHOR end-to-end", X.admit(fint, kt, t_e), reft, kt, "test", tops),
                       "multi_e2e": X.claim_summary("ANCHOR-multi end-to-end", X.admit(fint, kt, X.multi_thresholds("test", kt, g_e, t_e, tops)), reft, kt, "test", tops)}
    json.dump(res, open(X.HOME / "work/e8/e23_results.json", "w"), indent=1, default=float)
    print("E23_DONE")


if __name__ == "__main__":
    main()
