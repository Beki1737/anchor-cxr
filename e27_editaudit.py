#!/usr/bin/env python3
"""E27 (round 50): does the stated-claim certificate cover the EDITED report (ANCHOR-edit)?  ~/venvs/anchor, CPU
(small GPU only if probe arrays need it). Validation folds 2-3 and MIMIC only; never the ReXGradient test split.

ANCHOR-edit keeps a draft sentence only if every positive claim in it matches an admitted stated claim (same finding,
same side, or no side stated); it appends realised sentences for admitted claims not covered. In claim space the
edited text should therefore assert nothing that ANCHOR did not admit. Two gaps: (i) the extractor may read a kept
sentence differently from the gating step (descriptors, comparisons, wording), (ii) a draft that states one finding
twice, with and without a side, yields two claims where ANCHOR admitted one.

Input: claims extracted from e8edit_<split>.parquet with corpus_extract_v2 gentext (r49), admitted stated claims of
frozen ANCHOR (marginal threshold, the gate used for ANCHOR-edit; multi printed as a check).

PRE-STATED (round 50, before any run):
 H27  valid folds 2-3: realised risk of the edited text (capped FP of its extracted positive claims) <= realised
      stated-claim risk of ANCHOR + 0.005, AND reports whose edited text has more false findings than ANCHOR's
      admitted claims <= 1%.
 Disproof: either bound fails -> ANCHOR-edit is reported under a scope sentence, not under the certificate.
 MIMIC: descriptive (the frozen bound does not hold there anyway).
Aggregates only."""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import e2x_common as X

E4, F = X.E4, X.F


def load_ref(split):
    for d in [split, f"{split}_test", "mimic_test", "mimic"] if split == "mimic" else [split]:
        p = X.HOME / "work/corpus_v2" / d
        if p.exists():
            return E4.load_claims(p), d
    raise FileNotFoundError(f"no work/corpus_v2 dir for {split}")


def capped(fp):
    return np.minimum(np.asarray(fp, float), 3) / 3


def boot_diff(a, b, pat, reps=2000, seed=27):
    """patient-cluster bootstrap CI of mean(b) - mean(a); a, b aligned per study, pat = patient id per study."""
    rng = np.random.default_rng(seed)
    up, inv = np.unique(pat, return_inverse=True)
    sa, sb, n = np.bincount(inv, a), np.bincount(inv, b), np.bincount(inv)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(up), len(up))
        d.append(sb[i].sum() / n[i].sum() - sa[i].sum() / n[i].sum())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["valid", "mimic"], required=True)
    a = ap.parse_args(argv)
    S = a.split
    fz = json.load(open(F.OUT / "freeze.json"))
    print(f"E27 {S} | freeze.json {X.md5(F.OUT / 'freeze.json')}")
    C = F.load_cands(S); C["score"] = F.apply_scorer(C, fz["scorer_full"], fz["delta"])
    ref, refdir = load_ref(S)
    keys_all = sorted(C.key.unique())
    keys = sorted(pd.read_parquet(F.OUT / "eval_keys_valid.parquet").key) if S == "valid" else keys_all
    probe = F.probe_arrays(S, keys_all)
    fin = X.final_claims(C, probe, fz["tau"], ref)
    adm = X.admit(fin, keys, fz["t_full"])
    adm_m = X.admit(fin, keys, X.multi_thresholds(S, keys, fz["group_thresholds"], fz["t_full"], fz["group_tops"]))
    G = E4.load_claims(X.HOME / f"work/corpus_gen/e8edit_{S}")
    pat_map = C.groupby("key").patient.first().to_dict() if "patient" in C.columns else {}
    pat = np.array([pat_map.get(k, k) for k in keys])
    print(f"refs from work/corpus_v2/{refdir}; edited texts with claims: {sum(1 for k in keys if G.get(k))}/{len(keys)}")

    A = {k: set(zip(g.finding, g.lat)) for k, g in adm.groupby("key")}
    AL = {k: dict(zip(zip(g.finding, g.lat), g.label)) for k, g in adm.groupby("key")}
    AM = {k: set(zip(g.finding, g.lat)) for k, g in adm_m.groupby("key")}
    fp_a, fp_e, tp_e, cov_e, nref = [], [], [], [], []
    cat = {"exact admitted claim": 0, "admitted finding, other or no side": 0, "finding not admitted": 0}
    in_m = in_mm = tot_claims = 0
    for k in keys:
        R = set(ref.get(k, {})); ad = A.get(k, set()); ec = set(G.get(k, {}))
        fp_a.append(int(sum(1 for c in ad if not AL[k][c])) if k in AL else 0)
        adf, admf = {f for f, _ in ad}, {f for f, _ in AM.get(k, set())}
        fe = 0
        for c in ec:
            tot_claims += 1; in_m += c[0] in adf; in_mm += c[0] in admf
            if not E4.matches(c, R):
                fe += 1
                cat["exact admitted claim" if c in ad else ("admitted finding, other or no side" if c[0] in adf
                                                            else "finding not admitted")] += 1
        fp_e.append(fe); tp_e.append(len(ec) - fe)
        cov_e.append(sum(1 for r in R if any(E4.matches(c, {r}) for c in ec))); nref.append(len(R))
    fp_a, fp_e = np.array(fp_a), np.array(fp_e)
    La, Le = capped(fp_a), capped(fp_e)
    lo, hi = boot_diff(La, Le, pat)
    excess = float(np.mean(fp_e > fp_a))
    print(f"\n=== E27 {S.upper()} ({len(keys)} studies): edited text vs ANCHOR admitted stated claims (frozen marginal) ===")
    print(f"  gate check: edited-text claims whose finding ANCHOR admitted: marginal {in_m / max(tot_claims, 1):.3f} | multi {in_mm / max(tot_claims, 1):.3f}")
    print(f"  ANCHOR admitted (stated)  false/rpt {fp_a.mean():.3f} risk {La.mean():.4f}")
    print(f"  ANCHOR-edit (extracted)   false/rpt {fp_e.mean():.3f} risk {Le.mean():.4f} | claims/rpt {(fp_e + np.array(tp_e)).mean():.2f} "
          f"precision {np.sum(tp_e) / max(np.sum(tp_e) + fp_e.sum(), 1):.3f} recall {np.sum(cov_e) / max(np.sum(nref), 1):.3f}")
    print(f"  risk difference edit minus admitted: {Le.mean() - La.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] (patient-cluster)")
    print(f"  reports with more false findings in the edited text: {excess:.4f} ({int((fp_e > fp_a).sum())}); fewer: {np.mean(fp_e < fp_a):.4f}")
    print(f"  false claims in edited text by origin: {cat}")
    if S == "valid":
        ok = (Le.mean() <= La.mean() + 0.005) and (excess <= 0.01)
        print(f"  H27: {'HOLDS' if ok else 'FAILS'} (risk margin {Le.mean() - La.mean():+.4f} vs 0.005, excess share {excess:.4f} vs 0.01)")
    out = {"n": len(keys), "risk_admitted": float(La.mean()), "risk_edit": float(Le.mean()), "diff_ci": [lo, hi],
           "excess_share": excess, "false_by_origin": cat}
    json.dump(out, open(X.HOME / f"work/e8/e27_{S}.json", "w"), indent=1, default=float)
    print("E27_DONE")


if __name__ == "__main__":
    main()
