#!/usr/bin/env python3
"""E18 (round 44): TWO-SIDED certified admission. Validation + MIMIC only; never touches the ReXGradient test split.
CPU only (reads the work/e18 cache written by e18_cache.py). Aggregates only.

Attack answered: "a false-finding guarantee alone is met by saying nothing". We certify, jointly, with probability
>= 1 - delta (Learn-then-Test, Hoeffding-Bentkus p-values, family-wise error control):
    R_fp(lam, lam_c) = E[min(FP,3)/3]                                    <= a1 = 0.10  (same loss as the paper)
    R_cm(lam_c)      = E[missed critical refs / critical refs | >= 1 critical ref]    <= a2
with a separate admission threshold lam_c for the 8 critical findings (F.CRIT) and lam for all other findings.
Candidates = FUS pool (whole vocabulary + out-of-vocabulary ANCHOR candidates), sides resolved BEFORE calibration.
Testing: Bonferroni over the lam_c grid (level delta/m); within each lam_c, fixed sequence over lam from strict to
permissive with p = max(p_fp, p_cm) (intersection-union test). Every certified pair is valid; we deploy the one with the
highest calibration recall.

PRE-STATED (round 44, before any run):
 H18a  for some a2 <= 0.50 a certified pair exists on calibration fold 1, and on folds 2-3 it keeps r_fp <= 0.10 and
       r_cm <= a2 while its critical miss is lower than the frozen single-threshold FUS-multi system.
 H18b  on MIMIC (two-sided recalibration on n_cal labelled studies, 100 patient-disjoint draws) the share of draws that
       violate EITHER target is <= delta = 0.10.
 Disproof: no certified pair for any a2 <= 0.50; or folds 2-3 violate a target; or critical miss not lower than
       frozen FUS-multi; or MIMIC violation share > 0.10.
Also printed: the label-noise tipping point eps* = (a1 - R) * 3 / TP_per_report. If at most a share eps* of the admitted
claims scored as true are in fact false (reference-label error), the realised risk still stays below a1."""
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e17_recal as E  # noqa: E402

D18 = HOME / "work/e18"
CRIT = set(E.F.CRIT)
A1, DELTA = 0.10, 0.10
A2S = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
_CDF = {}


def binom_cdf(n, a):
    if (n, a) not in _CDF:
        k = np.arange(n + 1)
        lg = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
        _CDF[(n, a)] = np.cumsum(np.exp(lg + k * math.log(a) + (n - k) * math.log(1 - a)))
    return _CDF[(n, a)]


def hb_p(r, n, a):
    """Hoeffding-Bentkus p-value (Bates et al. 2021) for H0: risk > a, given mean r of n losses in [0, 1]."""
    if n == 0:
        return 1.0
    rr = min(r, a)
    if rr <= 0:
        ph = (1 - a) ** n
    else:
        ph = math.exp(-n * (rr * math.log(rr / a) + (1 - rr) * math.log((1 - rr) / (1 - a))))
    pb = math.e * binom_cdf(n, a)[min(int(math.ceil(n * r - 1e-9)), n)]
    return min(ph, pb, 1.0)


def split_ref(ref, crit):
    return {k: {r: v for r, v in d.items() if (r[0] in CRIT) == crit} for k, d in ref.items()}


def build(d, keys, ref, extra=(), n_grid=240, n_grid_c=40):
    """FP/TP/coverage matrices for non-critical and critical candidates on their own ascending threshold grids.
    Column j admits claims with score >= grid[j]; the last column admits nothing."""
    cm = d.finding.isin(CRIT).to_numpy()
    ex = np.asarray([x for x in extra if np.isfinite(x)], float)
    gnc = np.unique(np.concatenate([np.quantile(d.score[~cm], np.linspace(0, 1, n_grid)), ex]))
    gc = np.unique(np.concatenate([np.quantile(d.score[cm], np.linspace(0, 1, n_grid_c)), ex]))
    return {"gnc": gnc, "gc": gc, "nc": E.matrices(d[~cm], keys, split_ref(ref, False), gnc),
            "c": E.matrices(d[cm], keys, split_ref(ref, True), gc)}


def idx(grid, t):
    return np.searchsorted(grid, np.asarray(t, float), side="left")


def risks(M, rows, j, jc):
    """j, jc: scalars or arrays aligned with rows."""
    FP = M["nc"][0][rows, j] + M["c"][0][rows, jc]
    TP = M["nc"][1][rows, j] + M["c"][1][rows, jc]
    cov = M["nc"][2][rows, j] + M["c"][2][rows, jc]; nref = M["nc"][3][rows] + M["c"][3][rows]
    covc, nrefc = M["c"][2][rows, jc], M["c"][3][rows]
    has = nrefc > 0
    return {"r_fp": float((np.minimum(FP, 3) / 3).mean()),
            "r_cm": float(((nrefc[has] - covc[has]) / nrefc[has]).mean()) if has.any() else 0.0,
            "recall": float(cov.sum() / max(nref.sum(), 1)), "crit_recall": float(covc.sum() / max(nrefc.sum(), 1)),
            "precision": float(TP.sum() / max((TP + FP).sum(), 1)), "false_per_report": float(FP.mean()),
            "claims_per_report": float((TP + FP).mean()), "tp_per_report": float(TP.mean())}


def ltt_two_sided(M, rows, a1, a2, delta, jcs=None):
    """All certified (j, jc) on the calibration rows."""
    if jcs is None:
        jcs = list(range(len(M["gc"]), -1, -1))
    lvl = delta / len(jcs)
    n = len(rows); nrefc = M["c"][3][rows]; has = nrefc > 0; nh = int(has.sum())
    FPnc = M["nc"][0][rows]
    out = []
    for jc in jcs:
        covc = M["c"][2][rows, jc]
        pcm = hb_p(float(((nrefc[has] - covc[has]) / nrefc[has]).mean()) if nh else 0.0, nh, a2)
        if pcm > lvl:
            continue
        rfp = (np.minimum(FPnc + M["c"][0][rows, jc][:, None], 3) / 3).mean(0)
        for j in range(len(M["gnc"]), -1, -1):                 # strict -> permissive, stop at first non-rejection
            if hb_p(float(rfp[j]), n, a1) <= lvl:
                out.append((j, jc))
            else:
                break
    return out


def pick(M, rows, cert):
    """certified pair with the highest calibration recall (vectorised)."""
    if not cert:
        return None
    snc, sc = M["nc"][2][rows].sum(0), M["c"][2][rows].sum(0)
    c = np.array(cert)
    best = int(np.argmax(snc[c[:, 0]] + sc[c[:, 1]]))
    return int(c[best, 0]), int(c[best, 1])


def single_ltt(M, rows, a1, delta):
    """one threshold for all findings (FP-only LTT), fixed sequence strict -> permissive on the merged grid."""
    allg = np.unique(np.concatenate([M["gnc"], M["gc"]]))[::-1]
    best = (len(M["gnc"]), len(M["gc"]))
    for t in allg:
        j, jc = int(idx(M["gnc"], t)), int(idx(M["gc"], t))
        if hb_p(risks(M, rows, j, jc)["r_fp"], len(rows), a1) <= delta:
            best = (j, jc)
        else:
            break
    return best


def fmt(r):
    return " ".join(f"{k} {v:.4f}" for k, v in r.items())


def main():
    out = {}
    thr = json.load(open(D18 / "thresholds.json"))
    d = pd.read_parquet(D18 / "fus_valid.parquet"); st = pd.read_parquet(D18 / "studies_valid.parquet")
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    keys = list(st.key); fold = st.fold.to_numpy(); tm = st.t_fus_multi.to_numpy()
    M = build(d, keys, ref, extra=[thr["t_fus"], *np.unique(tm)])
    cal, ev = np.where(fold == 1)[0], np.where(fold >= 2)[0]
    print(f"VALID: studies {len(keys)} | calibration fold 1 {len(cal)} (with >=1 critical reference finding "
          f"{int((M['c'][3][cal] > 0).sum())}) | evaluation folds 2-3 {len(ev)} (with critical {int((M['c'][3][ev] > 0).sum())}) | "
          f"grid sizes non-critical {len(M['gnc'])}, critical {len(M['gc'])}")
    fm = risks(M, ev, idx(M["gnc"], tm[ev]), idx(M["gc"], tm[ev]))
    f1 = risks(M, ev, int(idx(M["gnc"], thr["t_fus"])), int(idx(M["gc"], thr["t_fus"])))
    print(f"\nfolds 2-3, frozen FUS-multi : {fmt(fm)}\nfolds 2-3, frozen FUS (CRC) : {fmt(f1)}")
    sj = single_ltt(M, cal, A1, DELTA); s1 = risks(M, ev, *sj)
    print(f"folds 2-3, single-threshold LTT (FP only, a1 0.10, delta 0.10): {fmt(s1)}")
    out["valid"] = {"frozen_fus_multi": fm, "frozen_fus": f1, "single_ltt": s1, "two_sided": {}}
    print(f"\nTWO-SIDED LTT (a1 0.10, delta 0.10, Bonferroni over {len(M['gc']) + 1} critical thresholds)")
    print(f"{'a2':>5} {'certified':>10} {'cal r_fp':>9} {'cal r_cm':>9} | folds 2-3: {'r_fp':>7} {'r_cm':>7} {'recall':>7} "
          f"{'crit rec':>9} {'prec':>6} {'false/rpt':>9} {'claims':>7}")
    for a2 in A2S:
        cert = ltt_two_sided(M, cal, A1, a2, DELTA); sel = pick(M, cal, cert)
        if sel is None:
            print(f"{a2:>5.2f} {0:>10}   no certified pair"); out["valid"]["two_sided"][str(a2)] = None; continue
        rc, re = risks(M, cal, *sel), risks(M, ev, *sel)
        out["valid"]["two_sided"][str(a2)] = {"n_certified": len(cert), "cal": rc, "eval": re,
                                              "lam": float(M["gnc"][sel[0]]) if sel[0] < len(M["gnc"]) else None,
                                              "lam_c": float(M["gc"][sel[1]]) if sel[1] < len(M["gc"]) else None}
        print(f"{a2:>5.2f} {len(cert):>10} {rc['r_fp']:>9.4f} {rc['r_cm']:>9.4f} | {re['r_fp']:>18.4f} {re['r_cm']:>7.4f} {re['recall']:>7.3f} "
              f"{re['crit_recall']:>9.3f} {re['precision']:>6.3f} {re['false_per_report']:>9.3f} {re['claims_per_report']:>7.2f}", flush=True)
    print("\nLABEL-NOISE TIPPING POINT on folds 2-3: eps* = (0.10 - R) * 3 / TP_per_report")
    rows_ = [("frozen FUS-multi", fm), ("frozen FUS", f1), ("single LTT", s1)] + \
            [(f"two-sided a2={a2}", v["eval"]) for a2, v in out["valid"]["two_sided"].items() if v]
    for nm, r in rows_:
        eps = (A1 - r["r_fp"]) * 3 / max(r["tp_per_report"], 1e-9)
        print(f"  {nm:<20} R {r['r_fp']:.4f} TP/report {r['tp_per_report']:.3f} eps* {eps:.3f}")
    # ---------------- MIMIC: two-sided recalibration with patient-disjoint draws ----------------
    dm = pd.read_parquet(D18 / "fus_mimic.parquet"); sm = pd.read_parquet(D18 / "studies_mimic.parquet")
    refm = E4.load_claims(HOME / "work/corpus_v2/mimic")
    km = list(sm.key); pat = sm.patient.to_numpy(); tmm = sm.t_fus_multi.to_numpy()
    Mm = build(dm, km, refm, extra=[thr["t_fus"], *np.unique(tmm)])
    allr = np.arange(len(km))
    frm = risks(Mm, allr, idx(Mm["gnc"], tmm), idx(Mm["gc"], tmm))
    print(f"\nMIMIC ({len(km)} studies, {len(set(pat))} patients, {int((Mm['c'][3] > 0).sum())} with a critical finding)")
    print(f"MIMIC frozen FUS-multi, no recalibration: {fmt(frm)}")
    out["mimic"] = {"frozen_fus_multi": frm}
    up = np.array(sorted(set(pat))); pc = pd.Series(pat).value_counts(); rng = np.random.default_rng(44)
    jcs = list(range(len(Mm["gc"]), -1, -2))
    print("P(.) columns are shares of CERTIFIED draws; 'viol/all' = certified-and-violating draws / all draws (the LTT error rate)")
    print(f"{'n_cal':>6} {'a2':>5} {'certified':>10} {'P(fp>a1)':>9} {'P(cm>a2)':>9} {'P(either)':>10} {'viol/all':>9} {'r_fp':>7} {'r_cm':>7} "
          f"{'recall':>7} {'crit rec':>9} | single-LTT recall, crit rec, P(fp>a1)")
    for n in [200, 400]:
        for a2 in [0.50, 0.60, 0.70]:
            rows, srows = [], []
            for _ in range(100):
                cal_p, cnt = [], 0
                for p in rng.permutation(up):
                    cal_p.append(p); cnt += pc[p]
                    if cnt >= n:
                        break
                cmask = np.isin(pat, cal_p); c_, e_ = np.where(cmask)[0], np.where(~cmask)[0]
                sel = pick(Mm, c_, ltt_two_sided(Mm, c_, A1, a2, DELTA, jcs=jcs))
                rows.append(risks(Mm, e_, *sel) if sel else None)
                srows.append(risks(Mm, e_, *single_ltt(Mm, c_, A1, DELTA)))
            ok = [r for r in rows if r]
            sr = {k: float(np.mean([r[k] for r in srows])) for k in ["recall", "crit_recall"]}
            sr["p_fp_over"] = float(np.mean([r["r_fp"] > A1 for r in srows]))
            if not ok:
                print(f"{n:>6} {a2:>5.2f} {0:>10}   no certified pair in any draw"); out["mimic"][f"{n}|{a2}"] = {"certified_draws": 0, "single": sr}; continue
            row = {"certified_draws": len(ok), "p_fp_over": float(np.mean([r["r_fp"] > A1 for r in ok])),
                   "p_cm_over": float(np.mean([r["r_cm"] > a2 for r in ok])),
                   "p_either_over": float(np.mean([(r["r_fp"] > A1) or (r["r_cm"] > a2) for r in ok])),
                   "p_either_over_all_draws": float(sum((r["r_fp"] > A1) or (r["r_cm"] > a2) for r in ok) / len(rows)),
                   **{k: float(np.mean([r[k] for r in ok])) for k in ["r_fp", "r_cm", "recall", "crit_recall", "precision"]}, "single": sr}
            out["mimic"][f"{n}|{a2}"] = row
            print(f"{n:>6} {a2:>5.2f} {len(ok):>10} {row['p_fp_over']:>9.3f} {row['p_cm_over']:>9.3f} {row['p_either_over']:>10.3f} {row['p_either_over_all_draws']:>9.3f} "
                  f"{row['r_fp']:>7.4f} {row['r_cm']:>7.4f} {row['recall']:>7.3f} {row['crit_recall']:>9.3f} | "
                  f"{sr['recall']:.3f}, {sr['crit_recall']:.3f}, {sr['p_fp_over']:.3f}", flush=True)
    json.dump(out, open(D18 / "e18_twosided.json", "w"), indent=1, default=float)
    print("E18_DONE")


if __name__ == "__main__":
    main()
