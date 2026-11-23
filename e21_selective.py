#!/usr/bin/env python3
"""E21 (round 45): SELECTIVE two-sided certificate = "certify what is said, flag what cannot be certified".
Pivot after E18 (round 44): admission alone could not certify critical misses below 0.70 at false-finding risk 0.10,
because the candidate pool and the false-finding budget cap critical recall. Silence must therefore be paid for
openly: a report that may miss a critical finding is FLAGGED for full review instead of passing silently.
Validation + MIMIC only; never the ReXGradient test split. CPU only (work/e18 cache). Aggregates only.

Two parameters: admission threshold lam (all findings, as FUS) and deferral threshold mu. A study is flagged when its
RESIDUAL critical evidence r(x; lam) = max FUS score among critical candidates NOT admitted at lam is >= mu.
Certified jointly with probability >= 1 - delta (LTT, Hoeffding-Bentkus; Bonferroni over the lam grid, fixed
sequence over mu from flag-everything to flag-nothing, p = max(p_fp, p_scm)):
    R_fp(lam)       = E[min(FP,3)/3]                                       <= a1 = 0.10
    R_scm(lam, mu)  = P(report NOT flagged AND misses >= 1 critical finding) <= a2   (silent critical miss)
Selected pair: lowest flag rate, then highest recall, among certified pairs (calibration fold 1).
Baseline deferral rule: flag on MAXIMUM critical evidence (admitted or not), i.e. the D60 triage logic generalised.

PRE-STATED (round 45, before any run):
 H21a  at a1 0.10, a2 0.05, a certified pair exists on fold 1; on folds 2-3 both risks hold and the flag rate is
       <= 0.50 of all studies (and <= 0.30 of reference-normal studies).
 H21b  residual-evidence flagging flags fewer studies than maximum-evidence flagging at the same certified a2.
 H21c  MIMIC recalibration (n_cal 200 / 400, 100 patient-disjoint draws): certified-and-violating draws <= 0.10.
 Disproof: no certified pair at a2 <= 0.10; or flag rate > 0.70 (vacuous); or a folds 2-3 violation; or H21c fails.
Also printed: the E18 ceiling (critical miss when EVERY critical candidate is admitted) to explain the E18 negative."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e4a_pool as E4  # noqa: E402
import e18_twosided as T  # noqa: E402

D18 = HOME / "work/e18"
A1, DELTA = 0.10, 0.10
A2S = [0.02, 0.05, 0.10, 0.15]
NLAM, NMU = 30, 80


def fixed_grids(d, fold, extra):
    """lam and mu grids fixed BEFORE calibration: quantiles of FIT-fold (fold 0) scores only, so the LTT hypothesis
    family does not depend on the calibration sample (also reused unchanged for MIMIC)."""
    d0 = d[d.fold == 0]
    g = np.unique(np.concatenate([np.quantile(d0.score, np.linspace(0, 1, 400)), [x for x in extra if np.isfinite(x)]]))
    mus = np.unique(np.quantile(d0.score[d0.finding.isin(T.CRIT)], np.linspace(0, 1, NMU)))
    return g, np.r_[-np.inf, mus, np.inf]


def prepare(d, keys, ref, g, mus):
    """Shared-threshold matrices on one grid (critical and non-critical counted separately), residual and max critical
    evidence per study and threshold."""
    cm = d.finding.isin(T.CRIT).to_numpy()
    Mnc = T.E.matrices(d[~cm], keys, T.split_ref(ref, False), g); Mc = T.E.matrices(d[cm], keys, T.split_ref(ref, True), g)
    pos = {k: i for i, k in enumerate(keys)}
    dc = d[cm]; si = dc.key.map(pos).to_numpy(); sc = dc.score.to_numpy()
    width = max(int(pd.Series(si).value_counts().max()), 1) if len(si) else 1
    S = np.full((len(keys), width), -np.inf); fill = np.zeros(len(keys), int)
    for i, s in zip(si, sc):
        S[i, fill[i]] = s; fill[i] += 1
    G = np.r_[g, np.inf]                                                     # column j admits score >= G[j]; last admits nothing
    RES = np.stack([np.where(S < t, S, -np.inf).max(1) for t in G], 1)      # residual critical evidence per study x threshold
    MAXE = S.max(1)
    return {"g": g, "mus": mus, "nc": Mnc, "c": Mc, "res": RES, "max": MAXE}


def per_study(P, rows, j):
    FP = P["nc"][0][rows, j] + P["c"][0][rows, j]; TP = P["nc"][1][rows, j] + P["c"][1][rows, j]
    cov = P["nc"][2][rows, j] + P["c"][2][rows, j]; nref = P["nc"][3][rows] + P["c"][3][rows]
    covc, nrefc = P["c"][2][rows, j], P["c"][3][rows]
    return FP, TP, cov, nref, covc, nrefc


def evaluate(P, rows, j, mu, rule):
    FP, TP, cov, nref, covc, nrefc = per_study(P, rows, j)
    score = P["res"][rows, j] if rule == "residual" else P["max"][rows]
    flag = score >= mu
    miss = nrefc - covc > 0
    normal = nref == 0
    return {"r_fp": float((np.minimum(FP, 3) / 3).mean()), "r_scm": float((~flag & miss).mean()),
            "flag_rate": float(flag.mean()), "flag_rate_ref_normal": float(flag[normal].mean()) if normal.any() else float("nan"),
            "flag_rate_with_crit": float(flag[nrefc > 0].mean()) if (nrefc > 0).any() else float("nan"),
            "silent_share_of_crit_findings": float((nrefc - covc)[~flag].sum() / max(nrefc.sum(), 1)),
            "crit_miss_any_no_flag": float(miss.mean()), "recall": float(cov.sum() / max(nref.sum(), 1)),
            "crit_recall": float(covc.sum() / max(nrefc.sum(), 1)), "precision": float(TP.sum() / max((TP + FP).sum(), 1)),
            "false_per_report": float(FP.mean())}


def certify(P, rows, a1, a2, delta, rule):
    G = len(P["g"])
    lam_idx = np.unique(np.linspace(0, G, NLAM).round().astype(int))
    lvl = delta / len(lam_idx); n = len(rows)
    nrefc = P["c"][3][rows]
    out = []
    for j in lam_idx:
        FP, _, _, _, covc, _ = per_study(P, rows, j)
        pfp = T.hb_p(float((np.minimum(FP, 3) / 3).mean()), n, a1)
        if pfp > lvl:
            continue
        score = P["res"][rows, j] if rule == "residual" else P["max"][rows]
        miss = nrefc - covc > 0
        for mu in P["mus"]:                                                       # ascending mu: flag everything -> flag nothing
            if T.hb_p(float((~(score >= mu) & miss).mean()), n, a2) <= lvl:
                out.append((int(j), float(mu)))
            else:
                break
    return out


def choose(P, rows, cert, rule):
    if not cert:
        return None
    best, key = None, None
    for j, mu in cert:
        r = evaluate(P, rows, j, mu, rule); k = (-r["flag_rate"], r["recall"])
        if key is None or k > key:
            best, key = (j, mu), k
    return best


def fmt(r):
    return (f"r_fp {r['r_fp']:.4f} r_scm {r['r_scm']:.4f} flag {r['flag_rate']:.3f} (ref-normal {r['flag_rate_ref_normal']:.3f}, "
            f"with-critical {r['flag_rate_with_crit']:.3f}) silent-share-of-critical {r['silent_share_of_crit_findings']:.3f} "
            f"recall {r['recall']:.3f} crit_recall {r['crit_recall']:.3f} precision {r['precision']:.3f} false/rpt {r['false_per_report']:.3f}")


def main():
    out = {}
    thr = json.load(open(D18 / "thresholds.json"))
    d = pd.read_parquet(D18 / "fus_valid.parquet"); st = pd.read_parquet(D18 / "studies_valid.parquet")
    ref = E4.load_claims(HOME / "work/corpus_v2/valid")
    keys = list(st.key); fold = st.fold.to_numpy(); tm = st.t_fus_multi.to_numpy()
    cal, ev = np.where(fold == 1)[0], np.where(fold >= 2)[0]
    # E18 ceiling on the separate-threshold matrices
    M = T.build(d, keys, ref, extra=[thr["t_fus"], *np.unique(tm)])
    ceil = T.risks(M, ev, len(M["gnc"]), 0)
    print(f"E18 CEILING (folds 2-3): every critical candidate admitted, no other claims -> r_cm {ceil['r_cm']:.4f}, crit_recall "
          f"{ceil['crit_recall']:.3f}, r_fp {ceil['r_fp']:.4f}, false/rpt {ceil['false_per_report']:.3f}")
    out["e18_ceiling"] = ceil
    g0, mu0 = fixed_grids(d, None, extra=[thr["t_fus"], *np.unique(tm)])
    print(f"grids fixed from fit fold 0: lam {len(g0)} values (Bonferroni over {NLAM}), mu {len(mu0)} values")
    P = prepare(d, keys, ref, g0, mu0)
    jf = int(T.idx(P["g"], thr["t_fus"]))
    base = evaluate(P, ev, jf, np.inf, "residual")
    print(f"\nfolds 2-3, frozen FUS (lam {thr['t_fus']:.3f}), no flagging: {fmt(base)}")
    out["valid"] = {"frozen_fus_noflag": base}
    for rule in ["residual", "max"]:
        print(f"\nSELECTIVE TWO-SIDED LTT, flag rule = {rule} critical evidence (a1 0.10, delta 0.10; cal fold 1 n {len(cal)}, eval folds 2-3 n {len(ev)})")
        out["valid"][rule] = {}
        for a2 in A2S:
            cert = certify(P, cal, A1, a2, DELTA, rule); sel = choose(P, cal, cert, rule)
            if sel is None:
                print(f"  a2 {a2:.2f}: no certified pair"); out["valid"][rule][str(a2)] = None; continue
            rc, re = evaluate(P, cal, *sel, rule), evaluate(P, ev, *sel, rule)
            out["valid"][rule][str(a2)] = {"n_certified": len(cert), "lam": float(P["g"][sel[0]]) if sel[0] < len(P["g"]) else None,
                                           "mu": sel[1], "cal": rc, "eval": re}
            print(f"  a2 {a2:.2f}: certified {len(cert)} | lam {out['valid'][rule][str(a2)]['lam']} mu {sel[1]:.3f} | cal flag {rc['flag_rate']:.3f} "
                  f"r_scm {rc['r_scm']:.4f}\n     folds 2-3: {fmt(re)}", flush=True)
    # ---------------- MIMIC: recalibrate (lam, mu) on n_cal labelled studies ----------------
    dm = pd.read_parquet(D18 / "fus_mimic.parquet"); sm = pd.read_parquet(D18 / "studies_mimic.parquet")
    refm = E4.load_claims(HOME / "work/corpus_v2/mimic")
    km = list(sm.key); pat = sm.patient.to_numpy()
    Pm = prepare(dm, km, refm, g0, mu0)
    allr = np.arange(len(km))
    out["mimic"] = {"frozen_fus_noflag": evaluate(Pm, allr, int(T.idx(Pm["g"], thr["t_fus"])), np.inf, "residual")}
    print(f"\nMIMIC ({len(km)} studies) frozen FUS, no flagging: {fmt(out['mimic']['frozen_fus_noflag'])}")
    sel_v = out["valid"]["residual"].get("0.05")
    if sel_v:
        jv = int(T.idx(Pm["g"], sel_v["lam"])) if sel_v["lam"] is not None else len(Pm["g"])
        fr = evaluate(Pm, allr, jv, sel_v["mu"], "residual"); out["mimic"]["frozen_valid_pair_a2_0.05"] = fr
        print(f"MIMIC with the valid-certified (lam, mu) at a2 0.05, NOT recalibrated: {fmt(fr)}")
    up = np.array(sorted(set(pat))); pc = pd.Series(pat).value_counts(); rng = np.random.default_rng(45)
    print("viol/all = certified-and-violating draws / all draws (the LTT error rate, should be <= 0.10)")
    for n in [200, 400]:
        for a2 in [0.05, 0.10]:
            rows = []
            for _ in range(100):
                cal_p, cnt = [], 0
                for p in rng.permutation(up):
                    cal_p.append(p); cnt += pc[p]
                    if cnt >= n:
                        break
                cmask = np.isin(pat, cal_p); c_, e_ = np.where(cmask)[0], np.where(~cmask)[0]
                sel = choose(Pm, c_, certify(Pm, c_, A1, a2, DELTA, "residual"), "residual")
                rows.append(evaluate(Pm, e_, *sel, "residual") if sel else None)
            ok = [r for r in rows if r]
            if not ok:
                print(f"  n_cal {n} a2 {a2:.2f}: no certified pair in any draw"); continue
            viol = sum((r["r_fp"] > A1) or (r["r_scm"] > a2) for r in ok) / len(rows)
            row = {"certified_draws": len(ok), "viol_all": viol, **{k: float(np.nanmean([r[k] for r in ok])) for k in
                   ["r_fp", "r_scm", "flag_rate", "flag_rate_ref_normal", "silent_share_of_crit_findings", "recall", "crit_recall"]}}
            out["mimic"][f"{n}|{a2}"] = row
            print(f"  n_cal {n} a2 {a2:.2f}: certified {len(ok)}/100, viol/all {viol:.3f} | r_fp {row['r_fp']:.4f} r_scm {row['r_scm']:.4f} "
                  f"flag {row['flag_rate']:.3f} (ref-normal {row['flag_rate_ref_normal']:.3f}) silent-share {row['silent_share_of_crit_findings']:.3f} "
                  f"recall {row['recall']:.3f} crit_recall {row['crit_recall']:.3f}", flush=True)
    json.dump(out, open(D18 / "e21_selective.json", "w"), indent=1, default=float)
    print("E21_DONE")


if __name__ == "__main__":
    main()
