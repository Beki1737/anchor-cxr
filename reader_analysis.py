#!/usr/bin/env python3
"""D74 reader-study analysis (~/venvs/anchor_eval; CPU). Aggregates only; never prints report text or study keys.
  0. completeness check: aborts if the sheets look ungraded (templates)
  1. per-system error rates (significant false / missed, wrong side, minor), share of reports with 0 significant errors,
     paired ANCHOR-edit vs B0 per study (Wilcoxon signed-rank + paired bootstrap), by stratum
  2. preference: ANCHOR-edit vs B0 vs equal, exact sign test (ties excluded)
  3. intra-reader reliability on the 10 hidden repeats
  4. metric validity: Kendall tau between reader significant errors and CRIMSON false+missed counts per report
  5. E1 extractor: second-rater accuracy and Cohen kappa vs the first rater"""
import math
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
R = HOME / "work/reader"
G = ["false_significant", "missed_significant", "wrong_side", "false_minor", "missed_minor"]
SYSCOND = {"B0": "b0_medgemma4b", "ANCHOR-edit": "e8edit"}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = (a == b).mean(); pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    s = pd.read_csv(R / "reader_sheet.csv"); k = pd.read_csv(R / "reader_key.csv")
    cols = [f"{x}_{g}" for x in "AB" for g in G]
    filled = s[cols].apply(num).notna().mean().mean()
    pref_filled = s.preferred.astype(str).str.strip().str.upper().isin(["A", "B", "EQUAL"]).mean()
    print(f"completeness: grade cells filled {filled:.1%}, preference filled {pref_filled:.1%} ({len(s)} items)")
    if filled < 0.5:
        print("READER SHEET LOOKS UNGRADED (template). Copy the radiologist's file over reader_sheet.csv and rerun."); return
    d = s.merge(k, on="item")
    long = []
    for _, r in d.iterrows():
        for x in "AB":
            row = {"item": r["item"], "key": r["key"], "system": r[x], "stratum": r["stratum"], "repeat": bool(pd.notna(num(pd.Series([r["repeat_of"]]))[0])),
                   "preferred_me": str(r["preferred"]).strip().upper() == x, "pref_equal": str(r["preferred"]).strip().upper() == "EQUAL",
                   "confidence": num(pd.Series([r["confidence_1to5"]]))[0]}
            row.update({g: num(pd.Series([r[f"{x}_{g}"]]))[0] for g in G})
            long.append(row)
    L = pd.DataFrame(long)
    for g in G:
        L[g] = L[g].fillna(0)
    L["sig"] = L.false_significant + L.missed_significant + L.wrong_side
    M = L[~L.repeat]
    print(f"\n#### 1. errors per report ({M.key.nunique()} studies, repeats excluded) ####")
    print(f"{'system':<13}" + "".join(f"{g:>20}" for g in G) + f"{'sig total':>11}{'0 sig err':>11}")
    for sysn, g in M.groupby("system"):
        print(f"{sysn:<13}" + "".join(f"{g[c].mean():>20.3f}" for c in G) + f"{g.sig.mean():>11.3f}{(g.sig == 0).mean():>11.3f}")
    W = M.pivot_table(index="key", columns="system", values="sig")
    dlt = (W["ANCHOR-edit"] - W["B0"]).to_numpy()
    rng = np.random.default_rng(0)
    bt = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(5000)]
    try:
        from scipy.stats import wilcoxon
        pw = float(wilcoxon(W["ANCHOR-edit"], W["B0"], zero_method="wilcox").pvalue) if (dlt != 0).any() else 1.0
    except Exception:
        pw = float("nan")
    print(f"paired significant errors ANCHOR-edit minus B0: {dlt.mean():+.3f} per report, 95% CI [{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}], "
          f"Wilcoxon p={pw:.3g} | fewer {np.mean(dlt < 0):.2f}, equal {np.mean(dlt == 0):.2f}, more {np.mean(dlt > 0):.2f}")
    for st, g in M.groupby("stratum"):
        w = g.pivot_table(index="key", columns="system", values="sig")
        print(f"  {st:<15} n={len(w):>3}  B0 {w['B0'].mean():.3f} | ANCHOR-edit {w['ANCHOR-edit'].mean():.3f} | diff {(w['ANCHOR-edit'] - w['B0']).mean():+.3f}")
    for c in ["false_significant", "missed_significant", "wrong_side"]:
        w = M.pivot_table(index="key", columns="system", values=c)
        print(f"  {c:<20} B0 {w['B0'].mean():.3f} vs ANCHOR-edit {w['ANCHOR-edit'].mean():.3f}")
    print("\n#### 2. preference (first showings) ####")
    P = M[M.system == "ANCHOR-edit"]
    wa = int(P.preferred_me.sum()); eq = int(P.pref_equal.sum()); wb = len(P) - wa - eq
    print(f"  prefers ANCHOR-edit {wa} | B0 {wb} | equal {eq}  -> sign test p={sign_test(wa, wb):.3g}; mean confidence {L[~L.repeat].confidence.mean():.2f}")
    print("\n#### 3. intra-reader reliability (hidden repeats) ####")
    rep = d[num(d.repeat_of).notna()]
    same_sig, same_pref, n = [], [], 0
    for _, r2 in rep.iterrows():
        r1 = d[d.item == int(float(r2.repeat_of))].iloc[0]
        for sysn in ["B0", "ANCHOR-edit"]:
            x1 = "A" if r1["A"] == sysn else "B"; x2 = "A" if r2["A"] == sysn else "B"
            v1 = sum(num(pd.Series([r1[f"{x1}_{c}"]])).fillna(0)[0] for c in ["false_significant", "missed_significant", "wrong_side"])
            v2 = sum(num(pd.Series([r2[f"{x2}_{c}"]])).fillna(0)[0] for c in ["false_significant", "missed_significant", "wrong_side"])
            same_sig.append((v1, v2))
        p1 = {"A": r1["A"], "B": r1["B"]}.get(str(r1.preferred).strip().upper(), "equal")
        p2 = {"A": r2["A"], "B": r2["B"]}.get(str(r2.preferred).strip().upper(), "equal")
        same_pref.append(p1 == p2); n += 1
    if n:
        a = np.array(same_sig)
        print(f"  {n} repeated studies: significant-error count identical {np.mean(a[:, 0] == a[:, 1]):.2f}, within 1 {np.mean(abs(a[:, 0] - a[:, 1]) <= 1):.2f}, "
              f"kappa (counts as categories) {kappa(a[:, 0], a[:, 1]):.2f} | same preferred system {np.mean(same_pref):.2f}")
    print("\n#### 4. metric validity: reader vs CRIMSON (per report) ####")
    f = HOME / "work/judge/test/crimson_scores.parquet"
    if f.exists():
        from scipy.stats import kendalltau
        cs = pd.read_parquet(f)
        rows = []
        for sysn, cond in SYSCOND.items():
            c = cs[cs.cond == cond].set_index("key")
            g = M[M.system == sysn].set_index("key")
            j = g.join(c[["score"] + [x for x in ["n_false_findings", "n_missing_findings"] if x in c]], how="inner")
            rows.append(j)
        J = pd.concat(rows)
        if "n_false_findings" in J:
            t1 = kendalltau(J.sig, J.n_false_findings + J.n_missing_findings)
            print(f"  Kendall tau(reader significant errors, CRIMSON false+missed) = {t1.statistic:.3f} (p={t1.pvalue:.2g}, n={len(J)})")
        t2 = kendalltau(J.sig, -J.score)
        print(f"  Kendall tau(reader significant errors, -CRIMSON score) = {t2.statistic:.3f} (p={t2.pvalue:.2g})")
    print("\n#### 6. power check: per-component paired tests and CRIMSON on the SAME studies ####")
    try:
        from scipy.stats import wilcoxon
        for c in ["false_significant", "missed_significant", "false_minor", "missed_minor"]:
            w = M.pivot_table(index="key", columns="system", values=c)
            dd = w["ANCHOR-edit"] - w["B0"]
            p = float(wilcoxon(w["ANCHOR-edit"], w["B0"]).pvalue) if (dd != 0).any() else 1.0
            print(f"  {c:<20} ANCHOR-edit minus B0 {dd.mean():+.3f} | fewer {np.mean(dd < 0):.2f} more {np.mean(dd > 0):.2f} | Wilcoxon p={p:.3g}")
    except Exception as ex:
        print("  scipy unavailable:", ex)
    if f.exists():
        c = cs[cs.cond.isin(SYSCOND.values()) & cs.key.isin(set(M.key))].pivot_table(index="key", columns="cond", values="score")
        dd = (c["e8edit"] - c["b0_medgemma4b"]).to_numpy()
        bt = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
        print(f"  CRIMSON on these {len(dd)} studies: ANCHOR-edit minus B0 {dd.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}, {np.percentile(bt, 97.5):+.4f}] "
              f"(full test +0.069); implied n for 80% power on the reader endpoint at the observed effect: "
              f"{int(math.ceil(((1.96 + 0.84) * np.std(dlt, ddof=1) / max(abs(dlt.mean()), 1e-9)) ** 2))}")
    print("\n#### 5. E1 extractor second rater ####")
    e2p, e1p = R / "e1_rater2.csv", HOME / "work/e1_label_200_v2_heldout.csv"
    if e2p.exists() and e1p.exists():
        e2, e1 = pd.read_csv(e2p), pd.read_csv(e1p)
        j = e1.merge(e2, on="row", suffixes=("_r1", "_r2"))
        for c in ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]:
            a, b = num(j[f"{c}_r1"]), num(j[f"{c}_r2"])
            m = a.notna() & b.notna()
            if m.sum() < 10:
                print(f"  {c}: too few graded rows (rater 2 filled {int(b.notna().sum())})"); continue
            print(f"  {c:<24} rater2 mean {b[m].mean():.3f} | rater1 {a[m].mean():.3f} | agreement {(a[m] == b[m]).mean():.3f} | kappa {kappa(a[m], b[m]):.3f} (n={int(m.sum())})")
    print("READER_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
