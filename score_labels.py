#!/usr/bin/env python3
"""Score the E1 labelling sheet(s): field accuracy with Wilson 95% CIs, gate check, and inter-rater kappa
if a second sheet exists (~/anchor/work/e1_label_200_rater2.csv). Prints aggregates only."""
import math
import pathlib
import sys

import pandas as pd

W = pathlib.Path.home() / "anchor/work"
FIELDS = ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]
GATE = {"finding_and_polarity_ok": 0.90, "regions_ok": 0.85}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def load(p):
    df = pd.read_csv(p)
    for f in FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def kappa(a, b):
    m = a.notna() & b.notna()
    a, b = a[m].astype(int), b[m].astype(int)
    if len(a) == 0:
        return float("nan"), 0
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in (0, 1))
    return ((po - pe) / (1 - pe) if pe < 1 else 1.0), len(a)


def main():
    p1 = W / (sys.argv[1] if len(sys.argv) > 1 else "e1_label_200.csv")
    print(f"sheet: {p1.name}")
    if not p1.exists():
        sys.exit("no labelling sheet found")
    s = load(p1)
    lab = s[s.finding_and_polarity_ok.notna()]
    print(f"labelled rows: {len(lab)} / {len(s)}")
    if len(lab) < 50:
        print("fewer than 50 rows labelled: gate not evaluated yet"); return
    passed = True
    for f in FIELDS:
        x = lab[f].dropna()
        p, lo, hi = wilson(int(x.sum()), len(x))
        tag = ""
        if f in GATE:
            ok = p >= GATE[f]
            passed &= ok
            tag = f"  gate >= {GATE[f]:.2f}: {'PASS' if ok else 'FAIL'}"
        print(f"{f:<26} n={len(x):>3}  rate={p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]{tag}")
    nf = lab.extraction.astype(str).str.contains(r"\(no findings\)")
    for name, sel in [("sentences with extracted findings", ~nf), ("sentences with no findings", nf)]:
        x = lab[sel].finding_and_polarity_ok
        print(f"   {name}: n={len(x)} finding/polarity ok={x.mean():.3f}")
    print(f"E1 GATE: {'PASS' if passed else 'FAIL'}")
    p2 = p1.with_name(p1.stem + "_rater2.csv")
    if p2.exists():
        s2 = load(p2)
        for f in FIELDS:
            k, n = kappa(s[f], s2[f])
            print(f"inter-rater kappa {f:<24} n={n:>3}  kappa={k:.3f}")


if __name__ == "__main__":
    main()
