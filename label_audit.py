#!/usr/bin/env python3
"""E1 error audit on the graded sheet (~/anchor/work/e1_label_200.csv). Aggregates only, safe to share.
- per-field n and rate, split into rows WITH extracted findings vs '(no findings)' rows
  (a 0 in regions_ok / laterality_ok on a '(no findings)' row is a grading-convention slip: those should be blank)
- extracted laterality / polarity values among the rows graded wrong
- counts of error codes if an 'err_code' column was added (comma separated, codes listed in the guide)
"""
import pathlib
import sys

import pandas as pd

W = pathlib.Path.home() / "anchor/work"
FIELDS = ["finding_and_polarity_ok", "regions_ok", "laterality_ok", "missed_finding"]
CODES = ["POL", "HEDGE", "FAKE", "MISS_ABN", "MISS_NEG", "MISS_DEV", "MISS_CHRON", "NAME", "LAT_SIDE", "LAT_CONV", "SPLIT", "OTHER"]


def parts(extraction, idx):
    if "(no findings)" in str(extraction):
        return []
    return [p.split(" | ")[idx].strip() for p in str(extraction).split(" || ") if len(p.split(" | ")) > idx]


def main():
    p = W / (sys.argv[1] if len(sys.argv) > 1 else "e1_label_200.csv")
    s = pd.read_csv(p)
    for f in FIELDS:
        s[f] = pd.to_numeric(s[f], errors="coerce")
    nf = s.extraction.astype(str).str.contains(r"\(no findings\)", regex=True)
    print(f"sheet {p.name}: {len(s)} rows | with findings {int((~nf).sum())} | (no findings) {int(nf.sum())}")
    print(f"{'field':<26}{'all n/rate':>16}{'findings rows n/rate':>24}{'no-findings rows n/rate':>26}")
    for f in FIELDS:
        a, b, c = s[f].dropna(), s.loc[~nf, f].dropna(), s.loc[nf, f].dropna()
        fmt = lambda x: f"{len(x)}/{x.mean():.3f}" if len(x) else "0/-"
        print(f"{f:<26}{fmt(a):>16}{fmt(b):>24}{fmt(c):>26}")
    bad_lat = s[(~nf) & (s.laterality_ok == 0)]
    print("extracted laterality values in rows graded laterality_ok=0:",
          pd.Series([v for e in bad_lat.extraction for v in parts(e, 3)]).value_counts().to_dict())
    print("extracted laterality values in rows graded laterality_ok=1:",
          pd.Series([v for e in s[(~nf) & (s.laterality_ok == 1)].extraction for v in parts(e, 3)]).value_counts().to_dict())
    bad_pol = s[(~nf) & (s.finding_and_polarity_ok == 0)]
    print("extracted polarity values in rows graded finding_and_polarity_ok=0:",
          pd.Series([v for e in bad_pol.extraction for v in parts(e, 1)]).value_counts().to_dict())
    print("missed_finding=1 rows: with findings", int(((~nf) & (s.missed_finding == 1)).sum()),
          "| (no findings)", int((nf & (s.missed_finding == 1)).sum()))
    if "err_code" in s.columns:
        codes = [c.strip().upper() for x in s.err_code.dropna().astype(str) for c in x.replace(";", ",").split(",") if c.strip()]
        vc = pd.Series(codes).value_counts()
        print("error codes:", {c: int(vc.get(c, 0)) for c in CODES if vc.get(c, 0)},
              "| unknown codes:", {k: int(v) for k, v in vc.items() if k not in CODES})
    else:
        print("no err_code column yet")
    print("LABEL_AUDIT_DONE")


if __name__ == "__main__":
    main()
