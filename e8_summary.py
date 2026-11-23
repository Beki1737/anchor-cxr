#!/usr/bin/env python3
"""Final-table summary for any split: CRIMSON / GREEN means (all, normal, abnormal), paired diffs vs B0 and vs template,
CRIMSON false / missed and location errors per report. Aggregates only.
    python e8_summary.py --split test --tags test
    python e8_summary.py --split valid --tags valid,valid_e3,valid_e8 --keys ~/anchor/work/e8/eval_keys_valid.parquet
"""
import argparse
import pathlib
import sys

import pandas as pd

HOME = pathlib.Path.home() / "anchor"
J = HOME / "work/judge"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import paired, study_strata  # noqa: E402

SYS = ["normal_template", "b0_medgemma4b", "e3rag_medgemma4b_k3", "e8anchor", "e8edit", "e8multi", "e8vlmfree"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tags", default="test")
    ap.add_argument("--keys", default="")
    ap.add_argument("--extra-sys", default="", help="comma list of extra condition names, e.g. e16fus,e16fusedit")
    a = ap.parse_args(argv)
    SYS.extend(x for x in a.extra_sys.split(",") if x and x not in SYS)
    st = study_strata(HOME / f"work/corpus_v2/{a.split}")
    keys = set(pd.read_parquet(a.keys).key) if a.keys else None
    for judge in ["crimson", "green"]:
        parts = [pd.read_parquet(J / t / f"{judge}_scores.parquet") for t in a.tags.split(",") if (J / t / f"{judge}_scores.parquet").exists()]
        if not parts:
            continue
        sc = pd.concat(parts).drop_duplicates(["cond", "key"])
        if keys is not None:
            sc = sc[sc.key.isin(keys)]
        W = sc.pivot_table(index="key", columns="cond", values="score").join(st, how="inner")
        print(f"\n=== {judge.upper()} {a.split} (n={len(W)}; normal {int((W.stratum == 'normal').sum())}, abnormal {int((W.stratum == 'abnormal').sum())}) ===")
        print(f"{'system':<20}{'all':>7}{'normal':>8}{'abnormal':>10}   vs B0 [95% CI]                 vs template [95% CI]")
        for s in [x for x in SYS if x in W]:
            line = f"{s:<20}{W[s].mean():>7.3f}{W[W.stratum == 'normal'][s].mean():>8.3f}{W[W.stratum == 'abnormal'][s].mean():>10.3f}"
            for base in ["b0_medgemma4b", "normal_template"]:
                if s != base and base in W:
                    p = paired(W[[s, base]].dropna().rename(columns={s: "sys"}), "sys", base)
                    line += f"   {p['diff']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]"
                else:
                    line += " " * 32
            print(line)
        if judge == "crimson" and "n_false_findings" in sc.columns:
            print("CRIMSON per report: false | missed | location errors   (normal: false | abnormal: false/missed)")
            E = sc.merge(st[["stratum"]], left_on="key", right_index=True)
            for s in [x for x in SYS if x in set(E.cond)]:
                x = E[E.cond == s]; n_, ab = x[x.stratum == "normal"], x[x.stratum == "abnormal"]
                print(f"  {s:<20} {x.n_false_findings.mean():.2f} | {x.n_missing_findings.mean():.2f} | {x.n_location_errors.mean():.3f}"
                      f"   ({n_.n_false_findings.mean():.2f} | {ab.n_false_findings.mean():.2f}/{ab.n_missing_findings.mean():.2f})")
    print("E8_SUMMARY_DONE")


if __name__ == "__main__":
    main()
