#!/usr/bin/env python3
"""Direct paired comparisons between two systems' per-report judge scores (any split; ~/venvs/anchor_eval; CPU).
    python e16_paired.py --split valid --tags valid,valid_e8,valid_e16 --keys ~/anchor/work/e8/eval_keys_valid.parquet \
        --pairs e16fusmulti:e8anchor,e16fus:e8anchor,e16fusedit:e8edit,e16fusmulti:e8multi
Study bootstrap 95% CI (strata.paired), overall and by E1 normal / abnormal stratum. Aggregates only."""
import argparse
import pathlib
import sys

import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import paired, study_strata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid"); ap.add_argument("--tags", required=True)
    ap.add_argument("--keys", default=""); ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    st = study_strata(HOME / f"work/corpus_v2/{a.split}")
    keys = set(pd.read_parquet(a.keys).key) if a.keys else None
    for judge in ["crimson", "green"]:
        parts = [pd.read_parquet(HOME / "work/judge" / t / f"{judge}_scores.parquet") for t in a.tags.split(",")
                 if (HOME / "work/judge" / t / f"{judge}_scores.parquet").exists()]
        if not parts:
            continue
        sc = pd.concat(parts).drop_duplicates(["cond", "key"])
        if keys is not None:
            sc = sc[sc.key.isin(keys)]
        W = sc.pivot_table(index="key", columns="cond", values="score").join(st, how="inner")
        print(f"\n{judge.upper()} {a.split}")
        for pr in a.pairs.split(","):
            x, y = pr.split(":")
            if x not in W or y not in W:
                print(f"  {x} vs {y}: missing"); continue
            out = []
            for nm, sel in [("all", W.index == W.index), ("normal", ~W.abn), ("abnormal", W.abn)]:
                p = paired(W[sel][[x, y]].dropna().rename(columns={x: "sys"}), "sys", y)
                out.append(f"{nm} {p['diff']:+.4f} [{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]")
            print(f"  {x:<14} minus {y:<14} " + " | ".join(out))
    print("E16_PAIRED_DONE")


if __name__ == "__main__":
    main()
