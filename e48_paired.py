#!/usr/bin/env python3
"""Round 48: paired CRIMSON / GREEN differences from judge score files (study bootstrap + patient-cluster CI).
python e48_paired.py --split valid --tags valid,valid_e8,valid_e24 --pairs e24regionp:e24globalp,... [--keys eval]
~/venvs/anchor_eval, CPU. Aggregates only."""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata import paired  # noqa: E402


def cluster(d, pat, reps=2000, seed=48):
    up, inv = np.unique(pat, return_inverse=True); s, c = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(seed); b = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)); b.append((w @ s) / (w @ c))
    return np.quantile(b, [0.025, 0.975])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid"); ap.add_argument("--tags", required=True); ap.add_argument("--pairs", required=True)
    ap.add_argument("--keys", default="all", help="'eval' = valid folds 2-3")
    a = ap.parse_args()
    man = pd.read_parquet(HOME / f"work/manifest/rex_{a.split}.parquet").set_index("key")
    keys = set(pd.read_parquet(HOME / "work/e8/eval_keys_valid.parquet").key) if a.keys == "eval" else None
    for judge in ["crimson", "green"]:
        parts = [pd.read_parquet(p) for t in a.tags.split(",") for p in [HOME / "work/judge" / t / f"{judge}_scores.parquet"] if p.exists()]
        if not parts:
            continue
        sc = pd.concat(parts)
        if keys is not None:
            sc = sc[sc.key.isin(keys)]
        W = sc.drop_duplicates(["cond", "key"]).pivot_table(index="key", columns="cond", values="score")
        for pr in a.pairs.split(","):
            x_, y_ = pr.split(":")
            if x_ not in W or y_ not in W:
                print(f"{judge.upper():<8} {x_} vs {y_}: missing"); continue
            x = W[[x_, y_]].dropna(); p = paired(x.rename(columns={x_: "sys"}), "sys", y_)
            cl = cluster((x[x_] - x[y_]).to_numpy(), man.reindex(x.index).patient.astype(str).to_numpy())
            print(f"{judge.upper():<8} {x_} minus {y_}: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] "
                  f"(patient-cluster [{cl[0]:+.4f}, {cl[1]:+.4f}]) n {p['n']} | means {x[x_].mean():.4f} vs {x[y_].mean():.4f}")
    print("E48_PAIRED_DONE")


if __name__ == "__main__":
    main()
