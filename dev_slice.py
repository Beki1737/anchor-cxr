#!/usr/bin/env python3
"""Fix the development slice: 3,000 train studies (eligible, primary cohort), seed 7.
Used for prompt and hyper-parameter development only; never reported as results.
Later retrieval for these queries must exclude the query patient's own studies (stored here)."""
import pathlib

import pandas as pd

HOME = pathlib.Path.home() / "anchor/work/manifest"


def main(n=3000, seed=7):
    out = HOME / "dev_slice.parquet"
    if out.exists():
        d = pd.read_parquet(out)
        print(f"dev slice exists: {len(d)} studies, {d.patient.nunique()} patients (unchanged)")
        return
    man = pd.read_parquet(HOME / "rex_train.parquet")
    flags = pd.read_parquet(HOME / "flags_train.parquet")[["key", "cohort_primary", "seg_hard_fail"]]
    m = man[man.eligible].merge(flags, on="key")
    m = m[m.cohort_primary]
    d = m.sample(n=n, random_state=seed)[["key", "patient", "frontal_view", "seg_hard_fail"]].sort_values("key")
    d.to_parquet(out, index=False)
    print(f"dev slice written: {len(d)} studies, {d.patient.nunique()} patients, PA share {(d.frontal_view == 'PA').mean():.3f}, "
          f"seg_hard_fail {d.seg_hard_fail.mean():.4f}")


if __name__ == "__main__":
    main()
