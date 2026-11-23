#!/usr/bin/env python3
"""D30: primary cohort = age >= 18 or age missing (pre-specified before any model results; CXAS is an adult
anatomy model and paediatric films fail ~25% of the time). Segmentation failures are never used to exclude:
they are flagged and get the global-only fallback.
Writes ~/anchor/work/manifest/flags_{split}.parquet (key, age, cohort_primary, seg_hard_fail, seg_zone_only).
Prints aggregates only."""
import pathlib

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
ZONES = ["R_apical", "R_upper", "R_mid", "R_base", "L_apical", "L_upper", "L_mid", "L_base"]


def main():
    for split in ["train", "valid", "test"]:
        man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet")
        man = man[man.eligible]
        rows = []
        for sp in sorted((HOME / "work/regions" / split).glob("shard_*.npz")):
            z = np.load(sp)
            reg = list(z["regions"]); A = z["area"].astype(np.float32)
            lung_missing = (A[:, reg.index("R_lung")] == 0) | (A[:, reg.index("L_lung")] == 0)
            hard = (z["ok"] == 0) | lung_missing | (z["lung_area"] < 0.15) | (z["lung_area"] > 0.70)
            zone_empty = (A[:, [reg.index(r) for r in ZONES]] == 0).any(1)
            rows.append(pd.DataFrame({"key": z["keys"], "seg_hard_fail": hard, "seg_zone_only": (~hard) & zone_empty}))
        seg = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["key", "seg_hard_fail", "seg_zone_only"])
        f = man[["key", "age"]].merge(seg, on="key", how="left")
        f["seg_cached"] = f.seg_hard_fail.notna()
        f["seg_hard_fail"] = f.seg_hard_fail.fillna(True).astype(bool)      # not cached = treat as failed
        f["seg_zone_only"] = f.seg_zone_only.fillna(False).astype(bool)
        f["cohort_primary"] = f.age.isna() | (f.age >= 18)
        f.to_parquet(HOME / f"work/manifest/flags_{split}.parquet", index=False)
        pc = f[f.cohort_primary]
        print(f"{split:<5} eligible={len(f):>6} primary={len(pc):>6} (age missing {int(f.age.isna().sum())}) "
              f"paediatric<18={int((f.age < 18).sum()):>5} | seg cached={f.seg_cached.mean():.3f} "
              f"hard_fail all={f.seg_hard_fail.mean():.4f} primary={pc.seg_hard_fail.mean():.4f} "
              f"zone_only primary={pc.seg_zone_only.mean():.4f}")
        q = HOME / f"work/manifest/rexvqa_{split}_validated.parquet"
        if q.exists():
            qq = pd.read_parquet(q)
            on = qq.study_id.isin(set(pc.key))
            print(f"      ReXVQA validated questions on primary-cohort studies: {int(on.sum())} "
                  f"(location {int((on & (qq.task_name == 'Location and Distribution Assessment')).sum())})")


if __name__ == "__main__":
    main()
