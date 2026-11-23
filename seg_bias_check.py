#!/usr/bin/env python3
"""Is segmentation failure correlated with disease or age? (selection-bias check before deciding how to treat failures)
Reads full-cache shards + manifests; prints rates only (no text, no IDs)."""
import json
import pathlib
import re

import numpy as np
import pandas as pd

HOME = pathlib.Path.home() / "anchor"
KW = {"effusion": r"effusion", "opacity/consolidation": r"opacit|consolidat|airspace disease",
      "atelectasis/collapse": r"atelecta|collapse", "pneumothorax": r"pneumothora", "edema": r"edema|congestion",
      "cardiomegaly": r"cardiomegal|enlarged (cardiac|heart)", "support device": r"tube|catheter|line|pacemaker|wire",
      "normal-ish": r"no acute|clear lungs|lungs are clear|no active"}


def main():
    for split in ["test", "valid", "train"]:
        d = HOME / "work/regions" / split
        qs = d / "qa_summary.json"
        if not qs.exists():
            print(f"{split}: no qa_summary.json (cache unfinished?)"); continue
        s = json.load(open(qs))
        print(f"\n=== {split}: images={s['images']} read_errors={s['read_errors']} fail={s['fail_rate']} "
              f"invert={s['invert_suspect_rate']} lat_swap={s['laterality_swap_rate']} "
              f"throughput={s.get('throughput_img_s')} cudnn={s.get('cudnn_enabled')}")
        man = pd.read_parquet(HOME / f"work/manifest/rex_{split}.parquet").set_index("key")
        rows = []
        for sp in sorted(d.glob("shard_*.npz")):
            z = np.load(sp)
            reg = list(z["regions"]); A = z["area"].astype(np.float32)
            rows.append(pd.DataFrame({"key": z["keys"], "ok": z["ok"], "lung_area": z["lung_area"],
                                      "lung_missing": (A[:, reg.index("R_lung")] == 0) | (A[:, reg.index("L_lung")] == 0)}))
        q = pd.concat(rows, ignore_index=True)
        q["hard_fail"] = (q.ok == 0) | q.lung_missing | (q.lung_area < 0.15) | (q.lung_area > 0.70)
        text = (man.loc[q.key, "findings"].fillna("") + " " + man.loc[q.key, "impression"].fillna("")).str.lower().to_numpy()
        age = man.loc[q.key, "age"].to_numpy()
        view = man.loc[q.key, "frontal_view"].to_numpy()
        hf = q.hard_fail.to_numpy()
        print(f"hard failure rate: {hf.mean():.4f}  (n={int(hf.sum())})")
        print(f"{'keyword':<24}{'in failures':>12}{'in others':>12}")
        for name, pat in KW.items():
            m = np.array([bool(re.search(pat, t)) for t in text])
            print(f"{name:<24}{m[hf].mean():>12.3f}{m[~hf].mean():>12.3f}")
        for lo, hi in [(0, 16), (16, 40), (40, 65), (65, 200)]:
            sel = (age >= lo) & (age < hi)
            if sel.sum():
                print(f"age {lo:>3}-{hi:<3}: n={int(sel.sum()):>6}  hard failure rate={hf[sel].mean():.4f}")
        for v in ["AP", "PA"]:
            sel = view == v
            print(f"view {v}: hard failure rate={hf[sel].mean():.4f}")


if __name__ == "__main__":
    main()
