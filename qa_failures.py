#!/usr/bin/env python3
"""Break down E0 mask failures by reason and view; draw overlays of failed cases (local only, never upload).

    python qa_failures.py --tag test_pilot2000 --split test --n 12
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import region_cache as rc  # noqa: E402

HOME = pathlib.Path.home() / "anchor"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="test_pilot2000")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--regions", default=str(HOME / "work/regions"))
    ap.add_argument("--manifest", default=str(HOME / "work/manifest"))
    ap.add_argument("--out", default=str(HOME / "work/qa_overlays_fail"))
    a = ap.parse_args(argv)

    man = pd.read_parquet(pathlib.Path(a.manifest) / f"rex_{a.split}.parquet").set_index("key")
    parts = []
    for sp in sorted((pathlib.Path(a.regions) / a.tag).glob("shard_*.npz")):
        z = np.load(sp)
        regions = list(z["regions"])
        n = len(z["keys"])
        m64 = np.unpackbits(z["m64"], axis=-1)[..., :4096].reshape(n, len(regions), 64, 64).astype(bool)
        parts.append(dict(keys=z["keys"], area=z["area"].astype(np.float32), lung_area=z["lung_area"],
                          ok=z["ok"], h=z["h"], w=z["w"], m64=m64))
    keys = np.concatenate([p["keys"] for p in parts]); A = np.concatenate([p["area"] for p in parts])
    M = np.concatenate([p["m64"] for p in parts])
    df = pd.DataFrame({"key": keys, "lung_area": np.concatenate([p["lung_area"] for p in parts]),
                       "ok": np.concatenate([p["ok"] for p in parts]),
                       "h": np.concatenate([p["h"] for p in parts]), "w": np.concatenate([p["w"] for p in parts])})
    df["view"] = man.loc[df.key, "frontal_view"].to_numpy()
    ri = {r: regions.index(r) for r in regions}
    df["lung_missing"] = (A[:, ri["R_lung"]] == 0) | (A[:, ri["L_lung"]] == 0)
    df["lung_low"] = df.lung_area < 0.15
    df["lung_high"] = df.lung_area > 0.70
    empty = {z: A[:, ri[z]] == 0 for z in rc.ZONES}
    df["zones_empty"] = np.sum(np.stack(list(empty.values()), 1), 1)
    df["hard_fail"] = (df.ok == 0) | df.lung_missing | df.lung_low | df.lung_high
    df["zone_only"] = (~df.hard_fail) & (df.zones_empty > 0)

    print(f"images={len(df)}")
    print(f"hard failures (unreadable, a lung missing, lung area outside 15-70%): {df.hard_fail.mean():.4f}")
    for c in ["lung_missing", "lung_low", "lung_high"]:
        print(f"   {c}: {int(df[c].sum())}")
    print(f"zone-only (lungs fine, >=1 of 8 zones empty): {df.zone_only.mean():.4f}")
    print("   empty-zone counts among zone-only:", {z: int((m & df.zone_only.to_numpy()).sum()) for z, m in empty.items()})
    print("by view:")
    print(df.groupby("view")[["hard_fail", "zone_only"]].mean().round(4).to_string())
    print("original size (median h x w): hard_fail",
          df[df.hard_fail][["h", "w"]].median().tolist(), "| others", df[~df.hard_fail][["h", "w"]].median().tolist())

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    k = 0
    for label, sel in [("hard", df.hard_fail), ("zone", df.zone_only)]:
        for j, i in enumerate(np.flatnonzero((sel & (df.ok == 1)).to_numpy())[: a.n]):
            try:
                gray, _ = rc.load_gray01(man.loc[df.key.iat[i], "frontal_path"])
            except Exception:
                continue
            g512 = rc.resize(gray, 512)
            masks = M[i].repeat(8, axis=1).repeat(8, axis=2)
            rc.save_overlay(out / f"{label}_{j:02d}_{df.view.iat[i]}.png", g512, masks, f"{label} #{j} {df.view.iat[i]}")
            k += 1
    print(f"failure overlays written: {k} in {out} (local only)")


if __name__ == "__main__":
    main()
