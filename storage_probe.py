#!/usr/bin/env python3
"""Read-only storage probe for the lab cleanup request (D41). Sizes only, no file names with IDs printed.
- ReXGradient: bytes of images the method actually uses (one frontal per eligible study, all splits) vs all images
- lossless PNG recompression estimate on a sample (pixel-identical check), nothing overwritten
"""
import io
import os
import pathlib
import random

import numpy as np
import pandas as pd
from PIL import Image

HOME = pathlib.Path.home() / "anchor"


def walk_sizes(root):
    tot, n = 0, 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            for e in os.scandir(d):
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False) and e.name.lower().endswith(".png"):
                    tot += e.stat().st_size; n += 1
        except PermissionError:
            pass
    return tot, n


def main():
    root = HOME / "data/rexgradient"
    tot, n = walk_sizes(root)
    used = set()
    for s in ["train", "valid", "test"]:
        m = pd.read_parquet(HOME / f"work/manifest/rex_{s}.parquet")
        used |= set(m.loc[m.eligible, "frontal_path"].dropna())
        if "prior_frontal_path" in m.columns:
            used |= set(m.prior_frontal_path.dropna())
    ub = sum(os.path.getsize(p) for p in used if os.path.exists(p))
    print(f"ReXGradient PNGs: {n} files, {tot / 1e9:.1f} GB | used by ANCHOR (eligible frontals): {len(used)} files, "
          f"{ub / 1e9:.1f} GB | not used (laterals, extra frontals, non-chest): {(tot - ub) / 1e9:.1f} GB")
    sample = random.Random(0).sample(sorted(used), min(150, len(used)))
    before = after = 0; same = True
    for p in sample:
        raw = open(p, "rb").read()
        im = Image.open(io.BytesIO(raw)); a = np.array(im)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True, compress_level=9)
        b = np.array(Image.open(io.BytesIO(buf.getvalue())))
        same &= bool(np.array_equal(a, b) and a.dtype == b.dtype)
        before += len(raw); after += len(buf.getvalue())
    print(f"lossless re-compression sample n={len(sample)}: {after / before:.3f} of original size, pixel-identical={same} "
          f"-> est. saving on all PNGs {(1 - after / before) * tot / 1e9:.1f} GB")
    print("STORAGE_PROBE_DONE")


if __name__ == "__main__":
    main()
