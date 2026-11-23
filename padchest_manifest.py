#!/usr/bin/env python3
"""PadChest-GR manifest in ANCHOR format (local only) for the region cache and the E10 pointing game.
Writes ~/anchor/work/manifest/rex_padchest.parquet (+ flags_padchest.parquet). Keys are anonymous (pc00000...)."""
import json
import pathlib

import pandas as pd

HOME = pathlib.Path.home() / "anchor"
P = HOME / "data/padchest_gr"


def main():
    d = json.load(open(P / "grounded_reports_20240819.json"))
    files = {f.name: f for f in (P / "images").rglob("*.png")}
    rows = []
    for i, s in enumerate(d):
        f = files.get(pathlib.Path(s["ImageID"]).name)
        en = " ".join(x.get("sentence_en") or "" for x in s["findings"])
        rows.append({"key": f"pc{i:05d}", "image_id": s["ImageID"], "patient": f"pc{i:05d}", "findings": en, "impression": "",
                     "indication": "", "frontal_path": str(f) if f else None, "frontal_view": "PA",
                     "institution": "PADCHEST", "manufacturer": "PADCHEST", "prior_key": None})
    df = pd.DataFrame(rows)
    df["eligible"] = df.frontal_path.notna()
    df.to_parquet(HOME / "work/manifest/rex_padchest.parquet", index=False)
    pd.DataFrame({"key": df.key, "age": float("nan"), "cohort_primary": df.eligible}).to_parquet(HOME / "work/manifest/flags_padchest.parquet", index=False)
    print(f"PadChest-GR studies {len(df)} | image found {int(df.eligible.sum())}")
    print("PADCHEST_MANIFEST_DONE")


if __name__ == "__main__":
    main()
