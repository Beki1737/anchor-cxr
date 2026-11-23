#!/usr/bin/env python3
"""MIMIC-CXR external test manifest in ANCHOR format (local only; reports never leave devbox).
One frontal 1024-px JPG per study (manifest_frontal_1024.csv) + report sections parsed from mimic-cxr-reports.zip.
Writes ~/anchor/work/manifest/rex_mimic.parquet and flags_mimic.parquet. Prints aggregates only.
Studies without a FINDINGS section are kept in the manifest but marked eligible=False (standard practice)."""
import pathlib
import re
import zipfile

import pandas as pd

HOME = pathlib.Path.home() / "anchor"
D = HOME / "data"
HDR = re.compile(r"(?m)^\s*([A-Z][A-Z /()]{2,40}):")


def sections(text):
    parts, last, name = {}, 0, None
    for m in HDR.finditer(text):
        if name:
            parts[name] = text[last:m.start()]
        name, last = m.group(1).strip(), m.end()
    if name:
        parts[name] = text[last:]
    return {k: " ".join(v.split()) for k, v in parts.items()}


def main():
    m = pd.read_csv(D / "manifest_frontal_1024.csv")
    m = m[m.split.astype(str).str.lower() == "test"].copy()
    z = zipfile.ZipFile(D / "reports/mimic-cxr-reports.zip")
    names = set(z.namelist())
    rows = []
    for r in m.itertuples(index=False):
        sid, pid = str(int(r.subject_id)), str(int(r.study_id))
        path = f"files/p{sid[:2]}/p{sid}/s{pid}.txt"
        txt = z.read(path).decode("utf-8", "ignore") if path in names else ""
        s = sections(txt)
        ind = s.get("INDICATION") or s.get("HISTORY") or s.get("REASON FOR EXAMINATION") or s.get("CLINICAL INFORMATION") or ""
        img = pathlib.Path(str(r.path))
        if not img.is_absolute():
            img = D / "jpg1024" / img if not (D / img).exists() else D / img
        rows.append({"key": f"m{pid}", "patient": f"p{sid}", "findings": s.get("FINDINGS", ""), "impression": s.get("IMPRESSION", ""),
                     "indication": ind, "frontal_path": str(img) if img.exists() else None, "frontal_view": str(r.view),
                     "institution": "MIMIC", "manufacturer": "MIMIC", "prior_key": None, "has_report": bool(txt)})
    df = pd.DataFrame(rows).drop_duplicates("key")
    df["eligible"] = df.findings.str.len().gt(0) & df.frontal_path.notna()
    (HOME / "work/manifest").mkdir(parents=True, exist_ok=True)
    df.to_parquet(HOME / "work/manifest/rex_mimic.parquet", index=False)
    pd.DataFrame({"key": df.key, "age": float("nan"), "cohort_primary": df.eligible}).to_parquet(HOME / "work/manifest/flags_mimic.parquet", index=False)
    print(f"MIMIC test studies {len(df)} | report found {int(df.has_report.sum())} | image found {int(df.frontal_path.notna().sum())} | "
          f"with FINDINGS (eligible) {int(df.eligible.sum())} | with IMPRESSION {int((df.impression.str.len() > 0).sum())} | "
          f"views {df.frontal_view.value_counts().to_dict()}")
    print("MIMIC_MANIFEST_DONE")


if __name__ == "__main__":
    main()
