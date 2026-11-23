#!/usr/bin/env python3
"""ANCHOR step 1: leakage-safe study manifests for ReXGradient-160K (+ validated ReXVQA join).

Rules (from the r07 audit):
  * view labels come from *_metadata_view_position.json (metadata.json views are ~90% empty)
  * a study is eligible only if it has a Chest CR/DX frontal image (PA preferred over AP)
  * priors: same patient, same split, strictly earlier StudyDate (same-day studies are NOT priors, order unknown)
  * institutions are replaced by anonymous indices (I00 = largest in train); the name map stays local
  * ReXVQA: only validation_code == 1 (answer consistent with the report) is kept as an exact-label set
  * official splits are patient-disjoint; this is re-asserted here

Writes (local only, contains report text, never share):
  ~/anchor/work/manifest/rex_{train,valid,test}.parquet
  ~/anchor/work/manifest/rexvqa_{valid,test}_validated.parquet
  ~/anchor/work/manifest/institution_map.json
  ~/anchor/work/manifest/summary.json            (aggregates only, safe to share)
Prints aggregates only.
"""
import argparse
import hashlib
import json
import pathlib
import random
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

HOME = pathlib.Path.home() / "anchor"
REX = HOME / "data/rexgradient"
META = REX / "metadata"
VQA = HOME / "data/rexvqa/metadata"
OUT = HOME / "work/manifest"
SPLITS = ["train", "valid", "test"]
FRONT = {"PA": "PA", "POSTERO_ANTERIOR": "PA", "AP": "AP"}
LAT = {"LATERAL", "LL"}
OK_MOD = {"CR", "DX"}


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def load_split(split, root=META):
    d = json.load(open(root / f"{split}_metadata_view_position.json"))
    df = pd.DataFrame.from_dict(d, orient="index")
    df.index.name = "key"
    return df.reset_index()


def load_records(p):
    """ReXVQA/ReXGradient JSON may be a list of records or a dict keyed by id (the real files are dicts)."""
    d = json.load(open(p))
    if isinstance(d, dict):
        v = next(iter(d.values()))
        if isinstance(v, dict):
            df = pd.DataFrame.from_dict(d, orient="index")
            df.index.name = "_key"
            return df.reset_index()
        return pd.DataFrame(d)
    return pd.DataFrame(d)


def parse_age(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    m = re.match(r"\s*(\d+)\s*([YMWD]?)", str(a).upper())
    if not m:
        return np.nan
    v, u = int(m.group(1)), (m.group(2) or "Y")
    return {"Y": v, "M": v / 12, "W": v / 52, "D": v / 365}[u]


def norm_text(t):
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


def make_resolver(rex_root):
    roots = [rex_root / "images", rex_root]

    def resolve(p):
        if p is None:
            return None
        q = str(p)
        q = q[3:] if q.startswith("../") else q.lstrip("/")
        parts = pathlib.PurePosixPath(q).parts
        for r in roots:
            for c in ([r / q] + ([r / pathlib.PurePosixPath(*parts[1:])] if len(parts) > 1 else [])):
                if c.exists():
                    return str(c)
        return None
    return resolve


def per_study(df):
    def g(L, i):
        return L[i] if isinstance(L, (list, tuple)) and i < len(L) else None
    rows = []
    for r in df.itertuples(index=False):
        paths = list(r.ImagePath) if isinstance(r.ImagePath, (list, tuple)) else []
        cands, lat, chest_any = [], None, False
        for i in range(len(paths)):
            v = str(g(r.ImageViewPosition, i)).upper()
            mod = str(g(r.ImageModality, i)).upper()
            bp = str(g(r.ImageBodyPart, i)).lower()
            chest = bp == "chest" and mod in OK_MOD
            chest_any |= chest
            if chest and v in FRONT:
                cands.append((0 if FRONT[v] == "PA" else 1, i, FRONT[v]))
            if chest and v in LAT and lat is None:
                lat = paths[i]
        cands.sort()
        f = cands[0] if cands else None
        rows.append({
            "key": r.key, "patient": r.PatientID, "accession": r.AccessionNumber,
            "institution_raw": r.InstitutionName, "manufacturer_raw": r.Manufacturer,
            "sex": r.PatientSex, "age": parse_age(r.PatientAge),
            "study_date": pd.to_datetime(str(r.StudyDate), format="%Y%m%d", errors="coerce"),
            "study_description": r.StudyDescription,
            "indication": r.Indication, "comparison": r.Comparison,
            "findings": r.Findings, "impression": r.Impression,
            "n_images": len(paths), "chest_ok": bool(chest_any),
            "frontal_rel": paths[f[1]] if f else None, "frontal_view": f[2] if f else None,
            "frontal_shape_meta": str(g(r.ImageShape, f[1])) if f else None,
            "lateral_rel": lat,
        })
    return pd.DataFrame(rows)


def add_priors(df):
    """Latest strictly-earlier study of the same patient in the same split that has a frontal image."""
    df = df.sort_values(["split", "patient", "study_date", "key"], kind="stable").reset_index(drop=True)
    prior_key = [None] * len(df)
    prior_gap = [np.nan] * len(df)
    has_f = df.frontal_rel.notna().to_numpy()
    for _, idx in df.groupby(["split", "patient"], sort=False).indices.items():
        if len(idx) < 2:
            continue
        dates = df.study_date.to_numpy()[idx]
        for a in range(1, len(idx)):
            for b in range(a - 1, -1, -1):
                if pd.notna(dates[b]) and pd.notna(dates[a]) and dates[b] < dates[a] and has_f[idx[b]]:
                    prior_key[idx[a]] = df.key.iat[idx[b]]
                    prior_gap[idx[a]] = (dates[a] - dates[b]) / np.timedelta64(1, "D")
                    break
    df["prior_key"] = prior_key
    df["prior_gap_days"] = prior_gap
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--vqa", default=str(VQA))
    ap.add_argument("--rex-root", default=str(REX))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--png-sample", type=int, default=300)
    a = ap.parse_args(argv)
    meta, vqa, rex_root, out = map(pathlib.Path, (a.meta, a.vqa, a.rex_root, a.out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    log("loading metadata")
    parts = []
    for s in SPLITS:
        d = per_study(load_split(s, meta))
        d["split"] = s
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    # leakage asserts
    pats = {s: set(df.patient[df.split == s]) for s in SPLITS}
    accs = {s: set(df.accession[df.split == s]) for s in SPLITS}
    for x, y in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        assert not (pats[x] & pats[y]), f"patient overlap {x}/{y}"
        assert not (accs[x] & accs[y]), f"accession overlap {x}/{y}"
    summary["patient_disjoint"] = True

    # anonymous institution / manufacturer indices
    inst_rank = df[df.split == "train"].institution_raw.value_counts()
    imap = {name: f"I{i:02d}" for i, name in enumerate(inst_rank.index)}
    for name in df.institution_raw.unique():
        imap.setdefault(name, f"IX{len(imap):02d}")
    mrank = df[df.split == "train"].manufacturer_raw.value_counts()
    mmap = {name: f"M{i:02d}" for i, name in enumerate(mrank.index)}
    for name in df.manufacturer_raw.unique():
        mmap.setdefault(name, f"MX{len(mmap):02d}")
    df["institution"] = df.institution_raw.map(imap)
    df["manufacturer"] = df.manufacturer_raw.map(mmap)
    json.dump({"institution": {str(k): v for k, v in imap.items()},
               "manufacturer": {str(k): v for k, v in mmap.items()}},
              open(out / "institution_map.json", "w"), indent=1)
    df = df.drop(columns=["institution_raw", "manufacturer_raw"])

    log("priors")
    df = add_priors(df)

    # dedupe signal for the retrieval corpus (train only)
    df["findings_hash"] = df.findings.map(lambda t: hashlib.md5(norm_text(t).encode()).hexdigest())
    tr_counts = df[df.split == "train"].findings_hash.value_counts()
    df["findings_dup_in_train"] = df.findings_hash.map(tr_counts).fillna(0).astype(int)

    log("resolving image paths")
    resolve = make_resolver(rex_root)
    df["frontal_path"] = df.frontal_rel.map(resolve)
    df["lateral_path"] = df.lateral_rel.map(resolve)
    df["eligible"] = df.chest_ok & df.frontal_rel.notna()

    for s in SPLITS:
        d = df[df.split == s]
        e = d[d.eligible]
        inst_ge200 = int((e.institution.value_counts() >= 200).sum())
        summ = {
            "studies": len(d), "chest_ok": int(d.chest_ok.sum()), "eligible_frontal": len(e),
            "frontal_PA_share": round(float((e.frontal_view == "PA").mean()), 3) if len(e) else None,
            "with_lateral": int(e.lateral_rel.notna().sum()),
            "with_prior_strict": int(e.prior_key.notna().sum()),
            "prior_gap_days_p50": float(np.nanmedian(e.prior_gap_days)) if e.prior_key.notna().any() else None,
            "frontal_path_resolved": int(e.frontal_path.notna().sum()),
            "age_parsed": round(float(d.age.notna().mean()), 3),
            "institutions_eligible": int(e.institution.nunique()), "institutions_ge200_eligible": inst_ge200,
            "unique_findings_share": round(float(e.findings_hash.nunique() / max(len(e), 1)), 3),
        }
        summary[s] = summ
        log(f"{s}: {summ}")
    tr = df[(df.split == "train") & df.eligible]
    summary["train_corpus_dedup_unique_findings"] = int(tr.findings_hash.nunique())

    # image sanity on a random sample of resolved frontal PNGs
    paths = df[(df.split == "test") & df.frontal_path.notna()].frontal_path.tolist()
    random.Random(0).shuffle(paths)
    modes, hs, ws = {}, [], []
    for p in paths[: a.png_sample]:
        try:
            with Image.open(p) as im:
                modes[im.mode] = modes.get(im.mode, 0) + 1
                hs.append(im.height); ws.append(im.width)
        except Exception as e:
            modes[f"error:{type(e).__name__}"] = modes.get(f"error:{type(e).__name__}", 0) + 1
    if hs:
        summary["png_sample"] = {"n": len(hs), "modes": modes,
                                 "h_p5_p50_p95": np.percentile(hs, [5, 50, 95]).astype(int).tolist(),
                                 "w_p5_p50_p95": np.percentile(ws, [5, 50, 95]).astype(int).tolist()}
    else:
        summary["png_sample"] = {"n": 0, "note": "no resolved PNGs yet (extraction unfinished?)", "modes": modes}
    log(f"png sample: {summary['png_sample']}")

    for s in SPLITS:
        df[df.split == s].drop(columns=["split"]).to_parquet(out / f"rex_{s}.parquet", index=False)

    log("ReXVQA validated join")
    for s in ["valid", "test"]:
        q = load_records(vqa / f"{s}_vqa_data.json")
        q["vc"] = pd.to_numeric(q.validation_code, errors="coerce")
        keep = [c for c in ["study_id", "task_name", "category", "subcategory", "class", "question", "options",
                            "correct_answer"] if c in q.columns]
        qv = q[q.vc == 1][keep].copy()
        elig = set(df[(df.split == s) & df.eligible].key)
        qv["study_eligible"] = qv.study_id.isin(elig)
        qv.to_parquet(out / f"rexvqa_{s}_validated.parquet", index=False)
        by_task = qv[qv.study_eligible].task_name.value_counts().to_dict()
        summary[f"rexvqa_{s}"] = {"all_questions": len(q), "validated": len(qv),
                                  "validated_on_eligible_studies": int(qv.study_eligible.sum()), "by_task": by_task}
        log(f"rexvqa {s}: {summary[f'rexvqa_{s}']}")

    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=str)
    log("MANIFEST_DONE")


if __name__ == "__main__":
    main()
