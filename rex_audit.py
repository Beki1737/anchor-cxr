#!/usr/bin/env python3
"""Aggregate-only audit of ReXGradient-160K + ReXVQA.
Prints counts, rates and masked templates only. Never prints report text, IDs or institution names
(safe to paste into a chat under the data use agreements)."""
import collections
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

H = pathlib.Path.home() / "anchor/data"
M = H / "rexgradient/metadata"
V = H / "rexvqa/metadata"
SPLITS = ["train", "valid", "test"]
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-_]{7,}")


def mask(s):
    return ID_RE.sub("<id>", str(s))


def hdr(t):
    print(f"\n===== {t} =====", flush=True)


def load_json(p):
    d = json.load(open(p))
    if isinstance(d, dict):
        df = pd.DataFrame.from_dict(d, orient="index")
        df.index.name = "_key"
        return df.reset_index()
    return pd.DataFrame(d)


def flat(series):
    out = []
    for v in series.dropna():
        if isinstance(v, (list, tuple, np.ndarray)):
            out.extend(list(v))
        else:
            out.append(v)
    return out


def to_date(x):
    return pd.to_datetime(x.astype(str), format="%Y%m%d", errors="coerce")


def section(name, fn):
    try:
        fn()
    except Exception as e:  # keep going, report the failure
        print(f"  [section {name} failed: {type(e).__name__}: {e}]")


meta = {s: load_json(M / f"{s}_metadata.json") for s in SPLITS}
vpj = {s: load_json(M / f"{s}_metadata_view_position.json") for s in SPLITS}
csvs = {s: pd.read_csv(M / f"{s}_metadata.csv", low_memory=False) for s in SPLITS}


def s1():
    hdr("1. sizes")
    for s in SPLITS:
        df = meta[s]
        print(f"{s:6} studies={len(df)} patients={df.PatientID.nunique()} "
              f"institutions={df.InstitutionName.nunique()} manufacturers={df.Manufacturer.nunique()}")


def s2():
    hdr("2. key structure (masked)")
    for s in SPLITS:
        df, c = meta[s], csvs[s]
        print(f"{s}: json key template {collections.Counter(mask(k) for k in df['_key'].head(2000)).most_common(2)}")
        print(f"  csv.id template {collections.Counter(mask(k) for k in c['id'].head(2000)).most_common(2)}")
        for col in ["_key", "AccessionNumber", "StudyInstanceUid"]:
            print(f"  frac csv.id found in json.{col}: {c['id'].isin(set(df[col])).mean():.3f}")
        print(f"  json vs view_position.json same key order: {list(df['_key']) == list(vpj[s]['_key'])}")


def s3():
    hdr("3. leakage: patient and accession overlap across splits")
    P = {s: set(meta[s].PatientID) for s in SPLITS}
    A = {s: set(meta[s].AccessionNumber) for s in SPLITS}
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        shared = P[a] & P[b]
        print(f"{a}&{b}: shared patients={len(shared)}  {b} studies from shared patients="
              f"{int(meta[b].PatientID.isin(shared).sum())}  shared accessions={len(A[a] & A[b])}")


def s4():
    hdr("4. institution shift (counts only, no names)")
    I = {s: set(meta[s].InstitutionName) for s in SPLITS}
    for s in ["valid", "test"]:
        unseen = I[s] - I["train"]
        print(f"{s}: institutions={len(I[s])} unseen_in_train={len(unseen)} "
              f"studies_at_unseen={int(meta[s].InstitutionName.isin(unseen).sum())}")
    for s in SPLITS:
        vc = meta[s].InstitutionName.value_counts()
        print(f"{s}: top1 share={vc.iloc[0]/vc.sum():.3f} top5 share={vc.iloc[:5].sum()/vc.sum():.3f} "
              f"median studies/inst={vc.median():.0f} inst>=200 studies={(vc >= 200).sum()} inst>=1000={(vc >= 1000).sum()}")


def s5():
    hdr("5. longitudinal priors")
    allm = pd.concat([meta[s][["PatientID", "StudyDate"]].assign(split=s) for s in SPLITS])
    allm["d"] = to_date(allm.StudyDate)
    allm = allm.sort_values(["PatientID", "d"], kind="stable")
    g = allm.groupby("PatientID")
    allm["prev_d"] = g["d"].shift(1)
    allm["prev_split"] = g["split"].shift(1)
    for s in SPLITS:
        sub = allm[allm.split == s]
        has = sub.prev_d.notna()
        gap = (sub.d - sub.prev_d).dt.days[has]
        print(f"{s}: with earlier study (any split)={has.mean():.3f}  same-day prior={(gap == 0).mean() if len(gap) else 0:.3f}  "
              f"gap days median={gap.median() if len(gap) else 'NA'}  prior split={sub.prev_split.value_counts().to_dict()}")
        spp = meta[s].groupby("PatientID").size()
        print(f"   studies/patient: mean={spp.mean():.2f} max={spp.max()} share patients>=2={(spp >= 2).mean():.3f}")
    print(f"date range years: {int(allm.d.dt.year.min())} to {int(allm.d.dt.year.max())}")


def s6():
    hdr("6. images, views, shapes")
    for s in SPLITS:
        df = meta[s]
        n_img = df.ImagePath.apply(lambda L: len(L) if isinstance(L, (list, tuple)) else 0)
        print(f"{s}: images total={int(n_img.sum())} per study dist={n_img.value_counts().sort_index().head(6).to_dict()}")
        v1 = collections.Counter(str(x) for x in flat(df.ImageViewPosition))
        v2 = collections.Counter(str(x) for x in flat(vpj[s].ImageViewPosition))
        print(f"   views metadata.json: {v1.most_common(8)}")
        print(f"   views view_position.json: {v2.most_common(8)}")
        m2 = vpj[s].set_index("AccessionNumber").ImageViewPosition
        diff = sum(1 for a, L in zip(df.AccessionNumber, df.ImageViewPosition) if list(L) != list(m2.get(a, [])))
        print(f"   studies where view lists differ between files: {diff}")
        front = vpj[s].ImageViewPosition.apply(lambda L: any(str(x).upper() in ("PA", "AP") for x in L)).sum()
        print(f"   studies with >=1 PA/AP (view_position.json): {int(front)} / {len(df)}")
        print(f"   modality {collections.Counter(map(str, flat(df.ImageModality))).most_common(4)}")
        print(f"   bodypart {collections.Counter(map(str, flat(df.ImageBodyPart))).most_common(4)}")
        print(f"   path templates {collections.Counter(mask(p) for p in flat(df.ImagePath)[:20000]).most_common(3)}")
        hs, ws = [], []
        for L in df.ImageShape.dropna():
            for sh in (L if isinstance(L, (list, tuple)) else [L]):
                nums = sh if isinstance(sh, (list, tuple)) else [int(n) for n in re.findall(r"\d+", str(sh))]
                if len(nums) >= 2:
                    hs.append(nums[0]); ws.append(nums[1])
        if hs:
            print(f"   shape h p5/p50/p95={np.percentile(hs, [5, 50, 95]).astype(int).tolist()} "
                  f"w p5/p50/p95={np.percentile(ws, [5, 50, 95]).astype(int).tolist()}")


def s7():
    hdr("7. report sections (lengths only)")
    for s in SPLITS:
        df = meta[s]
        for col in ["Indication", "Comparison", "Findings", "Impression"]:
            x = df[col].fillna("").astype(str)
            wc = x.str.split().str.len()
            print(f"{s:6} {col:<11} empty={(x.str.strip() == '').mean():.3f} words p50={wc.median():.0f} p95={wc.quantile(.95):.0f}")
        none_like = df.Comparison.fillna("").str.strip().str.lower().str.rstrip(".").isin(["", "none", "n/a", "na"]).mean()
        print(f"   comparison none-like share={none_like:.3f}")
        dup = df.Findings.fillna("").duplicated(keep=False).mean()
        print(f"   share of studies whose Findings text is duplicated elsewhere in split={dup:.3f}")


def s8():
    hdr("8. interstitial_pattern_bbox.json (structure only)")
    b = json.load(open(M / "interstitial_pattern_bbox.json"))
    first = b[0] if isinstance(b, list) else next(iter(b.items()))
    print(f"container={type(b).__name__} n={len(b)} first_type={type(first).__name__}")
    if isinstance(first, dict):
        print(f"keys={list(first)}")
        bx = first.get("boxes")
        print(f"boxes type={type(bx).__name__} len={len(bx) if hasattr(bx, '__len__') else 'NA'} "
              f"elem={type(bx[0]).__name__ if bx else 'NA'} elem_len={len(bx[0]) if bx and hasattr(bx[0], '__len__') else 'NA'}")
        imgs = [d.get("image") for d in b]
        print(f"image templates {collections.Counter(mask(i) for i in imgs).most_common(2)}")
        for s in SPLITS:
            paths = [str(p) for p in flat(meta[s].ImagePath)]
            full, names = set(paths), {pathlib.PurePath(p).name for p in paths}
            hits = sum(1 for i in imgs if str(i) in full or pathlib.PurePath(str(i)).name in names)
            print(f"   bbox images matched (full path or file name) in {s}: {hits}")


def s9():
    hdr("9. ReXVQA")
    keys = {s: {c: set(meta[s][c]) for c in ["_key", "AccessionNumber", "StudyInstanceUid"]} for s in SPLITS}
    for s in SPLITS:
        q = load_json(V / f"{s}_vqa_data.json")
        q["vc"] = pd.to_numeric(q.validation_code, errors="coerce")
        print(f"{s}: questions={len(q)} studies={q.study_id.nunique()}")
        print(pd.crosstab(q.task_name, q.vc.fillna(-9).astype(int), margins=True).to_string())
        for c, ks in keys[s].items():
            print(f"   frac study_id in ReXGradient {s}.{c}: {q.study_id.isin(ks).mean():.3f}")
        print(f"   options per question {collections.Counter(q.options.apply(len)).most_common(4)}")
        print(f"   questions/study p50={q.groupby('study_id').size().median():.0f}")
        del q
    readme = (H / "rexvqa/README.md")
    if readme.exists():
        print("\nREADME lines mentioning validation:")
        for line in readme.read_text(errors="ignore").splitlines():
            if re.search(r"validat|valid_code|quality|filter", line, re.I):
                print("   " + line.strip()[:200])


for name, fn in [("1", s1), ("2", s2), ("3", s3), ("4", s4), ("5", s5), ("6", s6), ("7", s7), ("8", s8), ("9", s9)]:
    section(name, fn)
print("\nAUDIT_DONE")
