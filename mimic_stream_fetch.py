#!/usr/bin/env python3
"""
Selective, streaming download of MIMIC-CXR-JPG v2.1.0 (frontal PA/AP views only),
plus the free-text reports (which are NOT inside MIMIC-CXR-JPG).

Per patient-group folder (p10 ... p19):
  1. fetch raw JPGs (wget -c, resumes partial files)
  2. verify SHA256 against PhysioNet SHA256SUMS.txt (on by default)
  3. resize (longest side = --size, default 1024), save JPEG q95
  4. delete that raw folder only if every file succeeded
Peak disk ~ one raw folder (~35 to 60 GB) + resized output (~40 to 60 GB total).

Auth: create <CREDENTIALS_FILE> on the server once (chmod 600):
     Configure PhysioNet credentials per your local environment.
     wget uses configured credentials automatically for non-interactive jobs.

Steps:
    python mimic_stream_fetch.py --root /path/mimic --dry-run      # metadata + counts only
    python mimic_stream_fetch.py --root /path/mimic --folders p10  # one folder test
    nohup python mimic_stream_fetch.py --root /path/mimic \
        > fetch_all.log 2>&1 & echo $! > fetch.pid ; disown -a
"""
import argparse
import csv
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0"
REPORTS_URLS = [  # reports live in the MIMIC-CXR (DICOM) project
    "https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip",
    "https://physionet.org/files/mimic-cxr/2.0.0/mimic-cxr-reports.zip",
]
REQUIRED_META = ["mimic-cxr-2.0.0-metadata.csv.gz", "mimic-cxr-2.0.0-split.csv.gz"]
OPTIONAL_META = [
    "mimic-cxr-2.0.0-chexpert.csv.gz",
    "mimic-cxr-2.0.0-negbio.csv.gz",
    "mimic-cxr-2.1.0-test-set-labeled.csv",
    "SHA256SUMS.txt",
    "IMAGE_FILENAMES",
    "README",
    "LICENSE.txt",
]
FRONTAL = {"PA", "AP"}
ALL_FOLDERS = [f"p{i}" for i in range(10, 20)]
MIN_FREE_GB = 120
MANIFEST_COLS = ["dicom_id", "subject_id", "study_id", "view", "study_date",
                 "study_time", "split", "path", "orig_wh", "new_wh"]


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg, flush=True)


def wget_file(url, dest_dir):
    return subprocess.run(["wget", "-c", "-q", "-P", str(dest_dir), url]).returncode == 0


def fetch_metadata(meta_dir, optional=True):
    meta_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_META:
        if not (meta_dir / name).exists() and not wget_file(f"{BASE}/{name}", meta_dir):
            sys.exit(f"Failed to fetch required file {name}. Check <CREDENTIALS_FILE> and PhysioNet access.")
    if not optional:
        return
    for name in OPTIONAL_META:
        if not (meta_dir / name).exists() and not wget_file(f"{BASE}/{name}", meta_dir):
            log(f"optional file not fetched: {name}")


def fetch_reports(reports_dir):
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / "mimic-cxr-reports.zip"
    if target.exists() and target.stat().st_size > 0:
        return True
    for url in REPORTS_URLS:
        if wget_file(url, reports_dir) and target.exists():
            log(f"reports fetched from {url}")
            return True
    log("WARNING: reports zip not fetched. Download mimic-cxr-reports.zip manually from MIMIC-CXR.")
    return False


def read_gz_csv(path):
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def build_index(meta_dir, splits=None, one_per_study=False):
    """Frontal images grouped by pXX folder.
    splits: optional set like {"test"} to keep only those official splits.
    one_per_study: keep one frontal image per study (PA preferred over AP, then smallest dicom_id)."""
    split_of = {r["dicom_id"]: r["split"] for r in read_gz_csv(meta_dir / "mimic-cxr-2.0.0-split.csv.gz")}
    n_all = 0
    kept = []
    for r in read_gz_csv(meta_dir / "mimic-cxr-2.0.0-metadata.csv.gz"):
        n_all += 1
        if r.get("ViewPosition", "") not in FRONTAL:
            continue
        split = split_of.get(r["dicom_id"], "NA")
        if splits and split not in splits:
            continue
        sid = r["subject_id"]
        kept.append({
            "rel": f"files/p{sid[:2]}/p{sid}/s{r['study_id']}/{r['dicom_id']}.jpg",
            "dicom_id": r["dicom_id"], "subject_id": sid,
            "study_id": r["study_id"], "view": r["ViewPosition"],
            "study_date": r.get("StudyDate", ""), "study_time": r.get("StudyTime", ""),
            "split": split,
        })
    if one_per_study:
        best = {}
        for it in kept:
            key = (0 if it["view"] == "PA" else 1, it["dicom_id"])
            cur = best.get(it["study_id"])
            if cur is None or key < cur[0]:
                best[it["study_id"]] = (key, it)
        kept = [v[1] for v in best.values()]
    groups = defaultdict(list)
    for it in sorted(kept, key=lambda x: x["rel"]):
        groups[f"p{it['subject_id'][:2]}"].append(it)
    return groups, n_all


def load_sha(meta_dir):
    p = meta_dir / "SHA256SUMS.txt"
    if not p.exists():
        return {}
    sums = {}
    for line in p.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            name = parts[1].lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            sums[name] = parts[0]
    return sums


def download_folder(items, raw_dir, list_dir, jobs):
    """wget -c on every listed item: resumes partial files, skips complete ones."""
    if not items:
        return
    list_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(jobs):
        chunk = items[k::jobs]
        if not chunk:
            continue
        lf = list_dir / f"urls_{k}.txt"
        lf.write_text("\n".join(f"{BASE}/{it['rel']}" for it in chunk) + "\n")
        cmd = ["wget", "-c", "-q", "-x", "-nH", "--cut-dirs=3", "-P", str(raw_dir), "-i", str(lf)]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()


def process_one(args):
    rel, raw_dir, out_dir, size, expected_sha = args
    src = Path(raw_dir) / rel
    dst = Path(out_dir) / rel
    if dst.exists():
        return rel, "ok", "", ""
    if not src.exists():
        return rel, "missing", "", ""
    if expected_sha:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != expected_sha:
            return rel, "sha_mismatch", "", ""
    try:
        with Image.open(src) as im:
            orig = f"{im.width}x{im.height}"
            im.thumbnail((size, size), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part.jpg")
            im.save(tmp, "JPEG", quality=95)
            os.replace(tmp, dst)  # atomic: no half-written outputs after a crash
            return rel, "ok", orig, f"{im.width}x{im.height}"
    except Exception as e:
        return rel, f"error:{type(e).__name__}", "", ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="working root, e.g. /path/mimic")
    ap.add_argument("--folders", nargs="+", default=ALL_FOLDERS)
    ap.add_argument("--size", type=int, default=1024, help="longest side after resize")
    ap.add_argument("--wget-jobs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--no-verify-sha", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--skip-reports", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                    help="stop before a folder if free disk drops below this")
    ap.add_argument("--splits", nargs="+", default=None, choices=["train", "validate", "test"],
                    help="keep only these official splits, e.g. --splits test")
    ap.add_argument("--one-per-study", action="store_true",
                    help="one frontal image per study (PA preferred)")
    ap.add_argument("--no-optional-meta", action="store_true",
                    help="skip optional metadata (SHA256SUMS etc.), useful on slow links")
    a = ap.parse_args(argv)

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    meta_dir, raw_dir = root / "meta", root / "raw"
    out_dir = root / f"jpg{a.size}"
    fetch_metadata(meta_dir, optional=not a.no_optional_meta)
    if not a.skip_reports:
        fetch_reports(root / "reports")
    groups, n_all = build_index(meta_dir, set(a.splits) if a.splits else None, a.one_per_study)
    log(f"metadata rows={n_all}  selected frontal={sum(len(v) for v in groups.values())}"
        f"  splits={a.splits or 'all'}  one_per_study={a.one_per_study}")
    for f in a.folders:
        log(f"  {f}: {len(groups.get(f, []))} frontal images")
    log(f"free disk at root: {shutil.disk_usage(root).free / 1e9:.0f} GB")
    if a.dry_run:
        return

    sha = {} if a.no_verify_sha else load_sha(meta_dir)
    if not a.no_verify_sha and not sha:
        log("WARNING: SHA256SUMS.txt missing, integrity check disabled")

    manifest = root / f"manifest_frontal_{a.size}.csv"
    done_ids = set()
    if manifest.exists():
        with open(manifest, newline="") as f:
            done_ids = {r["dicom_id"] for r in csv.DictReader(f)}
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if not done_ids and mf.tell() == 0:
            w.writerow(MANIFEST_COLS)
        for f in a.folders:
            items = groups.get(f, [])
            if not items:
                continue
            free_gb = shutil.disk_usage(root).free / 1e9
            if free_gb < a.min_free_gb:
                sys.exit(f"Only {free_gb:.0f} GB free, need {a.min_free_gb:.0f}. Stopping before {f}.")
            pending = [it for it in items if not (out_dir / it["rel"]).exists()]
            log(f"{f}: {len(items)} frontal, {len(pending)} still to fetch ({a.wget_jobs} parallel wget)")
            download_folder(pending, raw_dir, root / "lists" / f, a.wget_jobs)
            missing = [it for it in pending if not (raw_dir / it["rel"]).exists()]
            if missing:
                log(f"{f}: retry pass for {len(missing)} missing files")
                download_folder(missing, raw_dir, root / "lists" / f"{f}_retry", 1)

            log(f"{f}: verify + resize with {a.workers} workers")
            by_rel = {it["rel"]: it for it in items}
            jobs = [(it["rel"], str(raw_dir), str(out_dir), a.size, sha.get(it["rel"], ""))
                    for it in items]
            bad = []
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for rel, status, orig, new in ex.map(process_one, jobs, chunksize=64):
                    if status != "ok":
                        bad.append((rel, status))
                        continue
                    it = by_rel[rel]
                    if it["dicom_id"] in done_ids:
                        continue
                    done_ids.add(it["dicom_id"])
                    w.writerow([it["dicom_id"], it["subject_id"], it["study_id"], it["view"],
                                it["study_date"], it["study_time"], it["split"],
                                str(out_dir / rel), orig, new])
            mf.flush()

            if bad:
                bad_file = root / f"failed_{f}.txt"
                bad_file.write_text("\n".join(f"{r}\t{s}" for r, s in bad) + "\n")
                log(f"{f}: {len(bad)} failures in {bad_file}; raw kept. Rerun same command to resume.")
            elif not a.keep_raw:
                shutil.rmtree(raw_dir / "files" / f, ignore_errors=True)
                log(f"{f}: all ok, raw deleted")
            log(f"{f}: done. free disk {shutil.disk_usage(root).free / 1e9:.0f} GB")
    log("finished")


if __name__ == "__main__":
    main()
