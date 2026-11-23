#!/usr/bin/env python3
"""Resumable PadChest-GR download from the public B2DROP share over WebDAV.
Lists the share recursively, then fetches each file with `curl -C -` (resume), in parallel.
Re-running is safe: complete files are skipped by size.

    python padchest_fetch.py --out ~/anchor/data/padchest_gr --list-only
    nohup python padchest_fetch.py --out ~/anchor/data/padchest_gr > padchest.log 2>&1 &
"""
import argparse
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HOST = "https://b2drop.bsc.es"
PREFIX = "/nextcloud/public.php/webdav/"
TOKEN = "PadChest-GR"
NS = {"d": "DAV:"}


def log(m):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + m, flush=True)


def parse_propfind(xml_bytes, current_rel):
    """Return [(rel_path, is_dir, size)] for children of current_rel (self entry dropped)."""
    out = []
    root = ET.fromstring(xml_bytes)
    for resp in root.findall("d:response", NS):
        href = urllib.parse.unquote(resp.find("d:href", NS).text)
        rel = href.split(PREFIX, 1)[-1] if PREFIX in href else href.lstrip("/")
        prop = resp.find("d:propstat/d:prop", NS)
        is_dir = prop is not None and prop.find("d:resourcetype/d:collection", NS) is not None
        size_el = prop.find("d:getcontentlength", NS) if prop is not None else None
        size = int(size_el.text) if size_el is not None and size_el.text else 0
        if rel.rstrip("/") == current_rel.rstrip("/"):
            continue
        out.append((rel.rstrip("/") if is_dir else rel, is_dir, size))
    return out


def listdir(rel, retries=5):
    url = HOST + PREFIX + urllib.parse.quote(rel)
    for k in range(retries):
        try:
            r = requests.request("PROPFIND", url, auth=(TOKEN, ""), headers={"Depth": "1"}, timeout=300)
            r.raise_for_status()
            return parse_propfind(r.content, rel)
        except Exception as e:
            log(f"PROPFIND retry {k+1} for '{rel}': {e}")
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"cannot list {rel}")


def walk(rel=""):
    files = []
    for child, is_dir, size in listdir(rel):
        if is_dir:
            files.extend(walk(child + "/"))
        else:
            files.append((child, size))
    return files


def fetch(item, out):
    rel, size = item
    dst = out / rel
    if dst.exists() and dst.stat().st_size == size:
        return rel, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = HOST + PREFIX + urllib.parse.quote(rel)
    cmd = ["curl", "-sS", "-f", "-C", "-", "--retry", "30", "--retry-delay", "15", "--retry-all-errors",
           "-u", f"{TOKEN}:", "-o", str(dst), url]
    for attempt in range(5):
        subprocess.run(cmd)
        if dst.exists() and dst.stat().st_size == size:
            return rel, "ok"
        time.sleep(30)
    return rel, f"incomplete({dst.stat().st_size if dst.exists() else 0}/{size})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = walk("")
    total = sum(s for _, s in files)
    (out / "padchest_listing.tsv").write_text("".join(f"{r}\t{s}\n" for r, s in files))
    tops = {}
    for r, s in files:
        top = r.split("/")[0] if "/" in r else "(root)"
        n, b = tops.get(top, (0, 0)); tops[top] = (n + 1, b + s)
    log(f"files={len(files)} total={total/1e9:.1f} GB")
    for t, (n, b) in sorted(tops.items()):
        log(f"  {t}: {n} files, {b/1e9:.2f} GB")
    if a.list_only:
        return
    bad = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for i, (rel, status) in enumerate(ex.map(lambda it: fetch(it, out), files), 1):
            if status not in ("ok", "skip"):
                bad.append((rel, status))
            if i % 200 == 0 or i == len(files):
                log(f"progress {i}/{len(files)} files, failures so far {len(bad)}")
    if bad:
        (out / "padchest_failed.tsv").write_text("".join(f"{r}\t{s}\n" for r, s in bad))
        log(f"DONE_WITH_FAILURES {len(bad)} (rerun the same command to resume)")
    else:
        log("PADCHEST_DONE")


if __name__ == "__main__":
    main()
