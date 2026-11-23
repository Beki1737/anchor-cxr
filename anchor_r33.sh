#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r33.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; P=~/anchor/data/padchest_gr
echo "ANCHOR r33 start $(date)"

echo; echo "################ A. PadChest-GR extraction (Python zipfile, background) ################"
if [ "$(find $P/images -name '*.png' 2>/dev/null | wc -l)" -gt 4000 ]; then echo "already extracted"
else
  nohup python3 - > ~/anchor/logs/padchest_extract.log 2>&1 <<'EOF' &
import zipfile, pathlib, os, time
P = pathlib.Path.home() / "anchor/data/padchest_gr"
j = P / "PadChest_GR_joined.zip"
want = sum(f.stat().st_size for f in sorted((P / "Padchest_GR_files").glob("PadChest_GR.zip.*")))
if not j.exists() or j.stat().st_size != want:
    with open(j, "wb") as out:
        for f in sorted((P / "Padchest_GR_files").glob("PadChest_GR.zip.*")):
            with open(f, "rb") as src:
                while True:
                    b = src.read(1 << 24)
                    if not b: break
                    out.write(b)
t0 = time.time()
z = zipfile.ZipFile(j)
names = z.namelist(); print("entries", len(names), flush=True)
z.extractall(P / "images")
n = sum(1 for _ in (P / "images").rglob("*.png"))
print(f"PADCHEST_EXTRACT_DONE {n} pngs in {time.time() - t0:.0f}s", flush=True)
if n > 4000:
    os.remove(j); print("joined zip removed (split parts kept)")
EOF
  disown; echo "launched: tail -3 ~/anchor/logs/padchest_extract.log"
fi

echo; echo "################ B. PadChest-GR annotation format (aggregates only) ################"
python3 - <<'EOF'
import json, collections
d = json.load(open("${ANCHOR_DATA_DIR:-./data}/padchest_gr/grounded_reports_20240819.json"))
F = [f for s in d for f in s["findings"]]
print(f"studies {len(d)} | findings {len(F)} | keys seen: {sorted(set(k for f in F for k in f))}")
print(f"abnormal {sum(bool(f.get('abnormal')) for f in F)} | with boxes {sum(bool(f.get('boxes')) for f in F)} | with prior {sum(s.get('PreviousStudyID') is not None for s in d)}")
bx = [b for f in F for b in (f.get("boxes") or [])]
print("box sample (first 3):", bx[:3])
if bx and isinstance(bx[0], (list, tuple)):
    for i in range(len(bx[0])):
        v = [b[i] for b in bx if len(b) > i and isinstance(b[i], (int, float))]
        if v: print(f"  coord[{i}] min {min(v):.3f} max {max(v):.3f}")
print("labels (top 40):", collections.Counter(l for f in F for l in (f.get("labels") or [])).most_common(40))
print("locations (top 30):", collections.Counter(l for f in F for l in (f.get("locations") or [])).most_common(30))
EOF

echo; echo "################ C. launch E9 MIMIC external validation (frozen config) ################"
if pgrep -f mimic_run.sh > /dev/null; then echo "already running"
else nohup bash $C/mimic_run.sh > ~/anchor/logs/mimic_run.log 2>&1 & disown; echo "launched: tail -60 ~/anchor/logs/mimic_run.log (MIMIC_RUN_DONE)"; fi
sleep 60; head -5 ~/anchor/logs/mimic_run.log
echo "ANCHOR r33 end $(date)"
