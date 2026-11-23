#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r32.log
exec > >(tee "$LOG") 2>&1
P=~/anchor/data/padchest_gr; D=~/anchor/data
echo "ANCHOR r32 start $(date)"

echo; echo "################ A. PadChest-GR images: extraction (background) ################"
ls -la $P/Padchest_GR_files | awk '{print $5, $NF}'
file $P/Padchest_GR_files/PadChest_GR.zip.001 | cut -c1-200
which 7z 7za unzip 2>/dev/null
if [ -d $P/images ] && [ "$(find $P/images -name '*.png' | wc -l)" -gt 4000 ]; then echo "already extracted"
else
  mkdir -p $P/images
  nohup bash -c "
    if command -v 7z >/dev/null; then 7z x -y -o$P/images $P/Padchest_GR_files/PadChest_GR.zip.001
    else cat $P/Padchest_GR_files/PadChest_GR.zip.0* > $P/PadChest_GR_joined.zip && unzip -q -o $P/PadChest_GR_joined.zip -d $P/images && rm -f $P/PadChest_GR_joined.zip; fi
    echo PADCHEST_EXTRACT_DONE \$(find $P/images -name '*.png' | wc -l) pngs" > ~/anchor/logs/padchest_extract.log 2>&1 & disown
  echo "extraction launched: tail -2 ~/anchor/logs/padchest_extract.log"
fi

echo; echo "################ B. PadChest-GR annotation format (aggregates only) ################"
python3 - <<'EOF'
import json, collections
d = json.load(open("${ANCHOR_DATA_DIR:-./data}/padchest_gr/grounded_reports_20240819.json"))
nf = [len(s["findings"]) for s in d]
F = [f for s in d for f in s["findings"]]
print(f"studies {len(d)} | findings {len(F)} | abnormal findings {sum(f['abnormal'] for f in F)} | with boxes {sum(bool(f['boxes']) for f in F)} | with prior {sum(s['PreviousStudyID'] is not None for s in d)}")
bx = [b for f in F for b in (f["boxes"] or [])]
print("box sample (first 4):", bx[:4])
for i, name in enumerate(["x0/v0", "y0/v1", "x1/v2", "y1/v3"]):
    vals = [b[i] for b in bx if isinstance(b, (list, tuple)) and len(b) > i and isinstance(b[i], (int, float))]
    if vals: print(f"  coord {name}: min {min(vals):.3f} max {max(vals):.3f}")
print("box element type:", type(bx[0]).__name__, (len(bx[0]) if hasattr(bx[0], '__len__') else ''))
lab = collections.Counter(l for f in F for l in (f["labels"] or []))
loc = collections.Counter(l for f in F for l in (f["locations"] or []))
print("labels (top 40):", lab.most_common(40))
print("locations (top 30):", loc.most_common(30))
print("image id ends with .png:", d[0]["ImageID"].endswith(".png"), "| progression values:", collections.Counter(str(f["progression"]) for f in F).most_common(5))
EOF
python3 - <<'EOF'
import zipfile, io, csv
z = zipfile.ZipFile("${ANCHOR_DATA_DIR:-./data}/padchest_gr/master_table.csv.zip")
n = z.namelist(); print("master_table files:", n)
rows = list(csv.reader(io.TextIOWrapper(z.open(n[0]), encoding="utf-8")))
print("master_table columns:", rows[0], "| rows:", len(rows) - 1)
EOF

echo; echo "################ C. MIMIC-CXR reports + images ################"
python3 - <<'EOF'
import zipfile, collections
z = zipfile.ZipFile("${ANCHOR_DATA_DIR:-./data}/reports/mimic-cxr-reports.zip")
n = z.namelist()
txt = [x for x in n if x.endswith(".txt")]
print(f"zip entries {len(n)} | txt {len(txt)} | top-level folders: {sorted(set(x.split('/')[0] for x in n))[:5]}")
print("depth pattern:", collections.Counter(len(x.split('/')) for x in txt).most_common(3))
EOF
head -1 $D/manifest_frontal_1024.csv | cut -c1-400; echo "rows: $(($(wc -l < $D/manifest_frontal_1024.csv) - 1))"
echo "jpg1024 files: $(find $D/jpg1024 -name '*.jpg' | wc -l) | raw jpg: $(find $D/raw -name '*.jpg' | wc -l)"
ls $D/meta | head
echo "ANCHOR r32 end $(date)"
