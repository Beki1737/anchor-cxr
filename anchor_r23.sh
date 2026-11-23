#!/usr/bin/env bash
# ANCHOR round 23 = (A) leftover ReXGradient archive parts: verify extraction, delete only with DEL_PARTS=1,
# (B) relaunch E3 with retries, (C) CheXbert strata (independent check of D40), (D) launch E1 v2 chain. MobaXterm:
#   DEL_PARTS=1 bash ~/anchor/code/anchor_r23.sh
# Log: ~/anchor/logs/r23.log (send back).
# Needs in ~/anchor/code/: corpus_extract_v2.py, chexbert_strata.py, e1v2_chain.sh, valid_e3.sh (updated)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r23.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; R=~/anchor/data/rexgradient
echo "ANCHOR r23 start $(date)"

echo; echo "################ A. ReXGradient leftover archive parts ################"
df -h ~ | tail -1
ls -la $R | grep -E "deid_png|\.tar|part" | awk '{printf "%6.1f GB  %s\n", $5/1e9, $NF}'
PARTS_GB=$(du -cb $R/deid_png.part* 2>/dev/null | tail -1 | awk '{printf "%.1f", $1/1e9}')
echo "parts total: ${PARTS_GB} GB | other rexgradient top-level:"; du -sh $R/* $R/.cache 2>/dev/null | grep -v deid_png.part | sort -h | tail -5
source ~/venvs/anchor/bin/activate
python - <<'EOF'
import os, pathlib, pandas as pd
H = pathlib.Path.home() / "anchor"
miss = tot = 0
for s in ["train", "valid", "test"]:
    m = pd.read_parquet(H / f"work/manifest/rex_{s}.parquet")
    p = m.loc[m.eligible, "frontal_path"].dropna()
    tot += len(p); miss += int((~p.map(os.path.exists)).sum())
n = sum(1 for r, _, fs in os.walk(H / "data/rexgradient/images") for f in fs if f.endswith(".png"))
ok = miss == 0 and n >= 273004
print(f"extraction check: {n} PNGs (metadata 273,004), eligible frontals missing {miss}/{tot} -> {'EXTRACTION_VERIFIED' if ok else 'NOT_VERIFIED'}")
open("/tmp/anchor_r23_verify", "w").write("OK" if ok else "NO")
EOF
deactivate
if [ "${DEL_PARTS:-0}" = 1 ] && [ "$(cat /tmp/anchor_r23_verify)" = OK ]; then
  rm -f $R/deid_png.part*; echo "deleted leftover archive parts (${PARTS_GB} GB)"
elif [ "${DEL_PARTS:-0}" = 1 ]; then echo "NOT deleting: extraction not verified"; fi
rm -f /tmp/anchor_r23_verify
df -h ~ | tail -1

echo; echo "################ B. relaunch E3 (retrieval cached, generation with retries) ################"
if pgrep -f valid_e3.sh > /dev/null; then echo "already running"
else nohup bash $C/valid_e3.sh > ~/anchor/logs/valid_e3.log 2>&1 & disown; echo "launched: tail ~/anchor/logs/valid_e3.log"; fi

echo; echo "################ C. CheXbert strata (independent of E1) ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/chexbert_strata.py 2>&1 | grep -E "label|CheXbert|CHEX|crimson|green|RECHECK|recheck|Error"
deactivate

echo; echo "################ D. launch E1 v2 chain (devcheck -> valid -> test -> train -> held-out sheet) ################"
if pgrep -f e1v2_chain.sh > /dev/null; then echo "already running"
else nohup bash $C/e1v2_chain.sh > ~/anchor/logs/e1v2.log 2>&1 & disown; echo "launched: tail ~/anchor/logs/e1v2.log"; fi
sleep 60; echo "-- E3 so far:"; tail -4 ~/anchor/logs/valid_e3.log
echo "ANCHOR r23 end $(date)"
