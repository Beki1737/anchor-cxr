#!/usr/bin/env bash
# ANCHOR round 12 = full-cache QA + segmentation bias check + E1 corpus-extraction pilot. MobaXterm:
#   bash ~/anchor/code/anchor_r12.sh
# Log: ~/anchor/logs/r12.log (send back). Labelling sheet: ~/anchor/work/e1_label_200.csv (local only, never upload)
# Needs in ~/anchor/code/: seg_bias_check.py, corpus_extract.py
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r12.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data
echo "ANCHOR r12 start $(date)"
source ~/venvs/anchor/bin/activate

echo; echo "################ A. full region cache: status, QA, bias check ################"
tail -3 ~/anchor/logs/region_full.log
grep -E "REGION_CACHE_DONE|FAILED_|ALL_REGIONS_DONE" ~/anchor/logs/region_full.log
du -sh ~/anchor/work/regions/{test,valid,train} 2>/dev/null
python "$C/seg_bias_check.py"

echo; echo "################ B. E1 pilot: corpus extraction on 2,000 train reports ################"
MODEL=$(find ${HF_MODELS_DIR:-./models} -maxdepth 3 -type d -iname "*Qwen2.5-14B-Instruct" 2>/dev/null | grep -viE "awq|gptq|int4|int8" | head -1)
echo "model: ${MODEL:-NOT FOUND}"
if [ -z "$MODEL" ]; then
  echo "Qwen2.5 dirs available:"; find ${HF_MODELS_DIR:-./models} -maxdepth 3 -type d -iname "*Qwen2.5*" | head -20
else
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  python "$C/corpus_extract.py" --split train --limit 2000 --label 200 --model "$MODEL"
fi

echo; echo "################ C. background downloads ################"
echo "MIMIC raw: $(find "$D/raw" -name '*.jpg' | wc -l) / 3041"; tail -1 "$D/fetch_mimic_test.log"
echo "reports zip bytes: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null)"; tail -1 "$D/reports/fetch_reports.log" 2>/dev/null
echo "ANCHOR r12 end $(date)"
