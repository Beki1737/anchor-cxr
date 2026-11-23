#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r14.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r14 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }

echo; echo "################ A. corpus extraction progress ################"
L=$(grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1); echo "$L"
python3 - "$L" <<'PY'
import re, sys
m = re.search(r"(\d+)/(\d+) reports, ([\d.]+) reports/s", sys.argv[1] or "")
if m:
    done, tot, r = int(m[1]), int(m[2]), float(m[3])
    print(f"current split ETA: {(tot - done) / max(r, 1e-6) / 3600:.1f} h (then valid + test ~{15374 / max(r, 1e-6) / 3600:.1f} h)")
PY
grep -E "FAILED_|ALL_CORPUS_DONE" ~/anchor/logs/corpus_full.log

echo; echo "################ B. development slice ################"
source ~/venvs/anchor/bin/activate
python "$C/dev_slice.py"

echo; echo "################ C. B0 baseline: MedGemma-4B on 200 dev studies ################"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
python "$C/gen_b0.py" --model ${HF_MODELS_DIR:-./models}/Google/medgemma-4b-it --name medgemma4b --keys dev --limit 200 2>&1 \
  | grep -vE "^\(EngineCore|^\(MainProcess|Capturing CUDA graphs|Loading safetensors|it/s\]$"
deactivate

echo; echo "################ D. E2 harness smoke test (anchor_eval venv) ################"
source ~/venvs/anchor_eval/bin/activate
uv pip install pyarrow pandas --constraint ~/anchor/eval_pins.txt 2>&1 | tail -1
GEN=~/anchor/work/gen/b0_medgemma4b_dev200.parquet
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "eval GPU: $CUDA_VISIBLE_DEVICES"
if [ -f "$GEN" ]; then
  python "$C/eval_smoke.py" --gen "$GEN" --metrics radgraph,chexbert,ratescore,bleu,bertscore
  nohup bash -c "source ~/venvs/anchor_eval/bin/activate; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $C/eval_smoke.py --gen $GEN --metrics green" \
    > ~/anchor/logs/green_smoke.log 2>&1 & disown
  echo "GREEN launched in background (downloads a 7B judge first): tail -15 ~/anchor/logs/green_smoke.log"
else echo "no B0 generations found; skipping scoring"; fi
deactivate

echo; echo "################ E. downloads ################"
echo "MIMIC raw: $(find ~/anchor/data/raw -name '*.jpg' | wc -l) / 3041"
echo "reports zip bytes: $(stat -c %s ~/anchor/data/reports/mimic-cxr-reports.zip 2>/dev/null)"
echo "ANCHOR r14 end $(date)"
