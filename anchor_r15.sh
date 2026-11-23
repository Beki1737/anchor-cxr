#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r15.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; G=~/anchor/work/gen
echo "ANCHOR r15 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
FILT='^\(EngineCore|^\(MainProcess|Capturing CUDA graphs|Loading safetensors|it/s\]$|^$'

echo; echo "################ A. progress ################"
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
echo "-- GREEN smoke (background):"; tail -8 ~/anchor/logs/green_smoke.log 2>/dev/null
echo "-- model downloads:"; tail -2 ~/anchor/models/dl_models.log

echo; echo "################ B. B0 for Qwen2.5-VL-7B and Lingshu-7B (200 dev studies) ################"
source ~/venvs/anchor/bin/activate
for pair in "qwen25vl7b:$HOME/anchor/models/Qwen/Qwen2.5-VL-7B-Instruct" "lingshu7b:$HOME/anchor/models/lingshu-medical-mllm/Lingshu-7B"; do
  NAME=${pair%%:*}; MP=${pair#*:}
  if [ -f "$G/b0_${NAME}_dev200.parquet" ]; then echo "$NAME already generated"; continue; fi
  if [ ! -f "$MP/config.json" ]; then echo "$NAME: model not found at $MP"; continue; fi
  echo "-- $NAME"
  python "$C/gen_b0.py" --model "$MP" --name "$NAME" --keys dev --limit 200 2>&1 | grep -vE "$FILT"
done
deactivate

echo; echo "################ C. harness on all B0 outputs (anchor_eval venv) ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "eval GPU: $CUDA_VISIBLE_DEVICES"
FIRST=1
for f in "$G"/b0_*_dev200.parquet; do
  echo; echo "=========== $(basename "$f") ==========="
  if [ "$FIRST" = 1 ]; then M="radgraph,f1chexbert,radcliq,ratescore"; FIRST=0; else M="radgraph,f1chexbert,radcliq"; fi
  python "$C/eval_smoke.py" --gen "$f" --metrics "$M" 2>&1 | grep -vE "Computing|Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|can be ignored"
done
if ! pgrep -f "eval_smoke.py.*crimson" > /dev/null; then
  nohup bash -c "source ~/venvs/anchor_eval/bin/activate; CUDA_VISIBLE_DEVICES=$(lowgpu) python $C/eval_smoke.py --gen $G/b0_medgemma4b_dev200.parquet --metrics crimson" \
    > ~/anchor/logs/crimson_smoke.log 2>&1 & disown
  echo "CRIMSON launched in background: tail -12 ~/anchor/logs/crimson_smoke.log"
fi
deactivate

echo; echo "MIMIC raw: $(find ~/anchor/data/raw -name '*.jpg' | wc -l) / 3041"
echo "ANCHOR r15 end $(date)"
