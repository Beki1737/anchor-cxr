#!/usr/bin/env bash
# ANCHOR round 17 = (A) RaTEScore retry with a functional test instead of uv pip check,
# (B) D34 batched vLLM port of GREEN + CRIMSON, validated against radeval. MobaXterm:
#   bash ~/anchor/code/anchor_r17.sh
# Log: ~/anchor/logs/r17.log (send back). Needs in ~/anchor/code/: judge_vllm.py (new), eval_smoke.py
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r17.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; G=~/anchor/work/gen; J="python $C/judge_vllm.py"
echo "ANCHOR r17 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
FILT='^\(EngineCore|^\(MainProcess|^\(APIServer|Capturing CUDA graphs|Loading safetensors|it/s\]$|^INFO|^WARNING|^$'
EFILT='Computing|Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|can be ignored|NLTK|incr_download|^radeval|^available|^numpy|RadEval\.'

echo; echo "################ 0. progress ################"
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
grep -E "FAILED_|ALL_CORPUS_DONE" ~/anchor/logs/corpus_full.log
echo "MIMIC raw: $(find ~/anchor/data/raw -name '*.jpg' 2>/dev/null | wc -l) / 3041"

echo; echo "################ A. RaTEScore: spaCy 3.8 / thinc 8.3, kept only if RaTEScore actually scores ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "eval GPU: $CUDA_VISIBLE_DEVICES"
OLD_SPACY=$(uv pip show spacy 2>/dev/null | awk '/^Version/{print $2}'); OLD_THINC=$(uv pip show thinc 2>/dev/null | awk '/^Version/{print $2}')
OLD_WEASEL=$(uv pip show weasel 2>/dev/null | awk '/^Version/{print $2}')
echo "before: spacy $OLD_SPACY thinc $OLD_THINC weasel $OLD_WEASEL"
uv pip install "spacy>=3.8,<3.9" "thinc>=8.3.4,<8.4" --constraint ~/anchor/eval_pins.txt 2>&1 | tail -3
echo "-- uv pip check (a medspacy pin warning is expected and acceptable):"; uv pip check 2>&1 | tail -4
python "$C/eval_smoke.py" --gen "$G/b0_medgemma4b_dev200.parquet" --metrics ratescore 2>&1 | grep -vE "$EFILT" | tee /tmp/r17_rate.txt
if grep -q "\[ratescore\] ok" /tmp/r17_rate.txt && python -c "import medspacy" 2>/dev/null; then
  echo "RATESCORE_FIXED (spaCy 3.8 kept)"; uv pip freeze > ~/anchor/requirements.anchor_eval.lock
  for m in lingshu7b qwen25vl7b; do
    echo "-- ratescore $m"; python "$C/eval_smoke.py" --gen "$G/b0_${m}_dev200.parquet" --metrics ratescore 2>&1 | grep -E "^ratescore|\[ratescore\]"
  done
else
  echo "RATESCORE_STILL_BROKEN: reverting to spacy $OLD_SPACY"
  uv pip install "spacy==$OLD_SPACY" "thinc==$OLD_THINC" "weasel==$OLD_WEASEL" --constraint ~/anchor/eval_pins.txt 2>&1 | tail -1
fi
rm -f /tmp/r17_rate.txt

echo; echo "################ B1. judge port: pairs + prompts (radeval code, anchor_eval) ################"
$J pairs 2>&1 | tail -2
for j in green crimson; do $J prep --judge $j 2>&1 | grep -E "^\[|Error|error" ; done
deactivate

echo; echo "################ B2. judge port: vLLM generation (anchor venv) ################"
source ~/venvs/anchor/bin/activate
for j in green crimson; do
  export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "-- $j on GPU $CUDA_VISIBLE_DEVICES"
  $J gen --judge $j 2>&1 | grep -vE "$FILT" | tail -8
done
deactivate

echo; echo "################ B3. judge port: scoring + radeval reference subset + comparison (anchor_eval) ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "eval GPU: $CUDA_VISIBLE_DEVICES"
for j in green crimson; do
  $J score --judge $j 2>&1 | grep -vE "$EFILT" | tail -9
  K=16; [ $j = crimson ] && K=10
  $J refcheck --judge $j --k $K 2>&1 | grep -E "^\[|Error|error" | tail -4
  TOL=0.02; [ $j = crimson ] && TOL=0.04
  $J compare --judge $j --tol-agg $TOL 2>&1 | tail -14
done
deactivate
echo "ANCHOR r17 end $(date)"
