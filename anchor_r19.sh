#!/usr/bin/env bash
# ANCHOR round 19 = (A) GREEN vLLM port without GREEN's custom tokenizer code (token ids in, ids out),
# fallback: install matplotlib in the anchor venv and use the repo code; (B) validation vs radeval;
# (C) launch background large-n scoring on the valid cohort (valid_eval.sh). MobaXterm:
#   bash ~/anchor/code/anchor_r19.sh
# Log: ~/anchor/logs/r19.log (send back). Needs in ~/anchor/code/: judge_vllm.py, eval_smoke.py, valid_eval.sh
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r19.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; J="python $C/judge_vllm.py"
echo "ANCHOR r19 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
EFILT='Computing|Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|can be ignored|NLTK|incr_download|^radeval|^available|^numpy|RadEval\.'
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
tail -1 ~/anchor/logs/b0_valid.log

echo; echo "################ A. GREEN generation, tokenizer-free (anchor venv) ################"
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "GPU $CUDA_VISIBLE_DEVICES"
$J gen --judge green > ~/anchor/logs/green_gen.log 2>&1
grep -E "^\[green\]|JUDGE_GEN_DONE" ~/anchor/logs/green_gen.log | tail -4
if ! grep -q JUDGE_GEN_DONE ~/anchor/logs/green_gen.log; then
  echo "-- tokenizer-free path failed, last error:"; grep -E "Error|error" ~/anchor/logs/green_gen.log | tail -3
  echo "-- fallback: matplotlib into anchor venv (pinned torch/numpy untouched), repo code path"
  uv pip install matplotlib --constraint ~/anchor/pins.txt 2>&1 | tail -2
  $J gen --judge green --green-remote > ~/anchor/logs/green_gen.log 2>&1
  grep -E "^\[green\]|JUDGE_GEN_DONE|Error" ~/anchor/logs/green_gen.log | tail -4
fi
deactivate

echo; echo "################ B. GREEN score + compare with radeval (anchor_eval) ################"
source ~/venvs/anchor_eval/bin/activate
$J score --judge green 2>&1 | grep -vE "$EFILT" | tail -9
$J compare --judge green --tol-agg 0.02 2>&1 | tail -14
for j in green crimson; do for b in normal_template floor_shuffled; do
  [ -f ~/anchor/work/judge/dev200/${j}_scores.parquet ] && $J paired --judge $j --cond-a b0_medgemma4b --cond-b $b
done; done
deactivate

echo; echo "################ C. background: large-n scoring on valid cohort ################"
if pgrep -f valid_eval.sh > /dev/null; then echo "already running"
else nohup bash $C/valid_eval.sh > ~/anchor/logs/valid_eval.log 2>&1 & disown; echo "launched: tail -20 ~/anchor/logs/valid_eval.log"; fi
echo "ANCHOR r19 end $(date)"
