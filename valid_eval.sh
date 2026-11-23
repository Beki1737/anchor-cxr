#!/usr/bin/env bash
# Background chain (launched by anchor_r19.sh, do not run by hand): large-n scoring of MedGemma B0 on the
# valid primary cohort (6,743 studies) vs normal template and shuffled floor.
# CRIMSON + GREEN (vLLM port) first, then paired bootstrap, then fast metrics. Log: ~/anchor/logs/valid_eval.log
set -u
C=~/anchor/code; G=~/anchor/work/gen; J="python $C/judge_vllm.py"; T="--tag valid"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
echo "valid_eval start $(date)"
until grep -q GEN_B0_DONE ~/anchor/logs/b0_valid.log 2>/dev/null; do sleep 60; done
echo "B0 valid ready $(date)"; tail -2 ~/anchor/logs/b0_valid.log

JUDGES="crimson"
grep -q PORT_VALID ~/anchor/work/judge/dev200/green_compare.json 2>/dev/null && JUDGES="crimson green"
echo "judges: $JUDGES"

source ~/venvs/anchor_eval/bin/activate
$J pairs $T --split valid --gens b0_medgemma4b_valid.parquet --ref-conds floor_shuffled,normal_template
for j in $JUDGES; do $J prep $T --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate

source ~/venvs/anchor/bin/activate
for j in $JUDGES; do
  export CUDA_VISIBLE_DEVICES=$(lowgpu)
  $J gen $T --judge $j > ~/anchor/logs/valid_gen_$j.log 2>&1
  grep -E "^\[|JUDGE_GEN_DONE|Error" ~/anchor/logs/valid_gen_$j.log | tail -3
done
deactivate

source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu)
for j in $JUDGES; do
  $J score $T --judge $j 2>&1 | grep -E "mean=|JUDGE_SCORE_DONE|Error"
  $J paired $T --judge $j --cond-a b0_medgemma4b --cond-b normal_template
  $J paired $T --judge $j --cond-a b0_medgemma4b --cond-b floor_shuffled
done
echo "VALID_JUDGES_DONE $(date)"
python $C/eval_smoke.py --gen $G/b0_medgemma4b_valid.parquet --split valid --skip-ceiling \
  --metrics radgraph,f1chexbert,ratescore 2>&1 | grep -E "^\[|^radgraph|^f1chexbert|^ratescore|EVAL_SMOKE_DONE"
deactivate
echo "VALID_EVAL_DONE $(date)"
