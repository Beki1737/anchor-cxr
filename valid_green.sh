#!/usr/bin/env bash

set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; T="--tag valid"; D=~/anchor/work/judge/valid
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
echo "valid_green start $(date)"
until [ -f $D/pairs.parquet ]; do sleep 60; done

source ~/venvs/anchor_eval/bin/activate
$J prep $T --judge green --only-conds b0_medgemma4b,normal_template 2>&1 | grep -E "^\[|Error"
deactivate

source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "GPU $CUDA_VISIBLE_DEVICES"
$J gen $T --judge green --gpu-util 0.55 > ~/anchor/logs/valid_gen_green.log 2>&1
if ! grep -q JUDGE_GEN_DONE ~/anchor/logs/valid_gen_green.log; then
  echo "gpu-util 0.55 failed (GPU busy?), retry at 0.35"; export CUDA_VISIBLE_DEVICES=$(lowgpu)
  $J gen $T --judge green --gpu-util 0.35 > ~/anchor/logs/valid_gen_green.log 2>&1
fi
grep -E "^\[|JUDGE_GEN_DONE|Error" ~/anchor/logs/valid_gen_green.log | tail -3
deactivate

source ~/venvs/anchor_eval/bin/activate
$J score $T --judge green 2>&1 | grep -E "mean=|JUDGE_SCORE_DONE|Error"
$J paired $T --judge green --cond-a b0_medgemma4b --cond-b normal_template
until grep -q VALID_JUDGES_DONE ~/anchor/logs/valid_eval.log 2>/dev/null; do sleep 60; done
$J diag $T --cond-a b0_medgemma4b --cond-b normal_template
deactivate
echo "VALID_GREEN_DONE $(date)"
