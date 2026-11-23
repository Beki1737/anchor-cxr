#!/usr/bin/env bash

set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; source $C/gpu_wait.sh
echo "e16_judge start $(date)"
source ~/venvs/anchor/bin/activate
for j in crimson green; do
  U=0.35; N=36; [ $j = green ] && { U=0.30; N=31; }
  grep -q JUDGE_GEN_DONE ~/anchor/logs/valid_e16_gen_$j.log 2>/dev/null || \
    gpu_run $N ~/anchor/logs/valid_e16_gen_$j.log JUDGE_GEN_DONE $J gen --tag valid_e16 --judge $j --gpu-util $U
done
deactivate; source ~/venvs/anchor_eval/bin/activate
for j in crimson green; do $J score --tag valid_e16 --judge $j 2>&1 | grep -E "parse_failed|Error"; done
python $C/e8_summary.py --split valid --tags valid,valid_e8,valid_e16 --keys ~/anchor/work/e8/eval_keys_valid.parquet \
  --extra-sys e16exh,e16exhedit,e16fus,e16fusedit,e16fusmulti
deactivate
echo "E16_JUDGE_DONE $(date)"
