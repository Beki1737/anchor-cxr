#!/usr/bin/env bash
# Background (launched by anchor_r29.sh): waits for the E5 judges, then judges the E6 report variants
# (side-resolved, VLM-free, prior channel) and prints the combined summary. Log: ~/anchor/logs/valid_e6.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; T="--tag valid_e6"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
echo "valid_e6 start $(date)"
until grep -q VALID_E5_JUDGES_DONE ~/anchor/logs/valid_e5.log 2>/dev/null; do sleep 120; done
echo "E5 judges done, starting E6 judges $(date)"
GENS=e5anchorsr_fp30_valid.parquet,e5anchorsr_any30_valid.parquet,e5poolonly_fp30_valid.parquet,e5prior_fp30_valid.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs $T --split valid --gens $GENS --ref-conds none
for j in crimson green; do $J prep $T --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate
source ~/venvs/anchor/bin/activate
for j in crimson green; do
  export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35; [ $j = green ] && U=0.55
  for t in 1 2 3; do
    $J gen $T --judge $j --gpu-util $U > ~/anchor/logs/valid_e6_gen_$j.log 2>&1
    grep -q JUDGE_GEN_DONE ~/anchor/logs/valid_e6_gen_$j.log && break
    sleep 90; export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35
  done
  grep -E "^\[|JUDGE_GEN_DONE|Error" ~/anchor/logs/valid_e6_gen_$j.log | tail -2
done
deactivate
source ~/venvs/anchor_eval/bin/activate
for j in crimson green; do $J score $T --judge $j 2>&1 | grep -E "parse_failed|JUDGE_SCORE_DONE|Error"; done
python $C/e5_summary.py
deactivate
echo "VALID_E6_DONE $(date)"
