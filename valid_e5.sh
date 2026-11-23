#!/usr/bin/env bash
# Background (launched by anchor_r28.sh): E5 judges on the evaluation half -> summary -> fast metrics.
# Log: ~/anchor/logs/valid_e5.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; T="--tag valid_e5"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
echo "valid_e5 start $(date)"
GENS=e5b0real_valid.parquet,e5anchor_any15_valid.parquet,e5anchor_any30_valid.parquet,e5anchor_fp15_valid.parquet,e5anchor_fp30_valid.parquet,e5oracle_valid.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs $T --split valid --gens $GENS --ref-conds none
for j in crimson green; do $J prep $T --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate
source ~/venvs/anchor/bin/activate
for j in crimson green; do
  export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35; [ $j = green ] && U=0.55
  for t in 1 2 3; do
    $J gen $T --judge $j --gpu-util $U > ~/anchor/logs/valid_e5_gen_$j.log 2>&1
    grep -q JUDGE_GEN_DONE ~/anchor/logs/valid_e5_gen_$j.log && break
    sleep 90; export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35
  done
  grep -E "^\[|JUDGE_GEN_DONE|Error" ~/anchor/logs/valid_e5_gen_$j.log | tail -2
done
deactivate
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu)
for j in crimson green; do $J score $T --judge $j 2>&1 | grep -E "parse_failed|JUDGE_SCORE_DONE|Error"; done
python $C/e5_summary.py
echo "VALID_E5_JUDGES_DONE $(date)"
for g in ${GENS//,/ }; do
  python $C/eval_smoke.py --gen ~/anchor/work/gen/$g --split valid --only-gen \
    --metrics radgraph,f1chexbert,ratescore 2>&1 | grep -E "^radgraph|^f1chexbert|^ratescore" | sed "s/^/$g /"
done
deactivate
echo "VALID_E5_DONE $(date)"
