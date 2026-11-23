#!/usr/bin/env bash
# Background (launched by anchor_r26.sh): E4b build -> verify (MedGemma, Qwen2.5-VL) -> analyze. Log: ~/anchor/logs/e4b.log
set -u
C=~/anchor/code; X="python $C/e4b_verify.py"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
source ~/venvs/anchor/bin/activate
echo "e4b start $(date)"
export CUDA_VISIBLE_DEVICES=$(lowgpu)
[ -f ~/anchor/work/e4b/candidates.parquet ] || $X build --n 1500 2>&1 | grep -E "E4b|E4B|Error"
unset CUDA_VISIBLE_DEVICES
for m in medgemma qwenvl; do
  for t in 1 2 3; do
    [ -f ~/anchor/work/e4b/verify_$m.parquet ] && break
    $X verify --model $m > ~/anchor/logs/e4b_verify_$m.log 2>&1
    grep -q E4B_VERIFY_DONE ~/anchor/logs/e4b_verify_$m.log && break; sleep 90
  done
  grep -E "^verify|using GPU|Error" ~/anchor/logs/e4b_verify_$m.log | tail -2
done
$X analyze
echo "E4B_CHAIN_DONE $(date)"
