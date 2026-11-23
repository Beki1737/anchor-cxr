#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r27.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r27 start $(date)"
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "GPU $CUDA_VISIBLE_DEVICES"
for s in train eval score; do
  python $C/e4c_probe.py $s 2>&1 | grep -vE "Warning|warn"
done
unset CUDA_VISIBLE_DEVICES
echo; echo "################ E4b re-analysis (votes, zero-shot verifiers, trained probe) ################"
python $C/e4b_verify.py analyze
echo "ANCHOR r27 end $(date)"
