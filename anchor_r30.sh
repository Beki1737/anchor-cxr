#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r30.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r30 start $(date)"
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/e7_refine.py 2>&1 | grep -vE "Warning|warn"
unset CUDA_VISIBLE_DEVICES
deactivate
if pgrep -f valid_e7.sh > /dev/null; then echo "E7 judges already running"
else nohup bash $C/valid_e7.sh > ~/anchor/logs/valid_e7.log 2>&1 & disown; echo "E7 judges launched: tail -80 ~/anchor/logs/valid_e7.log (VALID_E7_DONE)"; fi
echo "ANCHOR r30 end $(date)"
