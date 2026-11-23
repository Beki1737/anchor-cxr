#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r40.log) 2>&1
C=~/anchor/code; source $C/gpu_wait.sh
echo "ANCHOR r40 start $(date)"
echo "-- allowed GPUs ($ANCHOR_GPUS) and who uses them:"
nvidia-smi -i "$ANCHOR_GPUS" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi -i "$ANCHOR_GPUS" --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | while IFS=, read bus pid mem; do
  echo "   pid $pid user $(ps -o user= -p $pid 2>/dev/null) $mem"; done
source ~/venvs/anchor/bin/activate
wait_free 4; python $C/e17b_pac.py 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
deactivate
for l in e9_mimic e16_judge sc_run; do echo "-- $l:"; tail -2 ~/anchor/logs/$l.log; done
echo "ANCHOR r40 end $(date)"
