#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r28.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r28 start $(date)"
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
[ -f ~/anchor/work/e5/candidates.parquet ] || python $C/e5_anchor.py build 2>&1 | grep -E "E4b|E4B|Error"
python $C/e4c_probe.py score --dir e5 2>&1 | grep -E "probe|E4C|Error"
python $C/e5_anchor.py run
unset CUDA_VISIBLE_DEVICES
deactivate
if pgrep -f valid_e5.sh > /dev/null; then echo "judges already running"
else nohup bash $C/valid_e5.sh > ~/anchor/logs/valid_e5.log 2>&1 & disown; echo "judges launched: tail -40 ~/anchor/logs/valid_e5.log (VALID_E5_DONE)"; fi
echo "ANCHOR r28 end $(date)"
