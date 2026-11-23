#!/usr/bin/env bash
# ANCHOR round 29 = E6 deep-dive (H6 side, H7 VLM-free, H8 prior, H9 subgroup guarantee, H10 autonomous normal,
# per-finding table) + background judges for the new report variants (start after the E5 judges finish).
# MobaXterm:   bash ~/anchor/code/anchor_r29.sh        Log: ~/anchor/logs/r29.log (send back). Foreground ~10 to 20 min.
# Needs in ~/anchor/code/: e6_deep.py, valid_e6.sh (new), e5_summary.py (updated)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r29.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r29 start $(date)"
echo "-- E5 judges so far:"; grep -E "JUDGE_GEN_DONE|parse_failed|E5_SUMMARY_DONE|VALID_E5" ~/anchor/logs/valid_e5.log | tail -5
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/e6_deep.py 2>&1 | grep -vE "Warning|warn"
unset CUDA_VISIBLE_DEVICES
deactivate
if pgrep -f valid_e6.sh > /dev/null; then echo "E6 judges already queued"
else nohup bash $C/valid_e6.sh > ~/anchor/logs/valid_e6.log 2>&1 & disown; echo "E6 judges queued (start after E5 judges): tail -60 ~/anchor/logs/valid_e6.log"; fi
echo "ANCHOR r29 end $(date)"
