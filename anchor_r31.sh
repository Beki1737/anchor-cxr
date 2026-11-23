#!/usr/bin/env bash
# ANCHOR round 31 = FREEZE the final configuration on valid (scorer, focal offset, CRC thresholds, multi-group thresholds,
# side resolver, triage with CRC + high-probability thresholds, code checksums), write valid-eval reports, then launch the
# ONE-SHOT test run in background (B0 + claims + E3 + candidates + probe + frozen apply + judges + metrics, ~6 to 8 h).
# MobaXterm:   bash ~/anchor/code/anchor_r31.sh        Log: ~/anchor/logs/r31.log (send back now), test_run.log later.
# Needs in ~/anchor/code/: e8_final.py, e8_summary.py, test_run.sh (new), e4b_verify.py (updated)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r31.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r31 start $(date)"
echo "-- E7 fast metrics (if finished):"; grep -E "^e7edit|^b0_med" ~/anchor/logs/valid_e7.log | tail -9
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
if [ -f ~/anchor/work/e8/freeze.json ]; then echo "freeze.json exists: NOT re-freezing (one-shot rule)"; else python $C/e8_final.py freeze 2>&1 | grep -vE "Warning|warn"; fi
unset CUDA_VISIBLE_DEVICES
deactivate
if pgrep -f test_run.sh > /dev/null; then echo "test run already running"
else nohup bash $C/test_run.sh > ~/anchor/logs/test_run.log 2>&1 & disown; echo "ONE-SHOT test run launched: tail -60 ~/anchor/logs/test_run.log (TEST_RUN_DONE)"; fi
echo "ANCHOR r31 end $(date)"
