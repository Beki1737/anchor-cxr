#!/usr/bin/env bash
# ANCHOR round 35 = relaunch E9 (MIMIC) and E10 (PadChest-GR) as GPU-waiting background chains (lab GPUs are full).
# MobaXterm:   bash ~/anchor/code/anchor_r35.sh        Log: ~/anchor/logs/r35.log
# Needs in ~/anchor/code/: gpu_wait.sh, padchest_run.sh (new), mimic_run.sh, gen_b0.py, corpus_extract.py (updated)
set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r35.log) 2>&1
C=~/anchor/code
echo "ANCHOR r35 start $(date)"
pkill -f mimic_run.sh 2>/dev/null; pkill -f padchest_run.sh 2>/dev/null; sleep 2
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
nohup bash $C/padchest_run.sh > ~/anchor/logs/padchest_run.log 2>&1 & disown
nohup bash $C/mimic_run.sh > ~/anchor/logs/mimic_run.log 2>&1 & disown
echo "launched: tail -40 ~/anchor/logs/padchest_run.log (PADCHEST_RUN_DONE) | tail -60 ~/anchor/logs/mimic_run.log (MIMIC_RUN_DONE)"
sleep 20; tail -3 ~/anchor/logs/padchest_run.log; tail -3 ~/anchor/logs/mimic_run.log
echo "ANCHOR r35 end $(date)"
