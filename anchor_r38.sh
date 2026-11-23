#!/usr/bin/env bash
# ANCHOR round 38 = (a) fix judge memory (CRIMSON needs gpu-util 0.35) and relaunch the MIMIC + E16 judge chains,
#                   (b) E17 MIMIC recalibration: frozen scorers, threshold re-estimated on n_cal MIMIC studies.
# MobaXterm:   bash ~/anchor/code/anchor_r38.sh        Log: ~/anchor/logs/r38.log (send back). Foreground ~5 to 15 min.
# Needs in ~/anchor/code/: e17_recal.py, e16_judge.sh (new), e9_mimic.sh (updated)
set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r38.log) 2>&1
C=~/anchor/code; source $C/gpu_wait.sh
echo "ANCHOR r38 start $(date)"
pkill -u "$USER" -f anchor_r37.sh 2>/dev/null; pkill -u "$USER" -f e9_mimic.sh 2>/dev/null
pkill -u "$USER" -f "judge_vllm.py gen --tag valid_e16" 2>/dev/null; pkill -u "$USER" -f "judge_vllm.py gen --tag mimic" 2>/dev/null; sleep 5
nohup bash $C/e9_mimic.sh > ~/anchor/logs/e9_mimic.log 2>&1 & disown
nohup bash $C/e16_judge.sh > ~/anchor/logs/e16_judge.log 2>&1 & disown
echo "relaunched: e9_mimic.sh (E9_MIMIC_DONE), e16_judge.sh (E16_JUDGE_DONE); sc_run.sh untouched"
source ~/venvs/anchor/bin/activate
echo; echo "######## E17 MIMIC recalibration ########"
wait_free 6; python $C/e17_recal.py 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
deactivate
echo; tail -2 ~/anchor/logs/e9_mimic.log; tail -2 ~/anchor/logs/e16_judge.log; tail -2 ~/anchor/logs/sc_run.log
echo "ANCHOR r38 end $(date)"
