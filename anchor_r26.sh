#!/usr/bin/env bash
# ANCHOR round 26 = (A) CheXbert strata (import fixed), (B) background E4b: image-grounded claim verification
# on a broad candidate pool (MedGemma + Qwen2.5-VL, full image and region crop).
# MobaXterm:   bash ~/anchor/code/anchor_r26.sh        Log: ~/anchor/logs/r26.log (send back)
# Needs in ~/anchor/code/: e4b_verify.py, e4b_chain.sh, chexbert_strata.py (updated)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r26.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r26 start $(date)"

echo; echo "################ A. CheXbert strata ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/chexbert_strata.py 2>&1 | grep -vE "Warning|warn|incr_download|it/s\]|RECHECK" | tail -20
deactivate
unset CUDA_VISIBLE_DEVICES

echo; echo "################ B. background: E4b ################"
if pgrep -f e4b_chain.sh > /dev/null; then echo "already running"
else nohup bash $C/e4b_chain.sh > ~/anchor/logs/e4b.log 2>&1 & disown; echo "launched: tail -30 ~/anchor/logs/e4b.log (done at E4B_CHAIN_DONE)"; fi
sleep 240; tail -3 ~/anchor/logs/e4b.log
echo "ANCHOR r26 end $(date)"
