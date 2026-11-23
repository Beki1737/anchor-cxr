#!/usr/bin/env bash
# ANCHOR round 27 = E4c trained region-evidence probe on cached RAD-DINO region embeddings (train split only),
# valid evaluation, scoring of the E4b candidates, then E4b re-analysis with the probe + patient-bootstrap CIs vs B0.
# MobaXterm:   bash ~/anchor/code/anchor_r27.sh        Log: ~/anchor/logs/r27.log (send back). About 15 to 25 min.
# Needs in ~/anchor/code/: e4c_probe.py (new), e4b_verify.py (updated)
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
