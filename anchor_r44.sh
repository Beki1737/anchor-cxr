#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r44.log) 2>&1
C=~/anchor/code
source $C/gpu_wait.sh
echo "ANCHOR r44 start $(date) | allowed GPUs ${ANCHOR_GPUS}"
echo "frozen files (must be unchanged):"
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json 2>/dev/null || find ~/anchor/work -name "freeze*.json" -exec md5sum {} \;
source ~/venvs/anchor/bin/activate
echo; echo "######## E18 cache: frozen scorers on VALID + MIMIC (probe on GPU) ########"
if [ -f ~/anchor/work/e18/feat_mimic.npz ] && [ -f ~/anchor/work/e18/thresholds.json ]; then
  echo "cache exists, skipping"
else
  wait_free 6; python $C/e18_cache.py 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
fi
grep -q . ~/anchor/work/e18/thresholds.json 2>/dev/null || { echo "E18 cache missing: stop"; echo "R44_FAILED"; exit 1; }
export CUDA_VISIBLE_DEVICES=""
echo; echo "######## E18 two-sided certification ########"
time python $C/e18_twosided.py 2>&1 | grep -vE "Warning|warn"
echo; echo "######## E19 anytime-valid monitoring ########"
time python $C/e19_monitor.py 2>&1 | grep -vE "Warning|warn"
echo; echo "######## E20 label-free weighted CRC ########"
time python $C/e20_wcrc.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo; echo "frozen files after (must match the start):"
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json 2>/dev/null || find ~/anchor/work -name "freeze*.json" -exec md5sum {} \;
echo "ANCHOR r44 end $(date)"
echo "R44_DONE"
