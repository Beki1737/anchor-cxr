#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r45.log) 2>&1
C=~/anchor/code
echo "ANCHOR r45 start $(date)"
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json
ls -l ~/anchor/work/e18/
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=""
echo; echo "######## E20 label-free weighted CRC ########"
time python $C/e20_wcrc.py 2>&1 | grep -vE "Warning|warn"
echo; echo "######## E21 selective two-sided certificate ########"
time python $C/e21_selective.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo; echo "######## FUS vs ANCHOR paired, TEST second look (already computed in r43, printed only) ########"
awk '/^CRIMSON test$/,/E16_PAIRED_DONE/' ~/anchor/logs/e16_ext.log | head -40
grep -n "GREEN test" -A12 ~/anchor/logs/e16_ext.log | grep -iE "pair|minus| - |fusmulti:|fusedit:" | head -20
echo; md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json
echo "ANCHOR r45 end $(date)"
echo "R45_DONE"
