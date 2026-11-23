#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r43.log) 2>&1
C=~/anchor/code
echo "ANCHOR r43 start $(date)"
echo "######## E12 self-consistency baseline (valid) ########"
sed -n '/self-consistency samples found/,/E12_DONE/p' ~/anchor/logs/sc_run.log
for s in 0 1 2 3 4; do tail -1 ~/anchor/logs/sc_gentext_$s.log; done
source ~/venvs/anchor_eval/bin/activate
echo; echo "######## E16 direct paired (valid eval) ########"
python $C/e16_paired.py --split valid --tags valid,valid_e8,valid_e16 --keys ~/anchor/work/e8/eval_keys_valid.parquet \
  --pairs e16fusmulti:e8anchor,e16fusmulti:e8multi,e16fus:e8anchor,e16fusedit:e8edit,e16exh:e8anchor
echo; echo "######## reader power check ########"
python $C/reader_analysis.py 2>&1 | sed -n '/#### 6/,/#### 5/p'
deactivate
nohup bash $C/e16_ext.sh > ~/anchor/logs/e16_ext.log 2>&1 & disown
echo "launched e16_ext.sh (E16_EXT_DONE): freeze2 on test + MIMIC, judges on GPUs 0/1"
sleep 20; tail -3 ~/anchor/logs/e16_ext.log
echo "ANCHOR r43 end $(date)"
