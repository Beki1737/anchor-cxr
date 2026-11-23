#!/usr/bin/env bash
# ANCHOR round 43 = (a) print E12 self-consistency results, (b) direct paired FUS vs ANCHOR on valid,
# (c) reader power check, (d) launch e16_ext.sh: freeze2 ONCE on test (second look) + MIMIC, judges (background).
# MobaXterm:   bash ~/anchor/code/anchor_r43.sh        Log: ~/anchor/logs/r43.log (send back). Foreground ~1 min.
# Needs in ~/anchor/code/: e16_paired.py, e16_ext.sh (new), reader_analysis.py (updated)
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
