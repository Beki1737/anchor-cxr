#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r20.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; J="python $C/judge_vllm.py"
echo "ANCHOR r20 start $(date)"
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
echo "-- valid_eval so far:"; tail -12 ~/anchor/logs/valid_eval.log

source ~/venvs/anchor_eval/bin/activate
echo; echo "################ A. GREEN port verdict (radeval r15 aggregates) ################"
$J compare --judge green --tol-agg 0.02 \
  --ref-agg "ceiling_self=1.0,floor_shuffled=0.300,normal_template=0.407,b0_medgemma4b=0.574" 2>&1 | tail -14

echo; echo "################ B. judge disagreement diagnostics (dev200) ################"
$J diag --cond-a b0_medgemma4b --cond-b normal_template
deactivate

echo; echo "################ C. background: GREEN on valid cohort ################"
if pgrep -f valid_green.sh > /dev/null; then echo "already running"
else nohup bash $C/valid_green.sh > ~/anchor/logs/valid_green.log 2>&1 & disown; echo "launched: tail ~/anchor/logs/valid_green.log"; fi
sleep 120; cat ~/anchor/logs/valid_green.log
echo "ANCHOR r20 end $(date)"
