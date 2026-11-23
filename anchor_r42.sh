#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r42.log) 2>&1
C=~/anchor/code
echo "ANCHOR r42 start $(date)"
echo "-- reader files (modified time, size):"; ls -l --time-style=+%m-%d_%H:%M ~/anchor/work/reader/*.csv | awk '{print $6, $5, $7}'
source ~/venvs/anchor_eval/bin/activate
echo; echo "######## reader study ########"
python $C/reader_analysis.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo; echo "######## E16 valid summary (CRIMSON part) ########"
sed -n '/=== CRIMSON valid/,/=== GREEN valid/p' ~/anchor/logs/e16_judge.log
echo; echo "######## MIMIC judge summary ########"
sed -n '/=== CRIMSON mimic/,/E8_SUMMARY_DONE/p' ~/anchor/logs/e9_mimic.log
grep -E "^b0_medgemma4b_mimic (radgraph|f1chexbert|ratescore) +b0" ~/anchor/logs/e9_mimic.log
echo; echo "-- sc_run:"; tail -4 ~/anchor/logs/sc_run.log
nvidia-smi -i 0,1 --query-gpu=index,memory.free --format=csv,noheader
echo "ANCHOR r42 end $(date)"
