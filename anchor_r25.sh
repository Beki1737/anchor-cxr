#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r25.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r25 start $(date)"

echo; echo "################ A. CheXbert strata (independent check of D40) ################"
source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/chexbert_strata.py 2>&1 | grep -vE "Warning|warn|incr_download|it/s\]" | tail -25
deactivate

echo; echo "################ B. extractor A/B (14B v2 vs 32B fp8) ################"
tail -3 ~/anchor/logs/e1_abtest.log 2>/dev/null
source ~/venvs/anchor/bin/activate
[ -f ~/anchor/work/e1_abtest.csv ] && python $C/corpus_extract_v2.py abscore 2>&1 | tail -2
deactivate

echo; echo "################ C. background: E4a (claims from B0 outputs, then pool/verification test) ################"
if pgrep -f e4a_chain.sh > /dev/null; then echo "already running"
else nohup bash $C/e4a_chain.sh > ~/anchor/logs/e4a.log 2>&1 & disown; echo "launched: tail -30 ~/anchor/logs/e4a.log (done at E4A_CHAIN_DONE)"; fi
echo "ANCHOR r25 end $(date)"
