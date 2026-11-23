#!/usr/bin/env bash
# Background (launched by anchor_r25.sh): claims from MedGemma B0 outputs (v2 extractor) -> E4a pool/verification test.
# Log: ~/anchor/logs/e4a.log
set -u
C=~/anchor/code
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
source ~/venvs/anchor/bin/activate
echo "e4a start $(date)"
for t in 1 2 3 4; do
  python $C/corpus_extract_v2.py gentext --gen b0_medgemma4b_valid.parquet > ~/anchor/logs/e4a_gentext.log 2>&1
  grep -q GENTEXT_DONE ~/anchor/logs/e4a_gentext.log && break; sleep 120
done
grep -E "GENTEXT_DONE|Error" ~/anchor/logs/e4a_gentext.log | tail -2
export CUDA_VISIBLE_DEVICES=$(lowgpu)
for k in 10 5; do python $C/e4a_pool.py --k $k 2>&1 | grep -vE "Warning|warn"; done
echo "E4A_CHAIN_DONE $(date)"
