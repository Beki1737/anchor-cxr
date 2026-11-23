#!/usr/bin/env bash
# Background (launched by anchor_r23.sh): E1 v2 extractor. devcheck on dev-sheet reports -> full corpus
# (valid first: needed for strata; then test; then train) -> fresh held-out gate sheet. Log: ~/anchor/logs/e1v2.log
set -u
C=~/anchor/code; X="python $C/corpus_extract_v2.py"
FILT='^\(EngineCore|^\(MainProcess|^\(APIServer|Capturing CUDA graphs|Loading safetensors|it/s\]$|^INFO|^WARNING|^$'
source ~/venvs/anchor/bin/activate
echo "e1v2 start $(date)"
for t in 1 2 3; do
  $X devcheck 2>&1 | grep -vE "$FILT" > ~/anchor/logs/e1v2_devcheck.log
  grep -q DEVCHECK_DONE ~/anchor/logs/e1v2_devcheck.log && break; sleep 90
done
grep -vE "Traceback|^  " ~/anchor/logs/e1v2_devcheck.log | tail -6
for sp in valid test train; do
  for t in 1 2 3 4 5; do         # resumable shards: a failed vLLM init just restarts where it stopped
    $X full --split $sp 2>&1 | grep -vE "$FILT" > ~/anchor/logs/e1v2_$sp.log
    grep -q "CORPUS_V2_DONE" ~/anchor/logs/e1v2_$sp.log && break; sleep 120
  done
  tail -2 ~/anchor/logs/e1v2_$sp.log
done
$X sheet --n 200 --seed 1
echo "E1V2_DONE $(date)"
