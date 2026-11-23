#!/usr/bin/env bash
# ANCHOR round 39 = move EVERYTHING to GPUs 0 and 1 only (our allocation), with one shared queue.
#   stops all our chains and GPU jobs (only processes owned by $USER), relaunches e9_mimic / e16_judge / sc_run on the
#   new gpu_wait.sh, runs E17 (MIMIC recalibration) in front if r38 did not finish it. Supersedes r38.
# MobaXterm:   bash ~/anchor/code/anchor_r39.sh        Log: ~/anchor/logs/r39.log (send back)
# Needs in ~/anchor/code/: gpu_wait.sh, gen_b0.py, corpus_extract.py, region_cache.py (updated)
#                          + from r38 if not copied yet: e17_recal.py, e16_judge.sh, e9_mimic.sh
set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r39.log) 2>&1
C=~/anchor/code
echo "ANCHOR r39 start $(date)"
for p in anchor_r37.sh anchor_r38.sh e9_mimic.sh e16_judge.sh sc_run.sh mimic_run.sh padchest_run.sh; do pkill -u "$USER" -f "$p" 2>/dev/null; done
sleep 2
for p in judge_vllm.py corpus_extract_v2.py gen_b0.py e3_global_rag.py region_cache.py e17_recal.py e16_fusion.py e12_selfcons.py EngineCore; do
  pkill -u "$USER" -f "$p" 2>/dev/null
done
sleep 15
echo "-- our processes still on a GPU (should be none):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read pid mem; do
  [ "$(ps -o user= -p $pid 2>/dev/null)" = "$USER" ] && echo "   pid $pid $mem"
done
for f in gpu_wait.sh e9_mimic.sh e16_judge.sh sc_run.sh e17_recal.py; do [ -f $C/$f ] || echo "MISSING $C/$f"; done
grep -q 'ANCHOR_GPUS' $C/gpu_wait.sh || echo "OLD gpu_wait.sh: copy the new one first"
source $C/gpu_wait.sh
echo "-- allowed GPUs ($ANCHOR_GPUS):"; nvidia-smi -i "$ANCHOR_GPUS" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
rm -f ~/anchor/.gpu_start.lock
nohup bash $C/e9_mimic.sh > ~/anchor/logs/e9_mimic.log 2>&1 & disown
sleep 5; nohup bash $C/e16_judge.sh > ~/anchor/logs/e16_judge.log 2>&1 & disown
sleep 5; nohup bash $C/sc_run.sh > ~/anchor/logs/sc_run.log 2>&1 & disown
echo "relaunched on GPUs $ANCHOR_GPUS: e9_mimic.sh (E9_MIMIC_DONE), e16_judge.sh (E16_JUDGE_DONE), sc_run.sh (SC_RUN_DONE)"
if [ ! -f ~/anchor/work/e17/recal_mimic.json ]; then
  source ~/venvs/anchor/bin/activate
  echo; echo "######## E17 MIMIC recalibration ########"
  wait_free 6; python $C/e17_recal.py 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
  deactivate
else
  echo "E17 already done (r38): send ~/anchor/logs/r38.log"
fi
echo; sleep 30; for l in e9_mimic e16_judge sc_run; do echo "-- $l:"; tail -2 ~/anchor/logs/$l.log; done
echo "ANCHOR r39 end $(date)"
