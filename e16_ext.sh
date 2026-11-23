#!/usr/bin/env bash
# Background (launched by anchor_r43.sh): freeze2 (FUS family, selected on valid, D75) applied ONCE to test
# (declared second look) and to MIMIC (external), then CRIMSON + GREEN + summaries. GPUs 0/1 only.
# Log: ~/anchor/logs/e16_ext.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; W=~/anchor/work; source $C/gpu_wait.sh
echo "e16_ext start $(date)"
source ~/venvs/anchor/bin/activate
for S in test mimic; do
  [ -f $W/gen/e16fusmulti_$S.parquet ] || { wait_free 8; python $C/e16_fusion.py apply --split $S 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES; }
done
deactivate
for S in test mimic; do
  T=${S}_e16; G=e16fusmulti_$S.parquet,e16fus_$S.parquet,e16fusedit_$S.parquet
  source ~/venvs/anchor_eval/bin/activate
  $J pairs --tag $T --split $S --gens $G --ref-conds none
  for j in crimson green; do $J prep --tag $T --judge $j 2>&1 | grep -E "^\[|Error"; done
  deactivate; source ~/venvs/anchor/bin/activate
  for j in crimson green; do
    U=0.35; N=36; [ $j = green ] && { U=0.30; N=31; }
    grep -q JUDGE_GEN_DONE ~/anchor/logs/${T}_gen_$j.log 2>/dev/null || \
      gpu_run $N ~/anchor/logs/${T}_gen_$j.log JUDGE_GEN_DONE $J gen --tag $T --judge $j --gpu-util $U
  done
  deactivate; source ~/venvs/anchor_eval/bin/activate
  for j in crimson green; do $J score --tag $T --judge $j 2>&1 | grep -E "parse_failed|Error"; done
  python $C/e8_summary.py --split $S --tags $S,$T --extra-sys e16fusmulti,e16fus,e16fusedit
  python $C/e16_paired.py --split $S --tags $S,$T --pairs e16fusmulti:e8anchor,e16fusmulti:e8multi,e16fus:e8anchor,e16fusedit:e8edit,e16fusedit:b0_medgemma4b
  deactivate
done
echo "E16_EXT_DONE $(date)"
