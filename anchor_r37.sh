#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r37.log) 2>&1
C=~/anchor/code; J="python $C/judge_vllm.py"; source $C/gpu_wait.sh
echo "ANCHOR r37 start $(date)"
echo "######## MIMIC frozen ANCHOR (from e9_mimic.log) ########"
sed -n '/frozen ANCHOR on MIMIC/,/E8_MIMIC_DONE/p' ~/anchor/logs/e9_mimic.log | grep -vE "Warning|warn"
grep -E "radiologist-labelled|macro AUROC" ~/anchor/logs/e9_mimic.log
source ~/venvs/anchor/bin/activate
echo; echo "######## E16 select (valid) ########"
wait_free 6; python $C/e16_fusion.py select 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
deactivate
G=e16exh_valid.parquet,e16exhedit_valid.parquet,e16fus_valid.parquet,e16fusedit_valid.parquet,e16fusmulti_valid.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs --tag valid_e16 --split valid --gens $G --ref-conds none
for j in crimson green; do $J prep --tag valid_e16 --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate; source ~/venvs/anchor/bin/activate
for j in crimson green; do
  N=22; [ $j = green ] && N=30
  gpu_run $N ~/anchor/logs/valid_e16_gen_$j.log JUDGE_GEN_DONE $J gen --tag valid_e16 --judge $j --gpu-util $(share $((N - 2)))
done
deactivate; source ~/venvs/anchor_eval/bin/activate
for j in crimson green; do $J score --tag valid_e16 --judge $j 2>&1 | grep -E "parse_failed|Error"; done
python $C/e8_summary.py --split valid --tags valid,valid_e8,valid_e16 --keys ~/anchor/work/e8/eval_keys_valid.parquet \
  --extra-sys e16exh,e16exhedit,e16fus,e16fusedit,e16fusmulti
deactivate
echo; tail -2 ~/anchor/logs/e9_mimic.log; tail -2 ~/anchor/logs/sc_run.log
echo "ANCHOR r37 end $(date)"
