#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r47.log) 2>&1
C=~/anchor/code; J="python $C/judge_vllm.py"; source $C/gpu_wait.sh
echo "ANCHOR r47 start $(date) | allowed GPUs ${ANCHOR_GPUS}"
echo "######## PRE-REGISTRATION (fixed before running) ########"
md5sum $C/e22_prereg.md $C/e22_nre.py $C/e22_s3.py ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json
cat $C/e22_prereg.md
source ~/venvs/anchor/bin/activate
echo; echo "######## E22 build + select (valid) ########"
wait_free 10; python $C/e22_nre.py 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES
deactivate
grep -q . ~/anchor/work/e22/freeze3.json 2>/dev/null || { echo "E22 failed before judging"; echo "R47_FAILED"; exit 1; }
echo; echo "######## judges on valid folds 2-3 ########"
G=e22nremulti_valid.parquet,e22ctrlmulti_valid.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs --tag valid_e22 --split valid --gens $G --ref-conds none
for j in crimson green; do $J prep --tag valid_e22 --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate; source ~/venvs/anchor/bin/activate
for j in crimson green; do
  N=36; U=0.35; [ $j = green ] && { N=31; U=0.30; }
  gpu_run $N ~/anchor/logs/valid_e22_gen_$j.log JUDGE_GEN_DONE $J gen --tag valid_e22 --judge $j --gpu-util $U
done
deactivate; source ~/venvs/anchor_eval/bin/activate
for j in crimson green; do $J score --tag valid_e22 --judge $j 2>&1 | grep -E "parse_failed|Error"; done
echo; echo "######## S3 + overall verdict ########"
python $C/e22_s3.py
deactivate
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json ~/anchor/work/e22/freeze3.json
echo "ANCHOR r47 end $(date)"
echo "R47_DONE"
