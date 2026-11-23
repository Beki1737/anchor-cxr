#!/usr/bin/env bash
# Background (launched by anchor_r22.sh): E3 on the valid primary cohort.
# retrieval -> ret1 copy baseline -> MedGemma RAG k=3 -> CRIMSON + GREEN -> paired vs template + strata -> fast metrics.
# Log: ~/anchor/logs/valid_e3.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; T="--tag valid_e3"; E="python $C/e3_global_rag.py"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
FILT='^\(EngineCore|^\(MainProcess|^\(APIServer|Capturing CUDA graphs|Loading safetensors|it/s\]$|^INFO|^WARNING|^$'
echo "valid_e3 start $(date)"

source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu)
[ -f ~/anchor/work/retrieval/global_cls_valid.parquet ] || $E retrieve --split valid 2>&1 | grep -vE "$FILT"
[ -f ~/anchor/work/gen/ret1_cls_valid.parquet ] || $E ret1 --split valid
for t in 1 2 3; do      # shared GPUs: vLLM init can fail when neighbours free memory during profiling; retry
  $E gen --split valid --k 3 > ~/anchor/logs/valid_e3_gen.log 2>&1
  grep -q E3_GEN_DONE ~/anchor/logs/valid_e3_gen.log && break
  echo "E3 gen attempt $t failed: $(grep -E 'Error' ~/anchor/logs/valid_e3_gen.log | tail -1 | cut -c1-160)"; sleep 90
done
grep -E "^done|using GPU" ~/anchor/logs/valid_e3_gen.log
deactivate
[ -f ~/anchor/work/gen/e3rag_medgemma4b_k3_valid.parquet ] || { echo "E3 gen failed 3 times, stopping"; exit 1; }

GENS=ret1_cls_valid.parquet,e3rag_medgemma4b_k3_valid.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs $T --split valid --gens $GENS --ref-conds none
for j in crimson green; do $J prep $T --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate

source ~/venvs/anchor/bin/activate
for j in crimson green; do
  export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35; [ $j = green ] && U=0.55
  for t in 1 2 3; do
    $J gen $T --judge $j --gpu-util $U > ~/anchor/logs/valid_e3_gen_$j.log 2>&1
    grep -q JUDGE_GEN_DONE ~/anchor/logs/valid_e3_gen_$j.log && break
    sleep 90; export CUDA_VISIBLE_DEVICES=$(lowgpu); U=0.35
  done
  grep -E "^\[|JUDGE_GEN_DONE|Error" ~/anchor/logs/valid_e3_gen_$j.log | tail -3
done
deactivate

source ~/venvs/anchor_eval/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu)
for j in crimson green; do
  $J score $T --judge $j 2>&1 | grep -E "mean=|JUDGE_SCORE_DONE|Error"
  for s in ret1_cls e3rag_medgemma4b_k3; do
    $J paired $T --judge $j --extra-tags valid --cond-a $s --cond-b normal_template
    $J paired $T --judge $j --extra-tags valid --cond-a $s --cond-b b0_medgemma4b
  done
done
python $C/strata.py --tags valid,valid_e3 --out strata_paired_e3.json | grep -vE "b0_medgemma4b "
echo "VALID_E3_JUDGES_DONE $(date)"
for g in ${GENS//,/ }; do
  python $C/eval_smoke.py --gen ~/anchor/work/gen/$g --split valid --only-gen \
    --metrics radgraph,f1chexbert,ratescore 2>&1 | grep -E "^\[|^radgraph|^f1chexbert|^ratescore" | sed "s/^/$g /"
done
deactivate
echo "VALID_E3_DONE $(date)"
