#!/usr/bin/env bash
# Background (launched by anchor_r31.sh): ONE-SHOT test-split run with the frozen configuration.
#   valid judges for the frozen config -> MedGemma B0 on test -> claims -> E3 RAG on test -> candidates -> probe ->
#   frozen apply -> CRIMSON + GREEN -> summary -> fast metrics. Log: ~/anchor/logs/test_run.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
FILT='^\(EngineCore|^\(MainProcess|^\(APIServer|Capturing CUDA graphs|Loading safetensors|it/s\]$|^INFO|^WARNING|^$'
retry () { local log=$1 tok=$2; shift 2; for t in 1 2 3 4; do "$@" > "$log" 2>&1; grep -q "$tok" "$log" && return 0; sleep 90; done; echo "FAILED: $*"; return 1; }
judge () {   # tag split gens refconds
  local T=$1 S=$2 G=$3 RC=$4
  source ~/venvs/anchor_eval/bin/activate
  $J pairs --tag $T --split $S --gens $G --ref-conds $RC
  for j in crimson green; do $J prep --tag $T --judge $j 2>&1 | grep -E "^\[|Error"; done
  deactivate; source ~/venvs/anchor/bin/activate
  for j in crimson green; do
    U=0.35; [ $j = green ] && U=0.55
    export CUDA_VISIBLE_DEVICES=$(lowgpu); retry ~/anchor/logs/${T}_gen_$j.log JUDGE_GEN_DONE $J gen --tag $T --judge $j --gpu-util $U
    grep -E "^\[$j\] [0-9]+ gen" ~/anchor/logs/${T}_gen_$j.log
  done
  deactivate; source ~/venvs/anchor_eval/bin/activate
  for j in crimson green; do $J score --tag $T --judge $j 2>&1 | grep -E "parse_failed|Error"; done
  deactivate
}
echo "test_run start $(date)"

echo "#### 1. valid judges for the frozen configuration ####"
judge valid_e8 valid e8anchor_valid.parquet,e8edit_valid.parquet,e8multi_valid.parquet,e8vlmfree_valid.parquet none
source ~/venvs/anchor_eval/bin/activate
python $C/e8_summary.py --split valid --tags valid,valid_e3,valid_e8 --keys ~/anchor/work/e8/eval_keys_valid.parquet
deactivate

echo "#### 2. test inputs: MedGemma B0, claims, E3 RAG ####"
source ~/venvs/anchor/bin/activate
[ -f ~/anchor/work/gen/b0_medgemma4b_test.parquet ] || retry ~/anchor/logs/test_b0.log GEN_B0_DONE python $C/gen_b0.py --name medgemma4b --keys test
grep -E "^done" ~/anchor/logs/test_b0.log
retry ~/anchor/logs/test_gentext.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen b0_medgemma4b_test.parquet
export CUDA_VISIBLE_DEVICES=$(lowgpu)
[ -f ~/anchor/work/retrieval/global_cls_test.parquet ] || python $C/e3_global_rag.py retrieve --split test 2>&1 | grep -E "queries|RETRIEVAL_SANITY"
unset CUDA_VISIBLE_DEVICES
[ -f ~/anchor/work/gen/e3rag_medgemma4b_k3_test.parquet ] || retry ~/anchor/logs/test_e3.log E3_GEN_DONE python $C/e3_global_rag.py gen --split test --k 3

echo "#### 3. frozen ANCHOR on test ####"
export CUDA_VISIBLE_DEVICES=$(lowgpu)
python $C/e8_final.py build --split test 2>&1 | grep -E "E4b|Error"
python $C/e4c_probe.py score --dir e8_test --split test 2>&1 | grep -E "probe|Error"
python $C/e8_final.py apply --split test 2>&1 | grep -vE "Warning|warn"
unset CUDA_VISIBLE_DEVICES
deactivate

echo "#### 4. test judges ####"
judge test test b0_medgemma4b_test.parquet,e3rag_medgemma4b_k3_test.parquet,e8anchor_test.parquet,e8edit_test.parquet,e8multi_test.parquet,e8vlmfree_test.parquet normal_template
source ~/venvs/anchor_eval/bin/activate
python $C/e8_summary.py --split test --tags test
echo "TEST_JUDGES_DONE $(date)"
export CUDA_VISIBLE_DEVICES=$(lowgpu)
for g in b0_medgemma4b_test e8anchor_test e8edit_test; do
  python $C/eval_smoke.py --gen ~/anchor/work/gen/$g.parquet --split test --skip-ceiling --metrics radgraph,f1chexbert,ratescore 2>&1 \
    | grep -E "^radgraph|^f1chexbert|^ratescore" | sed "s/^/$g /"
done
deactivate
echo "TEST_RUN_DONE $(date)"
