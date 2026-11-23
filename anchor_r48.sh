#!/usr/bin/env bash
# ANCHOR round 48 (Prof. Jang's review). VALIDATION + MIMIC by default; the test split is read ONLY with TEST=1 (gate).
#   A  E23 end-to-end certificate (laterality inside the guarantee)            [small GPU, minutes]
#   B  E24 anatomical vs global vs prior evidence ablation                     [small GPU, minutes]
#   C  E25 clinically important findings (valid folds 2-3, MIMIC + radiologists) [small GPU, minutes]
#   D  E24 CRIMSON + GREEN on valid for the evidence variants                  [GPU, ~2 h]
#   E  E26 second generator: Lingshu-7B drafts -> claims -> ANCHOR-L (valid + MIMIC) -> judges   [GPU, ~8-12 h]
#   F  TEST=1 only: E23 --test, E25 --split test, E26 on test (+ judges)        [GATED, needs the author's go]
# Start: nohup bash ~/anchor/code/anchor_r48.sh > /dev/null 2>&1 &
# Later, ONLY after the author approves the test looks:  TEST_ONLY=1 nohup bash ~/anchor/code/anchor_r48.sh > /dev/null 2>&1 &
#   (TEST_ONLY writes ~/anchor/logs/r48_test.log)
# Watch: tail -f ~/anchor/logs/r48.log     Done at R48_DONE. Send back the WHOLE log (raw).
# Needs in ~/anchor/code/: e2x_common.py e23_e2e.py e24_evidence.py e25_findings.py e26_plugin.py e48_paired.py anchor_r48.sh
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r48.log; [ "${TEST_ONLY:-0}" = 1 ] && LOG=~/anchor/logs/r48_test.log
exec > >(tee $LOG) 2>&1
C=~/anchor/code; J="python $C/judge_vllm.py"; source $C/gpu_wait.sh
LS=~/anchor/models/lingshu-medical-mllm/Lingshu-7B
TEST=${TEST:-0}; TEST_ONLY=${TEST_ONLY:-0}; [ "$TEST_ONLY" = 1 ] && TEST=1
echo "ANCHOR r48 start $(date) | GPUs ${ANCHOR_GPUS} | TEST=$TEST"
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json $C/e2x_common.py $C/e23_e2e.py $C/e24_evidence.py $C/e25_findings.py $C/e26_plugin.py
judge () {   # tag split gens
  local T=$1 S=$2 G=$3
  source ~/venvs/anchor_eval/bin/activate
  $J pairs --tag $T --split $S --gens $G --ref-conds none
  for j in crimson green; do $J prep --tag $T --judge $j 2>&1 | grep -E "^\[|Error"; done
  deactivate; source ~/venvs/anchor/bin/activate
  for j in crimson green; do
    N=36; U=0.35; [ $j = green ] && { N=31; U=0.30; }
    gpu_run $N ~/anchor/logs/${T}_gen_$j.log JUDGE_GEN_DONE $J gen --tag $T --judge $j --gpu-util $U
  done
  deactivate; source ~/venvs/anchor_eval/bin/activate
  for j in crimson green; do $J score --tag $T --judge $j 2>&1 | grep -E "parse_failed|Error"; done
  deactivate
}
small () { wait_free 8; python "$@" 2>&1 | grep -vE "Warning|warn"; unset CUDA_VISIBLE_DEVICES; }

if [ "$TEST_ONLY" != 1 ]; then
source ~/venvs/anchor/bin/activate
echo; echo "######## A. E23 end-to-end certificate (valid) ########"; small $C/e23_e2e.py
echo; echo "######## B. E24 evidence ablation (valid) ########";       small $C/e24_evidence.py
echo; echo "######## C. E25 clinically important findings ########"; small $C/e25_findings.py --split valid; small $C/e25_findings.py --split mimic
deactivate

echo; echo "######## D. E24 judges (valid folds 2-3) ########"
judge valid_e24 valid e24globalp_valid.parquet,e24regionp_valid.parquet,e24bothp_valid.parquet,e24prior_valid.parquet
source ~/venvs/anchor_eval/bin/activate
python $C/e48_paired.py --split valid --keys eval --tags valid_e8,valid_e24 \
  --pairs e24regionp:e24globalp,e24bothp:e24globalp,e24regionp:e24prior,e24globalp:e24prior,e8anchor:e24bothp
deactivate

echo; echo "######## E. E26 second generator: Lingshu-7B ########"
source ~/venvs/anchor/bin/activate
if [ ! -f $LS/config.json ]; then
  mkdir -p $LS; (hf download lingshu-medical-mllm/Lingshu-7B --local-dir $LS || huggingface-cli download lingshu-medical-mllm/Lingshu-7B --local-dir $LS) 2>&1 | tail -2
fi
du -sh $LS
for S in valid mimic; do
  [ -f ~/anchor/work/gen/b0_lingshu7b_$S.parquet ] || gpu_run 36 ~/anchor/logs/r48_gen_$S.log GEN_B0_DONE python $C/gen_b0.py --model $LS --name lingshu7b --keys $S
  grep -E "^done" ~/anchor/logs/r48_gen_$S.log 2>/dev/null
  [ -d ~/anchor/work/corpus_gen/b0_lingshu7b_$S ] || gpu_run 46 ~/anchor/logs/r48_gentext_$S.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen b0_lingshu7b_$S.parquet
  tail -1 ~/anchor/logs/r48_gentext_$S.log 2>/dev/null
  [ -f ~/anchor/work/e26_$S/candidates.parquet ] || small $C/e26_plugin.py build --split $S
  [ -f ~/anchor/work/e26_$S/verify_probe.parquet ] || small $C/e4c_probe.py score --dir e26_$S --split $S
  small $C/e26_plugin.py run --split $S
done
deactivate
judge valid_e26 valid b0_lingshu7b_valid.parquet,e26anchorl_valid.parquet,e26editl_valid.parquet
judge mimic_e26 mimic b0_lingshu7b_mimic.parquet,e26anchorl_mimic.parquet,e26editl_mimic.parquet
source ~/venvs/anchor_eval/bin/activate
P=e26anchorl:b0_lingshu7b,e26editl:b0_lingshu7b,b0_lingshu7b:b0_medgemma4b,e26anchorl:e8anchor,e26editl:e8edit
python $C/e48_paired.py --split valid --keys eval --tags valid,valid_e8,valid_e26 --pairs $P
python $C/e48_paired.py --split mimic --tags mimic,mimic_e26 --pairs $P
deactivate
fi

P=e26anchorl:b0_lingshu7b,e26editl:b0_lingshu7b,b0_lingshu7b:b0_medgemma4b,e26anchorl:e8anchor,e26editl:e8edit
if [ "$TEST" = 1 ]; then
  echo; echo "######## F. TEST (GATED: approved by the author) ########"
  source ~/venvs/anchor/bin/activate
  small $C/e23_e2e.py --test
  small $C/e25_findings.py --split test
  [ -f ~/anchor/work/gen/b0_lingshu7b_test.parquet ] || gpu_run 36 ~/anchor/logs/r48_gen_test.log GEN_B0_DONE python $C/gen_b0.py --model $LS --name lingshu7b --keys test
  [ -d ~/anchor/work/corpus_gen/b0_lingshu7b_test ] || gpu_run 46 ~/anchor/logs/r48_gentext_test.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen b0_lingshu7b_test.parquet
  [ -f ~/anchor/work/e26_test/candidates.parquet ] || small $C/e26_plugin.py build --split test
  [ -f ~/anchor/work/e26_test/verify_probe.parquet ] || small $C/e4c_probe.py score --dir e26_test --split test
  small $C/e26_plugin.py run --split test
  deactivate
  judge test_e26 test b0_lingshu7b_test.parquet,e26anchorl_test.parquet,e26editl_test.parquet
  source ~/venvs/anchor_eval/bin/activate
  python $C/e48_paired.py --split test --tags test,test_e26 --pairs $P
  deactivate
fi
md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json ~/anchor/work/e8/freeze_e2e.json ~/anchor/work/e26/freeze_lingshu.json 2>/dev/null
echo "ANCHOR r48 end $(date)"
echo "R48_DONE"
