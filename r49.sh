#!/usr/bin/env bash
# r49 (2026-11-20). GPUs 0,1 only (ANCHOR_GPUS=0,1).
#   TEST=0 (default): A. claim extraction of ANCHOR-edit outputs, valid + MIMIC (for E27 audit)
#                     B. E25 MIMIC re-run after patch_e25.py (radiologist block)      -> ~/anchor/logs/r49.log
#   TEST=1          : C. GATED test looks approved by the author (D94 E25 test, D95 E26 Lingshu test),
#                        = the TEST block of anchor_r48.sh with every e23_e2e line removed -> ~/anchor/logs/r49_test.log
# Order:
#   1) python ~/anchor/code/patch_e25.py
#   2) nohup bash ~/anchor/code/r49.sh > /dev/null 2>&1 &          # no test data; done at R49_DONE
#   3) DRY=1 TEST=1 bash ~/anchor/code/r49.sh                      # prints the exact test block, runs nothing; paste it
#   4) only after Claude confirms the block AND the method-definition gate (D98) is answered:
#      TEST=1 nohup bash ~/anchor/code/r49.sh > /dev/null 2>&1 &     # done at R49_TEST_DONE
set -u
mkdir -p ~/anchor/logs
C=~/anchor/code; J="python $C/judge_vllm.py"; source $C/gpu_wait.sh
LS=~/anchor/models/lingshu-medical-mllm/Lingshu-7B
TEST=${TEST:-0}; DRY=${DRY:-0}
P=e26anchorl:b0_lingshu7b,e26editl:b0_lingshu7b,b0_lingshu7b:b0_medgemma4b,e26anchorl:e8anchor,e26editl:e8edit

judge () {   # tag split gens  (verbatim from anchor_r48.sh)
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

if [ "$TEST" = 1 ]; then
  BLK=$(mktemp /tmp/r49_testblock.XXXX.sh)
  sed -n '/^if \[ "\$TEST" = 1 \]; then/,$p' $C/anchor_r48.sh | grep -v -E "e23_e2e" > $BLK
  if [ "$DRY" = 1 ]; then
    echo "DRY RUN (nothing executed). anchor_r48.sh md5: $(md5sum $C/anchor_r48.sh | cut -c1-32)"
    echo "---- test block that TEST=1 would execute (e23 lines removed) ----"; cat -n $BLK
    echo "---- lines removed ----"; sed -n '/^if \[ "\$TEST" = 1 \]; then/,$p' $C/anchor_r48.sh | grep -n -E "e23_e2e" || echo "(none)"
    exit 0
  fi
  exec > >(tee ~/anchor/logs/r49_test.log) 2>&1
  echo "ANCHOR r49 TEST start $(date) | GPUs ${ANCHOR_GPUS} | gated looks: E25 test (D94), E26 test (D95); E23 --test NOT run"
  md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json ~/anchor/work/e26/freeze_lingshu.json \
         $C/e25_findings.py $C/e26_plugin.py $C/anchor_r48.sh $BLK
  source $BLK
  echo "ANCHOR r49 TEST end $(date)"; echo R49_TEST_DONE
  exit 0
fi

exec > >(tee ~/anchor/logs/r49.log) 2>&1
echo "ANCHOR r49 start $(date) | GPUs ${ANCHOR_GPUS} | TEST=0 (valid + MIMIC only)"
md5sum $C/e25_findings.py $C/corpus_extract_v2.py

echo; echo "######## A. claim extraction of ANCHOR-edit outputs (E27 input) ########"
source ~/venvs/anchor/bin/activate
for S in valid mimic; do
  [ -d ~/anchor/work/corpus_gen/e8edit_$S ] || gpu_run 46 ~/anchor/logs/r49_gentext_$S.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen e8edit_$S.parquet
  tail -1 ~/anchor/logs/r49_gentext_$S.log 2>/dev/null
  ls ~/anchor/work/corpus_gen/e8edit_$S 2>/dev/null | head -5
done

echo; echo "######## B. E25 MIMIC (patched radiologist block) ########"
small $C/e25_findings.py --split mimic
deactivate
echo "ANCHOR r49 end $(date)"; echo R49_DONE
