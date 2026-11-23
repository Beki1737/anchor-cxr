#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r36.log) 2>&1
C=~/anchor/code; source $C/gpu_wait.sh
echo "ANCHOR r36 start $(date)"
pkill -u "$USER" -f mimic_run.sh 2>/dev/null; pkill -u "$USER" -f "corpus_extract_v2.py full --split mimic" 2>/dev/null; sleep 3
echo "-- last extractor root cause (r35):"; grep -E "OutOfMemory|out of memory|less than desired|No available memory|ValueError|Free memory" ~/anchor/logs/mimic_corpus.log 2>/dev/null | sort -u | tail -3 | cut -c1-220
nohup bash $C/e9_mimic.sh > ~/anchor/logs/e9_mimic.log 2>&1 & disown
nohup bash $C/sc_run.sh > ~/anchor/logs/sc_run.log 2>&1 & disown
echo "background: e9_mimic.sh (E9_MIMIC_DONE) and sc_run.sh (SC_RUN_DONE) launched"

source ~/venvs/anchor/bin/activate
echo; echo "######## E11 ablation ########"
wait_free 6; python $C/e11_ablate.py 2>&1 | grep -vE "Warning|warn"
echo; echo "######## E10 + E10b PadChest-GR ########"
python $C/e10_pointing.py 2>&1 | grep -vE "Warning|warn" | sed -n '/E10b/,$p'
unset CUDA_VISIBLE_DEVICES; deactivate

source ~/venvs/anchor_eval/bin/activate
echo; echo "######## E15 statistics (test) ########"
python $C/e15_stats.py test test 2>&1 | grep -vE "Warning|warn"
echo; echo "######## E14 independent labeller (test) ########"
wait_free 4; python $C/e14_indep.py 2>&1 | grep -vE "Warning|warn|^\s*$" | tail -16
unset CUDA_VISIBLE_DEVICES; deactivate

echo; echo "######## generators on disk (for the plug-in experiment) ########"
ls -d ${HF_MODELS_DIR:-./models}/*/* 2>/dev/null | sed 's#.*/models/##' | grep -iE "med|rad|chex|maira|llava|lingshu|qwen.*vl|gemma|internvl|cxr|huatuo" | head -40
echo; tail -3 ~/anchor/logs/e9_mimic.log; tail -3 ~/anchor/logs/sc_run.log
echo "ANCHOR r36 end $(date)"
