#!/usr/bin/env bash
# Background (launched by anchor_r36.sh; replaces mimic_run.sh): E9 MIMIC-CXR external validation with the FROZEN ReXGradient configuration.
# Every GPU step waits until a card has enough free memory (shared lab GPUs) and resumes from finished outputs.
# Log: ~/anchor/logs/e9_mimic.log
set -u
C=~/anchor/code; J="python $C/judge_vllm.py"; W=~/anchor/work
source $C/gpu_wait.sh
echo "e9_mimic start $(date)"
source ~/venvs/anchor/bin/activate
[ -f $W/manifest/rex_mimic.parquet ] || python $C/mimic_manifest.py
if [ "$(ls $W/regions/mimic/shard_*.npz 2>/dev/null | wc -l)" -lt 3 ]; then
  wait_free 10; python $C/region_cache.py --split mimic --gpu $CUDA_VISIBLE_DEVICES --batch 16 --workers 16 2>&1 | grep -E "images|FAILED|Error" | tail -2; unset CUDA_VISIBLE_DEVICES
fi
if ! grep -q CORPUS_V2_DONE ~/anchor/logs/mimic_corpus.log 2>/dev/null; then
  gpu_run 46 ~/anchor/logs/mimic_corpus.log CORPUS_V2_DONE python $C/corpus_extract_v2.py full --split mimic --gpu-util 0.45
fi
tail -1 ~/anchor/logs/mimic_corpus.log
if [ ! -f $W/gen/b0_medgemma4b_mimic.parquet ]; then
  gpu_run 22 ~/anchor/logs/mimic_b0.log GEN_B0_DONE python $C/gen_b0.py --name medgemma4b --keys mimic --gpu-util $(share 20)
fi
grep -E "^done" ~/anchor/logs/mimic_b0.log
if ! grep -q GENTEXT_DONE ~/anchor/logs/mimic_gentext.log 2>/dev/null; then
  gpu_run 46 ~/anchor/logs/mimic_gentext.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen b0_medgemma4b_mimic.parquet --gpu-util 0.45
fi
if [ ! -f $W/retrieval/global_cls_mimic.parquet ]; then
  wait_free 6; python $C/e3_global_rag.py retrieve --split mimic 2>&1 | grep -E "queries|RETRIEVAL_SANITY"; unset CUDA_VISIBLE_DEVICES
fi
if [ ! -f $W/gen/e3rag_medgemma4b_k3_mimic.parquet ]; then
  gpu_run 22 ~/anchor/logs/mimic_e3.log E3_GEN_DONE python $C/e3_global_rag.py gen --split mimic --k 3 --gpu-util $(share 20)
fi
wait_free 8
[ -f $W/e8_mimic/candidates.parquet ] || python $C/e8_final.py build --split mimic 2>&1 | grep -E "E4b|Error"
[ -f $W/e8_mimic/verify_probe.parquet ] || python $C/e4c_probe.py score --dir e8_mimic --split mimic 2>&1 | grep -E "probe|Error"
echo "#### frozen ANCHOR on MIMIC (calibrated on ReXGradient valid only) ####"
python $C/e8_final.py apply --split mimic 2>&1 | grep -vE "Warning|warn"
echo "#### radiologist labels ####"
python $C/e9_mimic_labels.py 2>&1 | grep -vE "Warning|warn"
unset CUDA_VISIBLE_DEVICES
deactivate

GENS=b0_medgemma4b_mimic.parquet,e3rag_medgemma4b_k3_mimic.parquet,e8anchor_mimic.parquet,e8edit_mimic.parquet,e8multi_mimic.parquet,e8vlmfree_mimic.parquet
source ~/venvs/anchor_eval/bin/activate
$J pairs --tag mimic --split mimic --gens $GENS --ref-conds normal_template
for j in crimson green; do $J prep --tag mimic --judge $j 2>&1 | grep -E "^\[|Error"; done
deactivate
source ~/venvs/anchor/bin/activate
for j in crimson green; do
  U=0.35; N=36; [ $j = green ] && { U=0.30; N=31; }      # CRIMSON needs ~10k context: 0.21 left no KV cache (r37)
  grep -q JUDGE_GEN_DONE ~/anchor/logs/mimic_gen_$j.log 2>/dev/null || \
    gpu_run $N ~/anchor/logs/mimic_gen_$j.log JUDGE_GEN_DONE $J gen --tag mimic --judge $j --gpu-util $U
done
deactivate
source ~/venvs/anchor_eval/bin/activate
for j in crimson green; do $J score --tag mimic --judge $j 2>&1 | grep -E "parse_failed|Error"; done
python $C/e8_summary.py --split mimic --tags mimic
echo "MIMIC_JUDGES_DONE $(date)"
wait_free 10
for g in b0_medgemma4b_mimic e8anchor_mimic e8edit_mimic; do
  python $C/eval_smoke.py --gen $W/gen/$g.parquet --split mimic --skip-ceiling --metrics radgraph,f1chexbert,ratescore 2>&1 \
    | grep -E "^radgraph|^f1chexbert|^ratescore" | sed "s/^/$g /"
done
deactivate
echo "E9_MIMIC_DONE $(date)"
