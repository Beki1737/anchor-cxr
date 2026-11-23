#!/usr/bin/env bash
# ANCHOR round 24 = (A) score both E1 sheets (gate, CIs, error codes), (B) one-page E3 summary + CheXbert strata,
# (C) D40 strata re-run on the better v2 labels, (D) background: E1 extractor A/B test, Qwen2.5-14B v2 vs 32B (fp8),
# on a third sentence pool. MobaXterm:   bash ~/anchor/code/anchor_r24.sh
# Log: ~/anchor/logs/r24.log (send back).
# Needs in ~/anchor/code/: score_labels.py, strata.py, corpus_extract_v2.py, e3_summary.py (all updated/new)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r24.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r24 start $(date)"
source ~/venvs/anchor_eval/bin/activate

echo; echo "################ A. E1 sheets ################"
python $C/score_labels.py e1_label_200_v2_heldout.csv
python $C/label_audit.py e1_label_200_v2_heldout.csv
ls ~/anchor/work/ | grep -i rater2 || echo "no second-rater sheet yet (kappa needs one)"

echo; echo "################ B. E3 summary + CheXbert strata (from r23) ################"
python $C/e3_summary.py corpus_v2

echo; echo "################ C. D40 strata with E1 v2 labels (v1 labels were below gate) ################"
python $C/strata.py --tags valid,valid_e3 --corpus corpus_v2 --out strata_paired_v2.json | grep -E "crimson|STRATA|strata"
deactivate

echo; echo "################ D. background: extractor A/B test (14B v2 vs 32B fp8) ################"
M32=${HF_MODELS_DIR:-./models}/Qwen/Qwen2.5-32B-Instruct
if [ -f "$M32/config.json" ]; then
  nohup bash -c "source ~/venvs/anchor/bin/activate; for t in 1 2 3; do python $C/corpus_extract_v2.py abtest --model $M32 --quant fp8 --gpu-util 0.55 > ~/anchor/logs/e1_abtest.log 2>&1; grep -q ABTEST_DONE ~/anchor/logs/e1_abtest.log && break; sleep 90; done" \
    > /dev/null 2>&1 & disown
  echo "launched: tail -3 ~/anchor/logs/e1_abtest.log  (sheet ready when ABTEST_DONE)"
else
  echo "Qwen2.5-32B not found at $M32; available Qwen models:"; ls ${HF_MODELS_DIR:-./models}/Qwen/ 2>/dev/null
fi
echo "ANCHOR r24 end $(date)"
