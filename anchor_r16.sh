#!/usr/bin/env bash
# ANCHOR round 16 = harness repair (CheXbert patch, RaTEScore via spaCy 3.8 / thinc 8.3 with auto-revert)
# + rescoring all three B0 models on the fast metrics. MobaXterm:   bash ~/anchor/code/anchor_r16.sh
# Log: ~/anchor/logs/r16.log (send back). Needs the updated ~/anchor/code/eval_smoke.py
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r16.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; G=~/anchor/work/gen
echo "ANCHOR r16 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
grep -E "FAILED_|ALL_CORPUS_DONE" ~/anchor/logs/corpus_full.log

echo; echo "################ A. RaTEScore: spaCy/thinc built for numpy 2 (eval venv only, auto-revert) ################"
source ~/venvs/anchor_eval/bin/activate
OLD_SPACY=$(uv pip show spacy 2>/dev/null | awk '/^Version/{print $2}'); OLD_THINC=$(uv pip show thinc 2>/dev/null | awk '/^Version/{print $2}')
echo "before: spacy $OLD_SPACY, thinc $OLD_THINC, numpy $(python -c 'import numpy; print(numpy.__version__)')"
if python -c "import spacy" 2>/dev/null; then echo "spacy already imports fine"
else
  uv pip install "spacy>=3.8,<3.9" "thinc>=8.3.4,<8.4" --constraint ~/anchor/eval_pins.txt 2>&1 | tail -3
  if python -c "import spacy, thinc, medspacy; print('import ok: spacy', spacy.__version__, 'thinc', thinc.__version__)" && uv pip check 2>&1 | grep -q "compatible"; then
    echo "RaTEScore deps FIXED"; uv pip freeze > ~/anchor/requirements.anchor_eval.lock
  else
    echo "upgrade failed or broke dependencies; reverting"
    uv pip install "spacy==$OLD_SPACY" "thinc==$OLD_THINC" --constraint ~/anchor/eval_pins.txt 2>&1 | tail -1
  fi
fi
uv pip check 2>&1 | tail -2

echo; echo "################ B. fast metrics on all B0 outputs (radgraph, f1chexbert, ratescore) ################"
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "eval GPU: $CUDA_VISIBLE_DEVICES"
for f in "$G"/b0_*_dev200.parquet; do
  echo; echo "=========== $(basename "$f") ==========="
  python "$C/eval_smoke.py" --gen "$f" --metrics radgraph,f1chexbert,ratescore 2>&1 \
    | grep -vE "Computing|Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|can be ignored|NLTK|incr_download|RadEval\.|^radeval|^available|^numpy"
done
deactivate
echo "ANCHOR r16 end $(date)"
