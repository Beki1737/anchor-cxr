#!/usr/bin/env bash
# ANCHOR round 13. MobaXterm:   bash ~/anchor/code/anchor_r13.sh
# A: test CUDA 13 toolkit for vLLM JIT (venv only, applied only if it works)  B: adult cohort + segmentation flags
# C: full corpus extraction train/valid/test (background, hours)  D: evaluation venv with RadEval (background)
# E: score E1 labels if the sheet is filled  F: status
# Log: ~/anchor/logs/r13.log (send back). Needs in ~/anchor/code/: cohort_flags.py, score_labels.py, corpus_extract.py
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r13.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data; ACT=~/venvs/anchor/bin/activate
echo "ANCHOR r13 start $(date)"
source "$ACT"
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)

echo; echo "################ A. CUDA 13 toolkit for vLLM / FlashInfer JIT ################"
CU13=$(python -c "import nvidia; print(list(nvidia.__path__)[0] + '/cu13')")
echo "venv CUDA 13 dir: $CU13"; ls -d "$CU13"/bin/nvcc "$CU13"/include/cuda.h "$CU13"/lib* 2>&1 | sed "s#$HOME#~#"
SMALL=$(find ${HF_MODELS_DIR:-./models} -maxdepth 3 -type d -iname "*Qwen2.5-0.5B-Instruct" | head -1)
if grep -q "ANCHOR cuda13" "$ACT"; then echo "cuda13 already applied"
elif [ -x "$CU13/bin/nvcc" ] && [ -n "$SMALL" ]; then
  "$CU13/bin/nvcc" --version | tail -2
  CUDA_HOME="$CU13" PATH="$CU13/bin:$PATH" CUDA_VISIBLE_DEVICES=$GPU python - "$SMALL" > ~/anchor/logs/cuda13_test.log 2>&1 <<'PY'
import sys
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.15, max_model_len=1024)
out = llm.generate(["Reply with the word OK."], SamplingParams(max_tokens=5, temperature=0))
print("GEN_OK", repr(out[0].outputs[0].text))
PY
  W=$(grep -c "requires CUDA >= 12.9" ~/anchor/logs/cuda13_test.log); G=$(grep -c "GEN_OK" ~/anchor/logs/cuda13_test.log)
  echo "test: generation ok=$G, 'requires CUDA >= 12.9' warnings=$W"
  if [ "$G" -ge 1 ] && [ "$W" -eq 0 ]; then
    printf '\n# ANCHOR cuda13 (r13): torch is cu132, so JIT uses the venv CUDA 13 toolkit\nexport CUDA_HOME="%s"\nexport PATH="%s/bin:$PATH"\n' "$CU13" "$CU13" >> "$ACT"
    deactivate; source "$ACT"; echo "APPLIED: CUDA_HOME=$CUDA_HOME"
  else echo "NOT applied (keeping cuda-12.8); see ~/anchor/logs/cuda13_test.log"; tail -5 ~/anchor/logs/cuda13_test.log; fi
else echo "skipped (no CUDA 13 nvcc in venv or no small Qwen model)"; fi

echo; echo "################ B. D30 adult primary cohort + segmentation flags ################"
python "$C/cohort_flags.py"

echo; echo "################ C. full corpus extraction (background) ################"
MODEL=$(find ${HF_MODELS_DIR:-./models} -maxdepth 3 -type d -iname "*Qwen2.5-14B-Instruct" | grep -viE "awq|gptq|int4|int8" | head -1)
if pgrep -f "corpus_extract.py --split" > /dev/null; then echo "corpus extraction already running"
else
  nohup bash -c "source $ACT; for s in train valid test; do python $C/corpus_extract.py --split \$s --model $MODEL || echo FAILED_\$s; done; echo ALL_CORPUS_DONE \$(date)" \
    > ~/anchor/logs/corpus_full.log 2>&1 & disown
  echo "LAUNCHED train (107,637), valid, test with $MODEL; progress: grep reports/s ~/anchor/logs/corpus_full.log | tail -2"
fi

echo; echo "################ D. evaluation venv ~/venvs/anchor_eval (background) ################"
if [ -f ~/anchor/logs/eval_env.log ] && grep -q EVAL_ENV_DONE ~/anchor/logs/eval_env.log; then echo "eval env already built"
else
  nohup bash -c '
    [ -d ~/venvs/anchor_eval ] || uv venv ~/venvs/anchor_eval --python 3.11
    source ~/venvs/anchor_eval/bin/activate
    uv pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 || exit 1
    echo "torch==2.9.1" > ~/anchor/eval_pins.txt
    uv pip install radeval --constraint ~/anchor/eval_pins.txt || exit 1
    python -c "import torch, radeval; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"sm_120\", \"sm_120\" in torch.cuda.get_arch_list(), \"| radeval ok\")"
    uv pip freeze > ~/anchor/requirements.anchor_eval.lock
    echo EVAL_ENV_DONE $(date)' > ~/anchor/logs/eval_env.log 2>&1 & disown
  echo "LAUNCHED (log: ~/anchor/logs/eval_env.log)"
fi

echo; echo "################ E. E1 label scoring ################"
python "$C/score_labels.py"

echo; echo "################ F. downloads ################"
echo "MIMIC raw: $(find "$D/raw" -name '*.jpg' | wc -l) / 3041"
echo "reports zip bytes: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null)"; tail -1 "$D/reports/fetch_reports.log" 2>/dev/null
echo "ANCHOR r13 end $(date)"
