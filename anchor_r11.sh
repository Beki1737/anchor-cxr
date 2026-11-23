#!/usr/bin/env bash
# ANCHOR round 11 = venv cuDNN fix + failure breakdown + full region cache (background). MobaXterm:
#   bash ~/anchor/code/anchor_r11.sh
# Log: ~/anchor/logs/r11.log (send back). Full-cache log: ~/anchor/logs/region_full.log
# Needs in ~/anchor/code/: qa_failures.py (new), region_cache.py (r10 version)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r11.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data
echo "ANCHOR r11 start $(date)"
source ~/venvs/anchor/bin/activate
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
PROBE='import torch, sys
print("  cudnn runtime", torch.backends.cudnn.version())
try:
    m = torch.nn.Conv2d(3, 8, 3).cuda(); m(torch.randn(2, 3, 64, 64, device="cuda")); torch.cuda.synchronize(); print("  CONV_OK"); sys.exit(0)
except RuntimeError as e:
    print("  CONV_FAIL", str(e)[:100]); sys.exit(1)'

echo; echo "################ A. venv cuDNN upgrade to match the system 9.27 sub-library (venv only, auto-revert) ################"
SYSV=$(ls /usr/lib/x86_64-linux-gnu/libcudnn_engines_tensor_ir.so.9.* 2>/dev/null | head -1 | grep -oP '9\.\d+\.\d+$')
OLDV=$(uv pip show nvidia-cudnn-cu13 2>/dev/null | grep -i '^version' | awk '{print $2}')
echo "system tensor_ir version: ${SYSV:-?}   venv nvidia-cudnn-cu13: ${OLDV:-?}"
CUDNN_OK=0
if CUDA_VISIBLE_DEVICES=$GPU python -c "$PROBE"; then CUDNN_OK=1; echo "already working"
elif [ -n "$SYSV" ]; then
  MM=$(echo "$SYSV" | cut -d. -f1,2)
  if uv pip install --no-deps "nvidia-cudnn-cu13==${MM}.*" 2>&1 | tail -2 && CUDA_VISIBLE_DEVICES=$GPU python -c "$PROBE"; then
    CUDNN_OK=1; echo "FIXED: venv cuDNN now $(uv pip show nvidia-cudnn-cu13 | grep -i '^version' | awk '{print $2}')"
    uv pip freeze > ~/anchor/requirements.anchor.lock
  else
    echo "upgrade did not fix it; reverting to $OLDV"; uv pip install --no-deps "nvidia-cudnn-cu13==$OLDV" 2>&1 | tail -1
  fi
fi
NOCUDNN=""; [ "$CUDNN_OK" = "1" ] || NOCUDNN="--no-cudnn"
echo "cuDNN usable: $CUDNN_OK (region cache flag: ${NOCUDNN:-none})"

echo; echo "################ B. E0 pilot failure breakdown ################"
python "$C/qa_failures.py" --tag test_pilot2000 --split test --n 12

echo; echo "################ C. full region cache in background: test, valid, train (~123k images) ################"
if pgrep -f "region_cache.py --split" > /dev/null; then echo "a region_cache job is already running; not starting another"
else
  nohup bash -c "source ~/venvs/anchor/bin/activate; for s in test valid train; do python $C/region_cache.py --split \$s --batch 32 --workers 16 $NOCUDNN || echo FAILED_\$s; done; echo ALL_REGIONS_DONE \$(date)" \
    > ~/anchor/logs/region_full.log 2>&1 & disown
  echo "LAUNCHED (expected about 35 to 45 min at ~60 img/s; check: tail -3 ~/anchor/logs/region_full.log)"
fi

echo; echo "################ D. background downloads ################"
echo "MIMIC raw: $(find "$D/raw" -name '*.jpg' | wc -l) / 3041"; tail -1 "$D/fetch_mimic_test.log"
echo "reports zip bytes: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null)"
echo "ANCHOR r11 end $(date)"
