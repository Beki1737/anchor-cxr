#!/usr/bin/env bash
# ANCHOR round 10 = cuDNN fix + E0 pilot rerun. Run in MobaXterm:   bash ~/anchor/code/anchor_r10.sh
# Log: ~/anchor/logs/r10.log (send it back). Needs the updated ~/anchor/code/region_cache.py
# Changes only our own venv activate script (~/venvs/anchor/bin/activate); nothing system-wide.
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r10.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data; ACT=~/venvs/anchor/bin/activate
echo "ANCHOR r10 start $(date)"
echo "region_cache.py md5=$(md5sum < "$C/region_cache.py" | cut -c1-32)"
source "$ACT"
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "test GPU: $GPU"

echo; echo "################ A. cuDNN diagnosis ################"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "system cudnn (ldconfig):"; ldconfig -p 2>/dev/null | grep -i cudnn | head -6
echo "venv cudnn libs:"; find ~/venvs/anchor -name "libcudnn*.so*" 2>/dev/null | sed "s#$HOME#~#" | head -12
CUDNN_LIB=$(dirname "$(find ~/venvs/anchor -name 'libcudnn.so.9*' 2>/dev/null | head -1)" 2>/dev/null)
echo "venv cudnn dir: ${CUDNN_LIB:-<none>}"
PROBE='import torch, sys
print("  torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
try:
    m = torch.nn.Conv2d(3, 8, 3).cuda(); y = m(torch.randn(2, 3, 64, 64, device="cuda")); torch.cuda.synchronize()
    print("  CONV_OK", tuple(y.shape)); ok = True
except RuntimeError as e:
    print("  CONV_FAIL", str(e)[:120]); ok = False
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "cudnn" in l and "/" in l})
print("  loaded cudnn files:", [p.replace(__import__("os").path.expanduser("~"), "~") for p in libs][:8])
sys.exit(0 if ok else 1)'
try_variant () {  # $1 name, rest = env prefix
  local name=$1; shift
  echo "-- variant: $name"
  env "$@" CUDA_VISIBLE_DEVICES=$GPU python -c "$PROBE" && return 0 || return 1
}
FIX=""
if try_variant "as-is" ; then FIX="none"
elif try_variant "LD_LIBRARY_PATH unset" -u LD_LIBRARY_PATH ; then FIX="unset"
elif [ -n "${CUDNN_LIB:-}" ] && try_variant "venv cudnn first" LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}" ; then FIX="prepend"
fi
echo "chosen fix: ${FIX:-<none works>}"

if [ "$FIX" = "unset" ] || [ "$FIX" = "prepend" ]; then
  if ! grep -q "ANCHOR cudnn fix" "$ACT"; then
    {
      echo ""
      echo "# ANCHOR cudnn fix (r10): keep the system cuDNN out of this venv"
      if [ "$FIX" = "unset" ]; then echo "unset LD_LIBRARY_PATH"
      else echo "export LD_LIBRARY_PATH=\"$CUDNN_LIB:\${LD_LIBRARY_PATH:-}\""; fi
    } >> "$ACT"
    echo "fix appended to venv activate"
  else echo "fix already present in activate"; fi
  deactivate; source "$ACT"; echo "re-activated; LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
fi

echo; echo "################ B. E0 pilot rerun ################"
NOCUDNN=""; [ -z "$FIX" ] && NOCUDNN="--no-cudnn" && echo "no variant worked: running with cuDNN disabled"
rm -rf ~/anchor/work/regions/test_pilot2000
python "$C/region_cache.py" --split test --limit 2000 --overlays 24 --batch 32 --workers 16 $NOCUDNN
echo "overlays: $(ls ~/anchor/work/qa_overlays/*.png 2>/dev/null | wc -l) files in ~/anchor/work/qa_overlays/ (open in VS Code)"

echo; echo "################ C. background jobs ################"
echo "MIMIC raw: $(find "$D/raw" -name '*.jpg' | wc -l) / 3041"; tail -1 "$D/fetch_mimic_test.log"
echo "reports zip bytes: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null)"; tail -2 "$D/reports/fetch_reports.log" 2>/dev/null
echo "ANCHOR r10 end $(date)"
