#!/usr/bin/env bash
# ANCHOR round 22 = (A) E1 label audit, (B) storage probe + optional reversible deletes, (C) CheXbert labeler probe
# (independent normal/abnormal strata), (D) launch E3 global-RAG baseline in background. MobaXterm:
#   bash ~/anchor/code/anchor_r22.sh                                   # measure only
#   DEL_LINGSHU=1 DEL_IUTGZ=1 bash ~/anchor/code/anchor_r22.sh         # also delete Lingshu weights + IU X-ray tgz
# Log: ~/anchor/logs/r22.log (send back).
# Needs in ~/anchor/code/: label_audit.py, storage_probe.py, e3_global_rag.py, valid_e3.sh, strata.py, judge_vllm.py, eval_smoke.py
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r22.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r22 start $(date)"

echo; echo "################ A. E1 label audit (aggregates) ################"
source ~/venvs/anchor_eval/bin/activate
python $C/label_audit.py

echo; echo "################ B. storage probe (read-only unless DEL_* flags set) ################"
df -h ~ | tail -1
echo "-- rexgradient:"; du -sh ~/anchor/data/rexgradient/* ~/anchor/data/rexgradient/.cache 2>/dev/null | sort -h | tail -8
echo "-- padchest_gr:"; du -sh ~/anchor/data/padchest_gr/* 2>/dev/null | sort -h
echo "-- iu_xray:"; du -sh ~/anchor/data/iu_xray/* 2>/dev/null | sort -h; echo "IU pngs: $(find ~/anchor/data/iu_xray -name '*.png' | wc -l)"
echo "-- HF blobs by model:"; for d in ~/anchor/.cache/huggingface/hub/models--*; do echo "$(du -shL $d/snapshots 2>/dev/null | cut -f1) $(basename $d)"; done | sort -h | tail -8
echo "-- region cache dirs:"; ls ~/anchor/work/regions/
python $C/storage_probe.py
deactivate
if [ "${DEL_LINGSHU:-0}" = 1 ] && [ -d ~/anchor/models/lingshu-medical-mllm ]; then
  rm -rf ~/anchor/models/lingshu-medical-mllm ~/.cache/huggingface/hub/models--lingshu-medical-mllm--Lingshu-7B
  echo "deleted Lingshu-7B weights (re-download: hf download lingshu-medical-mllm/Lingshu-7B, ~25 min)"
fi
if [ "${DEL_IUTGZ:-0}" = 1 ] && [ -f ~/anchor/data/iu_xray/NLMCXR_png.tgz ]; then
  N=$(find ~/anchor/data/iu_xray -name '*.png' | wc -l)
  if [ "$N" -gt 7000 ]; then rm -f ~/anchor/data/iu_xray/NLMCXR_png.tgz; echo "deleted IU tgz ($N PNGs extracted)"; else echo "IU tgz kept: only $N PNGs extracted"; fi
fi
df -h ~ | tail -1

echo; echo "################ C. CheXbert labeler probe (for an independent normal/abnormal check) ################"
source ~/venvs/anchor_eval/bin/activate
python - <<'EOF' 2>&1 | grep -vE "Warning|warn"
import importlib, inspect, pkgutil, radeval, radeval.metrics as M
mods = [m.name for m in pkgutil.walk_packages(M.__path__, "radeval.metrics.") if "chexbert" in m.name.lower()]
print("chexbert modules:", mods)
for name in mods:
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        print(name, "import failed:", type(e).__name__); continue
    for cname, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ == name:
            meths = [f"{m}{inspect.signature(f)}" for m, f in inspect.getmembers(cls, inspect.isfunction)
                     if not m.startswith("__") and any(t in m.lower() for t in ["label", "forward", "call", "predict", "encode"])]
            print(f"  {name}.{cname}:", meths[:8])
EOF
deactivate

echo; echo "################ D. launch E3 (global RAG) in background ################"
if pgrep -f valid_e3.sh > /dev/null; then echo "already running"
else nohup bash $C/valid_e3.sh > ~/anchor/logs/valid_e3.log 2>&1 & disown; echo "launched: tail ~/anchor/logs/valid_e3.log"; fi
sleep 150; grep -E "queries|retrieval done|RETRIEVAL_SANITY|ret1|Error|error" ~/anchor/logs/valid_e3.log
echo "ANCHOR r22 end $(date)"
