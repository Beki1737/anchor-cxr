#!/usr/bin/env bash
# ANCHOR round 34 = E10 PadChest-GR zero-shot grounding (pointing game + side) with the ReXGradient-trained region probe.
# manifest -> region cache (CXAS + RAD-DINO on 4,555 images) -> pointing game. Runs alongside the MIMIC chain.
# MobaXterm:   bash ~/anchor/code/anchor_r34.sh        Log: ~/anchor/logs/r34.log (send back). ~5 to 10 min.
# Needs in ~/anchor/code/: padchest_manifest.py, e10_pointing.py (new)
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r34.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code
echo "ANCHOR r34 start $(date)"
echo "-- MIMIC chain so far:"; grep -vE "^\s*$" ~/anchor/logs/mimic_run.log | tail -4
source ~/venvs/anchor/bin/activate
python $C/padchest_manifest.py
if [ "$(ls ~/anchor/work/regions/padchest/shard_*.npz 2>/dev/null | wc -l)" -lt 5 ]; then
  python $C/region_cache.py --split padchest --workers 16 2>&1 | grep -E "images|shard_000|FAILED|Error|invert|hard" | tail -6
fi
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
python $C/e10_pointing.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo "ANCHOR r34 end $(date)"
