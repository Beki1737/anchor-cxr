#!/usr/bin/env bash
# Background (launched by anchor_r35.sh): E10 PadChest-GR pointing game. Waits for a GPU with 10 GB free.
# Log: ~/anchor/logs/padchest_run.log
set -u
C=~/anchor/code; source $C/gpu_wait.sh
echo "padchest_run start $(date)"
source ~/venvs/anchor/bin/activate
[ -f ~/anchor/work/manifest/rex_padchest.parquet ] || python $C/padchest_manifest.py
if [ "$(ls ~/anchor/work/regions/padchest/shard_*.npz 2>/dev/null | wc -l)" -lt 5 ]; then
  wait_free 10; python $C/region_cache.py --split padchest --gpu $CUDA_VISIBLE_DEVICES --batch 16 --workers 16 2>&1 | grep -E "images|FAILED|Error" | tail -3
fi
wait_free 4
python $C/e10_pointing.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo "PADCHEST_RUN_DONE $(date)"
