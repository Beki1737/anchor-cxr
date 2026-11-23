#!/usr/bin/env bash

set -u
C=~/anchor/code; source $C/gpu_wait.sh
echo "sc_run start $(date)"
source ~/venvs/anchor/bin/activate
if [ ! -f ~/anchor/work/gen/b0sc4_medgemma4b_valid.parquet ]; then
  gpu_run 24 ~/anchor/logs/sc_gen.log GEN_B0_DONE python $C/gen_b0.py --name medgemma4b --keys valid --n-samples 5 --temperature 0.7 --seed 0 --gpu-util $(share 22)
fi
grep -E "^done" ~/anchor/logs/sc_gen.log
until grep -q GENTEXT_DONE ~/anchor/logs/mimic_gentext.log 2>/dev/null || ! pgrep -u "$USER" -f e9_mimic.sh > /dev/null; do sleep 300; done
for s in 0 1 2 3 4; do
  grep -q "GENTEXT_DONE b0sc${s}_" ~/anchor/logs/sc_gentext_$s.log 2>/dev/null || \
    gpu_run 46 ~/anchor/logs/sc_gentext_$s.log GENTEXT_DONE python $C/corpus_extract_v2.py gentext --gen b0sc${s}_medgemma4b_valid.parquet --gpu-util 0.45
  tail -1 ~/anchor/logs/sc_gentext_$s.log
done
wait_free 6
python $C/e12_selfcons.py 2>&1 | grep -vE "Warning|warn"
deactivate
echo "SC_RUN_DONE $(date)"
