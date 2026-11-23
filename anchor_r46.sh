#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r46.log) 2>&1
echo "ANCHOR r46 start $(date)"; md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json
source ~/venvs/anchor/bin/activate; export CUDA_VISIBLE_DEVICES=""
time python ~/anchor/code/e21b_selective.py 2>&1 | grep -vE "Warning|warn"
deactivate; md5sum ~/anchor/work/e8/freeze.json ~/anchor/work/e16/freeze2.json
echo "R46_DONE"
