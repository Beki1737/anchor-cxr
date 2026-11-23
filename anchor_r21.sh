#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r21.log
exec > >(tee "$LOG") 2>&1
echo "ANCHOR r21 start $(date)"

echo; echo "################ A. status ################"
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1
grep -E "FAILED_|ALL_CORPUS_DONE|CORPUS_EXTRACT_DONE|: [0-9]+ reports$" ~/anchor/logs/corpus_full.log | tail -6
for t in train valid test; do echo "corpus $t shards: $(ls ~/anchor/work/corpus/$t/shard_*.parquet 2>/dev/null | wc -l)"; done
echo "MIMIC raw jpg: $(find ~/anchor/data/raw -name '*.jpg' 2>/dev/null | wc -l) / 3041"
pgrep -af "mimic_stream_fetch|corpus_extract|valid_eval|valid_green" | cut -c1-120

echo; echo "################ B1. storage inventory (read-only) ################"
df -h ~ | tail -1
echo "-- ~/anchor top level:"; du -sh ~/anchor/* ~/anchor/.cache 2>/dev/null | sort -h | tail -15
echo "-- ~/anchor/data:"; du -sh ~/anchor/data/* 2>/dev/null | sort -h | tail -12
echo "-- ~/anchor/models:"; du -sh ~/anchor/models/*/* 2>/dev/null | sort -h
echo "-- HF hub cache (judges, encoders):"; du -sh ~/anchor/.cache/huggingface/hub/* ~/.cache/huggingface/hub/* 2>/dev/null | sort -h | tail -15
echo "-- ~/anchor/work:"; du -sh ~/anchor/work/* 2>/dev/null | sort -h | tail -10
echo "-- venvs + user caches:"; du -sh ~/venvs/* ~/.cache/* 2>/dev/null | sort -h | tail -12
echo "-- archives still on disk (zip/tar/7z/gz > 1 GB):"
find ~/anchor ~/ -maxdepth 4 \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.*" -o -name "*.tgz" -o -name "*.7z" -o -name "*.gz" \) \
  -size +1G -printf "%s\t%p\n" 2>/dev/null | sort -u | awk -F'\t' '{printf "%.1f GB\t%s\n", $1/1e9, $2}'
echo "-- PadChest-GR folder listing (names only):"; ls ~/anchor/data/padchest* 2>/dev/null | head -20

echo; echo "################ B2. safe cleanup: package caches only (rebuilt on demand, no data, no models) ################"
BEFORE=$(df --output=avail -B1 ~ | tail -1)
source ~/venvs/anchor/bin/activate
echo "uv cache: $(du -sh "$(uv cache dir)" 2>/dev/null | cut -f1) at $(uv cache dir)"
uv cache clean 2>&1 | tail -1
deactivate
[ -d ~/.cache/pip ] && { echo "pip cache: $(du -sh ~/.cache/pip | cut -f1)"; rm -rf ~/.cache/pip; }
AFTER=$(df --output=avail -B1 ~ | tail -1)
echo "freed by safe cleanup: $(( (AFTER - BEFORE) / 1000000000 )) GB"
df -h ~ | tail -1

echo; echo "################ C. D40 strata: B0 vs template on normal vs abnormal valid studies ################"
source ~/venvs/anchor_eval/bin/activate
python ~/anchor/code/strata.py
deactivate
echo "ANCHOR r21 end $(date)"
