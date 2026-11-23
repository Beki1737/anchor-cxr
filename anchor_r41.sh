#!/usr/bin/env bash
# ANCHOR round 41 = build the reader-study packet (D74) while GPUs 0/1 are busy. CPU only, ~1 min.
# MobaXterm:   bash ~/anchor/code/anchor_r41.sh        Log: ~/anchor/logs/r41.log (send back; counts only)
# Needs in ~/anchor/code/: reader_packet.py, READER_GUIDE.md (new)
set -u
mkdir -p ~/anchor/logs
exec > >(tee ~/anchor/logs/r41.log) 2>&1
C=~/anchor/code
echo "ANCHOR r41 start $(date)"
source ~/venvs/anchor/bin/activate
python $C/reader_packet.py
deactivate
cp $C/READER_GUIDE.md ~/anchor/work/reader/
chmod -R go-rwx ~/anchor/work/reader
ls ~/anchor/work/reader | head; echo "images: $(ls ~/anchor/work/reader/images | wc -l)"
echo "-- GPU queue:"; nvidia-smi -i 0,1 --query-gpu=index,memory.free --format=csv,noheader
for l in e9_mimic e16_judge sc_run; do echo "-- $l:"; tail -1 ~/anchor/logs/$l.log; done
echo "ANCHOR r41 end $(date)"
