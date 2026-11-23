#!/usr/bin/env bash
# ANCHOR round 07
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r07.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data
echo "ANCHOR r07 start $(date)"
for f in mimic_stream_fetch.py rex_audit.py padchest_fetch.py cxas_probe.py; do
  [ -f "$C/$f" ] && echo "found $f  md5=$(md5sum < "$C/$f" | cut -c1-32)" || echo "MISSING $C/$f"
done
source ~/venvs/anchor/bin/activate

echo; echo "################ A. MIMIC test stream (kill old job, relaunch new version) ################"
echo "running before cleanup:"; ps -eo pid,etime,args | grep -E "mimic_stream_fetch|anchor/data/lists/" | grep -v grep
pkill -f mimic_stream_fetch.py; pkill -f "anchor/data/lists/"; sleep 3
echo "still running after cleanup (want none):"; ps -eo pid,args | grep -E "mimic_stream_fetch|anchor/data/lists/" | grep -v grep
WANT=25acc45b14db3ca25d79ad5d34bbb4d0
HAVE=$(md5sum < "$C/mimic_stream_fetch.py" | cut -c1-32)
if [ "$HAVE" != "$WANT" ]; then
  echo "MIMIC NOT LAUNCHED: mimic_stream_fetch.py md5 $HAVE != $WANT (upload the new file)"
else
  cd "$D"
  ARGS="--root $D --splits test --one-per-study --no-optional-meta --skip-reports"
  N=$(python "$C/mimic_stream_fetch.py" $ARGS --dry-run 2>&1 | grep -oP "selected frontal=\K[0-9]+")
  echo "selected test frontal studies=$N"
  if [ -n "$N" ] && [ "$N" -lt 4000 ]; then
    nohup python "$C/mimic_stream_fetch.py" $ARGS --keep-raw --wget-jobs 8 --min-free-gb 50 \
      > "$D/fetch_mimic_test.log" 2>&1 & echo $! > "$D/fetch_mimic_test.pid"; disown
    echo "MIMIC LAUNCHED pid $(cat "$D/fetch_mimic_test.pid")"
  else
    echo "MIMIC NOT LAUNCHED: unexpected count"
  fi
fi
echo "reports zip size now: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null) bytes"

echo; echo "################ B. ReXGradient: completeness + extraction ################"
cd "$D/rexgradient"
grep -E "REXG_DONE|REXVQA_DONE" "$D/dl_rex.log" || echo "download markers not found yet"
ls -l deid_png.part* | awk '{s+=$5; n++} END {printf "parts=%d total_bytes=%d\n", n, s}'
echo "README hints:"; grep -n -i -E "cat |tar |unzip|extract|sha|md5|checksum|part" README.md | head -15
TYPE=$(head -c 512 deid_png.part00 | file -b -); echo "part00 type: $TYPE"
case "$TYPE" in
  *gzip*) X="tar -xzf -" ;;
  *Zstandard*) X="tar --zstd -xf -" ;;
  *tar*|*POSIX*) X="tar -xf -" ;;
  *) X="" ;;
esac
if grep -q REXG_DONE "$D/dl_rex.log" && [ -n "$X" ]; then
  if [ -f "$D/rexgradient/extract.done" ]; then echo "already extracted"
  else
    mkdir -p "$D/rexgradient/images"
    nohup bash -c "cd '$D/rexgradient' && cat \$(ls deid_png.part* | sort) | $X -C '$D/rexgradient/images' && touch extract.done && echo EXTRACT_DONE \$(date)" \
      > "$D/rexgradient/extract.log" 2>&1 & disown
    echo "EXTRACTION LAUNCHED with: $X  (log: rexgradient/extract.log)"
  fi
else
  echo "EXTRACTION NOT LAUNCHED (download not finished or unknown archive type)"
fi

echo; echo "################ C. PadChest-GR: list share + launch resumable download ################"
mkdir -p "$D/padchest_gr"
python "$C/padchest_fetch.py" --out "$D/padchest_gr" --list-only
nohup python "$C/padchest_fetch.py" --out "$D/padchest_gr" --jobs 4 > "$D/padchest_gr/padchest.log" 2>&1 & disown
echo "PADCHEST LAUNCHED (log: padchest_gr/padchest.log)"

echo; echo "################ D. CXAS 159-class anatomy probe (CPU) ################"
python "$C/cxas_probe.py"

echo; echo "################ E. ReXGradient + ReXVQA aggregate audit (no text, no IDs) ################"
python "$C/rex_audit.py"

echo; echo "################ status ################"
df -h ~ | tail -1
echo "ANCHOR r07 end $(date)"
