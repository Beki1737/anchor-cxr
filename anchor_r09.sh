#!/usr/bin/env bash
set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r09.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data
echo "ANCHOR r09 start $(date)"
for f in build_rex_manifest.py region_cache.py; do
  [ -f "$C/$f" ] && echo "found $f md5=$(md5sum < "$C/$f" | cut -c1-32)" || echo "MISSING $f"
done
source ~/venvs/anchor/bin/activate

echo; echo "################ A. background jobs ################"
echo "-- MIMIC test stream:"; tail -2 "$D/fetch_mimic_test.log"
echo "   raw jpg: $(find "$D/raw" -name '*.jpg' 2>/dev/null | wc -l) / 3041 target   resized: $(find "$D/jpg1024" -name '*.jpg' 2>/dev/null | wc -l)"
echo "-- PadChest-GR:"; tail -2 "$D/padchest_gr/padchest.log"; du -sh "$D/padchest_gr" | cut -f1
echo "-- MIMIC reports zip:"
Z="$D/reports/mimic-cxr-reports.zip"
if python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); bad=z.testzip(); print('zip OK, files:', len(z.namelist())) if bad is None else sys.exit(1)" "$Z" 2>/dev/null; then
  echo "   reports zip complete"
else
  echo "   reports zip incomplete ($(stat -c %s "$Z" 2>/dev/null) bytes)"
  if pgrep -f "mimic-cxr-reports.zip" > /dev/null; then echo "   a download is still running, leaving it alone"
  else
    nohup wget -c --tries=0 --read-timeout=60 --waitretry=15 --retry-connrefused --progress=dot:mega \
      -P "$D/reports" https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip > "$D/reports/fetch_reports.log" 2>&1 & disown
    echo "   RESTARTED reports download (log: reports/fetch_reports.log)"
  fi
fi

echo; echo "################ B. ReXGradient image check and optional tar cleanup ################"
N=$(find "$D/rexgradient/images" -name '*.png' | wc -l); echo "png count: $N (metadata total 273004)"
if [ "$N" = "273004" ] && [ -f "$D/rexgradient/extract.done" ]; then
  if [ "${DELETE_PARTS:-0}" = "1" ]; then
    du -ch "$D"/rexgradient/deid_png.part* | tail -1; rm -f "$D"/rexgradient/deid_png.part*; echo "TAR PARTS DELETED"
  else echo "tar parts kept (run with DELETE_PARTS=1 to remove them)"; fi
else echo "count mismatch or extraction unfinished: tar parts kept"; fi

echo; echo "################ C. manifests (fixed ReXVQA loader) ################"
python "$C/build_rex_manifest.py" 2>&1 | grep -vE "^\[..:..:..\] (train|valid|test):"
echo "summary.json written: $(ls -la ~/anchor/work/manifest/summary.json 2>/dev/null | awk '{print $5" bytes"}')"

echo; echo "################ D. E0 pilot: region cache on 2,000 test images ################"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
python "$C/region_cache.py" --split test --limit 2000 --overlays 24 --batch 32 --workers 16
echo "overlays: $(ls ~/anchor/work/qa_overlays/*.png 2>/dev/null | wc -l) files in ~/anchor/work/qa_overlays/ (open in VS Code)"

echo; df -h ~ | tail -1
echo "ANCHOR r09 end $(date)"
