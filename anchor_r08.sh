#!/usr/bin/env bash
# ANCHOR round 08. Run in MobaXterm:   bash ~/anchor/code/anchor_r08.sh
# Log: ~/anchor/logs/r08.log  (send it back). Needs ~/anchor/code/build_rex_manifest.py
set -u
mkdir -p ~/anchor/logs ~/anchor/work
LOG=~/anchor/logs/r08.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; D=~/anchor/data
echo "ANCHOR r08 start $(date)"
[ -f "$C/build_rex_manifest.py" ] && echo "found build_rex_manifest.py md5=$(md5sum < "$C/build_rex_manifest.py" | cut -c1-32)" || echo "MISSING build_rex_manifest.py"
source ~/venvs/anchor/bin/activate

echo; echo "################ A. background jobs ################"
echo "-- ReXGradient extraction:"; tail -2 "$D/rexgradient/extract.log" 2>/dev/null; [ -f "$D/rexgradient/extract.done" ] && echo "extract.done present"
echo "   images dir: $(du -sh "$D/rexgradient/images" 2>/dev/null | cut -f1)   png count: $(find "$D/rexgradient/images" -name '*.png' 2>/dev/null | wc -l)"
echo "   top-level entries in images/: $(ls "$D/rexgradient/images" 2>/dev/null | head -3 | sed -E 's/[A-Za-z0-9]{8,}/<id>/g' | tr '\n' ' ')"
echo "-- MIMIC test stream:"; tail -3 "$D/fetch_mimic_test.log"
echo "   raw jpg: $(find "$D/raw" -name '*.jpg' 2>/dev/null | wc -l)   resized: $(find "$D/jpg1024" -name '*.jpg' 2>/dev/null | wc -l)"
echo "-- PadChest-GR:"; tail -2 "$D/padchest_gr/padchest.log"; du -sh "$D/padchest_gr" | cut -f1
echo "-- MIMIC reports zip bytes: $(stat -c %s "$D/reports/mimic-cxr-reports.zip" 2>/dev/null)"

echo; echo "################ B. install pyarrow (torch pinned) ################"
uv pip install pyarrow --constraint ~/anchor/pins.txt 2>&1 | tail -3
python -c "import torch, pyarrow; print('torch', torch.__version__, '| pyarrow', pyarrow.__version__)"

echo; echo "################ C. build leakage-safe manifests ################"
python "$C/build_rex_manifest.py"

echo; echo "################ D. Google Cloud CLI (user-space install, no sudo, no login yet) ################"
if [ -x ~/opt/google-cloud-sdk/bin/gcloud ]; then echo "gcloud already installed"
else
  mkdir -p ~/opt && cd ~/opt
  curl -sSL -o gcloud.tgz https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
    && tar -xzf gcloud.tgz && rm gcloud.tgz \
    && ./google-cloud-sdk/install.sh --quiet --usage-reporting false --path-update false --command-completion false > /dev/null
fi
~/opt/google-cloud-sdk/bin/gcloud --version 2>/dev/null | head -2 || echo "gcloud install failed"

echo; df -h ~ | tail -1
echo "ANCHOR r08 end $(date)"
