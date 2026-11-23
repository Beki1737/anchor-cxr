#!/usr/bin/env bash

set -u
mkdir -p ~/anchor/logs
LOG=~/anchor/logs/r18.log
exec > >(tee "$LOG") 2>&1
C=~/anchor/code; G=~/anchor/work/gen; J="python $C/judge_vllm.py"
echo "ANCHOR r18 start $(date)"
lowgpu () { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1; }
EFILT='Computing|Loading weights|LOAD REPORT|UNEXPECTED|^Key |^---|^Notes|can be ignored|NLTK|incr_download|^radeval|^available|^numpy|RadEval\.'
grep "reports/s" ~/anchor/logs/corpus_full.log | tail -1

echo; echo "################ A. GREEN port: vLLM generation (anchor venv) ################"
source ~/venvs/anchor/bin/activate
export CUDA_VISIBLE_DEVICES=$(lowgpu); echo "GPU $CUDA_VISIBLE_DEVICES"
$J gen --judge green > ~/anchor/logs/green_gen.log 2>&1
grep -E "^\[green\]|JUDGE_GEN_DONE|Error|error:" ~/anchor/logs/green_gen.log | tail -8
deactivate

echo; echo "################ A2. GREEN port: score + compare with radeval (reference subset already saved in r17) ################"
source ~/venvs/anchor_eval/bin/activate
$J score --judge green 2>&1 | grep -vE "$EFILT" | tail -9
$J compare --judge green --tol-agg 0.02 2>&1 | tail -14
deactivate

echo; echo "################ B. background: MedGemma B0 on valid primary cohort ################"
if [ -f "$G/b0_medgemma4b_valid.parquet" ]; then echo "already generated"
elif pgrep -f "gen_b0.py.*--keys valid" > /dev/null; then echo "already running"
else
  nohup bash -c "source ~/venvs/anchor/bin/activate; python $C/gen_b0.py --name medgemma4b --keys valid" \
    > ~/anchor/logs/b0_valid.log 2>&1 & disown
  echo "launched: tail -3 ~/anchor/logs/b0_valid.log"
fi
sleep 240; grep -E "studies/s|using GPU|Error" ~/anchor/logs/b0_valid.log | tail -3
echo "ANCHOR r18 end $(date)"
