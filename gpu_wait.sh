export ANCHOR_GPUS=${ANCHOR_GPUS:-0,1}
GPU_LOCK=~/anchor/.gpu_start.lock
wait_free () {
  local need=$1 idx free n=0
  while true; do
    read idx free < <(nvidia-smi -i "$ANCHOR_GPUS" --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ' ' | tr ',' ' ')
    if [ -n "$free" ] && [ "$free" -ge $(( need * 1024 )) ]; then
      export CUDA_VISIBLE_DEVICES=$idx
      echo "[gpu $(date +%H:%M)] GPU $idx has ${free} MiB free (need ${need} GB)"
      return 0
    fi
    [ $(( n % 15 )) -eq 0 ] && echo "[gpu $(date +%H:%M)] queued: need ${need} GB, best allowed GPU (${ANCHOR_GPUS}) has ${free:-?} MiB free"
    n=$(( n + 1 )); sleep 120
  done
}
share () { awk -v n="$1" 'BEGIN { printf "%.2f", n / 96 }'; }
gpu_run () {
  local need=$1 log=$2 tok=$3 t pid i; shift 3
  for t in 1 2 3 4 5 6 7 8; do
    exec 9>"$GPU_LOCK"; flock 9
    wait_free "$need"
    "$@" > "$log" 2>&1 & pid=$!
    for i in $(seq 1 30); do                      # hold the queue until the model is loaded (<= 5 min) or the job ends
      sleep 10; kill -0 $pid 2>/dev/null || break
      grep -qE "Graph capturing finished|init engine|Processed prompts|studies/s| requests " "$log" 2>/dev/null && [ $i -ge 6 ] && break
    done
    flock -u 9; exec 9>&-
    wait $pid
    unset CUDA_VISIBLE_DEVICES
    grep -q "$tok" "$log" && return 0
    echo "[gpu $(date +%H:%M)] attempt $t failed. root cause lines:"
    grep -E "OutOfMemory|out of memory|less than desired|No available memory|insufficient|ValueError|CUDA error|NCCL|Free memory" "$log" \
      | grep -v "Engine core initialization failed" | sort -u | tail -3 | cut -c1-220
    sleep 180
  done
  echo "FAILED: $*"; return 1
}
