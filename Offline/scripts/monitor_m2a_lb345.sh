#!/usr/bin/env bash
set -u

pid=${1:?usage: monitor_m2a_lb345.sh PID LOG TRACE_DIR}
log=${2:?usage: monitor_m2a_lb345.sh PID LOG TRACE_DIR}
trace_dir=${3:?usage: monitor_m2a_lb345.sh PID LOG TRACE_DIR}

while kill -0 "$pid" 2>/dev/null; do
  timestamp=$(date -Is)
  gpu=$(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader,nounits -i 3,4,5 | paste -sd ';' -)
  lb=$(curl -fsS --max-time 5 http://127.0.0.1:8025/_lb/status 2>/dev/null || printf '{"error":"unavailable"}')
  traces=$(find "$trace_dir" -type f -name '*.jsonl' -print0 2>/dev/null \
    | xargs -0 -r wc -l \
    | awk 'END { print $1 + 0 }')
  printf '%s gpu=%s lb=%s traces=%s\n' \
    "$timestamp" "$gpu" "$lb" "$traces" >> "$log"
  sleep 60
done
