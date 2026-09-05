#!/bin/sh
set -u

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
offline_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
output_root=${OUTPUT_ROOT:-$offline_root/outputs/test_only_manifest_20260905_topk7}
python_bin=${PYTHON_BIN:-/data/haozhen/miniconda3/envs/pipeline_repro/bin/python}
max_restarts=${MAX_SUPERVISOR_RESTARTS:-3}
restart_count=0
log_path=$output_root/_logs/supervisor.log

mkdir -p "$(dirname -- "$log_path")"

while :; do
  echo "START scheduler $(date '+%Y-%m-%dT%H:%M:%S%z') restart=$restart_count" \
    >> "$log_path"
  "$python_bin" "$script_dir/run_test_baseline_matrix.py" \
    --output-root "$output_root" "$@" >> "$log_path" 2>&1
  return_code=$?
  phase=$(
    if [ -f "$output_root/status.json" ]; then
      "$python_bin" -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("phase", ""))' \
        "$output_root/status.json" 2>/dev/null || true
    fi
  )
  echo "EXIT scheduler $(date '+%Y-%m-%dT%H:%M:%S%z') rc=$return_code phase=$phase" \
    >> "$log_path"
  if [ "$phase" = "complete" ]; then
    exit 0
  fi
  restart_count=$((restart_count + 1))
  if [ "$restart_count" -ge "$max_restarts" ]; then
    echo "ALERT restart limit reached; manual diagnosis required" >> "$log_path"
    exit 1
  fi
  echo "RECOVER retrying scheduler in 60 seconds" >> "$log_path"
  sleep 60
done
