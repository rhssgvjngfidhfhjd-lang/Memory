#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
offline_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
output_root=${1:-$offline_root/outputs/test_only_manifest_20260905_topk7}
scheduler_session=${2:-baseline_test_only_manifest}
status_path=$output_root/status.json
log_path=$output_root/_logs/monitor.log
interval_seconds=${MONITOR_INTERVAL_SECONDS:-1200}

mkdir -p "$(dirname -- "$log_path")"

while :; do
  {
    date '+CHECK %Y-%m-%dT%H:%M:%S%z'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | sed -n '4,6p'
    for port in 8001 8013 8014 8015; do
      if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
        || curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
        echo "port $port ok"
      else
        echo "port $port unavailable"
      fi
    done
    if [ -f "$status_path" ]; then
      /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c '
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
for section in ("smoke", "jobs", "judges"):
    rows = p.get(section, {})
    counts = {}
    for row in rows.values():
        state = row.get("status", "unknown")
        counts[state] = counts.get(state, 0) + 1
    print(section, counts)
print("phase", p.get("phase"), "updated_at", p.get("updated_at"))
' "$status_path"
    else
      echo "status file unavailable"
    fi
  } >> "$log_path" 2>&1

  phase=$(
    if [ -f "$status_path" ]; then
      /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("phase", ""))' \
        "$status_path" 2>/dev/null || true
    fi
  )
  if [ "$phase" = "complete" ]; then
    exit 0
  fi
  if [ "$phase" = "incomplete" ]; then
    echo "ALERT scheduler reported incomplete $(date '+%Y-%m-%dT%H:%M:%S%z')" \
      >> "$log_path"
  fi
  if ! tmux has-session -t "$scheduler_session" 2>/dev/null; then
    echo "ALERT scheduler tmux missing $(date '+%Y-%m-%dT%H:%M:%S%z')" \
      >> "$log_path"
  fi
  sleep "$interval_seconds"
done
