#!/bin/sh
set -u

output_root=${1:?Usage: monitor_test_hivemem_graph.sh OUTPUT_ROOT}
interval=${MONITOR_INTERVAL_SECONDS:-1800}
log_path=$output_root/_launcher_logs/monitor_30m.log

while tmux ls 2>/dev/null | grep -q '^hivemem_graph_.*_v5g2:'; do
  date '+%Y-%m-%dT%H:%M:%S%z' >> "$log_path"
  for benchmark in Mem-Gallery H2HMEM WorldMemArena; do
    status_path=$output_root/$benchmark/HiveMem/pipeline_status.json
    if [ -f "$status_path" ]; then
      python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(sys.argv[2], p.get("status"), p.get("stage"), p.get("updated_at"), p.get("error", ""))' "$status_path" "$benchmark" >> "$log_path" 2>&1
    else
      echo "$benchmark status_missing" >> "$log_path"
    fi
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '4,6p' >> "$log_path" 2>&1
  printf '\n' >> "$log_path"
  sleep "$interval"
done

date '+%Y-%m-%dT%H:%M:%S%z all_sessions_ended' >> "$log_path"
