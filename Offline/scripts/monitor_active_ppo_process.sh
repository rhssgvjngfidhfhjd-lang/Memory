#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT PORT TRAIN_PID" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
output="$1"
port="$2"
train_pid="$3"
control="$output/run_control"
interval_seconds="${PPO_MONITOR_INTERVAL_SECONDS:-1800}"
mkdir -p "$control"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

http_ok() {
  curl --silent --fail --noproxy '*' --max-time 20 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null
}

tcp_ok() {
  timeout 5 bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null
}

repair_endpoint() {
  local repair_script="../port2conti.sh"
  if [[ ! -f "$repair_script" ]]; then
    return 1
  fi
  if command -v flock >/dev/null 2>&1; then
    flock /tmp/aaa-memory-port2conti.lock bash "$repair_script" \
      >> "$control/port_repair.log" 2>&1 || true
  else
    bash "$repair_script" >> "$control/port_repair.log" 2>&1 || true
  fi
  http_ok || tcp_ok
}

while kill -0 "$train_pid" 2>/dev/null; do
  checkpoint_count=$(find "$output/checkpoints" -maxdepth 1 -name 'epoch_*.pt' 2>/dev/null | wc -l)
  cache_lines=$(wc -l < "$output/rollout_cache.jsonl" 2>/dev/null || echo 0)
  metrics_lines=$(wc -l < "$output/ppo_metrics.jsonl" 2>/dev/null || echo 0)
  if http_ok; then
    endpoint="ok"
  elif tcp_ok; then
    endpoint="busy"
  elif repair_endpoint; then
    endpoint="recovered"
  else
    endpoint="repair_failed"
  fi
  printf '%s endpoint=%s checkpoints=%s cache_lines=%s metrics_lines=%s train_pid=%s\n' \
    "$(timestamp)" "$endpoint" "$checkpoint_count" "$cache_lines" \
    "$metrics_lines" "$train_pid" >> "$control/monitor.log"
  sleep "$interval_seconds"
done
printf '%s endpoint=monitor_stopped train_pid=%s\n' \
  "$(timestamp)" "$train_pid" >> "$control/monitor.log"
