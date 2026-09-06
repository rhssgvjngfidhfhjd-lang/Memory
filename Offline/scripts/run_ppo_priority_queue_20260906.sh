#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

mem_config="configs/evidence_policy_memgallery_independent_bernoulli_fullval312_18001_20260905.json"
h2h_config="configs/evidence_policy_h2hmem.json"
wma_config="configs/evidence_policy_wma.json"

mem_output="outputs/evidence_policy_memgallery_graph5plus2_fullval_18000_20260906"
h2h_output="outputs/evidence_policy_h2hmem_graph5plus2_fullval_18001_20260906"
wma_output="outputs/evidence_policy_wma_graph5plus2_fullval_auto_20260906"
queue_control="outputs/ppo_priority_queue_20260906"

mkdir -p "$queue_control" \
  "$mem_output/run_control" \
  "$h2h_output/run_control" \
  "$wma_output/run_control"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

launch_runner() {
  local config="$1"
  local output="$2"
  local port="$3"
  local run_name="$4"
  local benchmark="$5"
  PPO_MONITOR_INTERVAL_SECONDS=1800 \
    bash scripts/run_ppo_full_endpoint_monitored.sh \
      "$config" "$output" "$port" "$run_name" "$benchmark" \
      >> "$output/run_control/runner.log" 2>&1 &
  LAUNCHED_PID=$!
}

printf '%s status=queue_started\n' "$(timestamp)" >> "$queue_control/status.log"

launch_runner \
  "$mem_config" "$mem_output" 18000 \
  "memgallery-ppo-graph5plus2-fullval-18000-20260906" memgallery
mem_pid=$LAUNCHED_PID
printf '%s status=started benchmark=memgallery port=18000 pid=%s\n' \
  "$(timestamp)" "$mem_pid" >> "$queue_control/status.log"

launch_runner \
  "$h2h_config" "$h2h_output" 18001 \
  "h2hmem-ppo-graph5plus2-fullval-18001-20260906" h2hmem
h2h_pid=$LAUNCHED_PID
printf '%s status=started benchmark=h2hmem port=18001 pid=%s\n' \
  "$(timestamp)" "$h2h_pid" >> "$queue_control/status.log"

completed_pid=""
wait -n -p completed_pid "$mem_pid" "$h2h_pid"
first_status=$?
if [[ "$completed_pid" == "$mem_pid" ]]; then
  first_benchmark="memgallery"
  wma_port=18000
  remaining_pid=$h2h_pid
  remaining_benchmark="h2hmem"
else
  first_benchmark="h2hmem"
  wma_port=18001
  remaining_pid=$mem_pid
  remaining_benchmark="memgallery"
fi
printf '%s status=priority_finished benchmark=%s exit_code=%s released_port=%s\n' \
  "$(timestamp)" "$first_benchmark" "$first_status" "$wma_port" \
  >> "$queue_control/status.log"

launch_runner \
  "$wma_config" "$wma_output" "$wma_port" \
  "wma-ppo-graph5plus2-fullval-auto-20260906" wma
wma_pid=$LAUNCHED_PID
printf '%s status=started benchmark=wma port=%s pid=%s\n' \
  "$(timestamp)" "$wma_port" "$wma_pid" >> "$queue_control/status.log"

wait "$remaining_pid"
remaining_status=$?
printf '%s status=priority_finished benchmark=%s exit_code=%s\n' \
  "$(timestamp)" "$remaining_benchmark" "$remaining_status" \
  >> "$queue_control/status.log"

wait "$wma_pid"
wma_status=$?
printf '%s status=finished benchmark=wma exit_code=%s\n' \
  "$(timestamp)" "$wma_status" >> "$queue_control/status.log"

if [[ $first_status -eq 0 && $remaining_status -eq 0 && $wma_status -eq 0 ]]; then
  printf '%s status=complete\n' "$(timestamp)" >> "$queue_control/status.log"
  exit 0
fi
printf '%s status=failed first=%s remaining=%s wma=%s\n' \
  "$(timestamp)" "$first_status" "$remaining_status" "$wma_status" \
  >> "$queue_control/status.log"
exit 1
