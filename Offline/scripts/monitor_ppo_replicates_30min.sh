#!/usr/bin/env bash
set -uo pipefail

WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
CONTROL_ROOT="$OFFLINE_ROOT/outputs/PPO/_0909PPO_repeats_control"
LOG="$CONTROL_ROOT/monitor_30min.log"
RUNS=(
  H2HMEM_0909PPO_a H2HMEM_0909PPO_b H2HMEM_0909PPO_c
  WMA_0909PPO_a WMA_0909PPO_b WMA_0909PPO_c
)

mkdir -p "$CONTROL_ROOT"

count_lines() {
  local path=$1
  [[ -f "$path" ]] && wc -l < "$path" || echo 0
}

while true; do
  now=$(date --iso-8601=seconds)
  {
    echo "===== $now ====="
    echo "queues:"
    for benchmark in h2hmem wma; do
      status_file="$CONTROL_ROOT/${benchmark}_queue.status"
      printf '  %s session=%s status=%s\n' \
        "$benchmark" \
        "$(tmux has-session -t "${benchmark}_ppo_0909_repeats" 2>/dev/null && echo alive || echo absent)" \
        "$([[ -f "$status_file" ]] && tr '\n' ' ' < "$status_file" || echo missing)"
    done
    echo "endpoints:"
    for port in 8014 8015; do
      model=$(curl -sf --max-time 10 "http://127.0.0.1:$port/v1/models" 2>/dev/null | \
        /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
        'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
      printf '  %s %s\n' "$port" "${model:-UNAVAILABLE}"
    done
    echo "gpus:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | sed -n '5,6p'
    echo "runs:"
    terminal=0
    for run in "${RUNS[@]}"; do
      root="$OFFLINE_ROOT/outputs/PPO/$run"
      status="$([[ -f "$root/run_control/status.txt" ]] && tr '\n' ' ' < "$root/run_control/status.txt" || echo pending)"
      checkpoints=$(find "$root/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' 2>/dev/null | wc -l)
      updates=$(count_lines "$root/ppo_metrics.jsonl")
      cache=$(count_lines "$root/rollout_cache.jsonl")
      test_rows=$(count_lines "$root/eval/test_ppo/rollouts.jsonl")
      judge_rows=$(count_lines "$root/eval/test_ppo/llm_judge_progress.jsonl")
      printf '  %s status=%s checkpoints=%s updates=%s cache=%s test=%s judge=%s\n' \
        "$run" "$status" "$checkpoints" "$updates" "$cache" "$test_rows" "$judge_rows"
      if [[ "$status" == complete* || "$status" == *failed* ]]; then
        terminal=$((terminal + 1))
      fi
    done
    df -h "$OFFLINE_ROOT" | tail -n 1 | sed 's/^/disk: /'
    if [[ $terminal -eq 6 ]]; then
      echo "all runs reached terminal state; monitor exiting"
    fi
  } >> "$LOG" 2>&1

  if [[ $terminal -eq 6 ]]; then
    exit 0
  fi
  sleep 1800
done
