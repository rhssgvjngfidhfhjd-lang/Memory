#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/haozhen/Memory-clean/Offline/outputs/PPO
RUN_DATE=${1:-$(date +%m%d)}
SUFFIX_LIST=${2:-a}
MAX_ITERATION=${3:-6}
INTERVAL_SECONDS=${4:-300}
LOG="$ROOT/_${RUN_DATE}_cost_reward_sweep_monitor.log"
IFS=',' read -r -a SUFFIXES <<< "$SUFFIX_LIST"
RUNS=()
for suffix in "${SUFFIXES[@]}"; do
  RUNS+=(
    "MemGallery_${RUN_DATE}PPO_${suffix}"
    "H2HMEM_${RUN_DATE}PPO_${suffix}"
    "WMA_${RUN_DATE}PPO_${suffix}"
  )
done

count_lines() {
  local path=$1
  [[ -f "$path" ]] && wc -l < "$path" || printf '0\n'
}

for ((iteration = 0; iteration <= MAX_ITERATION; iteration++)); do
  {
    printf '\n[%s] iteration=%s/%s\n' \
      "$(date --iso-8601=seconds)" "$iteration" "$MAX_ITERATION"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | sed -n '4,6p'
    for run in "${RUNS[@]}"; do
      root="$ROOT/$run"
      status=$(head -n 1 "$root/run_control/status.txt" 2>/dev/null || printf 'starting')
      updates=$(count_lines "$root/ppo_metrics.jsonl")
      train_rows=0
      for trace in "$root"/train/epoch_*_rollouts.jsonl; do
        [[ -f "$trace" ]] && train_rows=$((train_rows + $(wc -l < "$trace")))
      done
      test_rows=$(count_lines "$root/eval/test_ppo/rollouts.jsonl")
      judge_rows=$(count_lines "$root/eval/test_ppo/llm_judge_progress.jsonl")
      printf '%s | %s | updates=%s train_saved=%s test=%s judge=%s\n' \
        "$run" "$status" "$updates" "$train_rows" "$test_rows" "$judge_rows"
      tail -n 2 "$root/train.log" 2>/dev/null || true
    done
  } >> "$LOG" 2>&1
  [[ "$iteration" -lt "$MAX_ITERATION" ]] && sleep "$INTERVAL_SECONDS"
done
