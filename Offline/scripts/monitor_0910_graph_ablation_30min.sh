#!/usr/bin/env bash
set -uo pipefail

WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
CONTROL_ROOT="$OFFLINE_ROOT/outputs/Ablation/_0910_graph_ablation_control"
LOG="$CONTROL_ROOT/monitor_30min.log"
mkdir -p "$CONTROL_ROOT"

count_lines() {
  local path=$1
  if [[ -f "$path" ]]; then wc -l < "$path"; else echo 0; fi
}

snapshot() {
  {
    echo "===== $(date --iso-8601=seconds) ====="
    echo "queues:"
    for bench in memgallery h2hmem wma; do
      session="${bench}_0910_graph_ablation"
      if tmux has-session -t "$session" 2>/dev/null || \
         tmux has-session -t "${session}_b" 2>/dev/null; then
        state=alive
      else
        state=absent
      fi
      status_file="$CONTROL_ROOT/${bench}_queue.status"
      if [[ -f "$status_file" ]]; then
        status=$(tr '\n' ' ' < "$status_file")
      else
        status=pending
      fi
      echo "  $bench session=$state status=$status"
    done
    echo "endpoints:"
    for port in 8013 8014 8015; do
      model=$(curl -sf --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null | \
        /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
        'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
      echo "  $port ${model:-unavailable}"
    done
    echo "gpus:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed -n '4p;5p;6p'
    echo "runs:"
    for pair in MemGallery:memgallery H2HMEM:h2hmem WorldMemArena:wma; do
      output_benchmark=${pair%%:*}
      suffixes=("")
      if [[ "$output_benchmark" == WorldMemArena ]]; then suffixes+=("_b"); fi
      for suffix in "${suffixes[@]}"; do
       for base_run_id in \
         0910_graph_ablation_top5 \
         0910_graph_ablation_top7 \
         0910_graph_ablation_random2_seed42 \
         0910_graph_ablation_random2_seed43 \
         0910_graph_ablation_random2_seed44 \
         0910_graph_ablation_graph2; do
        run_id="${base_run_id}${suffix}"
        root="$OFFLINE_ROOT/outputs/$output_benchmark/HiveMem/$run_id"
        status_file="$root/run_control/status.txt"
        if [[ -f "$status_file" ]]; then
          status=$(tr '\n' ' ' < "$status_file")
        else
          status=pending
        fi
        qa=$(count_lines "$root/eval/test_ppo/rollouts.jsonl")
        judge=$(count_lines "$root/eval/test_ppo/llm_judge_progress.jsonl")
        cache=$(count_lines "$root/rollout_cache.jsonl")
        echo "  $output_benchmark/$run_id status=$status qa=$qa judge=$judge cache=$cache"
       done
      done
    done
    echo "recent errors:"
    find "$OFFLINE_ROOT/outputs/MemGallery/HiveMem" \
         "$OFFLINE_ROOT/outputs/H2HMEM/HiveMem" \
         "$OFFLINE_ROOT/outputs/WorldMemArena/HiveMem" \
         -maxdepth 3 -type f \( -name '*.log' -o -name '*.txt' \) 2>/dev/null \
      | while read -r path; do
          grep -Ein 'traceback|exception|(^|[^a-z])error([^a-z]|$)|failed|out of memory|oom' "$path" 2>/dev/null \
            | tail -n 2 | sed "s#^#$path:#"
        done | tail -n 30
    echo "disk: $(df -h /data | tail -n 1)"
  } >> "$LOG" 2>&1
}

while true; do
  snapshot
  terminal=0
  for bench in memgallery h2hmem wma; do
    status=$(awk '{print $1}' "$CONTROL_ROOT/${bench}_queue.status" 2>/dev/null || echo pending)
    if [[ "$status" == complete || "$status" == failed ]]; then
      terminal=$((terminal + 1))
    fi
  done
  if [[ "$terminal" -eq 3 ]]; then
    echo "===== monitor finished $(date --iso-8601=seconds) =====" >> "$LOG"
    exit 0
  fi
  sleep 1800
done
