#!/usr/bin/env bash
set -euo pipefail

RUN_TAG=${1:-A1_D_k4_lambda0_rep1_20260923}
shift || true
if (( $# == 0 )); then
  BENCHMARKS=(h2hmem wma)
else
  BENCHMARKS=("$@")
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "run_tag contains unsupported characters: $RUN_TAG" >&2
  exit 2
fi

WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
RUNNER="$OFFLINE_ROOT/scripts/run_a1_d_k4_lambda0_test_queue.sh"
CONTROL_ROOT="$OFFLINE_ROOT/outputs/Ablation/$RUN_TAG/_dispatcher"
LEASE_ROOT="$OFFLINE_ROOT/outputs/_gpu_leases"
CLAIM_ROOT="$OFFLINE_ROOT/outputs/_task_claims"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
mkdir -p "$CONTROL_ROOT" "$LEASE_ROOT" "$CLAIM_ROOT"

declare -A ENDPOINTS=(
  [3]=http://127.0.0.1:8013/v1
  [4]=http://127.0.0.1:8014/v1
  [5]=http://127.0.0.1:8015/v1
)

timestamp() { date --iso-8601=seconds; }

endpoint_model() {
  local endpoint=$1
  curl -sf --max-time 8 "$endpoint/models" | "$PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

endpoint_load() {
  local endpoint=$1
  local server_root=${endpoint%/v1}
  curl -sf --max-time 8 "$server_root/metrics" | "$PYTHON" -c '
import re,sys
text=sys.stdin.read()
def value(name):
    rows=re.findall(rf"^vllm:{name}[^\n]*\s([0-9.]+)$", text, re.M)
    return sum(float(row) for row in rows)
running=value("num_requests_running")
waiting=value("num_requests_waiting")
print(f"{running:.0f} {waiting:.0f}")
' 2>/dev/null
}

gpu_utilization() {
  nvidia-smi --id="$1" --query-gpu=utilization.gpu --format=csv,noheader,nounits \
    | tr -d ' '
}

stable_idle() {
  local gpu=$1 endpoint=${ENDPOINTS[$1]} check model load running waiting util
  for check in 1 2 3; do
    model=$(endpoint_model "$endpoint" || true)
    [[ "$model" == Qwen/Qwen3-VL-4B-Instruct ]] || return 1
    load=$(endpoint_load "$endpoint" || true)
    read -r running waiting <<< "$load"
    [[ "${running:-x}" == 0 && "${waiting:-x}" == 0 ]] || return 1
    util=$(gpu_utilization "$gpu" || true)
    [[ "$util" =~ ^[0-9]+$ && "$util" -le 5 ]] || return 1
    if (( check < 3 )); then sleep 20; fi
  done
}

claim_task() {
  local benchmark=$1
  local claim_dir="$CLAIM_ROOT/${RUN_TAG}_${benchmark}_a1d.claim"
  if mkdir "$claim_dir" 2>/dev/null; then
    printf '%s\n' "pid=$$" "run_tag=$RUN_TAG" "benchmark=$benchmark" \
      "status=queued" "created_at=$(timestamp)" > "$claim_dir/owner.txt"
    return 0
  fi
  echo "$(timestamp) duplicate_task_claim benchmark=$benchmark claim=$claim_dir" \
    | tee -a "$CONTROL_ROOT/dispatcher.log"
  return 1
}

set_claim_status() {
  local benchmark=$1 status=$2 gpu=${3:-}
  local claim_dir="$CLAIM_ROOT/${RUN_TAG}_${benchmark}_a1d.claim"
  printf '%s\n' "pid=$$" "run_tag=$RUN_TAG" "benchmark=$benchmark" \
    "status=$status" "gpu=$gpu" "updated_at=$(timestamp)" > "$claim_dir/owner.txt"
}

run_task() {
  local benchmark=$1
  local task_log="$CONTROL_ROOT/${benchmark}.log"
  if ! claim_task "$benchmark"; then return 0; fi
  while true; do
    local gpu endpoint owner_file
    for gpu in 3 4 5; do
      endpoint=${ENDPOINTS[$gpu]}
      owner_file="$LEASE_ROOT/gpu${gpu}.owner"
      exec 9>"$LEASE_ROOT/gpu${gpu}.lock"
      if ! flock -n 9; then
        exec 9>&-
        continue
      fi
      printf '%s\n' "pid=$$" "benchmark=$benchmark" "run_tag=$RUN_TAG" \
        "endpoint=$endpoint" "acquired_at=$(timestamp)" > "$owner_file"
      echo "$(timestamp) lease_candidate benchmark=$benchmark gpu=$gpu" | tee -a "$task_log"
      if stable_idle "$gpu"; then
        set_claim_status "$benchmark" running "$gpu"
        echo "$(timestamp) run_start benchmark=$benchmark gpu=$gpu endpoint=$endpoint" \
          | tee -a "$task_log"
        if "$RUNNER" "$benchmark" "$RUN_TAG" "$endpoint" >> "$task_log" 2>&1; then
          set_claim_status "$benchmark" complete "$gpu"
          echo "$(timestamp) run_complete benchmark=$benchmark gpu=$gpu" | tee -a "$task_log"
          rm -f "$owner_file"
          flock -u 9
          exec 9>&-
          return 0
        fi
        set_claim_status "$benchmark" failed "$gpu"
        echo "$(timestamp) run_failed benchmark=$benchmark gpu=$gpu" | tee -a "$task_log"
        rm -f "$owner_file"
        flock -u 9
        exec 9>&-
        return 1
      fi
      echo "$(timestamp) lease_released_not_idle benchmark=$benchmark gpu=$gpu" \
        | tee -a "$task_log"
      rm -f "$owner_file"
      flock -u 9
      exec 9>&-
    done
    set_claim_status "$benchmark" queued
    echo "$(timestamp) waiting_for_gpu benchmark=$benchmark" | tee -a "$task_log"
    sleep 60
  done
}

pids=()
for benchmark in "${BENCHMARKS[@]}"; do
  case "$benchmark" in memgallery|h2hmem|wma) ;; *) echo "unsupported benchmark: $benchmark" >&2; exit 2 ;; esac
  run_task "$benchmark" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
exit "$failed"
