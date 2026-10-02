#!/usr/bin/env bash
# Persistent two-GPU M2A queue.  The matrix runner uses one worker per endpoint.
set -u -o pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ID" >&2
  exit 2
fi

run_id=$1
workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=$offline/.venvs/mirix/bin/python
run_root=$offline/outputs/_runs/$run_id
log=$run_root/gpu45_launcher.log

mkdir -p "$run_root"
export PYTHONPATH=$offline/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

run_matrix_job() {
  local benchmark=$1
  local endpoint=$2
  local status_file=matrix_gpu45_$(printf '%s' "$benchmark" | tr '[:upper:]-' '[:lower:]_').json
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    printf '%s start benchmark=%s endpoint=%s attempt=%s\n' "$(date --iso-8601=seconds)" "$benchmark" "$endpoint" "$attempt" >> "$log"
  "$python_bin" "$offline/scripts/run_test_baseline_matrix.py" \
    --defaults "$workspace/Nvida_api/defaults_m2a_qwen3vl4b_vl2b_top7_gpu45.json" \
    --efficiency-config "$offline/configs/model_efficiency.json" \
    --output-root "$offline/outputs" \
    --run-id "$run_id" \
    --status-file-name "$status_file" \
    --endpoint "$endpoint" \
    --embedding-base-url http://127.0.0.1:8001/v1 \
    --baseline M2A \
    --benchmark "$benchmark" \
    --skip-smoke >> "$log" 2>&1
  code=$?
  if [[ $code -eq 0 ]]; then
    printf '%s complete benchmark=%s\n' "$(date --iso-8601=seconds)" "$benchmark" >> "$log"
    return
  fi
  printf '%s retry benchmark=%s exit=%s\n' "$(date --iso-8601=seconds)" "$benchmark" "$code" >> "$log"
  sleep 60
  done
}

new_result_complete() {
  local result_dir=$1
  local expected=$2
  RESULT_DIR=$result_dir EXPECTED=$expected "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

try:
    rows = json.loads((Path(os.environ["RESULT_DIR"]) / "results.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(rows) == int(os.environ["EXPECTED"]) and not any(row.get("error") for row in rows) else 1)
PY
}

run_new_job() {
  local benchmark=$1
  local endpoint=$2
  local expected=$3
  local module=$4
  local judge_name=$5
  local result_dir=$offline/outputs/$benchmark/M2A/$run_id
  local attempt=0
  while ! new_result_complete "$result_dir" "$expected"; do
    attempt=$((attempt + 1))
    printf '%s start benchmark=%s endpoint=%s attempt=%s\n' "$(date --iso-8601=seconds)" "$benchmark" "$endpoint" "$attempt" >> "$log"
    "$python_bin" -m "$module" \
      --baseline M2A \
      --result-dir "$result_dir" \
      --baseline-state-dir "$result_dir/memory/datasets" \
      --sample-concurrency 1 \
      --answer-concurrency 16 \
      --checkpoint-every 10 \
      --answer-model Qwen/Qwen3-VL-4B-Instruct \
      --answer-base-url "$endpoint" \
      --answer-temperature 0.0 \
      --num-predict 512 \
      --request-timeout 180 \
      --retries 2 \
      --no-think \
      --reasoning-effort minimal \
      --executor-model Qwen/Qwen3-VL-4B-Instruct \
      --executor-base-url "$endpoint" \
      --executor-temperature 0.0 \
      --executor-max-tokens 4096 \
      --m2a-salvage-truncated-updates \
      --m2a-skip-failed-build-points \
      --m2a-max-consecutive-failed-build-points 1000000000 \
      --embedding-model Qwen/Qwen3-VL-Embedding-2B \
      --embedding-base-url http://127.0.0.1:8001/v1 \
      --embedding-dim 2048 \
      --top-k 7 \
      --efficiency-config "$offline/configs/model_efficiency.json" \
      --resume >> "$log" 2>&1
    code=$?
    if ! new_result_complete "$result_dir" "$expected"; then
      printf '%s retry benchmark=%s exit=%s\n' "$(date --iso-8601=seconds)" "$benchmark" "$code" >> "$log"
      sleep 60
    fi
  done
  "$python_bin" "$offline/scripts/judge_results_llm_parallel.py" \
    --benchmark "$judge_name" --results "$result_dir/results.json" --out-dir "$result_dir" \
    --key-file "$workspace/Nvida_api/Openrouter_api" --model openai/gpt-4o-mini \
    --workers 32 --timeout 60 --retries 2 --max-tokens 512 --checkpoint-every 25 --resume >> "$log" 2>&1
  printf '%s complete benchmark=%s\n' "$(date --iso-8601=seconds)" "$benchmark" >> "$log"
}

# Fixed round-robin assignment: GPU 4 receives jobs 1,3,5; GPU 5 receives 2,4.
(
  run_matrix_job Mem-Gallery http://127.0.0.1:8014/v1
  run_matrix_job WorldMemArena http://127.0.0.1:8014/v1
  run_new_job MEMLENS http://127.0.0.1:8014/v1 173 benchmarks.memlens_harness.eval_memlens memlens
) &
gpu4_pid=$!
(
  run_matrix_job H2HMEM http://127.0.0.1:8015/v1
  run_new_job MemEye http://127.0.0.1:8015/v1 371 benchmarks.memeye_harness.eval_memeye memeye
) &
gpu5_pid=$!
wait "$gpu4_pid"
wait "$gpu5_pid"

printf '%s launcher_complete\n' "$(date --iso-8601=seconds)" >> "$log"
