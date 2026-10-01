#!/usr/bin/env bash
# Five-GPU queue: H2HMEM=GPU1, MemEye=GPU2, Mem-Gallery=GPU3,
# MEMLENS=GPU4, WorldMemArena=GPU5.
set -u -o pipefail

if [[ $# -ne 1 ]]; then echo "usage: $0 RUN_ID" >&2; exit 2; fi
run_id=$1
workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=$offline/.venvs/mirix/bin/python
defaults=$workspace/Nvida_api/defaults_m2a_qwen3vl4b_emb06_top5_gpu35.json
run_root=$offline/outputs/_runs/$run_id
log=$run_root/gpu35_launcher.log
mkdir -p "$run_root"
export PYTHONPATH=$offline/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

retry() {
  local name=$1; shift
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    printf '%s start benchmark=%s attempt=%s\n' "$(date --iso-8601=seconds)" "$name" "$attempt" >> "$log"
    "$@" >> "$log" 2>&1 && { printf '%s complete benchmark=%s\n' "$(date --iso-8601=seconds)" "$name" >> "$log"; return; }
    printf '%s retry benchmark=%s\n' "$(date --iso-8601=seconds)" "$name" >> "$log"
    sleep 60
  done
}

result_complete() {
  local benchmark=$1 expected=$2
  local results=$offline/outputs/$benchmark/M2A/$run_id/results.json
  RESULTS=$results EXPECTED=$expected "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

try:
    rows = json.loads(Path(os.environ["RESULTS"]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(rows) == int(os.environ["EXPECTED"]) and not any(row.get("error") for row in rows) else 1)
PY
}

matrix_job() {
  local benchmark=$1 endpoint=$2 status=$3 embedding_endpoint=$4
  "$python_bin" "$offline/scripts/run_test_baseline_matrix.py" \
    --defaults "$defaults" --efficiency-config "$offline/configs/model_efficiency.json" \
    --output-root "$offline/outputs" --run-id "$run_id" --status-file-name "$status" \
    --endpoint "$endpoint" --embedding-base-url "$embedding_endpoint" \
    --baseline M2A --benchmark "$benchmark" --top-k 5 --skip-smoke
}

new_job() {
  local benchmark=$1 endpoint=$2 module=$3 judge=$4 expected=$5 embedding_endpoint=$6
  local result=$offline/outputs/$benchmark/M2A/$run_id
  "$python_bin" -m "$module" --baseline M2A --result-dir "$result" \
    --baseline-state-dir "$result/memory/datasets" --sample-concurrency 16 --answer-concurrency 16 \
    --checkpoint-every 10 --answer-model Qwen/Qwen3-VL-4B-Instruct --answer-base-url "$endpoint" \
    --answer-temperature 0.0 --num-predict 512 --request-timeout 180 --retries 2 --no-think \
    --reasoning-effort minimal --executor-model Qwen/Qwen3-VL-4B-Instruct --executor-base-url "$endpoint" \
    --executor-temperature 0.0 --executor-max-tokens 1200 --m2a-salvage-truncated-updates \
    --m2a-skip-failed-build-points --m2a-max-consecutive-failed-build-points 1000000000 \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --embedding-base-url "$embedding_endpoint" --embedding-dim 2048 --top-k 5 \
    --efficiency-config "$offline/configs/model_efficiency.json" --resume || return $?
  "$python_bin" "$offline/scripts/judge_results_llm_parallel.py" --benchmark "$judge" \
    --results "$result/results.json" --out-dir "$result" --key-file "$workspace/Nvida_api/Openrouter_api" \
    --model openai/gpt-4o-mini --workers 32 --timeout 60 --retries 2 --max-tokens 512 \
    --checkpoint-every 25 --resume
}

( result_complete H2HMEM 360 || retry H2HMEM matrix_job H2HMEM http://127.0.0.1:8016/v1 matrix_gpu35_emb06_h2hmem.json http://127.0.0.1:8003/v1 ) &
gpu1_pid=$!
( result_complete MemEye 371 || retry MemEye new_job MemEye http://127.0.0.1:8017/v1 benchmarks.memeye_harness.eval_memeye memeye 371 http://127.0.0.1:8004/v1 ) &
gpu2_pid=$!
( result_complete Mem-Gallery 275 || retry Mem-Gallery matrix_job Mem-Gallery http://127.0.0.1:8013/v1 matrix_gpu35_emb06_mem_gallery.json http://127.0.0.1:8003/v1 ) &
gpu3_pid=$!
( result_complete MEMLENS 173 || retry MEMLENS new_job MEMLENS http://127.0.0.1:8014/v1 benchmarks.memlens_harness.eval_memlens memlens 173 http://127.0.0.1:8004/v1 ) &
gpu4_pid=$!
( result_complete WorldMemArena 440 || retry WorldMemArena matrix_job WorldMemArena http://127.0.0.1:8015/v1 matrix_gpu35_emb06_worldmemarena.json http://127.0.0.1:8002/v1 ) &
gpu5_pid=$!
wait "$gpu1_pid" "$gpu2_pid" "$gpu3_pid" "$gpu4_pid" "$gpu5_pid"
printf '%s launcher_complete\n' "$(date --iso-8601=seconds)" >> "$log"
