#!/usr/bin/env bash
# Persistent H2HMEM -> WorldMemArena queue for the 1200-token M2A run.
set -u -o pipefail

run_id=${1:?usage: run_m2a_h2_wma_top7_vl2b_1200_queue.sh RUN_ID}
workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=$offline/.venvs/mirix/bin/python
defaults=$workspace/Nvida_api/defaults_m2a_qwen3vl4b_vl2b_top7_1200_ramp.json
run_root=$offline/outputs/_runs/$run_id
endpoint=http://127.0.0.1:8025/v1
embedding_endpoint=http://127.0.0.1:8001/v1
log=$run_root/h2_wma_persistent_queue.log

mkdir -p "$run_root"
export PYTHONPATH=$offline/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1

result_complete() {
  local benchmark=$1 expected=$2
  local result_dir=$offline/outputs/$benchmark/M2A/$run_id
  RESULT_DIR=$result_dir EXPECTED=$expected "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED"])
try:
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    judge = json.loads((root / "llm_judge_metrics.json").read_text(encoding="utf-8"))
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
complete = (
    isinstance(results, list)
    and len(results) == expected
    and not any(row.get("error") for row in results)
    and int(metrics.get("count", -1)) == expected
    and all(metrics.get(key) is not None for key in ("f1", "em", "llm_judge"))
    and int(judge.get("count", -1)) == expected
    and int(judge.get("judge_errors", -1)) == 0
)
raise SystemExit(0 if complete else 1)
PY
}

job_running() {
  local benchmark=$1
  pgrep -f "run-id $run_id.*--benchmark $benchmark" >/dev/null 2>&1
}

run_matrix() {
  local benchmark=$1 status_file=$2
  "$python_bin" "$offline/scripts/run_test_baseline_matrix.py" \
    --defaults "$defaults" \
    --efficiency-config "$offline/configs/model_efficiency.json" \
    --split-manifest "$offline/configs/multimodal_split_manifest.json" \
    --output-root "$offline/outputs" \
    --run-id "$run_id" \
    --status-file-name "$status_file" \
    --endpoint "$endpoint" \
    --embedding-base-url "$embedding_endpoint" \
    --top-k 7 \
    --baseline M2A \
    --benchmark "$benchmark" \
    --skip-smoke
}

ensure_job() {
  local benchmark=$1 expected=$2 status_file=$3
  local attempt=0
  while ! result_complete "$benchmark" "$expected"; do
    if job_running "$benchmark"; then
      printf '%s wait_existing benchmark=%s\n' \
        "$(date --iso-8601=seconds)" "$benchmark" >> "$log"
      sleep 60
      continue
    fi
    attempt=$((attempt + 1))
    printf '%s start benchmark=%s attempt=%s endpoint=%s\n' \
      "$(date --iso-8601=seconds)" "$benchmark" "$attempt" "$endpoint" >> "$log"
    run_matrix "$benchmark" "$status_file" >> "$log" 2>&1
    code=$?
    printf '%s exit benchmark=%s attempt=%s code=%s\n' \
      "$(date --iso-8601=seconds)" "$benchmark" "$attempt" "$code" >> "$log"
    if ! result_complete "$benchmark" "$expected"; then
      sleep 60
    fi
  done
  printf '%s complete benchmark=%s\n' \
    "$(date --iso-8601=seconds)" "$benchmark" >> "$log"
}

printf '%s queue_start run_id=%s\n' "$(date --iso-8601=seconds)" "$run_id" >> "$log"
ensure_job H2HMEM 360 h2_queue_recovery_status.json
ensure_job WorldMemArena 440 wma_queue_status.json
printf '%s queue_complete run_id=%s\n' "$(date --iso-8601=seconds)" "$run_id" >> "$log"
