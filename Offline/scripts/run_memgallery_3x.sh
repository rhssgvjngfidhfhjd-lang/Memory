#!/usr/bin/env bash
# Three sequential full MemGallery evaluations:
#   run 1 reuses an existing completed memory bank;
#   runs 2 and 3 build independent memory banks from scratch.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
GROUP_OUT=${GROUP_OUT:-$ROOT/outputs/hive_mem_3runs}
SOURCE_INDEX=${SOURCE_INDEX:-$ROOT/outputs/hive_mem}
EMBED_DEVICE=${EMBED_DEVICE:-cuda:0}
EXPECTED_DATASETS=${EXPECTED_DATASETS:-20}
EXPECTED_QA=${EXPECTED_QA:-1711}

if [ ! -x "$PY" ]; then
  echo "Python executable not found: $PY (override with PY=/path/to/python)" >&2
  exit 1
fi

mkdir -p "$GROUP_OUT"
exec 9>"$GROUP_OUT/pipeline.lock"
if ! flock -n 9; then
  echo "Another run_memgallery_3x.sh process already holds $GROUP_OUT/pipeline.lock" >&2
  exit 1
fi

cd "$ROOT"

stamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

valid_count() {
  local path=$1
  local expected=$2
  "$PY" - "$path" "$expected" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
if isinstance(value, list):
    count = len(value)
elif isinstance(value, dict):
    count = value.get("count")
else:
    count = None
raise SystemExit(0 if count == expected else 1)
PY
}

valid_qa_run() {
  local result_root=$1
  local expected=$2
  "$PY" - "$result_root" "$expected" <<'PY'
import json
import sys
from pathlib import Path

from benchmarks.memgallery_harness.runner.prompts import prompt_manifest

root = Path(sys.argv[1])
expected = int(sys.argv[2])
try:
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
current_prompt = prompt_manifest()
valid = (
    isinstance(results, list)
    and len(results) == expected
    and not any(row.get("error") for row in results)
    and manifest.get("answer_errors") == 0
    and manifest.get("prompt_sha256") == current_prompt["prompt_sha256"]
)
raise SystemExit(0 if valid else 1)
PY
}

valid_judge_run() {
  local metrics=$1
  local results=$2
  local expected=$3
  valid_count "$metrics" "$expected" && [ "$metrics" -nt "$results" ]
}

run_eval_pair() {
  local index_root=$1
  local result_root=$2
  mkdir -p "$result_root"

  if ! valid_qa_run "$result_root/qa_vector_baseline" "$EXPECTED_QA"; then
    echo "[$(stamp)] vector QA start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
      --all-datasets --index-root "$index_root" --top-k 5 \
      --result-dir "$result_root/qa_vector_baseline"
  fi

  if ! valid_qa_run "$result_root/qa_graph_append2" "$EXPECTED_QA"; then
    echo "[$(stamp)] graph QA start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
      --all-datasets --index-root "$index_root" --top-k 5 \
      --graph-retrieval --graph-mode append --append-k 2 --expansion-bonus 0.05 \
      --result-dir "$result_root/qa_graph_append2"
  fi

  if ! valid_judge_run \
      "$result_root/qa_vector_baseline/llm_judge_metrics.json" \
      "$result_root/qa_vector_baseline/results.json" "$EXPECTED_QA"; then
    echo "[$(stamp)] vector judge start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" scripts/judge_results_llm_parallel.py \
      --results "$result_root/qa_vector_baseline/results.json"
  fi

  if ! valid_judge_run \
      "$result_root/qa_graph_append2/llm_judge_metrics.json" \
      "$result_root/qa_graph_append2/results.json" "$EXPECTED_QA"; then
    echo "[$(stamp)] graph judge start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" scripts/judge_results_llm_parallel.py \
      --results "$result_root/qa_graph_append2/results.json"
  fi
}

build_and_eval() {
  local run_name=$1
  local run_root="$GROUP_OUT/$run_name/c_insert_only"
  local index_root="$run_root"

  if [ ! -f "$run_root/build_manifest.json" ] || \
     [ "$(find "$run_root/datasets" -mindepth 2 -maxdepth 2 -type f -name memories.jsonl 2>/dev/null | wc -l)" -ne "$EXPECTED_DATASETS" ]; then
    echo "[$(stamp)] fresh build start/resume: $run_name (embedding=$EMBED_DEVICE)"
    PYTHONUNBUFFERED=1 "$PY" -m hive_mem.build_memories \
      --all-datasets --output-root "$run_root" --device "$EMBED_DEVICE"
  else
    echo "[$(stamp)] completed build found, skipping rebuild: $run_name"
  fi

  echo "[$(stamp)] edges start: $run_name"
  PYTHONUNBUFFERED=1 "$PY" -m hive_mem.build_memory_edges \
    "$index_root"/datasets/*

  run_eval_pair "$index_root" "$run_root/results"
  echo "[$(stamp)] completed: $run_name"
}

if [ "$(find "$SOURCE_INDEX/datasets" -mindepth 2 -maxdepth 2 -type f -name memories.jsonl | wc -l)" -ne "$EXPECTED_DATASETS" ]; then
  echo "Expected $EXPECTED_DATASETS completed memories.jsonl files under $SOURCE_INDEX" >&2
  exit 1
fi

echo "[$(stamp)] run 1 start: reuse $SOURCE_INDEX"
run_eval_pair "$SOURCE_INDEX" "$GROUP_OUT/run1_existing_v3/results"
echo "[$(stamp)] completed: run1_existing_v3"

build_and_eval run2_rebuild
build_and_eval run3_rebuild

echo "[$(stamp)] all three runs completed"
