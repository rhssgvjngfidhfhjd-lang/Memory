#!/usr/bin/env bash
# Three sequential full MemGallery evaluations:
#   run 1 reuses the completed hive_mem_v3 memory bank;
#   runs 2 and 3 build independent memory banks from scratch.
set -euo pipefail

PY=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
ROOT=/data/haozhen/Memory/Offline
GROUP_OUT="$ROOT/outputs/hive_mem_3runs"
SOURCE_INDEX="$ROOT/outputs/hive_mem_v3/c_insert_only/c_text_caption_image"
EMBED_DEVICE=${EMBED_DEVICE:-cuda:1}

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

run_eval_pair() {
  local index_root=$1
  local result_root=$2
  mkdir -p "$result_root"

  if ! valid_count "$result_root/qa_vector_baseline/results.json" 1711; then
    echo "[$(stamp)] vector QA start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
      --all-datasets --index-root "$index_root" --top-k 5 \
      --result-dir "$result_root/qa_vector_baseline"
  fi

  if ! valid_count "$result_root/qa_graph_append2/results.json" 1711; then
    echo "[$(stamp)] graph QA start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
      --all-datasets --index-root "$index_root" --top-k 5 \
      --graph-retrieval --graph-mode append --append-k 2 --expansion-bonus 0.05 \
      --result-dir "$result_root/qa_graph_append2"
  fi

  if ! valid_count "$result_root/qa_vector_baseline/llm_judge_metrics.json" 1711; then
    echo "[$(stamp)] vector judge start: $result_root"
    PYTHONUNBUFFERED=1 "$PY" scripts/judge_results_llm_parallel.py \
      --results "$result_root/qa_vector_baseline/results.json"
  fi

  if ! valid_count "$result_root/qa_graph_append2/llm_judge_metrics.json" 1711; then
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
     [ "$(find "$run_root/datasets" -mindepth 2 -maxdepth 2 -type f -name memories.jsonl 2>/dev/null | wc -l)" -ne 20 ]; then
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

if [ "$(find "$SOURCE_INDEX/datasets" -mindepth 2 -maxdepth 2 -type f -name memories.jsonl | wc -l)" -ne 20 ]; then
  echo "Expected 20 completed memories.jsonl files under $SOURCE_INDEX" >&2
  exit 1
fi

echo "[$(stamp)] run 1 start: reuse hive_mem_v3"
run_eval_pair "$SOURCE_INDEX" "$GROUP_OUT/run1_existing_v3/results"
echo "[$(stamp)] completed: run1_existing_v3"

build_and_eval run2_rebuild
build_and_eval run3_rebuild

echo "[$(stamp)] all three runs completed"
