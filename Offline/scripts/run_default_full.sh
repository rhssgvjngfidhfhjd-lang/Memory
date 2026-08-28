#!/bin/bash
# Full default-parameter HiveMem pipeline: build all 20 datasets -> edges ->
# QA (vector baseline + graph append) -> LLM judge.  All model/path defaults
# come from configs/defaults.json.  Designed to run detached (nohup).
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
OUT=${OUT:-$ROOT/outputs/hive_mem}
if [ ! -x "$PY" ]; then
  echo "Python executable not found: $PY (override with PY=/path/to/python)" >&2
  exit 1
fi
cd "$ROOT"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1

echo "=== DEFAULT FULL RUN start $(date) ==="
PYTHONUNBUFFERED=1 "$PY" -m hive_mem.build_memories \
  --all-datasets --output-root "$OUT"
echo "=== EDGES $(date) ==="
PYTHONUNBUFFERED=1 "$PY" -m hive_mem.build_memory_edges "$OUT"/datasets/*
echo "=== QA baseline $(date) ==="
PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
  --all-datasets --index-root "$OUT" --top-k 5 \
  --result-dir "$OUT/results/qa_vector_baseline"
echo "=== QA graph append $(date) ==="
PYTHONUNBUFFERED=1 "$PY" -m benchmarks.memgallery_harness.eval_memgallery \
  --all-datasets --index-root "$OUT" --top-k 5 \
  --graph-retrieval --graph-mode append --append-k 2 --expansion-bonus 0.05 \
  --result-dir "$OUT/results/qa_graph_append2"
echo "=== JUDGE $(date) ==="
for d in qa_vector_baseline qa_graph_append2; do
  PYTHONUNBUFFERED=1 "$PY" "$ROOT/scripts/judge_results_llm_parallel.py" \
    --results "$OUT/results/$d/results.json"
done
echo "=== DEFAULT FULL RUN ALL DONE $(date) ==="
