#!/bin/bash
# Full default-parameter HiveMem pipeline: build all 20 datasets -> edges ->
# QA (vector baseline + graph append) -> LLM judge.  All model/path defaults
# come from configs/defaults.json.  Designed to run detached (nohup).
set -u
PY=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
ROOT=/data/haozhen/Memory/Offline
OUT=$ROOT/outputs/hive_mem
cd "$ROOT"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/pipeline.log") 2>&1

echo "=== DEFAULT FULL RUN start $(date) ==="
PYTHONUNBUFFERED=1 $PY -m hive_mem.build_memories --all-datasets
echo "=== EDGES $(date) ==="
PYTHONUNBUFFERED=1 $PY -m hive_mem.build_memory_edges "$OUT"/datasets/*
echo "=== QA baseline $(date) ==="
COMMON="--all-datasets --index-root $OUT --top-k 5"
PYTHONUNBUFFERED=1 $PY -m benchmarks.memgallery_harness.eval_memgallery $COMMON \
  --result-dir "$OUT/results/qa_vector_baseline"
echo "=== QA graph append $(date) ==="
PYTHONUNBUFFERED=1 $PY -m benchmarks.memgallery_harness.eval_memgallery $COMMON \
  --graph-retrieval --graph-mode append --append-k 2 --expansion-bonus 0.05 \
  --result-dir "$OUT/results/qa_graph_append2"
echo "=== JUDGE $(date) ==="
for d in qa_vector_baseline qa_graph_append2; do
  PYTHONUNBUFFERED=1 $PY "$ROOT/scripts/judge_results_llm_parallel.py" \
    --results "$OUT/results/$d/results.json"
done
echo "=== DEFAULT FULL RUN ALL DONE $(date) ==="
