#!/bin/sh
set -u

ROOT=/data/haozhen/Offline/Offline
LOG_DIR="$ROOT/artifacts/simple_memory_pipeline/logs"
STATUS_LOG="$LOG_DIR/715test1_supervisor.log"

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$STATUS_LOG"
}

run_stage() {
  stage=$1
  shift
  log "START stage=$stage"
  "$@"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FAILED stage=$stage exit_code=$rc"
    exit "$rc"
  fi
  log "DONE stage=$stage"
}

log "PIPELINE START pid=$$"

for mode in a b c; do
  run_stage "build_$mode" sh -c "PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.build_memgallery_memories \
    --mode $mode --all-datasets \
    --executor-model Qwen/Qwen3-VL-4B-Instruct \
    --executor-base-url http://127.0.0.1:8000/v1 \
    --executor-api-key EMPTY \
    --device cuda:0 \
    > '$LOG_DIR/build_$mode.log' 2>&1"
done

run_stage qa_a sh -c "PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.run_memgallery_baseline \
  --all-datasets \
  --index-root artifacts/simple_memory_pipeline/a_text_only \
  --query-embedding-dir artifacts/query_embeddings \
  --result-dir artifacts/simple_memory_pipeline/results/a_qwen3vl4b \
  --answer-base-url http://127.0.0.1:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct \
  > '$LOG_DIR/qa_a.log' 2>&1"

run_stage qa_b sh -c "PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.run_memgallery_baseline \
  --all-datasets \
  --index-root artifacts/simple_memory_pipeline/b_text_caption \
  --query-embedding-dir artifacts/query_embeddings \
  --result-dir artifacts/simple_memory_pipeline/results/b_qwen3vl4b \
  --answer-base-url http://127.0.0.1:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct \
  > '$LOG_DIR/qa_b.log' 2>&1"

run_stage qa_c sh -c "PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.run_memgallery_baseline \
  --all-datasets \
  --index-root artifacts/simple_memory_pipeline/c_text_caption_image \
  --query-embedding-dir artifacts/query_embeddings \
  --result-dir artifacts/simple_memory_pipeline/results/c_qwen3vl4b \
  --answer-base-url http://127.0.0.1:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct \
  > '$LOG_DIR/qa_c.log' 2>&1"

log "PIPELINE COMPLETE"
