#!/bin/sh
set -u

ROOT=/data/haozhen/Offline/Offline
LOG_DIR="$ROOT/artifacts/simple_memory_pipeline/logs"
STATUS_LOG="$LOG_DIR/no_id_leak_qa_supervisor.log"
BASE_URL=http://127.0.0.1:8022/v1

mkdir -p "$LOG_DIR"
cd "$ROOT" || exit 1

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$STATUS_LOG"
}

run_mode() {
  mode=$1
  index_name=$2
  log "START mode=$mode"
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.run_memgallery_baseline \
    --all-datasets \
    --index-root "artifacts/simple_memory_pipeline/$index_name" \
    --query-embedding-dir artifacts/query_embeddings \
    --result-dir "artifacts/simple_memory_pipeline/results/${mode}_qwen3vl4b_no_id_leak" \
    --answer-base-url "$BASE_URL" \
    --answer-model Qwen/Qwen3-VL-4B-Instruct \
    > "$LOG_DIR/qa_${mode}_no_id_leak.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FAILED mode=$mode exit_code=$rc"
    exit "$rc"
  fi
  log "DONE mode=$mode"
}

: > "$STATUS_LOG"
log "PIPELINE START pid=$$ base_url=$BASE_URL"
run_mode a a_text_only
run_mode b b_text_caption
run_mode c c_text_caption_image
log "PIPELINE COMPLETE"
