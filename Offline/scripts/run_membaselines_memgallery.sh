#!/bin/sh
# Stage 2+3 driver for the Mem-Gallery memory baselines (simplemem / amem / m2a
# / omnisimplemem): embed -> QA (F1 + HIT@5) -> LLM judge (JudgeAcc).
#
# Stage 1 (memory build) runs separately in the WorldMemArena venv:
#   cd /data1/haozhen/Visual_Primitives/Offline/WorldMemArena
#   .venv/bin/python eval_framework/scripts/build_memgallery_baselines.py --baseline amem
#
# Usage (from anywhere):
#   scripts/run_membaselines_memgallery.sh <baseline> [step] [answer_port]
#     baseline: simplemem | amem | m2a | omnisimplemem
#     step:     embed | qa | judge | all   (default all)
#     answer_port: vLLM answer server port (default $ANSWER_PORT or 8022)
# Env overrides:
#   ANSWER_PORT (default 8022)  EMBED_DEVICE (default cuda:3)
#   MEMORY_ROOT_BASE (default artifacts/baseline_memories)

set -u

ROOT=/data1/haozhen/Visual_Primitives/Offline/Offline
BASELINE=${1:?usage: run_membaselines_memgallery.sh <baseline> [embed|qa|judge|all] [answer_port]}
STEP=${2:-all}
ANSWER_PORT=${3:-${ANSWER_PORT:-8022}}
EMBED_DEVICE=${EMBED_DEVICE:-cuda:3}
MEMORY_ROOT_BASE=${MEMORY_ROOT_BASE:-artifacts/baseline_memories}

MEMORY_ROOT="$MEMORY_ROOT_BASE/$BASELINE"
RESULT_DIR="$MEMORY_ROOT_BASE/results/${BASELINE}_qwen3vl4b"
LOG_DIR="$ROOT/$MEMORY_ROOT_BASE/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT" || exit 1

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"; }

run_embed() {
  log "EMBED start baseline=$BASELINE device=$EMBED_DEVICE"
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.embed_baseline_memories \
    --memory-root "$MEMORY_ROOT" \
    --device "$EMBED_DEVICE" \
    2>&1 | tee "$LOG_DIR/embed_${BASELINE}.log"
  rc=$?
  [ "$rc" -ne 0 ] && { log "EMBED FAILED rc=$rc"; exit "$rc"; }
  log "EMBED done"
}

run_qa() {
  log "QA start baseline=$BASELINE port=$ANSWER_PORT"
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m simple_memory_pipeline.run_memgallery_baseline \
    --all-datasets \
    --index-root "$MEMORY_ROOT" \
    --query-embedding-dir artifacts/query_embeddings \
    --result-dir "$RESULT_DIR" \
    --answer-base-url "http://127.0.0.1:${ANSWER_PORT}/v1" \
    --answer-model Qwen/Qwen3-VL-4B-Instruct \
    --top-k 5 \
    2>&1 | tee "$LOG_DIR/qa_${BASELINE}.log"
  rc=$?
  [ "$rc" -ne 0 ] && { log "QA FAILED rc=$rc"; exit "$rc"; }
  log "QA done -> $RESULT_DIR/metrics.json (F1 + HIT@5)"
}

run_judge() {
  log "JUDGE start baseline=$BASELINE model=openai/gpt-oss-120b timeout=60"
  PYTHONUNBUFFERED=1 python3 judge_results_llm_parallel.py \
    --results "$RESULT_DIR/results.json" \
    --model openai/gpt-oss-120b \
    --timeout 60 \
    --key-count 12 \
    --max-tokens 1024 \
    2>&1 | tee "$LOG_DIR/judge_${BASELINE}.log"
  rc=$?
  [ "$rc" -ne 0 ] && { log "JUDGE FAILED rc=$rc"; exit "$rc"; }
  log "JUDGE done -> $RESULT_DIR/llm_judge_metrics.json"
}

case "$STEP" in
  embed) run_embed ;;
  qa)    run_qa ;;
  judge) run_judge ;;
  all)   run_embed; run_qa; run_judge ;;
  *) echo "unknown step: $STEP (embed|qa|judge|all)"; exit 2 ;;
esac
log "PIPELINE COMPLETE baseline=$BASELINE step=$STEP"
