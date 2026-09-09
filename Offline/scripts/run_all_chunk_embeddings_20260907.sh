#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data/haozhen/Memory-clean/Offline
PYTHON_BIN=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
EMBEDDING_URL=http://127.0.0.1:8001/v1
LOG_DIR="$PROJECT_ROOT/outputs/_logs/chunk_embeddings"
LOG_FILE="$LOG_DIR/all_chunks_20260907.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
exec > >(tee -a "$LOG_FILE") 2>&1

run_one() {
  local input_path=$1
  local output_dir=$2
  echo "[$(date --iso-8601=seconds)] embedding $input_path"
  "$PYTHON_BIN" scripts/build_chunk_embeddings.py \
    --input "$input_path" \
    --output-dir "$output_dir" \
    --base-url "$EMBEDDING_URL" \
    --model Qwen/Qwen3-VL-Embedding-2B \
    --dim 2048 \
    --batch-size 8 \
    --include-images
}

run_one \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/qwen3_vl_embedding_2b/chunk_embeddings/chunks_no_profile

run_one \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunk_embeddings/chunks_lifelong

run_one \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong_512_balanced.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunk_embeddings/chunks_lifelong_512_balanced

run_one \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunk_embeddings/chunks_dyadic

run_one \
  data/h2hmem/chunks_multiparty.jsonl \
  data/h2hmem/chunk_embeddings/chunks_multiparty

echo "[$(date --iso-8601=seconds)] all chunk embeddings complete"
