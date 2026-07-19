#!/usr/bin/env bash
set -uo pipefail
cd /data1/haozhen/Visual_Primitives/Offline/Offline
LOG=exp_log_qwen8b/pipeline.log

echo "[pipeline] waiting for step1 (embed_chunks) to finish..." >> "$LOG"
while [ ! -f exp_log_qwen8b/01_embed_chunks.exit ]; do sleep 5; done
code=$(cat exp_log_qwen8b/01_embed_chunks.exit)
if [ "$code" != "0" ]; then
  echo "[pipeline] STEP1 FAILED exit=$code" >> "$LOG"
  exit 1
fi
echo "[pipeline] STEP1 DONE (embed_chunks)" >> "$LOG"

echo "[pipeline] STEP2 START (build_faiss)" >> "$LOG"
python build_faiss.py \
  --chunks artifacts/chunks.jsonl \
  --embedding-dir artifacts/embeddings_qwen8b \
  --index-dir artifacts/faiss_index_qwen8b \
  --dim 4096 > exp_log_qwen8b/02_build_faiss.log 2>&1
code=$?
if [ "$code" != "0" ]; then
  echo "[pipeline] STEP2 FAILED exit=$code" >> "$LOG"
  exit 1
fi
echo "[pipeline] STEP2 DONE (build_faiss)" >> "$LOG"

echo "[pipeline] STEP3 START (build_query_embeddings)" >> "$LOG"
python build_query_embeddings.py \
  --data-dir /data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data \
  --out-dir artifacts/query_embeddings_qwen8b \
  --model-name Qwen/Qwen3-VL-Embedding-8B \
  --dim 4096 \
  --devices 2,3 \
  --dtype bfloat16 \
  --local-files-only > exp_log_qwen8b/03_build_query_embeddings.log 2>&1
code=$?
if [ "$code" != "0" ]; then
  echo "[pipeline] STEP3 FAILED exit=$code" >> "$LOG"
  exit 1
fi
echo "[pipeline] STEP3 DONE (build_query_embeddings)" >> "$LOG"

echo "[pipeline] STEP4 START (run_memgallery QA)" >> "$LOG"
RESULT_DIR="artifacts/results/memgallery20_qwen3vl_embedding_8b_$(date +%Y%m%d_%H%M)"
PYTHONPATH=$PWD python -u offline_omni_memory/benchmarks/memgallery/run_memgallery.py \
  --all-datasets --max-qa 0 \
  --embedding-model Qwen/Qwen3-VL-Embedding-8B \
  --embedding-dim 4096 \
  --index-dir artifacts/faiss_index_qwen8b \
  --query-embedding-dir artifacts/query_embeddings_qwen8b \
  --result-dir "$RESULT_DIR" \
  --answer-backend openai \
  --answer-base-url http://localhost:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct \
  --answer-api-key EMPTY \
  --num-predict 8000 \
  --request-timeout 180 \
  --retries 2 \
  --no-think > exp_log_qwen8b/04_run_memgallery.log 2>&1
code=$?
if [ "$code" != "0" ]; then
  echo "[pipeline] STEP4 FAILED exit=$code" >> "$LOG"
  exit 1
fi
echo "[pipeline] STEP4 DONE (run_memgallery) -> $RESULT_DIR" >> "$LOG"
echo "[pipeline] ALL DONE" >> "$LOG"
