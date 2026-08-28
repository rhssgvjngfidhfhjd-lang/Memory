#!/bin/sh
# Serve the executor/answer model with vLLM (OpenAI-compatible, port 18000).
# Usage: GPUS=4 sh scripts/serve_vllm.sh
# Override TP_SIZE/MAX_NUM_SEQS for topology and concurrency experiments.
set -eu

GPUS=${GPUS:-0}
PORT=${PORT:-18000}
MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct}
SERVED_NAME=${SERVED_NAME:-Qwen/Qwen3-VL-4B-Instruct}
TP_SIZE=${TP_SIZE:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --mm-processor-cache-gb 0
