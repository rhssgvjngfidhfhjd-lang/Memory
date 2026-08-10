#!/bin/sh
# Serve the executor/answer model with vLLM (OpenAI-compatible, port 8000).
# Usage:  GPUS=2,3 sh scripts/serve_vllm.sh        (default GPUS=0,1)
set -eu

GPUS=${GPUS:-0,1}
PORT=${PORT:-8000}
MODEL=${MODEL:-/data/shared_models/Qwen3-VL-4B-Instruct}
SERVED_NAME=${SERVED_NAME:-Qwen/Qwen3-VL-4B-Instruct}

CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --mm-processor-cache-gb 0
