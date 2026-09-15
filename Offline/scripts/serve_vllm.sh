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
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
# Qwen3-VL's bundled chat template emits Hermes-style
# <tool_call>{"name": ..., "arguments": ...}</tool_call> envelopes.
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-hermes}

CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --mm-processor-cache-gb 0
