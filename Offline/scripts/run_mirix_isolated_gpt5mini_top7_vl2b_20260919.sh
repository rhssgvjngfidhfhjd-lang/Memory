#!/usr/bin/env bash
set -euo pipefail

cd /data/haozhen/Memory-clean

benchmark="$1"
source_dir="$2"
result_dir="$3"

exec /data/haozhen/miniconda3/envs/wma_py310/bin/python Offline/scripts/rerun_qa_from_frozen_retrieval.py \
  --benchmark "$benchmark" \
  --baseline MIRIX \
  --source-dir "$source_dir" \
  --result-dir "$result_dir" \
  --answer-base-url https://openrouter.ai/api/v1 \
  --answer-model openai/gpt-5-mini \
  --answer-api-key-file /data/haozhen/Memory-clean/Nvida_api/Openrouter_api \
  --temperature 0 \
  --max-tokens 512 \
  --reasoning-effort minimal \
  --timeout 180 \
  --retries 2 \
  --concurrency 16 \
  --checkpoint-every 10 \
  --top-k 7 \
  --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
  --isolate-final-answer \
  --no-strict-short-answer \
  --no-openrouter-json-first \
  --resume
