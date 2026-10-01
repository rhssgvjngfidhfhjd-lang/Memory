#!/usr/bin/env bash
set -euo pipefail

result_dir="$1"
benchmark="$2"

while [[ ! -f "$result_dir/results.json" || ! -f "$result_dir/run_manifest.json" ]]; do
  sleep 30
done

export PYTHONPATH=/data/haozhen/Memory-clean/Offline/src
exec /data/haozhen/miniconda3/envs/wma_py310/bin/python -u \
  /data/haozhen/Memory-clean/Offline/scripts/judge_results_llm_parallel.py \
  --benchmark "$benchmark" \
  --results "$result_dir/results.json" \
  --out-dir "$result_dir" \
  --key-file /data/haozhen/Memory-clean/Nvida_api/Openrouter_api \
  --model openai/gpt-4o-mini \
  --workers 32 \
  --timeout 60 \
  --retries 2 \
  --max-tokens 512 \
  --checkpoint-every 25 \
  --resume
