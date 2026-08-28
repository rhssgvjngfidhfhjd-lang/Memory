#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/haozhen/Memory/WorldMemArena"
OUTPUT_DIR="$REPO_ROOT/exp_log/amem_qwen3vl_embedding_2b_gpu23_20260812"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# Match Offline/configs/defaults.json for the answer/executor model.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_MODEL="Qwen/Qwen3-VL-4B-Instruct"
export OPENAI_MAX_TOKENS="512"
export OPENAI_TIMEOUT="180"

# A-Mem retrieval embeddings: use Offline's validated Qwen3-VL loader.
export OPENAI_EMBEDDING_MODEL="Qwen/Qwen3-VL-Embedding-2B"
export OFFLINE_ROOT="/data/haozhen/Memory/Offline"
export QWEN3VL_EMBEDDING_DEVICE="cpu"
export QWEN3VL_EMBEDDING_DTYPE="bfloat16"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

# GPU 2,3 are reserved for the already-running vLLM server. Child baseline
# processes intentionally see no CUDA device, so embedding cannot claim VRAM.
export DATA150_BASELINE_CUDA_VISIBLE_DEVICES=""

# Keep the existing judge provider, but use Offline's current key-pool path.
export JUDGE_KEY_POOL_FILE="/data/haozhen/Memory/Nvida_api/apikey"

exec /data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  -m eval_framework.cli \
  --run-data150-gpt \
  --dataset "$REPO_ROOT/WorldMemArena" \
  --dataset-type worldmemarena \
  --output-dir "$OUTPUT_DIR" \
  --progress-filename progress.csv \
  --data150-count 461 \
  --baselines A-Mem \
  --max-baseline-workers 1 \
  --per-baseline-workers 1
