#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/haozhen/Memory-clean
OFFLINE="$ROOT/Offline"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
SOURCE_MEMORY="$OFFLINE/outputs/H2HMEM/MMA/0907a/memory"
RESULT_DIR="$OFFLINE/outputs/H2HMEM/MMA/mma_qwen_reuse_isolated_rep2_20260923"
STATE_DIR="$RESULT_DIR/memory/datasets"
INPUT_DIR="$OFFLINE/outputs/_runs/mma_gpt5mini_api_resume_20260914_220323/_test_inputs/H2HMEM/dataset"
SPLIT_MANIFEST="$OFFLINE/configs/multimodal_split_manifest.json"
ENDPOINT=http://127.0.0.1:8015/v1

mkdir -p "$RESULT_DIR"
export PYTHONPATH="$OFFLINE/src"
export OPENAI_API_KEY=EMPTY
export MMA_QA_ONLY_REUSE=1
export MMA_ISOLATED_FINAL_ANSWER=1
export MMA_STORAGE_EMBEDDING_DIM=4096

if [[ ! -f "$STATE_DIR/.mma_reuse_manifest.json" ]]; then
  if [[ -d "$STATE_DIR" ]] && find "$STATE_DIR" -mindepth 1 -print -quit | grep -q .; then
    printf '%s refusing non-empty unaudited state directory: %s\n' \
      "$(date --iso-8601=seconds)" "$STATE_DIR" | tee -a "$RESULT_DIR/run.log"
    exit 2
  fi
  "$PYTHON" "$OFFLINE/scripts/prepare_mma_qa_only_reuse.py" \
    "$SOURCE_MEMORY" "$STATE_DIR" 2>&1 | tee -a "$RESULT_DIR/prepare.log"
fi

run_qa() {
  "$PYTHON" -m benchmarks.h2hmem_harness.eval_h2hmem \
    --baseline MMA \
    --result-dir "$RESULT_DIR" \
    --baseline-state-dir "$STATE_DIR" \
    --data-dir "$INPUT_DIR" \
    --sample-concurrency 1 \
    --sample-max-attempts 3 \
    --answer-concurrency 16 \
    --checkpoint-every 10 \
    --answer-model Qwen/Qwen3-VL-4B-Instruct \
    --answer-base-url "$ENDPOINT" \
    --answer-temperature 0.0 \
    --num-predict 512 \
    --executor-model Qwen/Qwen3-VL-4B-Instruct \
    --executor-base-url "$ENDPOINT" \
    --executor-temperature 0.0 \
    --executor-max-tokens 4096 \
    --executor-visual-input image \
    --embedding-model Qwen/Qwen3-VL-Embedding-2B \
    --embedding-base-url http://127.0.0.1:8001/v1 \
    --embedding-dim 2048 \
    --top-k 7 \
    --request-timeout 180 \
    --retries 2 \
    --resume \
    --mma-native-batch-size 5 \
    --split-manifest "$SPLIT_MANIFEST" \
    --split test \
    --efficiency-config "$OFFLINE/configs/model_efficiency.json" \
    --allow-answer-errors
}

printf '%s replicate_start source=%s endpoint=%s\n' \
  "$(date --iso-8601=seconds)" "$SOURCE_MEMORY" "$ENDPOINT" | tee -a "$RESULT_DIR/run.log"

qa_ok=0
for attempt in 1 2 3; do
  printf '%s qa_attempt=%s\n' "$(date --iso-8601=seconds)" "$attempt" \
    | tee -a "$RESULT_DIR/run.log"
  if run_qa 2>&1 | tee -a "$RESULT_DIR/run.log"; then
    qa_ok=1
    break
  fi
  sleep 30
done
if (( qa_ok == 0 )); then
  printf '%s qa_failed_after_three_attempts\n' "$(date --iso-8601=seconds)" \
    | tee -a "$RESULT_DIR/run.log"
  exit 1
fi

export OPENAI_API_KEY
OPENAI_API_KEY=$(tr -d '\r\n' < "$ROOT/Nvida_api/Openrouter_api")
"$PYTHON" -u "$OFFLINE/scripts/judge_results_llm_parallel.py" \
  --benchmark h2hmem \
  --results "$RESULT_DIR/results.json" \
  --out-dir "$RESULT_DIR" \
  --key-file "$ROOT/Nvida_api/Openrouter_api" \
  --model openai/gpt-4o-mini \
  --workers 32 \
  --timeout 60 \
  --retries 2 \
  --max-tokens 512 \
  --checkpoint-every 25 \
  --resume 2>&1 | tee -a "$RESULT_DIR/llm_judge.log"

printf '%s replicate_complete\n' "$(date --iso-8601=seconds)" | tee -a "$RESULT_DIR/run.log"
