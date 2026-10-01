#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/haozhen/Memory-clean
OFFLINE="$ROOT/Offline"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
KEY_FILE="$ROOT/Nvida_api/Openrouter_api"
SPLIT_MANIFEST="$OFFLINE/configs/multimodal_split_manifest.json"
GPT_SOURCE_INPUTS="$OFFLINE/outputs/_runs/mma_gpt5mini_api_resume_20260914_220323/_test_inputs"
RUN_LOG_ROOT="$OFFLINE/outputs/_runs/mma_reuse_isolated_20260921"

mkdir -p "$RUN_LOG_ROOT"
export PYTHONPATH="$OFFLINE/src"
export MMA_QA_ONLY_REUSE=1
export MMA_ISOLATED_FINAL_ANSWER=1

timestamp() { date --iso-8601=seconds; }

run_harness() {
  local benchmark=$1 model=$2 endpoint=$3 result_dir=$4 state_dir=$5 storage_dim=$6
  local log="$result_dir/run.log"
  local reasoning_args=()
  local efficiency_config="$OFFLINE/configs/model_efficiency.json"
  if [[ "$model" == openai/* ]]; then
    reasoning_args=(--reasoning-effort minimal)
    efficiency_config="$ROOT/Nvida_api/config_gpt-5-mini"
  fi
  local common=(
    --baseline MMA
    --result-dir "$result_dir"
    --baseline-state-dir "$state_dir"
    --answer-concurrency 16
    --checkpoint-every 10
    --answer-model "$model"
    --answer-base-url "$endpoint"
    --answer-temperature 0.0
    --num-predict 512
    --executor-model "$model"
    --executor-base-url "$endpoint"
    --executor-temperature 0.0
    --executor-max-tokens 4096
    --executor-visual-input image
    --embedding-model Qwen/Qwen3-VL-Embedding-2B
    --embedding-base-url http://127.0.0.1:8001/v1
    --embedding-dim 2048
    --top-k 7
    --request-timeout 180
    --retries 2
    --resume
    --mma-native-batch-size 5
    --split-manifest "$SPLIT_MANIFEST"
    --split test
    --efficiency-config "$efficiency_config"
  )
  export MMA_STORAGE_EMBEDDING_DIM="$storage_dim"
  printf '%s benchmark=%s start model=%s endpoint=%s storage_dim=%s\n' \
    "$(timestamp)" "$benchmark" "$model" "$endpoint" "$storage_dim" | tee -a "$log"
  case "$benchmark" in
    memgallery)
      "$PYTHON" -m benchmarks.memgallery_harness.eval_memgallery \
        "${common[@]}" "${reasoning_args[@]}" --sample-concurrency 3 \
        --data-dir "$GPT_SOURCE_INPUTS/Mem-Gallery/data" --all-datasets \
        --retrieval-memory-tokenizer Qwen/Qwen3-VL-4B-Instruct \
        --allow-answer-errors 2>&1 | tee -a "$log"
      ;;
    wma)
      "$PYTHON" -m benchmarks.wma_harness.eval_wma \
        "${common[@]}" "${reasoning_args[@]}" --sample-concurrency "$([[ "$model" == openai/* ]] && echo 3 || echo 2)" \
        --data-dir "$GPT_SOURCE_INPUTS/WorldMemArena/lifelong" \
        --allow-answer-errors 2>&1 | tee -a "$log"
      ;;
    h2hmem)
      "$PYTHON" -m benchmarks.h2hmem_harness.eval_h2hmem \
        "${common[@]}" "${reasoning_args[@]}" --sample-concurrency 1 \
        --data-dir "$GPT_SOURCE_INPUTS/H2HMEM/dataset" \
        --allow-answer-errors 2>&1 | tee -a "$log"
      ;;
    *)
      printf 'unknown benchmark: %s\n' "$benchmark" >&2
      return 2
      ;;
  esac
  local rc=${PIPESTATUS[0]}
  printf '%s benchmark=%s finish rc=%s\n' "$(timestamp)" "$benchmark" "$rc" | tee -a "$log"
  return "$rc"
}

run_gpt() {
  if [[ ! -s "$KEY_FILE" ]]; then
    printf 'OpenRouter key file is missing or empty: %s\n' "$KEY_FILE" >&2
    exit 2
  fi
  export OPENAI_API_KEY
  OPENAI_API_KEY=$(tr -d '\r\n' < "$KEY_FILE")
  export MMA_STORAGE_EMBEDDING_DIM=2048
  case ${1:?benchmark required} in
    memgallery)
      run_harness memgallery openai/gpt-5-mini https://openrouter.ai/api/v1 \
        "$OFFLINE/outputs/Mem-Gallery/MMA/mma_gpt5mini_reuse_isolated_20260921" \
        "$OFFLINE/outputs/Mem-Gallery/MMA/mma_gpt5mini_reuse_isolated_20260921/memory/datasets" 2048
      ;;
    wma)
      run_harness wma openai/gpt-5-mini https://openrouter.ai/api/v1 \
        "$OFFLINE/outputs/WorldMemArena/MMA/mma_gpt5mini_reuse_isolated_20260921" \
        "$OFFLINE/outputs/WorldMemArena/MMA/mma_gpt5mini_reuse_isolated_20260921/memory/datasets" 2048
      ;;
    h2hmem)
      run_harness h2hmem openai/gpt-5-mini https://openrouter.ai/api/v1 \
        "$OFFLINE/outputs/H2HMEM/MMA/mma_gpt5mini_reuse_isolated_20260921" \
        "$OFFLINE/outputs/H2HMEM/MMA/mma_gpt5mini_reuse_isolated_20260921/memory/datasets" 2048
      ;;
  esac
}

run_qwen() {
  export OPENAI_API_KEY=EMPTY
  export MMA_STORAGE_EMBEDDING_DIM=4096
  case ${1:?benchmark required} in
    memgallery)
      run_harness memgallery Qwen/Qwen3-VL-4B-Instruct http://127.0.0.1:8014/v1 \
        "$OFFLINE/outputs/Mem-Gallery/MMA/mma_qwen_reuse_isolated_20260921" \
        "$OFFLINE/outputs/Mem-Gallery/MMA/mma_qwen_reuse_isolated_20260921/memory/datasets" 4096
      ;;
    wma)
      run_harness wma Qwen/Qwen3-VL-4B-Instruct http://127.0.0.1:8014/v1 \
        "$OFFLINE/outputs/WorldMemArena/MMA/mma_qwen_reuse_isolated_20260921" \
        "$OFFLINE/outputs/WorldMemArena/MMA/mma_qwen_reuse_isolated_20260921/memory/datasets" 4096
      ;;
    h2hmem)
      run_harness h2hmem Qwen/Qwen3-VL-4B-Instruct http://127.0.0.1:8015/v1 \
        "$OFFLINE/outputs/H2HMEM/MMA/mma_qwen_reuse_isolated_20260921" \
        "$OFFLINE/outputs/H2HMEM/MMA/mma_qwen_reuse_isolated_20260921/memory/datasets" 4096
      ;;
  esac
}

metric_value() {
  local port=$1 metric=$2
  curl -fsS --max-time 5 "http://127.0.0.1:${port}/metrics" 2>/dev/null \
    | awk -v name="$metric" '$1 ~ ("^vllm:" name "\\{") {sum += $2} END {print sum + 0}'
}

wait_for_qwen_gpus() {
  local stable=0
  while (( stable < 5 )); do
    local workers running waiting port
    workers=$(pgrep -fc 'zero_penalty_queue_worker.sh gpu[345]' || true)
    running=0
    waiting=0
    for port in 8013 8014 8015; do
      running=$((running + $(metric_value "$port" num_requests_running)))
      waiting=$((waiting + $(metric_value "$port" num_requests_waiting)))
    done
    if (( workers == 0 && running == 0 && waiting == 0 )); then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '%s qwen_wait workers=%s running=%s waiting=%s stable=%s/5\n' \
      "$(timestamp)" "$workers" "$running" "$waiting" "$stable" \
      | tee -a "$RUN_LOG_ROOT/qwen_wait.log"
    (( stable < 5 )) && sleep 60
  done
}

start_tmux_once() {
  local session=$1 command=$2
  if tmux has-session -t "$session" 2>/dev/null; then
    printf '%s tmux_exists session=%s\n' "$(timestamp)" "$session"
    return 0
  fi
  tmux new-session -d -s "$session" "cd '$ROOT' && $command"
  printf '%s tmux_started session=%s\n' "$(timestamp)" "$session"
}

scheduler() {
  start_tmux_once mma_gpt_reuse_memgallery_0921 "bash '$0' gpt memgallery"
  start_tmux_once mma_gpt_reuse_wma_0921 "bash '$0' gpt wma"
  start_tmux_once mma_gpt_reuse_h2h_0921 "bash '$0' gpt h2hmem"
  wait_for_qwen_gpus
  start_tmux_once mma_qwen_reuse_wma_0921 "bash '$0' qwen wma"
  start_tmux_once mma_qwen_reuse_h2h_0921 "bash '$0' qwen h2hmem"
  printf '%s scheduler_launched_all\n' "$(timestamp)" | tee -a "$RUN_LOG_ROOT/qwen_wait.log"
}

case ${1:-} in
  scheduler) scheduler ;;
  gpt) run_gpt "${2:?benchmark required}" ;;
  qwen) run_qwen "${2:?benchmark required}" ;;
  *) printf 'usage: %s scheduler|gpt BENCHMARK|qwen BENCHMARK\n' "$0" >&2; exit 2 ;;
esac
