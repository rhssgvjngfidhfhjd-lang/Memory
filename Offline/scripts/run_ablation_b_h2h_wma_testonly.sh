#!/usr/bin/env bash
set -uo pipefail

WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
OUTPUT_ROOT="$OFFLINE_ROOT/outputs/Ablation/B_evidence_testonly_20260923"
LEASE_ROOT="$OFFLINE_ROOT/outputs/_gpu_leases"
TASK_CLAIM="$OUTPUT_ROOT/task.claim"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"
EXPECTED_MODEL=Qwen/Qwen3-VL-4B-Instruct
POLL_SECONDS=60

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUTPUT_ROOT" "$LEASE_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/claims"

timestamp() {
  date --iso-8601=seconds
}

set_scheduler_status() {
  printf '%s %s\n' "$1" "$(timestamp)" | tee "$OUTPUT_ROOT/scheduler_status.txt"
}

cleanup() {
  if [[ -d "$TASK_CLAIM" ]]; then
    rm -f "$TASK_CLAIM/owner.txt"
    rmdir "$TASK_CLAIM" 2>/dev/null || true
  fi
}

if ! mkdir "$TASK_CLAIM" 2>/dev/null; then
  echo "task already claimed: $TASK_CLAIM" >&2
  exit 0
fi
trap cleanup EXIT INT TERM
printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(timestamp)" \
  > "$TASK_CLAIM/owner.txt"

gpu_endpoint() {
  case "$1" in
    3) echo http://127.0.0.1:8013/v1 ;;
    4) echo http://127.0.0.1:8014/v1 ;;
    5) echo http://127.0.0.1:8015/v1 ;;
    *) return 1 ;;
  esac
}

gpu_is_available() {
  local gpu=$1 endpoint=$2 model uuid util apps pid owner command api_pid port engine_ok=0
  model=$(curl -sf --max-time 10 "$endpoint/models" 2>/dev/null | "$PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
  [[ "$model" == "$EXPECTED_MODEL" ]] || return 1

  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu" 2>/dev/null) || return 1
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ') || return 1
  [[ "$util" =~ ^[0-9]+$ && "$util" -le 5 ]] || return 1

  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F', ' -v uuid="$uuid" '$1 == uuid {print $2}')
  [[ -n "$apps" ]] || return 1
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
    command=$(ps -o args= -p "$pid" 2>/dev/null)
    [[ "$owner" == "$(id -un)" ]] || return 1
    case "$command" in
      *VLLM::EngineCore*) engine_ok=1 ;;
      *serve_embeddings.py*|*serve_qwen3_embedding06_2048.py*) ;;
      *) return 1 ;;
    esac
  done <<< "$apps"
  [[ "$engine_ok" == 1 ]] || return 1

  port=${endpoint#http://127.0.0.1:}
  port=${port%%/*}
  api_pid=$(ss -ltnp 2>/dev/null | awk -v port=":${port}" '$4 ~ port {print $0}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)
  if [[ -n "$api_pid" ]]; then
    owner=$(ps -o user= -p "$api_pid" 2>/dev/null | xargs)
    command=$(ps -o args= -p "$api_pid" 2>/dev/null)
    [[ "$owner" == "$(id -un)" ]] || return 1
    [[ "$command" == *"vllm.entrypoints.openai.api_server"* ]] || return 1
    [[ "$command" == *"--served-model-name $EXPECTED_MODEL"* ]] || return 1
  fi
  return 0
}

acquire_gpu() {
  local gpu endpoint fd
  while true; do
    for gpu in 3 4 5; do
      endpoint=$(gpu_endpoint "$gpu")
      exec {fd}>"$LEASE_ROOT/gpu${gpu}.lock"
      if flock -n "$fd"; then
        if gpu_is_available "$gpu" "$endpoint"; then
          GPU=$gpu
          ENDPOINT=$endpoint
          LEASE_FD=$fd
          printf 'gpu=%s endpoint=%s pid=%s acquired=%s\n' \
            "$GPU" "$ENDPOINT" "$$" "$(timestamp)" > "$OUTPUT_ROOT/gpu_lease.txt"
          set_scheduler_status "gpu_acquired gpu=$GPU endpoint=$ENDPOINT"
          return 0
        fi
        flock -u "$fd"
      fi
      eval "exec ${fd}>&-"
    done
    set_scheduler_status "waiting_for_gpu candidates=3,4,5 poll_seconds=$POLL_SECONDS"
    sleep "$POLL_SECONDS"
  done
}

benchmark_spec() {
  case "$1" in
    H2HMEM)
      BASE_CONFIG="$OFFLINE_ROOT/outputs/PPO/H2HMEM_0918PPO_k4_lambda000_a/config.json"
      CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/H2HMEM_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
      EXPECTED_QA=360
      JUDGE_BENCHMARK=h2hmem
      ;;
    WMA)
      BASE_CONFIG="$OFFLINE_ROOT/outputs/PPO/WMA_0918PPO_k4_lambda000_a/config.json"
      CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/WMA_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
      EXPECTED_QA=440
      JUDGE_BENCHMARK=worldmemarena
      ;;
    *) return 1 ;;
  esac
}

job_spec() {
  case "$1" in
    1) printf '%s\t%s\t%s\n' B1_summary_only summary-only "" ;;
    2) printf '%s\t%s\t%s\n' B2_without_summary ppo summary ;;
    3) printf '%s\t%s\t%s\n' B3_without_dialogue ppo dialogue ;;
    4) printf '%s\t%s\t%s\n' B4_without_caption ppo caption ;;
    5) printf '%s\t%s\t%s\n' B5_without_image ppo image ;;
    6) printf '%s\t%s\t%s\n' B6_without_vp ppo vp ;;
    *) return 1 ;;
  esac
}

prepare_config() {
  local destination=$1 output_dir=$2 disabled=$3
  "$PYTHON" - "$BASE_CONFIG" "$destination" "$output_dir" "$ENDPOINT" "$disabled" <<'PY'
import json, sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
config = json.loads(source.read_text(encoding="utf-8"))
config["output_dir"] = sys.argv[3]
config["model"]["base_url"] = sys.argv[4]
config["reward"]["cost_tradeoff_lambda"] = 0.0
config["ppo"]["skip_invalid_response"] = False
config["evidence"]["disabled_types"] = [sys.argv[5]] if sys.argv[5] else []
if str(config.get("benchmark", "")).lower() == "wma":
    config["excluded_categories"] = []
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
PY
}

validate_eval() {
  local result=$1 disabled=$2 strategy=$3
  "$PYTHON" - "$result" "$EXPECTED_QA" "$disabled" "$strategy" "$JUDGE_BENCHMARK" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
disabled, strategy, benchmark = sys.argv[3:6]
metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (result / "rollouts.jsonl").open(encoding="utf-8") if line.strip()]
assert metrics["count"] == expected == len(rows), (metrics["count"], len(rows), expected)
assert metrics["errors"] == 0
assert len({row["query_id"] for row in rows}) == expected
order = ["summary", "dialogue", "caption", "image", "vp"]
disabled_index = order.index(disabled) if disabled else None
for row in rows:
    assert row["vector_k"] == 5, row["query_id"]
    assert 0 <= row["append_k_actual"] <= 2, row["query_id"]
    assert len(row["retrieval_final_ids"]) <= 7, row["query_id"]
    for action in row["actions"]:
        mask = action["mask"]
        if disabled_index is not None:
            assert mask[disabled_index] == "0", (row["query_id"], disabled, mask)
        if strategy == "summary-only":
            assert mask[1:] == "0000", (row["query_id"], mask)
if benchmark == "worldmemarena":
    assert sum(row["category"] == "MB" for row in rows) == 40
PY
}

validate_judge() {
  local result=$1
  "$PYTHON" - "$result" "$EXPECTED_QA" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
judge = json.loads((result / "llm_judge_metrics.json").read_text(encoding="utf-8"))
assert judge["count"] == expected
assert judge["valid_count"] == expected
assert judge["judge_errors"] == 0
assert judge["coverage"] == 1.0
assert judge["completion"] == 1.0
metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
assert metrics["llm_judge"] == judge["accuracy"]
PY
}

run_job() {
  local benchmark=$1 number=$2 name strategy disabled output config result attempt
  benchmark_spec "$benchmark" || return 1
  IFS=$'\t' read -r name strategy disabled < <(job_spec "$number")
  output="$OUTPUT_ROOT/$benchmark/$name"
  config="$output/run_control/effective_config.json"
  result="$output/eval/test_$strategy"
  mkdir -p "$output/run_control"
  prepare_config "$config" "$output" "$disabled"

  if ! validate_eval "$result" "$disabled" "$strategy" >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      printf 'test_running attempt=%s %s\n' "$attempt" "$(timestamp)" > "$output/run_control/status.txt"
      args=(
        "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py"
        --config "$config" --output-dir "$output" --model-base-url "$ENDPOINT"
        eval --strategy "$strategy" --split test --device cpu
      )
      [[ "$strategy" == ppo ]] && args+=(--checkpoint "$CHECKPOINT")
      if "${args[@]}" 2>&1 | tee -a "$output/test.log"; then
        validate_eval "$result" "$disabled" "$strategy" && break
      fi
      sleep 60
    done
  fi
  validate_eval "$result" "$disabled" "$strategy" || return 1

  if ! validate_judge "$result" >/dev/null 2>&1; then
    for attempt in 1 2 3; do
      printf 'judge_running attempt=%s %s\n' "$attempt" "$(timestamp)" > "$output/run_control/status.txt"
      if "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
        --benchmark "$JUDGE_BENCHMARK" --results "$result/rollouts.jsonl" \
        --out-dir "$result" --key-file "$JUDGE_KEY_FILE" \
        --model openai/gpt-4o-mini --workers 32 --timeout 60 --retries 2 \
        --max-tokens 512 --checkpoint-every 25 --resume \
        2>&1 | tee -a "$result/llm_judge.log"; then
        validate_judge "$result" && break
      fi
      sleep 60
    done
  fi
  validate_judge "$result" || return 1
  printf 'complete %s\n' "$(timestamp)" > "$output/run_control/status.txt"
}

acquire_gpu

for benchmark in H2HMEM WMA; do
  for number in 1 2 3 4 5 6; do
    name=$(job_spec "$number" | cut -f1)
    claim="$OUTPUT_ROOT/claims/${benchmark}_${name}"
    if ! mkdir "$claim" 2>/dev/null; then
      printf '%s skip_claimed benchmark=%s job=%s\n' "$(timestamp)" "$benchmark" "$name" \
        | tee -a "$OUTPUT_ROOT/logs/worker.log"
      continue
    fi
    printf 'pid=%s gpu=%s started=%s\n' "$$" "$GPU" "$(timestamp)" > "$claim/owner.txt"
    set_scheduler_status "running gpu=$GPU benchmark=$benchmark job=$name"
    if run_job "$benchmark" "$number"; then
      status=0
    else
      status=$?
    fi
    printf '%s\n' "$status" > "$claim/exit_code.txt"
    printf '%s finish gpu=%s benchmark=%s job=%s status=%s\n' \
      "$(timestamp)" "$GPU" "$benchmark" "$name" "$status" \
      | tee -a "$OUTPUT_ROOT/logs/worker.log"
    [[ "$status" == 0 ]] || exit "$status"
  done
done

set_scheduler_status "complete gpu=$GPU"
