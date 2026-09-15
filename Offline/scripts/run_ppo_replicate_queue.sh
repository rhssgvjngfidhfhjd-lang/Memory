#!/usr/bin/env bash
set -uo pipefail

BENCHMARK=${1:?usage: run_ppo_replicate_queue.sh h2hmem|wma}
WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
CONTROL_ROOT="$OFFLINE_ROOT/outputs/PPO/_0909PPO_repeats_control"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"
WANDB_PROJECT=hivemem-evidence-policy-v2
WANDB_ENTITY=rhssgvjngfidhfhjd-nanyang-technological-university-singapore
EXPECTED_SOURCE_BUNDLE=895a5d7abcabae1304ecfdfaad2d024b5b914b53333b123b583989d16267f15a
EXPECTED_JUDGE_PROTOCOL_SHA=c83ddf6f54c2140af045a14f999713c664958bfe52e488b9c92ead952bf3a05c
EXPECTED_JUDGE_PROTOCOL_SNAPSHOT=evaluation_protocol_bundle@2026-09-09-wma-context-fix

case "$BENCHMARK" in
  h2hmem)
    PREFIX=H2HMEM
    ENDPOINT=http://127.0.0.1:8014/v1
    JUDGE_BENCHMARK=h2hmem
    EXPECTED_TRAIN=1063
    EXPECTED_VALIDATION=363
    EXPECTED_TEST=360
    EXPECTED_PROMPT_SHA=4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f
    ;;
  wma)
    PREFIX=WMA
    ENDPOINT=http://127.0.0.1:8015/v1
    JUDGE_BENCHMARK=worldmemarena
    EXPECTED_TRAIN=1265
    EXPECTED_VALIDATION=385
    EXPECTED_TEST=440
    EXPECTED_PROMPT_SHA=6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250
    ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$CONTROL_ROOT/logs"
QUEUE_LOG="$CONTROL_ROOT/logs/${BENCHMARK}_queue.log"
QUEUE_STATUS="$CONTROL_ROOT/${BENCHMARK}_queue.status"

timestamp() {
  date --iso-8601=seconds
}

set_queue_status() {
  printf '%s %s\n' "$1" "$(timestamp)" | tee "$QUEUE_STATUS"
}

set_run_status() {
  local output_dir=$1
  local stage=$2
  printf '%s %s\n' "$stage" "$(timestamp)" | tee "$output_dir/run_control/status.txt"
}

run_logged() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee -a "$log_path"
  return "${PIPESTATUS[0]}"
}

source_bundle_sha() {
  (
    cd "$WORKSPACE" || exit 1
    sha256sum \
      answer_prompts.py \
      Offline/src/benchmarks/h2hmem_harness/prompts.py \
      Offline/src/benchmarks/wma_harness/runner/prompts.py \
      Offline/scripts/evidence_policy.py \
      Offline/src/evidence_policy/rollout.py | sha256sum | awk '{print $1}'
  )
}

prompt_sha() {
  "$PYTHON" - "$BENCHMARK" <<'PY'
import sys
if sys.argv[1] == "h2hmem":
    from benchmarks.h2hmem_harness import prompts
else:
    from benchmarks.wma_harness.runner import prompts
print(prompts.prompt_sha256())
PY
}

wait_for_endpoint() {
  while true; do
    model=$(curl -sf --max-time 10 "$ENDPOINT/models" 2>/dev/null | "$PYTHON" -c \
      'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
    if [[ "$model" == "Qwen/Qwen3-VL-4B-Instruct" ]]; then
      return 0
    fi
    printf '%s endpoint_wait endpoint=%s model=%s\n' "$(timestamp)" "$ENDPOINT" "${model:-unavailable}" | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

validate_training_artifacts() {
  "$PYTHON" - "$1" "$EXPECTED_TRAIN" "$EXPECTED_VALIDATION" "$EXPECTED_PROMPT_SHA" <<'PY'
import json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
expected_train = int(sys.argv[2])
expected_validation = int(sys.argv[3])
expected_prompt_sha = sys.argv[4]
expected_checkpoints = {f"epoch_{epoch:03d}.pt" for epoch in range(6)}
actual_checkpoints = {path.name for path in (root / "checkpoints").glob("epoch_*.pt")}
assert actual_checkpoints == expected_checkpoints, (actual_checkpoints, expected_checkpoints)
assert (root / "checkpoints" / "initial.pt").is_file()

updates = [json.loads(line) for line in (root / "ppo_metrics.jsonl").open() if line.strip()]
assert len(updates) == math.ceil(expected_train / 16) * 6, len(updates)

for epoch in range(6):
    path = root / "train" / f"epoch_{epoch:03d}_rollouts.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == expected_train, (path, len(rows))
    assert not any(row.get("error") for row in rows), path
    assert {row.get("prompt_sha256") for row in rows} == {expected_prompt_sha}, path

validation_paths = [root / "validation" / "initial_metrics.json"]
for epoch in range(6):
    validation_paths.extend(
        [
            root / "validation" / f"epoch_{epoch:03d}_half_metrics.json",
            root / "validation" / f"epoch_{epoch:03d}_metrics.json",
        ]
    )
for path in validation_paths:
    event = json.loads(path.read_text())
    metrics = event["metrics"]
    assert metrics["count"] == expected_validation, (path, metrics["count"])
    assert metrics["errors"] == 0, path
PY
}

validate_test_artifacts() {
  "$PYTHON" - "$1" "$EXPECTED_TEST" "$EXPECTED_PROMPT_SHA" "$BENCHMARK" <<'PY'
import json, re, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
prompt_sha = sys.argv[3]
benchmark = sys.argv[4]
metrics = json.loads((result / "metrics.json").read_text())
rows = [json.loads(line) for line in (result / "rollouts.jsonl").open() if line.strip()]
assert metrics["count"] == expected == len(rows)
assert metrics["errors"] == 0
assert metrics["cached_rollouts"] == 0
assert len({row["query_id"] for row in rows}) == expected
assert {row.get("prompt_sha256") for row in rows} == {prompt_sha}
assert all(re.fullmatch(r"\s*<answer>[^<]+</answer>\s*", row["answer_raw_response"], re.S) for row in rows)
if benchmark == "wma":
    assert sum(row["category"] == "MB" for row in rows) == 40
PY
}

validate_judge_artifacts() {
  "$PYTHON" - "$1" "$EXPECTED_TEST" "$EXPECTED_JUDGE_PROTOCOL_SHA" "$EXPECTED_JUDGE_PROTOCOL_SNAPSHOT" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
protocol_sha = sys.argv[3]
protocol_snapshot = sys.argv[4]
metrics = json.loads((result / "llm_judge_metrics.json").read_text())
assert metrics["count"] == expected
assert metrics["valid_count"] == expected
assert metrics["judge_errors"] == 0
assert metrics["coverage"] == 1.0
assert metrics["completion"] == 1.0
assert metrics["protocol_snapshot"] == protocol_snapshot
checkpoint = json.loads((result / "llm_judge_checkpoint.json").read_text())
assert checkpoint["signature"]["judge"]["protocol_sha256"] == protocol_sha
summary = json.loads((result / "summary.json").read_text())
assert summary.get("llm_judge") == metrics.get("accuracy")
PY
}

run_train() {
  local run_name=$1
  local output_dir=$2
  local config=$3
  local log_path="$output_dir/train.log"
  local resume_args=()
  local attempt
  for attempt in 1 2 3; do
    wait_for_endpoint
    set_run_status "$output_dir" "train_running attempt=$attempt"
    if run_logged "$log_path" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
        --config "$config" \
        --output-dir "$output_dir" \
        --model-base-url "$ENDPOINT" \
        train --device cpu \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --wandb-name "$run_name" \
        "${resume_args[@]}"; then
      return 0
    fi
    latest=$(find "$output_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' 2>/dev/null | sort | tail -n 1)
    if [[ -z "$latest" && -f "$output_dir/checkpoints/initial.pt" ]]; then
      latest="$output_dir/checkpoints/initial.pt"
    fi
    resume_args=()
    if [[ -n "$latest" ]]; then
      resume_args=(--resume "$latest")
    fi
    printf '%s train_retry run=%s attempt=%s resume=%s\n' "$(timestamp)" "$run_name" "$attempt" "${latest:-fresh}" | tee -a "$QUEUE_LOG"
    sleep 300
  done
  return 1
}

run_test() {
  local output_dir=$1
  local config=$2
  local checkpoint="$output_dir/checkpoints/epoch_005.pt"
  local log_path="$output_dir/test.log"
  local attempt
  for attempt in 1 2 3; do
    wait_for_endpoint
    set_run_status "$output_dir" "test_running attempt=$attempt"
    if run_logged "$log_path" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
        --config "$config" \
        --output-dir "$output_dir" \
        --model-base-url "$ENDPOINT" \
        eval --strategy ppo --split test \
        --checkpoint "$checkpoint" --device cpu; then
      return 0
    fi
    sleep 60
  done
  return 1
}

run_judge() {
  local output_dir=$1
  local result="$output_dir/eval/test_ppo"
  local attempt
  for attempt in 1 2 3; do
    set_run_status "$output_dir" "judge_running attempt=$attempt"
    if run_logged "$result/llm_judge.log" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
        --benchmark "$JUDGE_BENCHMARK" \
        --results "$result/rollouts.jsonl" \
        --out-dir "$result" \
        --key-file "$JUDGE_KEY_FILE" \
        --model openai/gpt-4o-mini \
        --workers 32 --timeout 60 --retries 2 --max-tokens 512 \
        --checkpoint-every 25 --resume; then
      return 0
    fi
    sleep 60
  done
  return 1
}

upload_wandb_charts() {
  local output_dir=$1
  local run_name=$2
  local control="$output_dir/run_control/wandb.json"
  if [[ ! -s "$control" ]]; then
    printf '%s wandb_upload_skipped run=%s reason=missing_control\n' "$(timestamp)" "$run_name" | tee -a "$QUEUE_LOG"
    return 1
  fi
  local run_id
  run_id=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_id", ""))' "$control")
  if [[ -z "$run_id" ]]; then
    printf '%s wandb_upload_skipped run=%s reason=missing_run_id\n' "$(timestamp)" "$run_name" | tee -a "$QUEUE_LOG"
    return 1
  fi
  run_logged "$output_dir/wandb_final_upload.log" \
    "$PYTHON" "$OFFLINE_ROOT/scripts/upload_evidence_policy_wandb.py" \
      --run-dir "$output_dir" \
      --project "$WANDB_PROJECT" \
      --entity "$WANDB_ENTITY" \
      --name "$run_name" \
      --run-id "$run_id" \
      --tag new-answer-prompt \
      --tag three-seed-replicate \
      --charts-only --skip-workspace
}

set_queue_status running
for spec in a:42 b:43 c:44; do
  suffix=${spec%%:*}
  seed=${spec##*:}
  run_name="${PREFIX}_0909PPO_${suffix}"
  output_dir="$OFFLINE_ROOT/outputs/PPO/$run_name"
  config="$CONTROL_ROOT/configs/$run_name.json"

  if [[ -f "$output_dir/run_control/status.txt" ]] && grep -q '^complete ' "$output_dir/run_control/status.txt"; then
    printf '%s skip_complete run=%s\n' "$(timestamp)" "$run_name" | tee -a "$QUEUE_LOG"
    continue
  fi
  if [[ ! -s "$config" ]]; then
    printf '%s missing_config run=%s config=%s\n' "$(timestamp)" "$run_name" "$config" | tee -a "$QUEUE_LOG"
    set_queue_status failed
    exit 1
  fi
  if [[ "$(source_bundle_sha)" != "$EXPECTED_SOURCE_BUNDLE" || "$(prompt_sha)" != "$EXPECTED_PROMPT_SHA" ]]; then
    printf '%s source_changed run=%s; refusing mixed-code replicate\n' "$(timestamp)" "$run_name" | tee -a "$QUEUE_LOG"
    set_queue_status source_changed
    exit 1
  fi

  mkdir -p "$output_dir/run_control"
  cp "$config" "$output_dir/run_control/input_config.json"
  printf '%s run_start name=%s seed=%s\n' "$(timestamp)" "$run_name" "$seed" | tee -a "$QUEUE_LOG"

  if ! run_train "$run_name" "$output_dir" "$config"; then
    set_run_status "$output_dir" train_failed
    set_queue_status failed
    exit 1
  fi
  if ! validate_training_artifacts "$output_dir"; then
    set_run_status "$output_dir" training_acceptance_failed
    set_queue_status failed
    exit 1
  fi
  if ! run_test "$output_dir" "$config"; then
    set_run_status "$output_dir" test_failed
    set_queue_status failed
    exit 1
  fi
  if ! validate_test_artifacts "$output_dir/eval/test_ppo"; then
    set_run_status "$output_dir" test_acceptance_failed
    set_queue_status failed
    exit 1
  fi
  if ! run_judge "$output_dir"; then
    set_run_status "$output_dir" judge_failed
    set_queue_status failed
    exit 1
  fi
  if ! validate_judge_artifacts "$output_dir/eval/test_ppo"; then
    set_run_status "$output_dir" judge_acceptance_failed
    set_queue_status failed
    exit 1
  fi

  if ! upload_wandb_charts "$output_dir" "$run_name"; then
    printf '%s wandb_final_upload_warning run=%s local_results_remain_authoritative\n' "$(timestamp)" "$run_name" | tee -a "$QUEUE_LOG"
  fi
  set_run_status "$output_dir" complete
  printf '%s run_complete name=%s seed=%s\n' "$(timestamp)" "$run_name" "$seed" | tee -a "$QUEUE_LOG"
done
set_queue_status complete
