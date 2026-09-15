#!/usr/bin/env bash
set -uo pipefail

BENCHMARK=${1:?usage: run_0911_cost_reward_ppo.sh memgallery|h2hmem|wma RUN_NAME}
RUN_NAME=${2:?usage: run_0911_cost_reward_ppo.sh memgallery|h2hmem|wma RUN_NAME}
COST_LAMBDA=${3:-0.1}
WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
OUTPUT_DIR="$OFFLINE_ROOT/outputs/PPO/$RUN_NAME"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"
WANDB_PROJECT=hivemem-evidence-policy-v2
WANDB_ENTITY=rhssgvjngfidhfhjd-nanyang-technological-university-singapore
RUN_DATE=$(date +%m%d)

case "$BENCHMARK" in
  memgallery)
    PREFIX=MemGallery
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy.json"
    ENDPOINT=http://127.0.0.1:8013/v1
    JUDGE_BENCHMARK=memgallery
    EXPECTED_TRAIN=940
    EXPECTED_VALIDATION=312
    EXPECTED_TEST=275
    ;;
  h2hmem)
    PREFIX=H2HMEM
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy_h2hmem.json"
    ENDPOINT=http://127.0.0.1:8014/v1
    JUDGE_BENCHMARK=h2hmem
    EXPECTED_TRAIN=1063
    EXPECTED_VALIDATION=363
    EXPECTED_TEST=360
    ;;
  wma)
    PREFIX=WMA
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy_wma.json"
    ENDPOINT=http://127.0.0.1:8015/v1
    JUDGE_BENCHMARK=worldmemarena
    EXPECTED_TRAIN=1265
    EXPECTED_VALIDATION=385
    EXPECTED_TEST=440
    ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

ENDPOINT=${EVIDENCE_POLICY_ENDPOINT:-$ENDPOINT}

if [[ ! "$COST_LAMBDA" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "cost lambda must be a non-negative decimal: $COST_LAMBDA" >&2
  exit 2
fi

if [[ ! "$RUN_NAME" =~ ^${PREFIX}_[0-9]{4}PPO_ ]]; then
  echo "run name $RUN_NAME does not match benchmark/date prefix ${PREFIX}_MMDDPPO_" >&2
  exit 2
fi

BASE_CONFIG="$CONFIG"
CONFIG="$OUTPUT_DIR/run_control/effective_config.json"

mkdir -p "$OUTPUT_DIR/run_control"
exec 9>"$OUTPUT_DIR/run_control/pipeline.lock"
flock 9

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT"
export CUDA_VISIBLE_DEVICES=""

timestamp() {
  date --iso-8601=seconds
}

set_status() {
  mkdir -p "$OUTPUT_DIR/run_control"
  printf '%s %s\n' "$1" "$(timestamp)" | tee "$OUTPUT_DIR/run_control/status.txt"
}

run_logged() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee -a "$log_path"
  return "${PIPESTATUS[0]}"
}

wait_for_endpoint() {
  local model
  while true; do
    model=$(curl -sf --max-time 10 "$ENDPOINT/models" 2>/dev/null | "$PYTHON" -c \
      'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
    if [[ "$model" == "Qwen/Qwen3-VL-4B-Instruct" ]]; then
      return 0
    fi
    printf '%s endpoint_wait endpoint=%s model=%s\n' \
      "$(timestamp)" "$ENDPOINT" "${model:-unavailable}" | tee -a "$OUTPUT_DIR/pipeline.log"
    sleep 60
  done
}

validate_preflight() {
  "$PYTHON" - "$CONFIG" "$BENCHMARK" "$EXPECTED_TRAIN" "$EXPECTED_VALIDATION" "$EXPECTED_TEST" "$COST_LAMBDA" "$OFFLINE_ROOT" <<'PY'
import json, sys
from pathlib import Path

config_path = Path(sys.argv[1])
benchmark = sys.argv[2]
expected = tuple(map(int, sys.argv[3:6]))
expected_lambda = float(sys.argv[6])
root = Path(sys.argv[7])
config = json.loads(config_path.read_text())
assert config["top_k"] == 5
assert config["graph_options"]["mode"] == "append"
assert config["graph_options"]["append_k"] == 2
assert config["ppo"]["epochs"] == 6
assert config["ppo"]["rollout_batch_size"] == 16
assert config["ppo"]["validation_at_start"] is True
assert config["ppo"]["validation_interval_fraction"] == 0.5
assert config["ppo"]["validation_limit"] == 0
reward = config["reward"]
assert reward["cost_enabled"] is True
assert reward["cost_tradeoff_lambda"] == expected_lambda
assert reward["cost_transform"] == "sqrt_incremental"
assert reward["window_size"] == 512
assert reward["min_window_size"] == 128
assert reward["lower_quantile"] == 0.05
assert reward["upper_quantile"] == 0.95
assert reward["initial_range_floor_ratio"] == 0.25
assert reward["range_epsilon"] == 1e-12
assert reward["std_epsilon"] == 1e-8
if benchmark == "wma":
    assert config["excluded_categories"] == []

manifest = json.loads((root / "configs/multimodal_split_manifest.json").read_text())
sources = {
    "memgallery": ("mem_gallery",),
    "h2hmem": ("h2hmem_dyadic", "h2hmem_multiparty"),
    "wma": ("worldmemarena_lifelong",),
}[benchmark]
counts = []
for split in ("train", "val", "test"):
    counts.append(sum(
        dataset["splits"][split]["question_count"]
        for dataset in manifest["datasets"]
        if dataset["data_source"] in sources
    ))
assert tuple(counts) == expected, (counts, expected)
PY
  "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
    --config "$CONFIG" audit-vp > "$OUTPUT_DIR/run_control/vp_audit.json"
  "$PYTHON" - "$OUTPUT_DIR/run_control/vp_audit.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["missing_records"] == 0, report
assert report["missing_crop_files"] == 0, report
PY
  if [[ "$BENCHMARK" == "h2hmem" ]]; then
    "$PYTHON" - "$CONFIG" "$OFFLINE_ROOT" <<'PY'
import json, sys
from pathlib import Path

config = json.load(open(sys.argv[1]))
root = Path(sys.argv[2])
bank = Path(config["memory_bank"])
if not bank.is_absolute():
    bank = (root / bank).resolve()
image_rows = 0
missing_captions = 0
for memories in bank.glob("datasets/*/memories.jsonl"):
    for line in memories.open(encoding="utf-8"):
        if not line.strip():
            continue
        metadata = (json.loads(line).get("metadata") or {})
        if metadata.get("image_paths"):
            image_rows += 1
            if not any(str(value).strip() for value in metadata.get("image_captions") or []):
                missing_captions += 1
assert image_rows > 0, "H2HMEM captioned bank has no image memories"
assert missing_captions == 0, (image_rows, missing_captions)
PY
  fi
}

validate_training() {
  "$PYTHON" - "$OUTPUT_DIR" "$EXPECTED_TRAIN" "$EXPECTED_VALIDATION" <<'PY'
import json, math, sys
from pathlib import Path

root = Path(sys.argv[1])
expected_train = int(sys.argv[2])
expected_validation = int(sys.argv[3])
expected_checkpoints = {f"epoch_{epoch:03d}.pt" for epoch in range(6)}
actual_checkpoints = {path.name for path in (root / "checkpoints").glob("epoch_*.pt")}
assert actual_checkpoints == expected_checkpoints, (actual_checkpoints, expected_checkpoints)
assert (root / "checkpoints" / "initial.pt").is_file()
updates = [json.loads(line) for line in (root / "ppo_metrics.jsonl").open() if line.strip()]
assert len(updates) == math.ceil(expected_train / 16) * 6, len(updates)
assert any(float(row.get("cost_normalizer_active", 0)) == 1 for row in updates)
for epoch in range(6):
    rows = [json.loads(line) for line in (root / "train" / f"epoch_{epoch:03d}_rollouts.jsonl").open() if line.strip()]
    assert len(rows) == expected_train, (epoch, len(rows))
    assert not any(row.get("error") or row.get("cost_error") for row in rows), epoch
    for row in rows:
        all_zero = not row.get("actions") or all(
            action.get("mask") == "00000" for action in row["actions"]
        )
        expected_version = (
            "ppo-empty-evidence-20260911-v1"
            if all_zero
            else "answer-prompts-custom-20260909-v1"
        )
        assert row.get("prompt_version") == expected_version, (
            epoch, row.get("query_id"), row.get("prompt_version"), expected_version
        )
validation_paths = [root / "validation" / "initial_metrics.json"]
for epoch in range(6):
    validation_paths.extend([
        root / "validation" / f"epoch_{epoch:03d}_half_metrics.json",
        root / "validation" / f"epoch_{epoch:03d}_metrics.json",
    ])
assert len(validation_paths) == 13
for path in validation_paths:
    metrics = json.loads(path.read_text())["metrics"]
    assert metrics["count"] == expected_validation, (path, metrics["count"])
    assert metrics["errors"] == 0, path
PY
}

validate_test() {
  "$PYTHON" - "$OUTPUT_DIR/eval/test_ppo" "$EXPECTED_TEST" "$BENCHMARK" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
benchmark = sys.argv[3]
metrics = json.loads((result / "metrics.json").read_text())
rows = [json.loads(line) for line in (result / "rollouts.jsonl").open() if line.strip()]
assert metrics["count"] == expected == len(rows)
assert metrics["errors"] == 0
assert metrics.get("cached_rollouts", 0) == 0
assert len({row["query_id"] for row in rows}) == expected
for row in rows:
    all_zero = not row.get("actions") or all(
        action.get("mask") == "00000" for action in row["actions"]
    )
    expected_version = (
        "ppo-empty-evidence-20260911-v1"
        if all_zero
        else "answer-prompts-custom-20260909-v1"
    )
    assert row.get("prompt_version") == expected_version, (
        row.get("query_id"), row.get("prompt_version"), expected_version
    )
assert not any(row.get("error") for row in rows)
if benchmark == "wma":
    assert sum(row["category"] == "MB" for row in rows) == 40
PY
}

validate_judge() {
  "$PYTHON" - "$OUTPUT_DIR/eval/test_ppo" "$EXPECTED_TEST" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
metrics = json.loads((result / "llm_judge_metrics.json").read_text())
assert metrics["count"] == expected
assert metrics["valid_count"] == expected
assert metrics["judge_errors"] == 0
assert metrics["coverage"] == 1.0
assert metrics["completion"] == 1.0
summary = json.loads((result / "summary.json").read_text())
assert summary["llm_judge"] == metrics["accuracy"]
PY
}

run_train() {
  local resume_args=()
  local attempt latest
  latest=$(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' 2>/dev/null | sort | tail -n 1)
  if [[ -z "$latest" && -f "$OUTPUT_DIR/checkpoints/initial.pt" ]]; then
    latest="$OUTPUT_DIR/checkpoints/initial.pt"
  fi
  [[ -n "$latest" ]] && resume_args=(--resume "$latest")
  for attempt in 1 2 3; do
    wait_for_endpoint
    set_status "train_running attempt=$attempt"
    if run_logged "$OUTPUT_DIR/train.log" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
        --config "$CONFIG" --output-dir "$OUTPUT_DIR" --model-base-url "$ENDPOINT" \
        train --device cpu --wandb --wandb-project "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" --wandb-name "$RUN_NAME" "${resume_args[@]}"; then
      return 0
    fi
    latest=$(find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' 2>/dev/null | sort | tail -n 1)
    if [[ -z "$latest" && -f "$OUTPUT_DIR/checkpoints/initial.pt" ]]; then
      latest="$OUTPUT_DIR/checkpoints/initial.pt"
    fi
    resume_args=()
    [[ -n "$latest" ]] && resume_args=(--resume "$latest")
    printf '%s train_retry attempt=%s resume=%s\n' "$(timestamp)" "$attempt" "${latest:-fresh}" | tee -a "$OUTPUT_DIR/pipeline.log"
    sleep 300
  done
  return 1
}

run_test() {
  local attempt
  for attempt in 1 2 3; do
    wait_for_endpoint
    set_status "test_running attempt=$attempt"
    if run_logged "$OUTPUT_DIR/test.log" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
        --config "$CONFIG" --output-dir "$OUTPUT_DIR" --model-base-url "$ENDPOINT" \
        eval --strategy ppo --split test \
        --checkpoint "$OUTPUT_DIR/checkpoints/epoch_005.pt" --device cpu; then
      return 0
    fi
    sleep 60
  done
  return 1
}

run_judge() {
  local result="$OUTPUT_DIR/eval/test_ppo"
  local attempt
  for attempt in 1 2 3; do
    set_status "judge_running attempt=$attempt"
    if run_logged "$result/llm_judge.log" \
      "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
        --benchmark "$JUDGE_BENCHMARK" --results "$result/rollouts.jsonl" \
        --out-dir "$result" --key-file "$JUDGE_KEY_FILE" \
        --model openai/gpt-4o-mini --workers 32 --timeout 60 --retries 2 \
        --max-tokens 512 --checkpoint-every 25 --resume; then
      return 0
    fi
    sleep 60
  done
  return 1
}

upload_wandb() {
  local control="$OUTPUT_DIR/run_control/wandb.json"
  local run_id
  [[ -s "$control" ]] || return 1
  run_id=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("run_id", ""))' "$control")
  [[ -n "$run_id" ]] || return 1
  run_logged "$OUTPUT_DIR/wandb_final_upload.log" \
    "$PYTHON" "$OFFLINE_ROOT/scripts/upload_evidence_policy_wandb.py" \
      --run-dir "$OUTPUT_DIR" --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY" \
      --name "$RUN_NAME" --run-id "$run_id" --tag cost-reward --tag "$RUN_DATE" \
      --tag "lambda-$COST_LAMBDA" \
      --charts-only --skip-workspace
}

cp "$BASE_CONFIG" "$OUTPUT_DIR/run_control/input_config.json"
"$PYTHON" - "$BASE_CONFIG" "$CONFIG" "$COST_LAMBDA" <<'PY'
import json, sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
config = json.loads(source.read_text(encoding="utf-8"))
config["reward"]["cost_tradeoff_lambda"] = float(sys.argv[3])
split_manifest_value = config.get("split_manifest")
if split_manifest_value:
    split_manifest = Path(split_manifest_value)
    if not split_manifest.is_absolute():
        config["split_manifest"] = str((source.parent / split_manifest).resolve())
destination.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
sha256sum \
  "$CONFIG" \
  "$WORKSPACE/answer_prompts.py" \
  "$OFFLINE_ROOT/src/benchmarks/memgallery_harness/runner/prompts.py" \
  "$OFFLINE_ROOT/src/benchmarks/h2hmem_harness/prompts.py" \
  "$OFFLINE_ROOT/src/benchmarks/wma_harness/runner/prompts.py" \
  "$OFFLINE_ROOT/src/benchmarks/answer_response.py" \
  "$OFFLINE_ROOT/src/benchmarks/memgallery_harness/runner/answer_client.py" \
  "$OFFLINE_ROOT/scripts/evidence_policy.py" \
  "$OFFLINE_ROOT/src/evidence_policy/ppo.py" \
  "$OFFLINE_ROOT/src/evidence_policy/rollout.py" \
  > "$OUTPUT_DIR/run_control/source_sha256.txt"

if ! validate_preflight; then set_status preflight_failed; exit 1; fi
if validate_training >/dev/null 2>&1; then
  set_status training_already_complete
else
  if ! run_train; then set_status train_failed; exit 1; fi
  if ! validate_training; then set_status training_acceptance_failed; exit 1; fi
fi
if validate_test >/dev/null 2>&1; then
  set_status test_already_complete
else
  if ! run_test; then set_status test_failed; exit 1; fi
  if ! validate_test; then set_status test_acceptance_failed; exit 1; fi
fi
if validate_judge >/dev/null 2>&1; then
  set_status judge_already_complete
else
  if ! run_judge; then set_status judge_failed; exit 1; fi
  if ! validate_judge; then set_status judge_acceptance_failed; exit 1; fi
fi
if ! upload_wandb; then
  printf '%s wandb_final_upload_warning local_results_authoritative\n' "$(timestamp)" | tee -a "$OUTPUT_DIR/pipeline.log"
fi
set_status complete
