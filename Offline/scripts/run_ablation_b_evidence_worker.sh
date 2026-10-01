#!/usr/bin/env bash
set -uo pipefail

MODE=${1:?usage: run_ablation_b_evidence_worker.sh smoke|worker gpu1|gpu3|gpu4|gpu5}
GPU_LABEL=${2:?usage: run_ablation_b_evidence_worker.sh smoke|worker gpu1|gpu3|gpu4|gpu5}

WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
EXPERIMENT_ROOT="$OFFLINE_ROOT/outputs/Ablation/B_evidence_20260922"
BASE_CONFIG="$OFFLINE_ROOT/outputs/PPO/MemGallery_0918PPO_k4_lambda000_a/config.json"
BASE_CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/MemGallery_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"

case "$GPU_LABEL" in
  gpu1) ENDPOINT=http://127.0.0.1:8016/v1 ;;
  gpu3) ENDPOINT=http://127.0.0.1:8013/v1 ;;
  gpu4) ENDPOINT=http://127.0.0.1:8014/v1 ;;
  gpu5) ENDPOINT=http://127.0.0.1:8015/v1 ;;
  *) echo "unsupported GPU label: $GPU_LABEL" >&2; exit 2 ;;
esac

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$EXPERIMENT_ROOT"/{claims,logs,smoke}

timestamp() {
  date --iso-8601=seconds
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
  local destination=$1
  local output_dir=$2
  local disabled_type=$3
  "$PYTHON" - "$BASE_CONFIG" "$destination" "$output_dir" "$ENDPOINT" "$disabled_type" <<'PY'
import json, sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
config = json.loads(source.read_text(encoding="utf-8"))
config["output_dir"] = sys.argv[3]
config["model"]["base_url"] = sys.argv[4]
config["reward"]["cost_tradeoff_lambda"] = 0.0
config["evidence"]["disabled_types"] = [sys.argv[5]] if sys.argv[5] else []
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

validate_eval() {
  local result_dir=$1
  local expected_count=$2
  local disabled_type=$3
  local strategy=$4
  "$PYTHON" - "$result_dir" "$expected_count" "$disabled_type" "$strategy" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
disabled = sys.argv[3]
strategy = sys.argv[4]
metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in (result / "rollouts.jsonl").open(encoding="utf-8") if line.strip()]
assert metrics["count"] == expected == len(rows), (metrics["count"], len(rows), expected)
assert metrics["errors"] == 0, metrics["errors"]
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
PY
}

run_smoke() {
  local number=$1
  local name strategy disabled
  IFS=$'\t' read -r name strategy disabled < <(job_spec "$number")
  local output="$EXPERIMENT_ROOT/smoke/$name"
  local config="$output/effective_config.json"
  local result="$output/eval/test_$strategy"
  if [[ -f "$result/metrics.json" ]]; then
    validate_eval "$result" 5 "$disabled" "$strategy" && return 0
  fi
  prepare_config "$config" "$output" "$disabled"
  local args=(
    "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py"
    --config "$config" --output-dir "$output" --model-base-url "$ENDPOINT"
    eval --strategy "$strategy" --split test --device cpu --limit 5
  )
  if [[ "$strategy" == ppo ]]; then
    args+=(--checkpoint "$BASE_CHECKPOINT")
  fi
  "${args[@]}" 2>&1 | tee -a "$EXPERIMENT_ROOT/logs/smoke_${name}_${GPU_LABEL}.log"
  validate_eval "$result" 5 "$disabled" "$strategy"
}

run_summary_full() {
  local name=B1_summary_only
  local output="$EXPERIMENT_ROOT/$name"
  local config="$output/run_control/effective_config.json"
  local result="$output/eval/test_summary-only"
  mkdir -p "$output/run_control"
  prepare_config "$config" "$output" ""
  printf 'eval_running %s\n' "$(timestamp)" > "$output/run_control/status.txt"
  if ! validate_eval "$result" 275 "" summary-only >/dev/null 2>&1; then
    "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
      --config "$config" --output-dir "$output" --model-base-url "$ENDPOINT" \
      eval --strategy summary-only --split test --device cpu \
      2>&1 | tee -a "$output/test.log" || return 1
  fi
  validate_eval "$result" 275 "" summary-only || return 1
  printf 'judge_running %s\n' "$(timestamp)" > "$output/run_control/status.txt"
  "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
    --benchmark memgallery --results "$result/rollouts.jsonl" --out-dir "$result" \
    --key-file "$JUDGE_KEY_FILE" --model openai/gpt-4o-mini --workers 32 \
    --timeout 60 --retries 2 --max-tokens 512 --checkpoint-every 25 --resume \
    2>&1 | tee -a "$result/llm_judge.log" || return 1
  "$PYTHON" - "$result" <<'PY'
import json, sys
from pathlib import Path

result = Path(sys.argv[1])
judge = json.loads((result / "llm_judge_metrics.json").read_text(encoding="utf-8"))
assert judge["count"] == 275
assert judge["valid_count"] == 275
assert judge["judge_errors"] == 0
assert judge["coverage"] == 1.0
PY
  [[ $? -eq 0 ]] || return 1
  printf 'complete %s\n' "$(timestamp)" > "$output/run_control/status.txt"
}

run_ppo_full() {
  local number=$1
  local name strategy disabled
  IFS=$'\t' read -r name strategy disabled < <(job_spec "$number")
  local run_name="MemGallery_0922PPO_ablation${name}_a"
  EVIDENCE_POLICY_ENDPOINT="$ENDPOINT" \
  EVIDENCE_POLICY_BASE_CONFIG="$BASE_CONFIG" \
  EVIDENCE_POLICY_OUTPUT_ROOT="$EXPERIMENT_ROOT" \
  EVIDENCE_POLICY_DISABLED_TYPE="$disabled" \
    "$OFFLINE_ROOT/scripts/run_0911_cost_reward_ppo.sh" memgallery "$run_name" 0
}

if [[ "$MODE" == smoke ]]; then
  case "$GPU_LABEL" in
    gpu1) jobs=(1 5) ;;
    gpu3) jobs=(2 6) ;;
    gpu4) jobs=(3) ;;
    gpu5) jobs=(4) ;;
  esac
  for number in "${jobs[@]}"; do
    run_smoke "$number" || exit 1
  done
  exit 0
fi

if [[ "$MODE" != worker ]]; then
  echo "unsupported mode: $MODE" >&2
  exit 2
fi

if [[ "$GPU_LABEL" == gpu1 ]]; then
  jobs=(1 2 3 4 5 6)
else
  jobs=(2 3 4 5 6)
fi

while true; do
  claimed=""
  for number in "${jobs[@]}"; do
    claim="$EXPERIMENT_ROOT/claims/job${number}"
    if mkdir "$claim" 2>/dev/null; then
      printf '%s %s\n' "$GPU_LABEL" "$(timestamp)" > "$claim/owner.txt"
      claimed=$number
      break
    fi
  done
  [[ -n "$claimed" ]] || break
  name=$(job_spec "$claimed" | cut -f1)
  printf '%s start job=%s name=%s endpoint=%s\n' \
    "$(timestamp)" "$claimed" "$name" "$ENDPOINT" \
    | tee -a "$EXPERIMENT_ROOT/logs/worker_${GPU_LABEL}.log"
  if [[ "$claimed" == 1 ]]; then
    run_summary_full
  else
    run_ppo_full "$claimed"
  fi
  status=$?
  printf '%s finish job=%s name=%s status=%s\n' \
    "$(timestamp)" "$claimed" "$name" "$status" \
    | tee -a "$EXPERIMENT_ROOT/logs/worker_${GPU_LABEL}.log"
  printf '%s\n' "$status" > "$EXPERIMENT_ROOT/claims/job${claimed}/exit_code.txt"
done

printf '%s queue_complete\n' "$(timestamp)" \
  | tee -a "$EXPERIMENT_ROOT/logs/worker_${GPU_LABEL}.log"
