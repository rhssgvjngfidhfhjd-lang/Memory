#!/usr/bin/env bash
set -euo pipefail

BENCHMARK=${1:?usage: run_a1_d_k4_lambda0_test_queue.sh memgallery|h2hmem|wma [run_tag] [endpoint]}
RUN_TAG=${2:-A1_D_k4_lambda0_20260922}
ENDPOINT_OVERRIDE=${3:-}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "run_tag contains unsupported characters: $RUN_TAG" >&2
  exit 2
fi
WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
EXPERIMENT_ROOT="$OFFLINE_ROOT/outputs/Ablation/$RUN_TAG"
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"

case "$BENCHMARK" in
  memgallery)
    LABEL=MemGallery
    CONFIG="$OFFLINE_ROOT/outputs/PPO/MemGallery_0918PPO_k4_lambda000_a/config.json"
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/MemGallery_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
    ENDPOINT=http://127.0.0.1:8013/v1
    EXPECTED_QA=275
    JUDGE_BENCHMARK=memgallery
    ;;
  h2hmem)
    LABEL=H2HMEM
    CONFIG="$OFFLINE_ROOT/outputs/PPO/H2HMEM_0918PPO_k4_lambda000_a/config.json"
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/H2HMEM_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
    ENDPOINT=http://127.0.0.1:8014/v1
    EXPECTED_QA=360
    JUDGE_BENCHMARK=h2hmem
    ;;
  wma)
    LABEL=WMA
    CONFIG="$OFFLINE_ROOT/outputs/PPO/WMA_0918PPO_k4_lambda000_a/config.json"
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/WMA_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt"
    ENDPOINT=http://127.0.0.1:8015/v1
    EXPECTED_QA=440
    JUDGE_BENCHMARK=worldmemarena
    ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

if [[ -n "$ENDPOINT_OVERRIDE" ]]; then
  ENDPOINT=${ENDPOINT_OVERRIDE%/}
fi

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT:$WORKSPACE"
export CUDA_VISIBLE_DEVICES=""
BENCHMARK_ROOT="$EXPERIMENT_ROOT/$LABEL"
LOG_DIR="$EXPERIMENT_ROOT/logs"
QUEUE_LOG="$LOG_DIR/${BENCHMARK}.log"
QUEUE_STATUS="$LOG_DIR/${BENCHMARK}.status"
mkdir -p "$LOG_DIR"

timestamp() { date --iso-8601=seconds; }
status() { printf '%s %s\n' "$1" "$(timestamp)" | tee "$QUEUE_STATUS"; }

run_logged() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee -a "$log_path"
  return "${PIPESTATUS[0]}"
}

preflight() {
  [[ -f "$CONFIG" && -f "$CHECKPOINT" && -s "$JUDGE_KEY_FILE" ]]
  local model
  model=$(curl -sf --max-time 10 "$ENDPOINT/models" | "$PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  [[ "$model" == Qwen/Qwen3-VL-4B-Instruct ]]
  "$PYTHON" - "$CONFIG" <<'PY'
import json, math, sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
assert int(config["top_k"]) == 5
assert config["retrieval_mode"] == "graph_append"
assert int(config["graph_options"]["degree_cap"]) == 4
assert int(config["graph_options"]["append_k"]) == 2
memory = Path(config["memory_bank"])
edges = list(memory.glob("datasets/*/reports/edges.json"))
assert edges, memory
for path in edges:
    report = json.loads(path.read_text())
    assert report["schema_version"] == 2, path
    assert report["degree_cap"] == 4, path
PY
}

validate_result() {
  local result_dir=$1 expected=$2 mode=$3 vector_k=$4 append_k=$5
  "$PYTHON" - "$result_dir" "$expected" "$mode" "$vector_k" "$append_k" "$BENCHMARK" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
expected, mode, vector_k, append_k = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
benchmark = sys.argv[6]
metrics = json.loads((root / "metrics.json").read_text())
rows = [json.loads(line) for line in (root / "rollouts.jsonl").open() if line.strip()]
assert metrics["count"] == expected == len(rows), (metrics.get("count"), len(rows), expected)
assert metrics["errors"] == 0
assert len({row["manifest_question_id"] for row in rows}) == expected
for row in rows:
    hits = row["retrieval_top_k"]
    vias = [hit["via"] for hit in hits]
    assert row["retrieval_mode"] == mode
    assert row["vector_k"] == vector_k
    assert vias[:vector_k] == ["vector"] * vector_k, vias
    if mode == "vector":
        assert len(hits) == vector_k == 7 and set(vias) == {"vector"}
    elif mode == "random_append":
        assert len(hits) == 7 and vias[vector_k:] == ["random"] * append_k
        assert row["retrieval_seed"] is not None
    else:
        assert all(via == "graph" for via in vias[vector_k:])
        assert len(hits) <= vector_k + append_k
    assert len({hit["memory_id"] for hit in hits}) == len(hits)
if benchmark == "wma":
    assert sum(row["category"] == "MB" for row in rows) == (40 if expected == 440 else sum(row["category"] == "MB" for row in rows))
PY
}

run_eval() {
  local output_dir=$1 limit=$2 mode=$3 vector_k=$4 append_k=$5 seed=$6
  local args=(
    "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py"
    --config "$CONFIG" --output-dir "$output_dir" --model-base-url "$ENDPOINT"
    --retrieval-mode "$mode" --top-k "$vector_k"
  )
  if [[ "$mode" != vector ]]; then
    args+=(--append-k "$append_k")
  fi
  if [[ "$mode" == random_append ]]; then
    args+=(--retrieval-seed "$seed")
  fi
  args+=(eval --strategy ppo --split test --checkpoint "$CHECKPOINT" --device cpu)
  if (( limit > 0 )); then
    args+=(--limit "$limit")
  fi
  mkdir -p "$output_dir"
  run_logged "$output_dir/qa.log" "${args[@]}"
}

run_judge() {
  local result_dir=$1
  run_logged "$result_dir/llm_judge.log" \
    "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
      --benchmark "$JUDGE_BENCHMARK" --results "$result_dir/rollouts.jsonl" \
      --out-dir "$result_dir" --key-file "$JUDGE_KEY_FILE" \
      --model openai/gpt-4o-mini --workers 32 --timeout 60 --retries 2 \
      --max-tokens 512 --checkpoint-every 25 --resume
  "$PYTHON" - "$result_dir" "$EXPECTED_QA" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
metrics = json.loads((root / "llm_judge_metrics.json").read_text())
assert metrics["count"] == metrics["valid_count"] == expected
assert metrics["judge_errors"] == 0
assert metrics["coverage"] == metrics["completion"] == 1.0
PY
}

SPECS=(
  'top7:vector:7:0:42'
  'random5plus2_seed42:random_append:5:2:42'
  'graph6plus1:graph_append:6:1:42'
  'graph4plus3:graph_append:4:3:42'
  'graph3plus4:graph_append:3:4:42'
)

status preflight
preflight

status smoke_running
for spec in "${SPECS[@]}"; do
  IFS=: read -r name mode vector_k append_k seed <<< "$spec"
  output_dir="$BENCHMARK_ROOT/smoke/$name"
  [[ ! -e "$output_dir" ]] || { echo "refusing to overwrite $output_dir" >&2; exit 1; }
  run_eval "$output_dir" 5 "$mode" "$vector_k" "$append_k" "$seed"
  validate_result "$output_dir/eval/test_ppo" 5 "$mode" "$vector_k" "$append_k"
done

status full_test_running
for spec in "${SPECS[@]}"; do
  IFS=: read -r name mode vector_k append_k seed <<< "$spec"
  output_dir="$BENCHMARK_ROOT/runs/$name"
  [[ ! -e "$output_dir" ]] || { echo "refusing to overwrite $output_dir" >&2; exit 1; }
  printf '%s run_start benchmark=%s run=%s\n' "$(timestamp)" "$BENCHMARK" "$name" | tee -a "$QUEUE_LOG"
  run_eval "$output_dir" 0 "$mode" "$vector_k" "$append_k" "$seed"
  validate_result "$output_dir/eval/test_ppo" "$EXPECTED_QA" "$mode" "$vector_k" "$append_k"
  run_judge "$output_dir/eval/test_ppo"
  printf '%s run_complete benchmark=%s run=%s\n' "$(timestamp)" "$BENCHMARK" "$name" | tee -a "$QUEUE_LOG"
done

status complete
