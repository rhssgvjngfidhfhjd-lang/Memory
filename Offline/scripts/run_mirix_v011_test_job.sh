#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Mem-Gallery|H2HMEM|WorldMemArena> <vllm-port>" >&2
  exit 2
fi

benchmark=$1
port=$2
offline_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$offline_root/.." && pwd)
python_bin=${PIPELINE_PYTHON:-/data/haozhen/miniconda3/envs/pipeline_repro/bin/python}
run_id=0913_mirix_v011_native_test
full_manifest=$offline_root/configs/multimodal_split_manifest.json
smoke_manifest=$offline_root/outputs/_runs/$run_id/_preflight/smoke_split_manifest.json
embedding_url=http://127.0.0.1:8001/v1
answer_url=http://127.0.0.1:${port}/v1

case "$benchmark" in
  Mem-Gallery)
    module=benchmarks.memgallery_harness.eval_memgallery
    data_dir=$workspace_root/Mem-Gallery/benchmark/data
    judge_benchmark=memgallery
    expected=275
    extra_args=(--all-datasets)
    ;;
  H2HMEM)
    module=benchmarks.h2hmem_harness.eval_h2hmem
    data_dir=$workspace_root/H2HMEM-main/dataset
    judge_benchmark=h2hmem
    expected=360
    extra_args=()
    ;;
  WorldMemArena)
    module=benchmarks.wma_harness.eval_wma
    data_dir=$workspace_root/WorldMemArena/WorldMemArena/lifelong
    judge_benchmark=worldmemarena
    expected=440
    extra_args=()
    ;;
  *)
    echo "unsupported benchmark: $benchmark" >&2
    exit 2
    ;;
esac

smoke_dir=$offline_root/outputs/_runs/$run_id/_smoke/$benchmark/MIRIX
result_dir=$offline_root/outputs/$benchmark/MIRIX/$run_id
mkdir -p "$smoke_dir" "$result_dir"
pipeline_log=$result_dir/pipeline.log
trap 'rc=$?; printf "%s EXIT %s\n" "$(date --iso-8601=seconds)" "$rc" >> "$pipeline_log"' EXIT

export PYTHONPATH=$offline_root/src
export PYTHONUNBUFFERED=1

common=(
  --baseline MIRIX
  --executor-base-url "$answer_url"
  --executor-model Qwen/Qwen3-VL-4B-Instruct
  --executor-max-tokens 2048
  --mirix-skip-failed-build-points
  --mirix-max-consecutive-failed-build-points 10
  --answer-base-url "$answer_url"
  --answer-model Qwen/Qwen3-VL-4B-Instruct
  --answer-temperature 0
  --num-predict 512
  --embedding-base-url "$embedding_url"
  --embedding-model Qwen/Qwen3-VL-Embedding-2B
  --embedding-dim 2048
  --top-k 7
  --request-timeout 180
  --retries 2
  --efficiency-config "$offline_root/configs/model_efficiency.json"
  --checkpoint-every 10
  --data-dir "$data_dir"
)

printf "%s SMOKE_START benchmark=%s port=%s\n" "$(date --iso-8601=seconds)" "$benchmark" "$port" >> "$pipeline_log"
"$python_bin" -u -m "$module" \
  "${common[@]}" "${extra_args[@]}" \
  --result-dir "$smoke_dir" \
  --baseline-state-dir "$smoke_dir/memory/datasets" \
  --sample-concurrency 1 \
  --answer-concurrency 1 \
  --split-manifest "$smoke_manifest" \
  --split test \
  --resume > "$smoke_dir/run.log" 2>&1

printf "%s FORMAL_START benchmark=%s expected=%s\n" "$(date --iso-8601=seconds)" "$benchmark" "$expected" >> "$pipeline_log"
"$python_bin" -u -m "$module" \
  "${common[@]}" "${extra_args[@]}" \
  --result-dir "$result_dir" \
  --baseline-state-dir "$result_dir/memory/datasets" \
  --sample-concurrency 4 \
  --answer-concurrency 16 \
  --split-manifest "$full_manifest" \
  --split test \
  --resume > "$result_dir/run.log" 2>&1

RESULT_DIR="$result_dir" EXPECTED_QA="$expected" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED_QA"])
required = (
    "results.json",
    "retrieval_trace.jsonl",
    "pipeline_qa.jsonl",
    "memory/memory_snapshot.jsonl",
    "run_manifest.json",
    "metrics.json",
    "efficiency_metrics.json",
    "call_trace.jsonl",
)
for name in required:
    if not (root / name).is_file():
        raise RuntimeError(f"missing formal output: {root / name}")
results = json.loads((root / "results.json").read_text())
traces = [json.loads(line) for line in (root / "retrieval_trace.jsonl").read_text().splitlines() if line]
pipeline = [json.loads(line) for line in (root / "pipeline_qa.jsonl").read_text().splitlines() if line]
if (len(results), len(traces), len(pipeline)) != (expected, expected, expected):
    raise RuntimeError(f"formal count mismatch {len(results)}/{len(traces)}/{len(pipeline)} != {expected}")
if any(row.get("error") for row in results):
    raise RuntimeError("formal results contain answer errors")
if any(not isinstance(row.get("top_k"), list) or len(row["top_k"]) > 7 for row in traces):
    raise RuntimeError("formal retrieval trace violates global Top-7")
manifest = json.loads((root / "run_manifest.json").read_text())
if manifest.get("selection_mode") != "strict_manifest" or manifest.get("split") != "test":
    raise RuntimeError("formal run did not use the strict test manifest")
if int(manifest.get("top_k", -1)) != 7 or int(manifest.get("questions", -1)) != expected:
    raise RuntimeError("formal manifest has the wrong Top-K or QA count")
PY

printf "%s JUDGE_START benchmark=%s\n" "$(date --iso-8601=seconds)" "$benchmark" >> "$pipeline_log"
"$python_bin" -u "$offline_root/scripts/judge_results_llm_parallel.py" \
  --benchmark "$judge_benchmark" \
  --results "$result_dir/results.json" \
  --out-dir "$result_dir" \
  --key-file "$workspace_root/Nvida_api/Openrouter_api" \
  --model openai/gpt-4o-mini \
  --workers 32 \
  --timeout 60 \
  --retries 2 \
  --max-tokens 512 \
  --checkpoint-every 25 \
  --resume > "$result_dir/llm_judge.log" 2>&1

RESULT_DIR="$result_dir" EXPECTED_QA="$expected" "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED_QA"])
judge = json.loads((root / "llm_judge_metrics.json").read_text())
metrics = json.loads((root / "metrics.json").read_text())
if int(judge.get("count", -1)) != expected or int(judge.get("valid_count", -1)) != expected:
    raise RuntimeError("Judge count does not match the formal test QA count")
if int(judge.get("judge_errors", -1)) != 0 or bool(judge.get("provisional", True)):
    raise RuntimeError("Judge output is incomplete")
if metrics.get("llm_judge") != judge.get("accuracy"):
    raise RuntimeError("Judge score was not merged into metrics.json")
PY

printf "%s PIPELINE_COMPLETE benchmark=%s\n" "$(date --iso-8601=seconds)" "$benchmark" >> "$pipeline_log"
