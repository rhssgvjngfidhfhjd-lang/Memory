#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Mem-Gallery|WorldMemArena|H2HMEM|MemEye|MEMLENS> <local-qwen3vl-endpoint>" >&2
  exit 2
fi

benchmark=$1
answer_endpoint=$2
workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
run_id=mirix_qwen35_9b_emb06_top5_4096_20260919
run_root=$offline/outputs/_runs/$run_id
python_bin=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
key_file=$workspace/Nvida_api/Openrouter_api
split_manifest=$offline/configs/multimodal_split_manifest.json
efficiency=$offline/outputs/_runs/m3_qwen35_9b_emb06_top5_smoke_20260915_114634/_config/qwen35_efficiency.json
api_url=https://openrouter.ai/api/v1
executor_model=qwen/qwen3.5-9b
answer_model=Qwen/Qwen3-VL-4B-Instruct
embedding_model=Qwen/Qwen3-Embedding-0.6B
embedding_url=http://127.0.0.1:8002/v1

case "$benchmark" in
  Mem-Gallery)
    module=benchmarks.memgallery_harness.eval_memgallery
    expected=275
    sample_concurrency=1
    slug=memgallery
    judge_benchmark=memgallery
    source_dir=$offline/outputs/Mem-Gallery/MIRIX/$run_id
    executor_extra=(--executor-hard-max-tokens 4096 --executor-visual-input image)
    extra=(
      --data-dir "$workspace/Mem-Gallery/benchmark/data"
      --all-datasets --split-manifest "$split_manifest" --split test
    )
    ;;
  WorldMemArena)
    module=benchmarks.wma_harness.eval_wma
    expected=440
    sample_concurrency=8
    slug=wma
    judge_benchmark=worldmemarena
    source_dir=$offline/outputs/WorldMemArena/MIRIX/$run_id
    executor_extra=(--executor-hard-max-tokens 4096 --executor-visual-input image)
    extra=(
      --data-dir "$workspace/WorldMemArena/WorldMemArena/lifelong"
      --split-manifest "$split_manifest" --split test --sample-attempts 3
    )
    ;;
  H2HMEM)
    module=benchmarks.h2hmem_harness.eval_h2hmem
    expected=360
    sample_concurrency=1
    slug=h2hmem
    judge_benchmark=h2hmem
    source_dir=$offline/outputs/H2HMEM/MIRIX/$run_id
    executor_extra=(--executor-hard-max-tokens 4096 --executor-visual-input image)
    extra=(
      --data-dir "$workspace/H2HMEM-main/dataset"
      --variant all --split-manifest "$split_manifest" --split test
    )
    ;;
  MemEye)
    module=benchmarks.memeye_harness.eval_memeye
    expected=371
    sample_concurrency=1
    slug=memeye
    judge_benchmark=memeye
    source_dir=$offline/outputs/MemEye/MIRIX/$run_id
    executor_extra=()
    extra=(--data-dir "$workspace/MemEye/data")
    ;;
  MEMLENS)
    module=benchmarks.memlens_harness.eval_memlens
    expected=173
    sample_concurrency=8
    slug=memlens
    judge_benchmark=memlens
    source_dir=$offline/outputs/MEMLENS/MIRIX/$run_id
    executor_extra=()
    extra=(
      --data-dir "$workspace/MEMLENS"
      --dataset-file dataset_32k.json
    )
    ;;
  *)
    echo "unsupported benchmark: $benchmark" >&2
    exit 2
    ;;
esac

isolated_root=$offline/outputs/_experiments/mirix_isolated_local_qwen3vl4b_20260919
isolated_dir=$isolated_root/top5_emb06_qwen35mem4096_$slug
log_dir=$run_root/logs
pipeline_log=$log_dir/$slug.pipeline.log
mkdir -p "$log_dir" "$source_dir" "$isolated_dir"

export OPENAI_API_KEY
OPENAI_API_KEY=$(tr -d '\r\n' < "$key_file")
export PYTHONPATH=$offline/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export MIRIX_PYTHON=$offline/.venvs/mirix/bin/python

stamp() {
  date --iso-8601=seconds
}

result_complete() {
  RESULT_DIR=$1 EXPECTED=$2 "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED"])
try:
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in (root / "retrieval_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(results) == expected and len(traces) == expected else 1)
PY
}

isolated_complete() {
  RESULT_DIR=$isolated_dir EXPECTED=$expected "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED"])
try:
    results = json.loads((root / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
ok = (
    len(results) == expected
    and not any(row.get("error") for row in results)
    and manifest.get("final_answer_isolated_from_baseline_agent") is True
)
raise SystemExit(0 if ok else 1)
PY
}

judge_complete() {
  RESULT_DIR=$isolated_dir EXPECTED=$expected "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULT_DIR"])
expected = int(os.environ["EXPECTED"])
try:
    judge = json.loads((root / "llm_judge_metrics.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
ok = (
    judge.get("count") == expected
    and judge.get("valid_count") == expected
    and judge.get("judge_errors") == 0
    and judge.get("provisional") is False
    and metrics.get("llm_judge") == judge.get("accuracy")
)
raise SystemExit(0 if ok else 1)
PY
}

echo "$(stamp) source_start benchmark=$benchmark" >> "$pipeline_log"
while ! result_complete "$source_dir" "$expected"; do
  "$python_bin" -u -m "$module" \
    --baseline MIRIX \
    --result-dir "$source_dir" \
    --baseline-state-dir "$source_dir/memory/datasets" \
    --sample-concurrency "$sample_concurrency" \
    --answer-concurrency 16 \
    --checkpoint-every 10 \
    --answer-model "$executor_model" \
    --answer-base-url "$api_url" \
    --answer-temperature 0.0 \
    --num-predict 512 \
    --executor-model "$executor_model" \
    --executor-base-url "$api_url" \
    --executor-temperature 0.0 \
    --executor-max-tokens 4096 \
    "${executor_extra[@]}" \
    --embedding-model "$embedding_model" \
    --embedding-base-url "$embedding_url" \
    --embedding-dim 2048 \
    --top-k 5 \
    --request-timeout 180 \
    --retries 2 \
    --reasoning-effort none \
    --efficiency-config "$efficiency" \
    --mirix-skip-failed-build-points \
    --mirix-max-consecutive-failed-build-points 10 \
    --allow-answer-errors \
    --resume \
    "${extra[@]}" >> "$pipeline_log" 2>&1
  status=$?
  if result_complete "$source_dir" "$expected"; then
    break
  fi
  echo "$(stamp) source_retry benchmark=$benchmark exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) isolated_qa_start benchmark=$benchmark" >> "$pipeline_log"
while ! isolated_complete; do
  "$python_bin" -u "$offline/scripts/rerun_qa_from_frozen_retrieval.py" \
    --benchmark "$benchmark" \
    --baseline MIRIX \
    --source-dir "$source_dir" \
    --result-dir "$isolated_dir" \
    --answer-base-url "$answer_endpoint" \
    --answer-model "$answer_model" \
    --max-tokens 512 \
    --top-k 5 \
    --concurrency 16 \
    --checkpoint-every 10 \
    --isolate-final-answer \
    --resume >> "$pipeline_log" 2>&1
  status=$?
  if isolated_complete; then
    break
  fi
  echo "$(stamp) isolated_qa_retry benchmark=$benchmark exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) judge_start benchmark=$benchmark" >> "$pipeline_log"
while ! judge_complete; do
  "$python_bin" -u "$offline/scripts/judge_results_llm_parallel.py" \
    --benchmark "$judge_benchmark" \
    --results "$isolated_dir/results.json" \
    --out-dir "$isolated_dir" \
    --key-file "$key_file" \
    --model openai/gpt-4o-mini \
    --workers 32 \
    --timeout 60 \
    --retries 2 \
    --max-tokens 512 \
    --checkpoint-every 25 \
    --resume >> "$pipeline_log" 2>&1
  status=$?
  if judge_complete; then
    break
  fi
  echo "$(stamp) judge_retry benchmark=$benchmark exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) complete benchmark=$benchmark source=$source_dir isolated=$isolated_dir" >> "$pipeline_log"
