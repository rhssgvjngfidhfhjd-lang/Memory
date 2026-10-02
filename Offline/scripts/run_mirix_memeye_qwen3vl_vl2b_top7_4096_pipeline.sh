#!/usr/bin/env bash
set -uo pipefail

workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
run_id=mirix_qwen3vl4b_vl2b_top7_4096_20260919
source_dir=$offline/outputs/MemEye/MIRIX/$run_id
isolated_dir=$offline/outputs/_experiments/mirix_isolated_local_qwen3vl4b_20260919/top7_vl2b_memeye
log_dir=$offline/outputs/_runs/$run_id/logs
pipeline_log=$log_dir/memeye.pipeline.log
answer_endpoint=http://127.0.0.1:8013/v1
embedding_endpoint=http://127.0.0.1:8001/v1
model=Qwen/Qwen3-VL-4B-Instruct
embedding_model=Qwen/Qwen3-VL-Embedding-2B
expected=371

mkdir -p "$source_dir" "$isolated_dir" "$log_dir"
export PYTHONPATH=$offline/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export MIRIX_PYTHON=$offline/.venvs/mirix/bin/python

stamp() {
  date --iso-8601=seconds
}

result_complete() {
  RESULT_DIR=$1 EXPECTED=$expected "$python_bin" - <<'PY'
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

echo "$(stamp) source_start benchmark=MemEye" >> "$pipeline_log"
while ! result_complete "$source_dir"; do
  "$python_bin" -u -m benchmarks.memeye_harness.eval_memeye \
    --baseline MIRIX \
    --result-dir "$source_dir" \
    --baseline-state-dir "$source_dir/memory/datasets" \
    --data-dir "$workspace/MemEye/data" \
    --sample-concurrency 4 \
    --answer-concurrency 16 \
    --checkpoint-every 10 \
    --answer-model "$model" \
    --answer-base-url "$answer_endpoint" \
    --answer-temperature 0.0 \
    --num-predict 512 \
    --no-think \
    --executor-model "$model" \
    --executor-base-url "$answer_endpoint" \
    --executor-temperature 0.0 \
    --executor-max-tokens 4096 \
    --embedding-model "$embedding_model" \
    --embedding-base-url "$embedding_endpoint" \
    --embedding-dim 2048 \
    --top-k 7 \
    --request-timeout 180 \
    --retries 2 \
    --efficiency-config "$offline/configs/model_efficiency.json" \
    --mirix-skip-failed-build-points \
    --mirix-max-consecutive-failed-build-points 10 \
    --resume >> "$pipeline_log" 2>&1
  status=$?
  if result_complete "$source_dir"; then
    break
  fi
  echo "$(stamp) source_retry benchmark=MemEye exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) isolated_qa_start benchmark=MemEye" >> "$pipeline_log"
while ! isolated_complete; do
  "$python_bin" -u "$offline/scripts/rerun_qa_from_frozen_retrieval.py" \
    --benchmark MemEye \
    --baseline MIRIX \
    --source-dir "$source_dir" \
    --result-dir "$isolated_dir" \
    --answer-base-url "$answer_endpoint" \
    --answer-model "$model" \
    --max-tokens 512 \
    --top-k 7 \
    --concurrency 16 \
    --checkpoint-every 10 \
    --isolate-final-answer \
    --resume >> "$pipeline_log" 2>&1
  status=$?
  if isolated_complete; then
    break
  fi
  echo "$(stamp) isolated_qa_retry benchmark=MemEye exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) judge_start benchmark=MemEye" >> "$pipeline_log"
while ! judge_complete; do
  "$python_bin" -u "$offline/scripts/judge_results_llm_parallel.py" \
    --benchmark memeye \
    --results "$isolated_dir/results.json" \
    --out-dir "$isolated_dir" \
    --key-file "$workspace/Nvida_api/Openrouter_api" \
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
  echo "$(stamp) judge_retry benchmark=MemEye exit=$status" >> "$pipeline_log"
  sleep 60
done

echo "$(stamp) complete benchmark=MemEye source=$source_dir isolated=$isolated_dir" >> "$pipeline_log"
