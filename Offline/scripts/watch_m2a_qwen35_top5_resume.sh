#!/usr/bin/env bash
# Keep the incomplete Qwen3.5-9B M2A runs alive without rerunning completed work.
set -u -o pipefail

workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=$offline/.venvs/mirix/bin/python
run_id=m2a_qwen35_9b_emb06_top5_1200_salvage_formal_20260919_173648
run_root=$offline/outputs/_runs/$run_id
interval_seconds=${1:-300}

is_complete() {
  local benchmark=$1
  local expected=$2
  local results=$offline/outputs/$benchmark/M2A/$run_id/results.json
  "$python_bin" - "$results" "$expected" <<'PY'
import json
import sys
from pathlib import Path

try:
    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(rows) == int(sys.argv[2]) and not any(row.get("error") for row in rows) else 1)
PY
}

is_running() {
  local benchmark=$1
  pgrep -f "run_m2a_openrouter_five_formal.py --run-id $run_id.*--benchmarks $benchmark" >/dev/null ||
    pgrep -f "outputs/$benchmark/M2A/$run_id" >/dev/null
}

launch() {
  local benchmark=$1
  local log=$run_root/watchdog_${benchmark,,}.log
  printf '%s restarting %s\n' "$(date --iso-8601=seconds)" "$benchmark" >> "$log"
  "$python_bin" "$offline/scripts/run_m2a_openrouter_five_formal.py" \
    --run-id "$run_id" --resume --benchmarks "$benchmark" >> "$log" 2>&1 &
}

while true; do
  for item in 'H2HMEM 360' 'WorldMemArena 440' 'MemEye 371' 'MEMLENS 173'; do
    set -- $item
    if ! is_complete "$1" "$2" && ! is_running "$1"; then
      launch "$1"
    fi
  done
  sleep "$interval_seconds"
done
