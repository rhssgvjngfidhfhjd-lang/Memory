#!/usr/bin/env bash
# Report the persistent GPU 4/5 M2A queue every 30 minutes.
set -u -o pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_ID" >&2
  exit 2
fi

run_id=$1
workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
run_root=$offline/outputs/_runs/$run_id
log=$offline/outputs/_runs/$run_id/monitor_30m.log

while true; do
  {
    printf '%s run_id=%s\n' "$(date --iso-8601=seconds)" "$run_id"
    for status in "$run_root"/matrix_gpu45_*.json; do
      [[ -f $status ]] || continue
      /data/haozhen/Memory-clean/Offline/.venvs/mirix/bin/python - "$status" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("phase=" + str(payload.get("phase")))
for name, row in sorted((payload.get("jobs") or {}).items()):
    print(f"{name}: {row.get('status')} attempt={row.get('attempt')} endpoint={row.get('endpoint')}")
PY
    done
    for benchmark in MemEye MEMLENS; do
      results="$offline/outputs/$benchmark/M2A/$run_id/results.json"
      if [[ -f $results ]]; then
        echo "$benchmark results=$( /data/haozhen/Memory-clean/Offline/.venvs/mirix/bin/python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$results" )"
      else
        echo "$benchmark results=not_written"
      fi
    done
    for port in 8014 8015; do
      curl -fsS --max-time 10 "http://127.0.0.1:$port/metrics" \
        | grep -E 'vllm:(num_requests_running|num_requests_waiting)' \
        | sed "s/^/port=$port /" || true
    done
  } | tee -a "$log"
  sleep 1800
done
