#!/usr/bin/env bash
# Restart the five-GPU launcher only when this run is incomplete and no launcher exists.
set -u -o pipefail

workspace=/data/haozhen/Memory-clean
offline=$workspace/Offline
python_bin=$offline/.venvs/mirix/bin/python
run_id=m2a_qwen3vl4b_emb06_top5_gpu35_formal_20260920_150153
launcher=$offline/scripts/run_m2a_local_gpu35_emb06_top5_resilient.sh
run_root=$offline/outputs/_runs/$run_id
log=$run_root/watchdog_60s.log
interval_seconds=${1:-60}

all_complete() {
  RUN_ID=$run_id OUTPUT_ROOT=$offline/outputs "$python_bin" - <<'PY'
import json
import os
from pathlib import Path

expected = {
    "Mem-Gallery": 275,
    "H2HMEM": 360,
    "WorldMemArena": 440,
    "MemEye": 371,
    "MEMLENS": 173,
}
root = Path(os.environ["OUTPUT_ROOT"])
run_id = os.environ["RUN_ID"]
for benchmark, count in expected.items():
    try:
        rows = json.loads((root / benchmark / "M2A" / run_id / "results.json").read_text())
    except (OSError, json.JSONDecodeError):
        raise SystemExit(1)
    if len(rows) != count or any(row.get("error") for row in rows):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

while ! all_complete; do
  health=""
  for port in 8002 8003 8004 8013 8014 8015 8016 8017; do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/health" >/dev/null || \
       curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null; then
      health="$health $port=ok"
    else
      health="$health $port=down"
    fi
  done
  workers=$(pgrep -fc "eval_(h2hmem|wma|memeye|memlens).*${run_id}" || true)
  printf '%s workers=%s%s\n' "$(date --iso-8601=seconds)" "$workers" "$health" >> "$log"
  if ! pgrep -f "run_m2a_local_gpu35_emb06_top5_resilient.sh $run_id" >/dev/null; then
    printf '%s restarting launcher\n' "$(date --iso-8601=seconds)" >> "$log"
    tmux new-session -d -s m2a_gpu35_emb06_top5 \
      "cd $workspace && exec $launcher $run_id"
  fi
  sleep "$interval_seconds"
done
printf '%s all_complete\n' "$(date --iso-8601=seconds)" >> "$log"
