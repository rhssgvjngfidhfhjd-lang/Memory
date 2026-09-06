#!/bin/sh
set -u

output_root=${1:?Usage: monitor_test_baseline_matrix.sh OUTPUT_ROOT TMUX_SESSION}
runner_session=${2:?Usage: monitor_test_baseline_matrix.sh OUTPUT_ROOT TMUX_SESSION}
interval=${MONITOR_INTERVAL_SECONDS:-1800}
python_bin=${PYTHON_BIN:-python3}
log_path=$output_root/_logs/monitor_30m.log
status_path=$output_root/status.json

mkdir -p "$(dirname -- "$log_path")"

while tmux has-session -t "$runner_session" 2>/dev/null; do
  date '+%Y-%m-%dT%H:%M:%S%z' >> "$log_path"
  "$python_bin" - "$status_path" >> "$log_path" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("status=missing")
    raise SystemExit
payload = json.loads(path.read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "phase": payload.get("phase"),
            "updated_at": payload.get("updated_at"),
            "jobs": {
                state: sum(
                    row.get("status") == state
                    for row in (payload.get("jobs") or {}).values()
                )
                for state in ("pending", "running", "retrying", "completed", "failed")
            },
            "judges": {
                state: sum(
                    row.get("status") == state
                    for row in (payload.get("judges") or {}).values()
                )
                for state in ("running", "completed", "failed")
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
)
PY
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader | sed -n '4,6p' >> "$log_path" 2>&1
  printf '\n' >> "$log_path"
  sleep "$interval"
done

date '+%Y-%m-%dT%H:%M:%S%z runner_session_ended' >> "$log_path"
