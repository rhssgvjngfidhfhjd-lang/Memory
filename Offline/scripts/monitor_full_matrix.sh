#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${repo_dir}/logs/full_matrix"
monitor_log="${log_dir}/monitor.log"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "${log_dir}"

matrix_session_alive() {
  tmux has-session -t full_matrix_24 2>/dev/null \
    || tmux has-session -t full_matrix_recovery 2>/dev/null
}

while matrix_session_alive; do
  {
    date '+CHECK %Y-%m-%dT%H:%M:%S%z'
    REPO_DIR="${repo_dir}" python - <<'PY'
import collections
import json
import os
from pathlib import Path

status_path = Path(os.environ["REPO_DIR"]) / "logs/full_matrix/status.json"
try:
    status = json.loads(status_path.read_text())
    counts = collections.Counter(job.get("status") for job in status["jobs"].values())
    print("status_updated", status.get("updated_at"), dict(counts))
    for name, job in status["jobs"].items():
        if job.get("status") in {"running", "retrying"}:
            print("active", name, "attempt", job.get("attempt"))
except Exception as exc:
    print("status_error", repr(exc))
PY
    for port in 8001 8013 8014 8015; do
      code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)"
      printf 'service %s %s\n' "${port}" "${code:-unreachable}"
    done
    PYTHONPATH="${repo_dir}/src" \
      "${python_bin}" \
      "${repo_dir}/scripts/verify_full_matrix.py" --allow-incomplete \
      | head -n 1
  } >> "${monitor_log}" 2>&1
  sleep 1800
done

date '+STOP %Y-%m-%dT%H:%M:%S%z matrix and recovery schedulers exited' >> "${monitor_log}"
