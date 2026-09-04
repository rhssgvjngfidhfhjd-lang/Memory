#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/../.."

log="Offline/outputs/ppo_endpoint_watchdog.log"
repair_script="./port2conti.sh"
ports=(18000 18001)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

endpoint_ok() {
  curl --silent --fail --noproxy '*' --max-time 10 \
    "http://127.0.0.1:$1/v1/models" >/dev/null
}

while pgrep -f 'scripts/evidence_policy.py.*(18000|18001)' >/dev/null; do
  repair_needed=0
  for port in "${ports[@]}"; do
    if endpoint_ok "$port"; then
      printf '%s port=%s endpoint=ok\n' "$(timestamp)" "$port" >> "$log"
    else
      printf '%s port=%s endpoint=unresponsive action=repair\n' \
        "$(timestamp)" "$port" >> "$log"
      repair_needed=1
    fi
  done

  if [[ $repair_needed -eq 1 ]]; then
    if command -v flock >/dev/null 2>&1; then
      flock /tmp/aaa-memory-port2conti.lock \
        bash "$repair_script" >> "$log" 2>&1 || true
    else
      bash "$repair_script" >> "$log" 2>&1 || true
    fi
    for port in "${ports[@]}"; do
      if endpoint_ok "$port"; then
        printf '%s port=%s endpoint=recovered\n' "$(timestamp)" "$port" >> "$log"
      else
        printf '%s port=%s endpoint=repair_failed\n' "$(timestamp)" "$port" >> "$log"
      fi
    done
  fi

  sleep 60
done

printf '%s status=stopped reason=no_active_ppo_process\n' "$(timestamp)" >> "$log"
