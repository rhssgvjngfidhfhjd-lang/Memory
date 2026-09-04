#!/usr/bin/env bash
set -u

threshold_kb="${RECOVERY_DISK_MIN_KB:-1048576}"
interval_seconds="${RECOVERY_DISK_INTERVAL:-60}"
log_path="${RECOVERY_DISK_LOG:-/var/tmp/haozhen_memory_recovery_20260830/disk_guard.log}"
sessions=(recover_mirix_wma recover_mma_wma recover_mirix_h2 recover_omni_wma)

mkdir -p "$(dirname "${log_path}")"

stop_tree() {
  local pid="$1"
  local child
  while read -r child; do
    [[ -n "${child}" ]] && stop_tree "${child}"
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill -STOP "${pid}" 2>/dev/null || true
}

while true; do
  active=0
  available_kb="$(df -Pk /data | awk 'NR == 2 {print $4}')"
  printf '%s available_kb=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${available_kb}" >> "${log_path}"

  for session in "${sessions[@]}"; do
    if ! tmux has-session -t "${session}" 2>/dev/null; then
      continue
    fi
    active=1
    if (( available_kb < threshold_kb )); then
      while read -r pane_pid; do
        [[ -n "${pane_pid}" ]] && stop_tree "${pane_pid}"
      done < <(tmux list-panes -t "${session}" -F '#{pane_pid}')
      printf '%s paused=%s threshold_kb=%s\n' \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${session}" "${threshold_kb}" >> "${log_path}"
    fi
  done

  if (( active == 0 )); then
    printf '%s stop=no-recovery-sessions\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" >> "${log_path}"
    exit 0
  fi
  if (( available_kb < threshold_kb )); then
    exit 0
  fi
  sleep "${interval_seconds}"
done
