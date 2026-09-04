#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CONFIG OUTPUT PORT WANDB_RUN_NAME BENCHMARK_TAG" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

config="$1"
output="$2"
port="$3"
wandb_run_name="$4"
benchmark_tag="$5"
control="$output/run_control"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

mkdir -p "$control"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

endpoint_ok() {
  curl --silent --fail --noproxy '*' --max-time 10 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null
}

repair_endpoint() {
  local repair_script="../port2conti.sh"
  if [[ ! -f "$repair_script" ]]; then
    printf '%s status=endpoint_repair_unavailable script=%s\n' \
      "$(timestamp)" "$repair_script" >> "$control/status.log"
    return 1
  fi

  printf '%s status=endpoint_repair_started port=%s\n' \
    "$(timestamp)" "$port" >> "$control/status.log"
  # Both PPO runners may notice a broken tunnel together.  Serialize the
  # idempotent repair script so they cannot race while creating SSH tunnels.
  if command -v flock >/dev/null 2>&1; then
    flock /tmp/aaa-memory-port2conti.lock \
      bash "$repair_script" >> "$control/port_repair.log" 2>&1 || true
  else
    bash "$repair_script" >> "$control/port_repair.log" 2>&1 || true
  fi

  if endpoint_ok; then
    printf '%s status=endpoint_recovered port=%s\n' \
      "$(timestamp)" "$port" >> "$control/status.log"
    return 0
  fi
  printf '%s status=endpoint_repair_failed port=%s\n' \
    "$(timestamp)" "$port" >> "$control/status.log"
  return 1
}

monitor() {
  while true; do
    checkpoint_count=$(find "$output/checkpoints" -maxdepth 1 -name 'epoch_*.pt' 2>/dev/null | wc -l)
    if [[ -f "$output/rollout_cache.jsonl" ]]; then
      cache_lines=$(wc -l < "$output/rollout_cache.jsonl")
    else
      cache_lines=0
    fi
    if [[ -f "$output/ppo_metrics.jsonl" ]]; then
      metrics_lines=$(wc -l < "$output/ppo_metrics.jsonl")
    else
      metrics_lines=0
    fi
    if endpoint_ok; then
      endpoint="ok"
    else
      endpoint="unresponsive"
      repair_endpoint || true
      if endpoint_ok; then
        endpoint="recovered"
      fi
    fi
    printf '%s endpoint=%s checkpoints=%s cache_lines=%s metrics_lines=%s\n' \
      "$(timestamp)" "$endpoint" "$checkpoint_count" "$cache_lines" "$metrics_lines" \
      >> "$control/monitor.log"
    sleep 1800
  done
}

monitor &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

printf '%s status=started port=%s config=%s\n' \
  "$(timestamp)" "$port" "$config" >> "$control/status.log"

while ! endpoint_ok; do
  printf '%s status=waiting_endpoint port=%s\n' \
    "$(timestamp)" "$port" >> "$control/status.log"
  repair_endpoint || true
  if endpoint_ok; then
    break
  fi
  sleep 50
done

printf '%s status=endpoint_ready port=%s\n' \
  "$(timestamp)" "$port" >> "$control/status.log"

attempt=0
while true; do
  resume_args=()
  latest_checkpoint=$(find "$output/checkpoints" -maxdepth 1 -name 'epoch_*.pt' 2>/dev/null | sort | tail -n 1)
  if [[ -n "$latest_checkpoint" ]]; then
    resume_args=(--resume "$latest_checkpoint")
  fi
  printf '%s status=training attempt=%s resume=%s\n' \
    "$(timestamp)" "$attempt" "${latest_checkpoint:-none}" >> "$control/status.log"
  .venv/bin/python scripts/evidence_policy.py \
    --config "$config" train --device cpu "${resume_args[@]}"
  train_status=$?
  if [[ $train_status -eq 0 ]]; then
    break
  fi
  attempt=$((attempt + 1))
  printf '%s status=train_failed exit_code=%s retry=%s\n' \
    "$(timestamp)" "$train_status" "$attempt" >> "$control/status.log"
  if [[ $attempt -ge 3 ]]; then
    printf '%s status=failed stage=train\n' "$(timestamp)" >> "$control/status.log"
    exit "$train_status"
  fi
  sleep 60
done

checkpoint="$output/checkpoints/epoch_005.pt"
if [[ ! -f "$checkpoint" ]]; then
  printf '%s status=failed stage=checkpoint_missing\n' "$(timestamp)" >> "$control/status.log"
  exit 1
fi

printf '%s status=evaluating checkpoint=%s\n' "$(timestamp)" "$checkpoint" >> "$control/status.log"
.venv/bin/python scripts/evidence_policy.py \
  --config "$config" eval --strategy ppo --split test \
  --checkpoint "$checkpoint" --device cpu
eval_status=$?
if [[ $eval_status -ne 0 ]]; then
  printf '%s status=failed stage=eval exit_code=%s\n' \
    "$(timestamp)" "$eval_status" >> "$control/status.log"
  exit "$eval_status"
fi

printf '%s status=uploading_wandb\n' "$(timestamp)" >> "$control/status.log"
.venv/bin/python scripts/upload_evidence_policy_wandb.py \
  --run-dir "$output" \
  --project hivemem-evidence-policy \
  --name "$wandb_run_name" \
  --run-id "$wandb_run_name" \
  --tag "$benchmark_tag" --tag ppo --tag multibinary-vp --tag step0
wandb_status=$?
if [[ $wandb_status -ne 0 ]]; then
  printf '%s status=failed stage=wandb exit_code=%s\n' \
    "$(timestamp)" "$wandb_status" >> "$control/status.log"
  exit "$wandb_status"
fi

printf '%s status=complete\n' "$(timestamp)" >> "$control/status.log"
