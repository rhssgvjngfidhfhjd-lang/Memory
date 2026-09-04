#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

config="configs/evidence_policy_memgallery_full.json"
output="outputs/evidence_policy_memgallery_multibinary_vp_full"
control="$output/run_control"
mkdir -p "$control"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
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
    if curl --silent --fail --max-time 10 http://127.0.0.1:18000/v1/models >/dev/null; then
      endpoint="ok"
    else
      endpoint="unresponsive"
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

printf '%s status=started\n' "$(timestamp)" >> "$control/status.log"

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
  --name memgallery-ppo-multibinary-vp-full-20260901 \
  --run-id memgallery-ppo-multibinary-vp-full-20260901 \
  --tag memgallery --tag ppo --tag multibinary-vp
wandb_status=$?
if [[ $wandb_status -ne 0 ]]; then
  printf '%s status=failed stage=wandb exit_code=%s\n' \
    "$(timestamp)" "$wandb_status" >> "$control/status.log"
  exit "$wandb_status"
fi

printf '%s status=complete\n' "$(timestamp)" >> "$control/status.log"
