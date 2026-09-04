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
checkpoint="$output/checkpoints/epoch_005.pt"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

endpoint_ok() {
  curl --silent --fail --noproxy '*' --max-time 10 \
    "http://127.0.0.1:${port}/v1/models" >/dev/null
}

mkdir -p "$control"
if [[ ! -f "$checkpoint" ]]; then
  printf '%s status=failed stage=checkpoint_missing checkpoint=%s\n' \
    "$(timestamp)" "$checkpoint" >> "$control/status.log"
  exit 1
fi

if ! endpoint_ok; then
  bash ../port2conti.sh >> "$control/port_repair.log" 2>&1 || true
fi
if ! endpoint_ok; then
  printf '%s status=failed stage=endpoint_unavailable port=%s\n' \
    "$(timestamp)" "$port" >> "$control/status.log"
  exit 1
fi

printf '%s status=evaluating_retry checkpoint=%s\n' \
  "$(timestamp)" "$checkpoint" >> "$control/status.log"
.venv/bin/python scripts/evidence_policy.py \
  --config "$config" eval --strategy ppo --split test \
  --checkpoint "$checkpoint" --device cpu \
  >> "$control/eval_retry.stdout.log" \
  2>> "$control/eval_retry.stderr.log"
eval_status=$?
if [[ $eval_status -ne 0 ]]; then
  printf '%s status=failed stage=eval_retry exit_code=%s\n' \
    "$(timestamp)" "$eval_status" >> "$control/status.log"
  exit "$eval_status"
fi

printf '%s status=uploading_wandb_retry\n' "$(timestamp)" >> "$control/status.log"
.venv/bin/python scripts/upload_evidence_policy_wandb.py \
  --run-dir "$output" \
  --project hivemem-evidence-policy \
  --name "$wandb_run_name" \
  --run-id "$wandb_run_name" \
  --tag "$benchmark_tag" --tag ppo --tag multibinary-vp --tag step0 \
  >> "$control/wandb_retry.stdout.log" \
  2>> "$control/wandb_retry.stderr.log"
wandb_status=$?
if [[ $wandb_status -ne 0 ]]; then
  printf '%s status=failed stage=wandb_retry exit_code=%s\n' \
    "$(timestamp)" "$wandb_status" >> "$control/status.log"
  exit "$wandb_status"
fi

printf '%s status=complete\n' "$(timestamp)" >> "$control/status.log"
