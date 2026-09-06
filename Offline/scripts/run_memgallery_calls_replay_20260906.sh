#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output="outputs/evidence_policy_memgallery_graph5plus2_fullval_18000_20260906_calls_replay"
control="$output/run_control"
mkdir -p "$control"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONUNBUFFERED=1

printf '%s status=started\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" \
  >> "$control/status.log"

.venv/bin/python scripts/evidence_policy.py \
  --config configs/evidence_policy_memgallery_independent_bernoulli_fullval312_18001_20260905.json \
  --output-dir "$output" \
  --model-base-url http://127.0.0.1:18000/v1 \
  eval --strategy ppo --split test \
  --checkpoint outputs/evidence_policy_memgallery_graph5plus2_fullval_18000_20260906/checkpoints/epoch_005.pt \
  --device cpu \
  >> "$control/eval.stdout.log" \
  2>> "$control/eval.stderr.log"

printf '%s status=complete\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" \
  >> "$control/status.log"
