#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

CONFIG="$ROOT/configs/evidence_policy.json"
TRAIN_PYTHON=${TRAIN_PYTHON:-}
VLLM_PYTHON=${VLLM_PYTHON:-}
VLLM_MODEL=${VLLM_MODEL:-}
VLLM_GPUS=${VLLM_GPUS:-0}
POLICY_GPUS=${POLICY_GPUS:-$VLLM_GPUS}
POLICY_DEVICE=${POLICY_DEVICE:-cuda:0}
VLLM_MEMORY_UTILIZATION=${VLLM_MEMORY_UTILIZATION:-0.85}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-600}
WANDB_PROJECT=${WANDB_PROJECT:-hivemem-evidence-policy}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_NAME=${WANDB_NAME:-}
RESUME=""
EPOCHS=""
MAX_TRAIN_EPISODES=""
SKIP_VLLM=0
KEEP_VLLM=0
NO_WANDB=0
DRY_RUN=0
WANDB_TAGS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_evidence_policy_ppo.sh [options]

Runs the local VLM, PPO Evidence Policy training, checkpoint/log generation,
and W&B upload. Project paths are resolved relative to this script.

Options:
  --config PATH                 Config JSON (default: configs/evidence_policy.json)
  --model PATH_OR_ID            vLLM model path or Hugging Face model ID
  --vlm-gpus IDS                CUDA devices for vLLM, e.g. 0 or 0,1
  --policy-gpus IDS             CUDA devices visible to the MLP trainer
  --policy-device DEVICE        Trainer device inside its CUDA view (default: cuda:0)
  --train-python PATH           Python with project training dependencies
  --vllm-python PATH            Python with vLLM (defaults to train Python)
  --resume CHECKPOINT           Resume instead of random initialization
  --epochs N                    Override config epochs
  --max-train-episodes N        Limit train questions per epoch (smoke tests)
  --skip-vllm                   Use the VLM endpoint already in the config
  --keep-vllm                   Do not stop a VLM started by this script
  --no-wandb                    Do not upload metrics after training
  --wandb-project NAME          W&B project
  --wandb-entity NAME           W&B entity/team
  --wandb-name NAME             W&B run name (defaults to output directory name)
  --wandb-tag TAG               Repeatable W&B tag
  --startup-timeout SECONDS     Maximum VLM startup wait (default: 600)
  --dry-run                     Print resolved settings and commands only
  -h, --help                    Show this help

Environment variables with the same purpose are also supported:
TRAIN_PYTHON, VLLM_PYTHON, VLLM_MODEL, VLLM_GPUS, POLICY_GPUS,
POLICY_DEVICE, VLLM_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN,
VLLM_TENSOR_PARALLEL_SIZE, STARTUP_TIMEOUT, WANDB_PROJECT,
WANDB_ENTITY, and WANDB_NAME.
EOF
}

while (($#)); do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    --model) VLLM_MODEL=$2; shift 2 ;;
    --vlm-gpus) VLLM_GPUS=$2; shift 2 ;;
    --policy-gpus) POLICY_GPUS=$2; shift 2 ;;
    --policy-device) POLICY_DEVICE=$2; shift 2 ;;
    --train-python) TRAIN_PYTHON=$2; shift 2 ;;
    --vllm-python) VLLM_PYTHON=$2; shift 2 ;;
    --resume) RESUME=$2; shift 2 ;;
    --epochs) EPOCHS=$2; shift 2 ;;
    --max-train-episodes) MAX_TRAIN_EPISODES=$2; shift 2 ;;
    --skip-vllm) SKIP_VLLM=1; shift ;;
    --keep-vllm) KEEP_VLLM=1; shift ;;
    --no-wandb) NO_WANDB=1; shift ;;
    --wandb-project) WANDB_PROJECT=$2; shift 2 ;;
    --wandb-entity) WANDB_ENTITY=$2; shift 2 ;;
    --wandb-name) WANDB_NAME=$2; shift 2 ;;
    --wandb-tag) WANDB_TAGS+=("$2"); shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

resolve_command() {
  local value=$1
  if [ -x "$value" ]; then
    printf '%s\n' "$value"
  elif command -v "$value" >/dev/null 2>&1; then
    command -v "$value"
  else
    echo "Command not found or not executable: $value" >&2
    return 1
  fi
}

if [ -z "$TRAIN_PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then
    TRAIN_PYTHON=python3
  else
    TRAIN_PYTHON=python
  fi
fi
TRAIN_PYTHON=$(resolve_command "$TRAIN_PYTHON")
VLLM_PYTHON=$(resolve_command "${VLLM_PYTHON:-$TRAIN_PYTHON}")

case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac
if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

mapfile -t CONFIG_VALUES < <(
  "$TRAIN_PYTHON" - "$CONFIG" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

config_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
output = Path(config["output_dir"])
if not output.is_absolute():
    output = root / output
base_url = str(config["model"]["base_url"]).rstrip("/")
parsed = urlparse(base_url)
if not parsed.hostname:
    raise SystemExit(f"Invalid model.base_url: {base_url}")
port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(output.resolve())
print(base_url)
print(config["model"]["name"])
print(parsed.hostname)
print(port)
PY
)
if ((${#CONFIG_VALUES[@]} != 5)); then
  echo "Failed to resolve output_dir and model settings from: $CONFIG" >&2
  exit 1
fi
RUN_DIR=${CONFIG_VALUES[0]}
VLM_BASE_URL=${CONFIG_VALUES[1]}
SERVED_MODEL=${CONFIG_VALUES[2]}
VLM_HOSTNAME=${CONFIG_VALUES[3]}
VLM_PORT=${CONFIG_VALUES[4]}
VLLM_MODEL=${VLLM_MODEL:-$SERVED_MODEL}
WANDB_NAME=${WANDB_NAME:-$(basename "$RUN_DIR")}

if [ -z "$VLLM_TENSOR_PARALLEL_SIZE" ]; then
  IFS=',' read -r -a VLLM_GPU_LIST <<< "$VLLM_GPUS"
  VLLM_TENSOR_PARALLEL_SIZE=${#VLLM_GPU_LIST[@]}
fi

TRAIN_CMD=(
  "$TRAIN_PYTHON" "$ROOT/scripts/evidence_policy.py"
  --config "$CONFIG" train --device "$POLICY_DEVICE"
)
if [ -n "$RESUME" ]; then
  TRAIN_CMD+=(--resume "$RESUME")
fi
if [ -n "$EPOCHS" ]; then
  TRAIN_CMD+=(--epochs "$EPOCHS")
fi
if [ -n "$MAX_TRAIN_EPISODES" ]; then
  TRAIN_CMD+=(--max-train-episodes "$MAX_TRAIN_EPISODES")
fi

WANDB_CMD=(
  "$TRAIN_PYTHON" "$ROOT/scripts/upload_evidence_policy_wandb.py"
  --run-dir "$RUN_DIR" --project "$WANDB_PROJECT" --name "$WANDB_NAME"
)
if [ -n "$WANDB_ENTITY" ]; then
  WANDB_CMD+=(--entity "$WANDB_ENTITY")
fi
for tag in "${WANDB_TAGS[@]}"; do
  WANDB_CMD+=(--tag "$tag")
done

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

echo "Resolved Evidence Policy run:"
echo "  root:           $ROOT"
echo "  config:         $CONFIG"
echo "  output:         $RUN_DIR"
echo "  VLM endpoint:   $VLM_BASE_URL"
echo "  VLM model:      $VLLM_MODEL"
echo "  VLM GPUs:       $VLLM_GPUS"
echo "  policy GPUs:    $POLICY_GPUS"
echo "  train Python:   $TRAIN_PYTHON"
echo "  vLLM Python:    $VLLM_PYTHON"
echo "  initialization: $([ -n "$RESUME" ] && echo resume || echo random)"
echo "Training command:"
print_command "${TRAIN_CMD[@]}"
if ((NO_WANDB == 0)); then
  echo "W&B command:"
  print_command "${WANDB_CMD[@]}"
fi
if ((DRY_RUN)); then
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the VLM health check." >&2
  exit 1
fi
"$TRAIN_PYTHON" - "$CONFIG" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
resolved = {}
for key in ("data_dir", "memory_bank", "query_cache"):
    path = Path(config[key])
    resolved[key] = path if path.is_absolute() else root / path
if config.get("profiles_file"):
    path = Path(config["profiles_file"])
    resolved["profiles_file"] = path if path.is_absolute() else root / path
missing = [f"{key}: {path}" for key, path in resolved.items() if not path.exists()]
query_cache = resolved["query_cache"]
for filename in ("vectors.npy", "metadata.jsonl"):
    path = query_cache / filename
    if not path.exists():
        missing.append(f"query_cache file: {path}")
if missing:
    raise SystemExit("Missing required inputs:\n  " + "\n  ".join(missing))
print("Input preflight passed")
PY
if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs"
if [ -z "$RESUME" ] && {
  [ -e "$RUN_DIR/ppo_metrics.jsonl" ] || [ -d "$RUN_DIR/checkpoints" ];
}; then
  echo "Refusing to start a fresh run in a non-empty training output: $RUN_DIR" >&2
  echo "Choose a new output_dir in the config or pass --resume CHECKPOINT." >&2
  exit 1
fi
if [ ! -f "$RUN_DIR/config.json" ]; then
  cp "$CONFIG" "$RUN_DIR/config.json"
fi

VLLM_PID=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "$VLLM_PID" ] && ((KEEP_VLLM == 0)); then
    echo "Stopping vLLM process $VLLM_PID"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

vlm_ready() {
  curl --connect-timeout 2 --max-time 5 --fail --silent \
    "$VLM_BASE_URL/models" >/dev/null
}

if ((SKIP_VLLM == 0)) && ! vlm_ready; then
  case "$VLM_HOSTNAME" in
    127.0.0.1|localhost|0.0.0.0) ;;
    *)
      echo "Cannot start a local vLLM for remote host $VLM_HOSTNAME." >&2
      echo "Start that endpoint separately and use --skip-vllm." >&2
      exit 1
      ;;
  esac
  VLLM_LOG="$RUN_DIR/logs/vllm.log"
  echo "Starting vLLM; log: $VLLM_LOG"
  CUDA_VISIBLE_DEVICES="$VLLM_GPUS" "$VLLM_PYTHON" \
    -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 --port "$VLM_PORT" \
    --model "$VLLM_MODEL" --served-model-name "$SERVED_MODEL" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$VLLM_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" --max-num-seqs 1 \
    --mm-processor-cache-gb 0 >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until vlm_ready; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "vLLM exited during startup. Last log lines:" >&2
      tail -n 50 "$VLLM_LOG" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for vLLM after $STARTUP_TIMEOUT seconds." >&2
      tail -n 50 "$VLLM_LOG" >&2
      exit 1
    fi
    sleep 5
  done
elif ! vlm_ready; then
  echo "Configured VLM endpoint is unavailable: $VLM_BASE_URL" >&2
  exit 1
else
  echo "Using the existing VLM endpoint: $VLM_BASE_URL"
fi

echo "Starting PPO training at $(date '+%Y-%m-%d %H:%M:%S %Z')"
if [ -n "$RESUME" ]; then
  CUDA_VISIBLE_DEVICES="$POLICY_GPUS" PYTHONUNBUFFERED=1 \
    "${TRAIN_CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
else
  CUDA_VISIBLE_DEVICES="$POLICY_GPUS" PYTHONUNBUFFERED=1 \
    "${TRAIN_CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
fi
echo "PPO training completed at $(date '+%Y-%m-%d %H:%M:%S %Z')"

if ((NO_WANDB == 0)); then
  echo "Uploading W&B charts"
  "${WANDB_CMD[@]}" | tee "$RUN_DIR/wandb_upload.json"
fi

echo "Evidence Policy pipeline completed: $RUN_DIR"
