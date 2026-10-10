#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/run.sh <command> [arguments...]

Commands:
  check          Check installed dependencies
  prepare        Download, prepare, or audit datasets
  serve          Start the local embedding service
  build          Build multimodal memory banks
  graph          Build Cross-Episode Affinity Graphs for existing memories
  gvv            Extract or export grounded visual views
  train          Train the Vertical Evidence Composition selector with PPO
  eval           Build, evaluate, and judge the graph retrieval method
  eval policy    Evaluate a trained policy or full evidence
  judge          Judge existing evaluation results

Examples:
  ./scripts/run.sh check --group train --group eval
  ./scripts/run.sh prepare chunks --benchmark memgallery
  ./scripts/run.sh build --benchmark memgallery --all-datasets
  ./scripts/run.sh train --benchmark memgallery
  ./scripts/run.sh eval policy --benchmark memgallery --strategy full-evidence --split test

Use <command> --help for command options. Settings are loaded from .env.
EOF
}

case "${1:-}" in
  ""|-h|--help|help)
    usage
    exit 0
    ;;
  check|prepare|serve|build|graph|gvv|train|eval|judge)
    hive_command="$1"
    shift
    ;;
  *)
    printf 'Unknown command: %s\n\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac

HIVE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HIVE_PROJECT_ROOT
if [[ -f "$HIVE_PROJECT_ROOT/.env" ]]; then
  set -a
  source "$HIVE_PROJECT_ROOT/.env"
  set +a
fi
export PYTHONPATH="$HIVE_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${HIVE_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$HIVE_CUDA_VISIBLE_DEVICES"
fi
HIVE_PYTHON="${HIVE_PYTHON:-python}"
cd -- "$HIVE_PROJECT_ROOT"

case "$hive_command" in
  check)
    exec "$HIVE_PYTHON" -m embedding.cli check "$@"
    ;;
  prepare)
    exec "$HIVE_PYTHON" -m embedding.cli data "$@"
    ;;
  serve)
    exec "$HIVE_PYTHON" -m embedding.backends serve "$@"
    ;;
  build)
    exec "$HIVE_PYTHON" -m src.build "$@"
    ;;
  graph)
    exec "$HIVE_PYTHON" -m src.graph "$@"
    ;;
  gvv)
    exec "$HIVE_PYTHON" -m gvv_extractor "$@"
    ;;
  train)
    exec "$HIVE_PYTHON" -m evidence_policy.cli train "$@"
    ;;
  eval)
    if [[ "${1:-}" == policy ]]; then
      shift
      exec "$HIVE_PYTHON" -m evidence_policy.cli eval "$@"
    fi
    exec "$HIVE_PYTHON" -m benchmarks.run_hivemem "$@"
    ;;
  judge)
    exec "$HIVE_PYTHON" -m benchmarks.judge "$@"
    ;;
esac
