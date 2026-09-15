#!/usr/bin/env python3
"""Run the MemVerse control with padded-2048 text embedding 0.6B/Top-5.

This is a deliberately isolated entrypoint.  It delegates execution to the
standard baseline matrix without changing that runner's default 2B/Top-7
protocol or its configuration files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULTS_PATH = ROOT / "configs" / "defaults.json"
EFFICIENCY_PATH = ROOT / "configs" / "model_efficiency.json"
SPLIT_MANIFEST_PATH = ROOT / "configs" / "multimodal_split_manifest.json"
OUTPUT_ROOT = ROOT / "outputs"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_NATIVE_DIM = 1024
EMBEDDING_DIM = 2048
EMBEDDING_BASE_URL = "http://127.0.0.1:8002/v1"
TOP_K = 5
BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena")


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run ID must be a safe path component")
    return value


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime("memverse_emb06_top5_%Y%m%d_%H%M%S")


def build_experiment_defaults(source: Path) -> dict[str, Any]:
    config = json.loads(source.read_text(encoding="utf-8"))
    config.update(
        {
            "top_k": TOP_K,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_native_dim": EMBEDDING_NATIVE_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            # Never let this control accidentally consume the default 2B query
            # cache. MemVerse embeds its LightRAG text online, but retaining an
            # isolated path makes the derived configuration safe to audit.
            "query_embedding_dir": "data/qwen3_embedding_0_6b/query_embeddings",
        }
    )
    return config


def _write_derived_defaults(
    *, source: Path, output_root: Path, run_id: str
) -> Path:
    config = build_experiment_defaults(source)
    run_root = output_root / "_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "memverse_embedding06_top5_defaults.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_safe_run_id, default=_default_run_id())
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="Repeat for answer endpoints; defaults to local ports 8013-8015.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=EMBEDDING_BASE_URL,
        help="Dedicated Qwen3-Embedding-0.6B OpenAI-compatible endpoint.",
    )
    parser.add_argument("--defaults", type=Path, default=DEFAULTS_PATH)
    parser.add_argument("--efficiency-config", type=Path, default=EFFICIENCY_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--reuse-state",
        action="store_true",
        help="Resume the exact same Run-ID and preserve unfinished MemVerse state.",
    )
    args = parser.parse_args()
    if args.skip_smoke and args.smoke_only:
        parser.error("--skip-smoke and --smoke-only cannot be combined")
    # Accept exactly the declared endpoint with or without its /v1 suffix.
    normalized = args.embedding_base_url.rstrip("/")
    if normalized not in {"http://127.0.0.1:8002", EMBEDDING_BASE_URL.rstrip("/")}:
        parser.error("this control entrypoint fixes embedding to port 8002")
    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    derived_defaults = _write_derived_defaults(
        source=args.defaults.expanduser().resolve(),
        output_root=output_root,
        run_id=args.run_id,
    )

    # Import only after this wrapper has parsed its own constrained interface.
    # The overrides below are process-local and leave the standard runner and
    # protocol files byte-for-byte unchanged.
    from scripts import run_test_baseline_matrix as matrix

    matrix.EMBEDDING_MODEL = EMBEDDING_MODEL
    matrix.PROTOCOL = {**matrix.PROTOCOL, "top_k": TOP_K}
    matrix.SMOKE_JOB_ORDER = tuple(("MemVerse", name) for name in BENCHMARKS)

    if args.reuse_state:
        os.environ["MEMVERSE_REUSE_STATE"] = "1"
    else:
        os.environ.pop("MEMVERSE_REUSE_STATE", None)

    endpoints = args.endpoint or [
        "http://127.0.0.1:8013/v1",
        "http://127.0.0.1:8014/v1",
        "http://127.0.0.1:8015/v1",
    ]
    delegated = [
        str(Path(matrix.__file__).resolve()),
        "--defaults",
        str(derived_defaults),
        "--efficiency-config",
        str(args.efficiency_config.expanduser().resolve()),
        "--split-manifest",
        str(args.split_manifest.expanduser().resolve()),
        "--output-root",
        str(output_root),
        "--run-id",
        args.run_id,
        "--embedding-base-url",
        EMBEDDING_BASE_URL,
        "--baseline",
        "MemVerse",
    ]
    for benchmark in BENCHMARKS:
        delegated.extend(["--benchmark", benchmark])
    for endpoint in endpoints:
        delegated.extend(["--endpoint", endpoint])
    if args.skip_smoke:
        delegated.append("--skip-smoke")
    if args.smoke_only:
        delegated.append("--smoke-only")

    print(
        json.dumps(
            {
                "experiment": "memverse_embedding06_top5",
                "run_id": args.run_id,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_native_dim": EMBEDDING_NATIVE_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
                "embedding_base_url": EMBEDDING_BASE_URL,
                "top_k": TOP_K,
                "answer_endpoints": endpoints,
                "derived_defaults": str(derived_defaults),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    sys.argv = delegated
    matrix.main()


if __name__ == "__main__":
    main()
