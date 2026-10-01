#!/usr/bin/env python3
"""Build or reuse one GPT-5-mini test bank, then run one policy test."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.io_utils import write_json_atomic  # noqa: E402
from evidence_policy.split_manifest import SplitManifestIndex  # noqa: E402
from run_test_hivemem_graph import (  # noqa: E402
    filter_chunks,
    selected_dataset_ids,
    source_chunk_paths,
)


MODEL = "openai/gpt-5-mini"
BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_URL = "http://127.0.0.1:8001/v1"
MANIFEST = ROOT / "configs/multimodal_split_manifest.json"
PROFILES = ROOT / "configs/profiles.json"
EFFICIENCY_CONFIG = WORKSPACE / "Nvida_api/config_gpt-5-mini"
JUDGE_KEY_FILE = WORKSPACE / "Nvida_api/Openrouter_api"

SETTINGS: dict[str, dict[str, Any]] = {
    "memgallery": {
        "label": "Mem-Gallery",
        "base_config": ROOT / "outputs/_runs/hivemem_k4_full_20260917/configs/memgallery.json",
        "checkpoint": ROOT / "outputs/PPO/MemGallery_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt",
        "expected_qa": 275,
        "judge_benchmark": "memgallery",
    },
    "h2hmem": {
        "label": "H2HMEM",
        "base_config": ROOT / "outputs/_runs/hivemem_k4_full_20260917/configs/h2hmem.json",
        "checkpoint": ROOT / "outputs/PPO/H2HMEM_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt",
        "expected_qa": 360,
        "judge_benchmark": "h2hmem",
    },
    "wma": {
        "label": "WorldMemArena",
        "base_config": ROOT / "outputs/_runs/hivemem_k4_full_20260917/configs/wma.json",
        "checkpoint": ROOT / "outputs/PPO/WMA_0918PPO_k4_lambda000_a/checkpoints/epoch_005.pt",
        "expected_qa": 440,
        "judge_benchmark": "worldmemarena",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run(command: list[str], log, status_path: Path, status: dict[str, Any], stage: str) -> None:
    status.update({"stage": stage, "updated_at": now(), "command": command})
    write_json_atomic(status_path, status)
    log.write(f"\n=== {stage} START {now()} ===\n")
    log.write("COMMAND " + " ".join(command) + "\n")
    log.flush()
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=child_env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write(f"=== {stage} EXIT {completed.returncode} {now()} ===\n")
    log.flush()
    if completed.returncode:
        raise RuntimeError(f"{stage} exited with {completed.returncode}")


def make_eval_config(
    setting: dict[str, Any],
    memory_dir: Path,
    run_dir: Path,
    cost_tradeoff_lambda: float,
    retrieval_mode: str,
    vector_top_k: int,
    graph_append_k: int,
    degree_cap: int,
) -> Path:
    config = json.loads(Path(setting["base_config"]).read_text(encoding="utf-8"))
    config["memory_bank"] = str(memory_dir)
    config["output_dir"] = str(run_dir)
    config["top_k"] = vector_top_k
    config["retrieval_mode"] = retrieval_mode
    config["graph_options"].update(
        {
            "mode": "append",
            "append_k": graph_append_k,
            "degree_cap": degree_cap,
            "expand_temporal": False,
            "expand_related": False,
            "expand_entity": False,
            "expand_attribute": False,
        }
    )
    config["model"] = {
        "name": MODEL,
        "base_url": BASE_URL,
        "api_key": "",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_tokens": 512,
        "timeout": 180,
        "retries": 2,
        "think": False,
        "reasoning_effort": "minimal",
    }
    config["efficiency_config"] = str(EFFICIENCY_CONFIG)
    config["qa_latency_denominator"] = "queries"
    config["reward"]["cost_tradeoff_lambda"] = cost_tradeoff_lambda
    if config.get("benchmark") == "wma":
        config["excluded_categories"] = []
    config_path = run_dir / "effective_config.json"
    write_json_atomic(config_path, config)
    return config_path


def validate_graph(
    memory_dir: Path, expected_datasets: set[str], expected_degree_cap: int
) -> None:
    dataset_dirs = sorted((memory_dir / "datasets").iterdir())
    available_datasets = {path.name for path in dataset_dirs}
    missing_datasets = expected_datasets - available_datasets
    if missing_datasets:
        raise RuntimeError(
            "Memory bank is missing test datasets: "
            + ", ".join(sorted(missing_datasets))
        )
    for dataset_dir in dataset_dirs:
        report = json.loads((dataset_dir / "reports/edges.json").read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != 2
            or report.get("degree_cap") != expected_degree_cap
        ):
            raise RuntimeError(
                f"Invalid K{expected_degree_cap} attribute graph: {dataset_dir}"
            )


def validate_results(result_dir: Path, expected_qa: int, effective_top_k: int) -> None:
    results = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    if len(results) != expected_qa:
        raise RuntimeError(f"Expected {expected_qa} QA rows, found {len(results)}")
    errors = [row for row in results if row.get("error")]
    if errors:
        raise RuntimeError(f"QA errors remain: {len(errors)}")
    for row in results:
        retrieval = row.get("retrieval") or row.get("retrieval_trace") or {}
        items = retrieval.get("top_k") or row.get("retrieved_memories") or []
        if len(items) > effective_top_k:
            raise RuntimeError(f"Retrieved more than Top-{effective_top_k} memories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--executor-concurrency", type=int, default=8)
    parser.add_argument(
        "--retrieval-mode", choices=("vector", "graph_append"), default="graph_append"
    )
    parser.add_argument("--vector-top-k", type=int, default=5)
    parser.add_argument("--graph-append-k", type=int, default=2)
    parser.add_argument("--degree-cap", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        help="Override the benchmark's default lambda=0 policy checkpoint.",
    )
    parser.add_argument(
        "--cost-tradeoff-lambda",
        type=float,
        default=0.0,
        help="Lambda stored in the selected policy checkpoint.",
    )
    parser.add_argument(
        "--reuse-memory-from",
        help="Reuse an existing GPT-5-mini test memory directory and skip rebuilding it.",
    )
    parser.add_argument(
        "--strategy",
        choices=("ppo", "full-evidence"),
        default="ppo",
    )
    args = parser.parse_args()
    if args.vector_top_k < 1 or args.graph_append_k < 0 or args.degree_cap < 1:
        parser.error("top-k and degree-cap must be positive; append-k cannot be negative")
    if args.retrieval_mode == "vector" and args.graph_append_k != 0:
        parser.error("vector retrieval requires --graph-append-k 0")
    setting = SETTINGS[args.benchmark]
    checkpoint = Path(args.checkpoint or setting["checkpoint"]).resolve()
    run_dir = Path(args.output_root).resolve() / setting["label"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    memory_dir = (
        Path(args.reuse_memory_from).resolve()
        if args.reuse_memory_from
        else run_dir / "memory"
    )
    result_dir = run_dir / f"eval/test_{args.strategy}"
    status_path = run_dir / "pipeline_status.json"
    log_path = run_dir / "logs/pipeline.log"
    log_path.parent.mkdir(parents=True)
    status: dict[str, Any] = {
        "benchmark": args.benchmark,
        "model": MODEL,
        "policy_checkpoint": (
            str(checkpoint) if args.strategy == "ppo" else None
        ),
        "cost_tradeoff_lambda": args.cost_tradeoff_lambda,
        "memory_bank": str(memory_dir),
        "memory_reused": bool(args.reuse_memory_from),
        "strategy": args.strategy,
        "deterministic": True,
        "graph": {
            "schema_version": 2,
            "retrieval_mode": args.retrieval_mode,
            "vector_top_k": args.vector_top_k,
            "append_k": args.graph_append_k,
            "degree_cap": args.degree_cap,
        },
        "started_at": now(),
        "status": "running",
    }
    write_json_atomic(status_path, status)
    try:
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise ValueError("OPENROUTER_API_KEY is empty")
        if args.strategy == "ppo" and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        manifest = SplitManifestIndex(MANIFEST)
        selected = selected_dataset_ids(setting["label"], manifest, "test")
        config_path = make_eval_config(
            setting,
            memory_dir,
            run_dir,
            args.cost_tradeoff_lambda,
            args.retrieval_mode,
            args.vector_top_k,
            args.graph_append_k,
            args.degree_cap,
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            if args.reuse_memory_from:
                status["stage"] = "memory_reuse_validation"
                status["updated_at"] = now()
                write_json_atomic(status_path, status)
            else:
                chunks = run_dir / "inputs/chunks_test.jsonl"
                status["selection"] = filter_chunks(
                    source_chunk_paths(setting["label"]), chunks, selected
                )
                run(
                    [
                        sys.executable, "-m", "hive_mem.build_memories",
                        "--chunks", str(chunks),
                        "--all-datasets",
                        "--output-root", str(memory_dir),
                        "--executor-model", MODEL,
                        "--executor-base-url", BASE_URL,
                        "--executor-api-key-env", "OPENROUTER_API_KEY",
                        "--executor-reasoning-effort", "minimal",
                        "--executor-max-tokens", "512",
                        "--executor-timeout", "180",
                        "--executor-retries", "2",
                        "--executor-concurrency", str(args.executor_concurrency),
                        "--executor-visual-input", "image",
                        "--embedding-model", "Qwen/Qwen3-VL-Embedding-2B",
                        "--embedding-base-url", EMBEDDING_URL,
                        "--embedding-api-key", "EMPTY",
                        "--embedding-dim", "2048",
                        "--profiles-file", str(PROFILES),
                    ],
                    log, status_path, status, "memory_build",
                )
                dataset_dirs = sorted((memory_dir / "datasets").iterdir())
                run(
                    [
                        sys.executable,
                        "-m",
                        "hive_mem.build_memory_edges",
                        "--degree-cap",
                        str(args.degree_cap),
                        *map(str, dataset_dirs),
                    ],
                    log, status_path, status, "graph_build",
                )
            validate_graph(memory_dir, selected, args.degree_cap)
            eval_command = [
                sys.executable, str(ROOT / "scripts/evidence_policy.py"),
                "--config", str(config_path),
                "--output-dir", str(run_dir),
                "--retrieval-mode", args.retrieval_mode,
                "--top-k", str(args.vector_top_k),
                "eval", "--strategy", args.strategy, "--split", "test",
                "--device", "cpu",
            ]
            if args.strategy == "ppo":
                eval_command.extend(["--checkpoint", str(checkpoint)])
            run(
                eval_command,
                log, status_path, status, f"{args.strategy}_test",
            )
            validate_results(
                result_dir,
                int(setting["expected_qa"]),
                args.vector_top_k + args.graph_append_k,
            )
            run(
                [
                    sys.executable, str(ROOT / "scripts/judge_results_llm_parallel.py"),
                    "--benchmark", str(setting["judge_benchmark"]),
                    "--results", str(result_dir / "results.json"),
                    "--out-dir", str(result_dir),
                    "--key-file", str(JUDGE_KEY_FILE),
                    "--model", "openai/gpt-4o-mini",
                    "--workers", "16", "--timeout", "60", "--retries", "2",
                    "--max-tokens", "512", "--checkpoint-every", "25", "--resume",
                ],
                log, status_path, status, "llm_judge",
            )
        status.update({"status": "completed", "stage": "completed", "finished_at": now()})
    except Exception as exc:
        status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": now()})
        write_json_atomic(status_path, status)
        raise
    write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
