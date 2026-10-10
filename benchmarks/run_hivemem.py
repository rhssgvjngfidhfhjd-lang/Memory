#!/usr/bin/env python3
"""Build, evaluate, and judge one strict test-split HiVe_mem vector-5 + graph-2 run."""

from __future__ import annotations

from src.utils import (
    CONFIG_ROOT, chunks_path, dataset_root, output_root, profiles_path,
    query_embedding_path, resolve_reference,
)
from src.utils import apply_environment, load_runtime_config, validate_embedding_settings

from src.utils import PROJECT_ROOT as _HIVE_PROJECT_ROOT

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = _HIVE_PROJECT_ROOT

from src.utils import write_json_atomic
from benchmarks.common.utils import require_service_url
from evidence_policy.evidence import SplitManifestIndex


CONFIG_PATH = CONFIG_ROOT / "experiments.json"
BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena")
BENCHMARK_ARGUMENT = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["protocols"]["hivemem_graph"]
    runtime = load_runtime_config()
    for key in ("judge_base_url", "judge_api_key_env", "answer_base_url", "answer_model",
                "executor_base_url", "executor_model", "executor_api_key_env", "degree_cap",
                "embedding_base_url", "embedding_model", "embedding_dim", "embedding_revision", "judge_model", "efficiency_config"):
        if key in runtime:
            config[key] = runtime[key]
    config = apply_environment(config)
    if int(config["vector_top_k"]) + int(config["graph_append_k"]) != int(
        config["effective_top_k"]
    ):
        raise ValueError("effective_top_k must equal vector_top_k + graph_append_k")
    return config


def config_path(value: str) -> Path:
    path = Path(resolve_reference(value)).expanduser()
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    return path.resolve()


def selected_dataset_ids(
    benchmark: str, manifest: SplitManifestIndex, split: str
) -> set[str]:
    if benchmark == "Mem-Gallery":
        return set(manifest.source_ids(split, "mem_gallery"))
    if benchmark == "WorldMemArena":
        return set(manifest.source_ids(split, "worldmemarena_lifelong"))
    selected = set()
    for source in ("h2hmem_dyadic", "h2hmem_multiparty"):
        for row in manifest.conversations(split, data_source=source):
            selected.add(f"{row.variant}_{row.source_id}")
    return selected


def source_chunk_paths(benchmark: str) -> tuple[Path, ...]:
    if benchmark == "Mem-Gallery":
        return (chunks_path("memgallery"),)
    if benchmark == "WorldMemArena":
        return (chunks_path("wma"),)
    return (
        chunks_path("h2hmem", "dyadic"),
        chunks_path("h2hmem", "multiparty"),
    )


def filter_chunks(
    sources: Iterable[Path], destination: Path, selected: set[str]
) -> dict[str, Any]:
    sources = tuple(sources)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared chunks missing: {missing}. Run scripts/run.sh prepare chunks for the benchmark first.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    counts: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8") as output:
        for source in sources:
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dataset = str((row.get("metadata") or {}).get("dataset") or "")
                    if not dataset:
                        raise ValueError(f"Missing metadata.dataset: {source}:{line_number}")
                    if dataset in selected:
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[dataset] = counts.get(dataset, 0) + 1
    if set(counts) != selected:
        raise ValueError(
            f"Chunk selection mismatch: missing={sorted(selected - set(counts))}, "
            f"extra={sorted(set(counts) - selected)}"
        )
    temporary.replace(destination)
    return {"datasets": sorted(counts), "chunks_per_dataset": counts}


def run(command: list[str], log, status_path: Path, status: dict[str, Any], stage: str) -> None:
    status.update({"stage": stage, "updated_at": now(), "command": command})
    write_json_atomic(status_path, status)
    log.write(f"\n=== {stage} START {now()} ===\n")
    log.write("COMMAND " + " ".join(command) + "\n")
    log.flush()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.write(f"=== {stage} EXIT {completed.returncode} {now()} ===\n")
    log.flush()
    if completed.returncode:
        raise RuntimeError(f"{stage} exited with {completed.returncode}")


def evaluation_command(
    benchmark: str,
    endpoint: str,
    embedding_url: str,
    result_dir: Path,
    memory_dir: Path,
    manifest_path: Path,
    config: dict[str, Any],
) -> list[str]:
    common = [
        "--index-root", str(memory_dir),
        "--result-dir", str(result_dir),
        "--split-manifest", str(manifest_path),
        "--split", str(config["split"]),
        "--sample-concurrency", str(config["sample_concurrency"]),
        "--answer-concurrency", str(config["answer_concurrency"]),
        "--answer-model", str(config["answer_model"]),
        "--answer-base-url", endpoint,
        "--answer-temperature", "0",
        "--embedding-model", str(config["embedding_model"]),
        "--embedding-base-url", embedding_url,
        "--embedding-dim", str(config["embedding_dim"]),
        "--embedding-revision", str(config.get("embedding_revision") or ""),
        "--top-k", str(config["vector_top_k"]),
        "--append-k", str(config["graph_append_k"]),
        "--degree-cap", str(config.get("degree_cap", 4)),
        "--request-timeout", str(config["request_timeout"]),
        "--retries", str(config["retries"]),
        "--efficiency-config", str(config_path(config["efficiency_config"])),
        "--resume",
    ]
    if benchmark == "Mem-Gallery":
        return [
            sys.executable, "-m", "benchmarks.memgallery_harness.eval_memgallery",
            *common,
            "--data-dir", str(dataset_root("memgallery")),
            "--all-datasets",
            "--query-embedding-dir", str(
                query_embedding_path("memgallery", str(config["embedding_model"]))
            ),
        ]
    if benchmark == "WorldMemArena":
        return [
            sys.executable, "-m", "benchmarks.wma_harness.eval_wma",
            *common,
            "--data-dir", str(dataset_root("wma")),
            "--query-embedding-dir", str(
                query_embedding_path("wma", str(config["embedding_model"]))
            ),
        ]
    return [
        sys.executable, "-m", "benchmarks.h2hmem_harness.eval_h2hmem",
        *common,
        "--data-dir", str(dataset_root("h2hmem")),
        "--query-embedding-dir", str(
            query_embedding_path("h2hmem", str(config["embedding_model"]))
        ),
        "--variant", "all",
    ]


def validate_outputs(
    benchmark: str, result_dir: Path, expected_count: int, effective_top_k: int
) -> None:
    results = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in (result_dir / "retrieval_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(results) != expected_count or len(traces) != expected_count:
        raise RuntimeError(
            f"{benchmark}: results/traces={len(results)}/{len(traces)}, "
            f"expected {expected_count}"
        )
    if any(row.get("error") for row in results):
        raise RuntimeError(f"{benchmark}: answer errors remain")
    oversized = [row for row in traces if len(row.get("top_k") or []) > effective_top_k]
    if oversized:
        raise RuntimeError(f"{benchmark}: retrieval trace exceeds {effective_top_k}")
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    if f"retrieval_hitrate@{effective_top_k}" not in metrics:
        raise RuntimeError(f"{benchmark}: effective Top-{effective_top_k} metric missing")


def main() -> None:
    runtime = load_runtime_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--endpoint", default=runtime.get("answer_base_url", ""))
    parser.add_argument("--executor-model", default=runtime.get("executor_model", ""))
    parser.add_argument("--executor-url", default=runtime.get("executor_base_url", ""))
    parser.add_argument("--executor-api-key-env", default=runtime.get("executor_api_key_env", ""),
                        help="Custom executor key variable; otherwise use EXECUTOR_API_KEY and common key fallbacks.")
    parser.add_argument("--embedding-url", default=runtime.get("embedding_base_url", ""))
    parser.add_argument("--judge-url", default=runtime.get("judge_base_url", ""))
    parser.add_argument("--embedding-model", default=runtime.get("embedding_model", ""))
    parser.add_argument("--embedding-dim", type=int, default=runtime.get("embedding_dim"))
    parser.add_argument("--embedding-revision", default=runtime.get("embedding_revision", ""))
    parser.add_argument("--judge-model", default=runtime.get("judge_model", ""))
    parser.add_argument("--degree-cap", type=int, default=runtime.get("degree_cap", 4))
    parser.add_argument("--output-root", default=str(output_root() / "evaluation"))
    args = parser.parse_args()
    args.endpoint = require_service_url(parser, args.endpoint,
                                        flag="--endpoint", env_name="HIVE_ANSWER_BASE_URL")
    args.executor_url = require_service_url(parser, args.executor_url,
                                            flag="--executor-url", env_name="HIVE_EXECUTOR_BASE_URL")
    args.executor_model = str(args.executor_model or "").strip()
    if not args.executor_model:
        parser.error("--executor-model is required; set HIVE_EXECUTOR_MODEL, provide --executor-model, or configure executor_model in HIVE_CONFIG")
    if args.degree_cap < 1:
        parser.error("--degree-cap must be positive")
    args.embedding_url = require_service_url(parser, args.embedding_url,
                                             flag="--embedding-url", env_name="HIVE_EMBEDDING_BASE_URL")
    args.judge_url = require_service_url(parser, args.judge_url,
                                         flag="--judge-url", env_name="HIVE_JUDGE_BASE_URL")
    args.embedding_model, args.embedding_dim = validate_embedding_settings(
        parser, args.embedding_model, args.embedding_dim,
    )
    args.judge_model = str(args.judge_model or "").strip()
    if not args.judge_model:
        parser.error("--judge-model is required; set HIVE_JUDGE_MODEL, provide --judge-model, or configure judge_model in HIVE_CONFIG")

    config = load_config()
    config.update(answer_base_url=args.endpoint, embedding_base_url=args.embedding_url,
                  judge_base_url=args.judge_url, embedding_model=args.embedding_model,
                  embedding_dim=args.embedding_dim, judge_model=args.judge_model,
                  embedding_revision=str(args.embedding_revision or "").strip(),
                  executor_model=args.executor_model, executor_base_url=args.executor_url,
                  executor_api_key_env=args.executor_api_key_env, degree_cap=args.degree_cap)
    manifest_path = config_path(config["split_manifest"])
    manifest = SplitManifestIndex(manifest_path)
    result_dir = Path(args.output_root).resolve() / args.benchmark / "HiVe_mem"
    memory_dir = result_dir / "memory"
    status_path = result_dir / "pipeline_status.json"
    log_path = result_dir / "logs" / "pipeline.log"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "benchmark": args.benchmark,
        "started_at": now(),
        "endpoint": args.endpoint,
        "executor_model": args.executor_model,
        "executor_url": args.executor_url,
        "embedding_url": args.embedding_url,
        "judge_url": args.judge_url,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "judge_model": args.judge_model,
        "result_dir": str(result_dir),
        "configuration": config,
        "status": "running",
    }
    write_json_atomic(result_dir / "graph_experiment_config.json", status)

    try:
        selected = selected_dataset_ids(args.benchmark, manifest, str(config["split"]))
        filtered_chunks = result_dir / "inputs" / "chunks_test.jsonl"
        selection_stats = filter_chunks(
            source_chunk_paths(args.benchmark), filtered_chunks, selected
        )
        status["selection"] = selection_stats
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            build = [
                sys.executable, "-m", "src.build",
                "--chunks", str(filtered_chunks),
                "--all-datasets",
                "--output-root", str(memory_dir),
                "--executor-model", args.executor_model,
                "--executor-base-url", args.executor_url,
                "--executor-timeout", str(config["request_timeout"]),
                "--executor-retries", str(config["retries"]),
                "--executor-concurrency", str(config["executor_concurrency"]),
                "--executor-max-tokens", str(config["executor_max_tokens"]),
                "--embedding-model", str(config["embedding_model"]),
                "--embedding-base-url", args.embedding_url,
                "--embedding-dim", str(config["embedding_dim"]),
                "--profiles-file", str(profiles_path()),
            ]
            if args.executor_api_key_env:
                build.extend(("--executor-api-key-env", args.executor_api_key_env))
            if config["embedding_revision"]:
                build.extend(("--embedding-revision", config["embedding_revision"]))
            run(build, log, status_path, status, "memory_build")
            dataset_dirs = sorted((memory_dir / "datasets").iterdir())
            if {path.name for path in dataset_dirs} != selected:
                raise RuntimeError("Built memory dataset set does not match test split")
            edges = [
                sys.executable, "-m", "src.graph",
                *[str(path) for path in dataset_dirs],
                "--degree-cap", str(args.degree_cap),
            ]
            run(edges, log, status_path, status, "edge_build")
            evaluate = evaluation_command(
                args.benchmark,
                args.endpoint,
                args.embedding_url,
                result_dir,
                memory_dir,
                manifest_path,
                config,
            )
            run(evaluate, log, status_path, status, "retrieve_and_answer")
            validate_outputs(
                args.benchmark,
                result_dir,
                int(config["expected_qa_counts"][args.benchmark]),
                int(config["effective_top_k"]),
            )
            judge = [
                sys.executable,
                "-m", "benchmarks.judge",
                "--benchmark", BENCHMARK_ARGUMENT[args.benchmark],
                "--results", str(result_dir / "results.json"),
                "--out-dir", str(result_dir),
                "--api-key-env", str(config.get("judge_api_key_env", "JUDGE_API_KEY")),
                "--base-url", args.judge_url,
                "--model", str(config["judge_model"]),
                "--workers", str(config["judge_workers"]),
                "--timeout", str(config["judge_timeout"]),
                "--retries", str(config["retries"]),
                "--max-tokens", str(config["judge_max_tokens"]),
                "--checkpoint-every", "25",
                "--resume",
            ]
            run(judge, log, status_path, status, "llm_judge")
        status.update({"status": "completed", "stage": "completed", "finished_at": now()})
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "finished_at": now(),
            }
        )
        write_json_atomic(status_path, status)
        raise
    write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
