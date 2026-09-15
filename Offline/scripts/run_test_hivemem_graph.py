#!/usr/bin/env python3
"""Run one strict test-split HiveMem vector-5 + graph-2 experiment.

One process owns one benchmark and one inference endpoint.  The launcher is
therefore suitable for three independent tmux sessions on GPUs 3, 4 and 5.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.io_utils import write_json_atomic  # noqa: E402
from evidence_policy.split_manifest import SplitManifestIndex  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "test_hivemem_graph_matrix.json"
BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena")
BENCHMARK_ARGUMENT = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if int(config["vector_top_k"]) + int(config["graph_append_k"]) != int(
        config["effective_top_k"]
    ):
        raise ValueError("effective_top_k must equal vector_top_k + graph_append_k")
    return config


def config_path(value: str) -> Path:
    path = Path(value).expanduser()
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
        return (ROOT / "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl",)
    if benchmark == "WorldMemArena":
        return (ROOT / "data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl",)
    return (
        ROOT / "data/h2hmem/chunks_dyadic.jsonl",
        ROOT / "data/h2hmem/chunks_multiparty.jsonl",
    )


def filter_chunks(
    sources: Iterable[Path], destination: Path, selected: set[str]
) -> dict[str, Any]:
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
        "--baseline", "HiveMem",
        "--index-root", str(memory_dir),
        "--result-dir", str(result_dir),
        "--split-manifest", str(manifest_path),
        "--split", str(config["split"]),
        "--sample-concurrency", str(config["sample_concurrency"]),
        "--answer-concurrency", str(config["answer_concurrency"]),
        "--answer-model", str(config["answer_model"]),
        "--answer-base-url", endpoint,
        "--answer-temperature", "0",
        "--executor-model", str(config["answer_model"]),
        "--executor-base-url", endpoint,
        "--executor-temperature", "0",
        "--executor-visual-input", "image",
        "--embedding-model", str(config["embedding_model"]),
        "--embedding-base-url", embedding_url,
        "--embedding-dim", str(config["embedding_dim"]),
        "--top-k", str(config["vector_top_k"]),
        "--graph-retrieval",
        "--graph-mode", str(config["graph_mode"]),
        "--append-k", str(config["graph_append_k"]),
        "--expansion-bonus", str(config["expansion_bonus"]),
        "--request-timeout", str(config["request_timeout"]),
        "--retries", str(config["retries"]),
        "--efficiency-config", str(config_path(config["efficiency_config"])),
        "--resume",
    ]
    if benchmark == "Mem-Gallery":
        return [
            sys.executable, "-m", "benchmarks.memgallery_harness.eval_memgallery",
            *common,
            "--data-dir", str(WORKSPACE / "Mem-Gallery/benchmark/data"),
            "--all-datasets",
            "--query-embedding-dir", str(
                ROOT / "data/qwen3_vl_embedding_2b/query_embeddings"
            ),
        ]
    if benchmark == "WorldMemArena":
        return [
            sys.executable, "-m", "benchmarks.wma_harness.eval_wma",
            *common,
            "--data-dir", str(WORKSPACE / "WorldMemArena/WorldMemArena/lifelong"),
            "--query-embedding-dir", str(
                ROOT / "data/wma_qwen3_vl_embedding_2b/query_embeddings_lifelong"
            ),
        ]
    return [
        sys.executable, "-m", "benchmarks.h2hmem_harness.eval_h2hmem",
        *common,
        "--data-dir", str(WORKSPACE / "H2HMEM-main/dataset"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    config = load_config()
    manifest_path = config_path(config["split_manifest"])
    manifest = SplitManifestIndex(manifest_path)
    result_dir = Path(args.output_root).resolve() / args.benchmark / "HiveMem"
    memory_dir = result_dir / "memory"
    status_path = result_dir / "pipeline_status.json"
    log_path = result_dir / "logs" / "pipeline.log"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "benchmark": args.benchmark,
        "started_at": now(),
        "endpoint": args.endpoint,
        "embedding_url": args.embedding_url,
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
                sys.executable, "-m", "hive_mem.build_memories",
                "--chunks", str(filtered_chunks),
                "--all-datasets",
                "--output-root", str(memory_dir),
                "--executor-model", str(config["answer_model"]),
                "--executor-base-url", args.endpoint,
                "--executor-timeout", str(config["request_timeout"]),
                "--executor-retries", str(config["retries"]),
                "--executor-concurrency", str(config["executor_concurrency"]),
                "--executor-visual-input", "image",
                "--embedding-model", str(config["embedding_model"]),
                "--embedding-base-url", args.embedding_url,
                "--embedding-dim", str(config["embedding_dim"]),
                "--profiles-file", str(ROOT / "configs/profiles.json"),
            ]
            run(build, log, status_path, status, "memory_build")
            dataset_dirs = sorted((memory_dir / "datasets").iterdir())
            if {path.name for path in dataset_dirs} != selected:
                raise RuntimeError("Built memory dataset set does not match test split")
            edges = [
                sys.executable, "-m", "hive_mem.build_memory_edges",
                *[str(path) for path in dataset_dirs],
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
                str(ROOT / "scripts/judge_results_llm_parallel.py"),
                "--benchmark", BENCHMARK_ARGUMENT[args.benchmark],
                "--results", str(result_dir / "results.json"),
                "--out-dir", str(result_dir),
                "--key-file", str(config_path(config["judge_key_file"])),
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
