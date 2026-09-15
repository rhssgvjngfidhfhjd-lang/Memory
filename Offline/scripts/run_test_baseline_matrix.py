#!/usr/bin/env python3
"""Run all non-HiveMem baselines on the manifest-defined test split.

The runner creates read-only staged dataset views containing only test
conversations, runs a shortest-job-first queue over the configured answer endpoints,
and starts OpenRouter judging on completed outputs without occupying a GPU
worker.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from evidence_policy.split_manifest import SplitManifestIndex  # noqa: E402
from benchmarks.h2hmem_harness.prompts import (  # noqa: E402
    prompt_sha256 as h2hmem_prompt_sha256,
)
from benchmarks.memgallery_harness.runner.prompts import (  # noqa: E402
    prompt_sha256 as memgallery_prompt_sha256,
)
from benchmarks.wma_harness.runner.prompts import (  # noqa: E402
    prompt_sha256 as wma_prompt_sha256,
)


EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
PROTOCOL_PATH = ROOT / "configs" / "test_baseline_matrix.json"
BENCHMARK_ARGUMENT = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
}
# Confirmed shortest-job-first order from docs/plan_baseline.md. MemVerse is
# deliberately last because its memory construction is substantially slower.
JOB_ORDER = (
    ("M3-Agent-caption", "H2HMEM"),
    ("M3-Agent-caption", "Mem-Gallery"),
    ("M3-Agent-caption", "WorldMemArena"),
    ("M2A", "Mem-Gallery"),
    ("M2A", "WorldMemArena"),
    ("M2A", "H2HMEM"),
    ("AUGUSTUSMemory", "H2HMEM"),
    ("AUGUSTUSMemory", "WorldMemArena"),
    ("AUGUSTUSMemory", "Mem-Gallery"),
    ("MMA", "Mem-Gallery"),
    ("MIRIX", "Mem-Gallery"),
    ("MMA", "H2HMEM"),
    ("OmniSimpleMem", "Mem-Gallery"),
    ("OmniSimpleMem", "H2HMEM"),
    ("MIRIX", "H2HMEM"),
    ("MMA", "WorldMemArena"),
    ("MIRIX", "WorldMemArena"),
    ("OmniSimpleMem", "WorldMemArena"),
    ("MemVerse", "Mem-Gallery"),
    ("MemVerse", "H2HMEM"),
    ("MemVerse", "WorldMemArena"),
)
SMOKE_JOB_ORDER = (
    ("M3-Agent-caption", "H2HMEM"),
    ("M3-Agent-caption", "Mem-Gallery"),
    ("M3-Agent-caption", "WorldMemArena"),
    ("M2A", "Mem-Gallery"),
    ("M2A", "H2HMEM"),
    ("M2A", "WorldMemArena"),
    ("AUGUSTUSMemory", "Mem-Gallery"),
    ("MMA", "Mem-Gallery"),
    ("MIRIX", "Mem-Gallery"),
    ("OmniSimpleMem", "Mem-Gallery"),
)
def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slug(value: str) -> str:
    return value.casefold().replace("-", "_").replace(" ", "_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_job_attempts(config: dict[str, Any]) -> int:
    attempts = int(config.get("formal_job_attempts", 2))
    if attempts < 1:
        raise ValueError("formal_job_attempts must be positive")
    return attempts


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_json(path)
    required = {
        "split",
        "run_id",
        "split_manifest",
        "top_k",
        "efficiency_config",
        "expected_qa_counts",
        "smoke_expected_qa_counts",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"Missing test matrix protocol keys: {missing}")
    if str(protocol["split"]).casefold() != "test":
        raise ValueError("The baseline matrix protocol must select the test split")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(protocol["run_id"])):
        raise ValueError("Protocol run_id must be a safe path component")
    expected_counts = protocol["expected_qa_counts"]
    smoke_counts = protocol["smoke_expected_qa_counts"]
    expected_benchmarks = set(BENCHMARK_ARGUMENT)
    if set(expected_counts) != expected_benchmarks:
        raise ValueError("Protocol expected_qa_counts has the wrong benchmarks")
    if set(smoke_counts) != expected_benchmarks:
        raise ValueError("Protocol smoke_expected_qa_counts has the wrong benchmarks")
    if int(protocol["top_k"]) < 1:
        raise ValueError("Protocol top_k must be positive")
    return protocol


PROTOCOL = load_protocol()
SPLIT_NAME = str(PROTOCOL["split"])
RUN_ID = str(PROTOCOL["run_id"])
EXPECTED_COUNTS = {
    benchmark: int(count)
    for benchmark, count in PROTOCOL["expected_qa_counts"].items()
}
SMOKE_EXPECTED_COUNTS = {
    benchmark: int(count)
    for benchmark, count in PROTOCOL["smoke_expected_qa_counts"].items()
}


def configured_split_manifest() -> Path:
    path = Path(str(PROTOCOL["split_manifest"])).expanduser()
    if not path.is_absolute():
        path = PROTOCOL_PATH.parent / path
    return path.resolve()


def configured_efficiency_config() -> Path:
    path = Path(str(PROTOCOL["efficiency_config"])).expanduser()
    if not path.is_absolute():
        path = PROTOCOL_PATH.parent / path
    return path.resolve()


def formal_result_dir(output_root: Path, job: Job) -> Path:
    return output_root / job.benchmark / job.method / RUN_ID


def run_artifact_root(output_root: Path) -> Path:
    return output_root / "_runs" / RUN_ID


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    api_key: str = "EMPTY",
    timeout: int = 180,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class Job:
    method: str
    benchmark: str

    @property
    def name(self) -> str:
        return f"{slug(self.method)}__{slug(self.benchmark)}"


class Status:
    def __init__(self, path: Path, endpoints: list[str], output_root: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "runner_pid": os.getpid(),
            "phase": "preflight",
            "started_at": now(),
            "updated_at": now(),
            "endpoints": endpoints,
            "output_root": str(output_root),
            "jobs": {},
            "judges": {},
        }
        self.write()

    def update_root(self, **values: Any) -> None:
        with self.lock:
            self.data.update(values)
            self._write_unlocked()

    def update(self, section: str, key: str, **values: Any) -> None:
        with self.lock:
            row = self.data.setdefault(section, {}).setdefault(key, {})
            row.update(values)
            self._write_unlocked()

    def write(self) -> None:
        with self.lock:
            self._write_unlocked()

    def _write_unlocked(self) -> None:
        self.data["updated_at"] = now()
        write_json_atomic(self.path, self.data)


@dataclass(frozen=True)
class Selection:
    memgallery: tuple[str, ...]
    wma: tuple[str, ...]
    h2h: tuple[tuple[str, str], ...]
    ordered_questions: tuple[tuple[str, tuple[str, ...]], ...]
    manifest_path: Path
    manifest_sha256: str

    def for_benchmark(self, benchmark: str) -> list[str]:
        if benchmark == "Mem-Gallery":
            return list(self.memgallery)
        if benchmark == "WorldMemArena":
            return list(self.wma)
        return [f"{variant}:{source_id}" for variant, source_id in self.h2h]

    def question_ids_for_benchmark(self, benchmark: str) -> tuple[str, ...]:
        try:
            return dict(self.ordered_questions)[benchmark]
        except KeyError as exc:
            raise KeyError(f"Unknown benchmark: {benchmark}") from exc


def selection_from_manifest(path: Path) -> Selection:
    index = SplitManifestIndex(path)
    h2h_sources = tuple(
        source for source in index.data_sources if source.startswith("h2hmem_")
    )
    h2h_rows = tuple(
        row
        for source in h2h_sources
        for row in index.conversations(SPLIT_NAME, data_source=source)
    )
    h2h = tuple((row.variant, row.source_id) for row in h2h_rows)
    ordered_questions = (
        (
            "Mem-Gallery",
            index.ordered_question_ids(SPLIT_NAME, data_source="mem_gallery"),
        ),
        (
            "H2HMEM",
            tuple(
                question_id
                for row in h2h_rows
                for question_id in row.question_ids
            ),
        ),
        (
            "WorldMemArena",
            index.ordered_question_ids(
                SPLIT_NAME, data_source="worldmemarena_lifelong"
            ),
        ),
    )
    selection = Selection(
        memgallery=index.source_ids(SPLIT_NAME, "mem_gallery"),
        wma=index.source_ids(SPLIT_NAME, "worldmemarena_lifelong"),
        h2h=h2h,
        ordered_questions=ordered_questions,
        manifest_path=index.path,
        manifest_sha256=index.file_sha256,
    )
    return selection


def load_selection(path: Path) -> Selection:
    selection = selection_from_manifest(path)
    actual = {
        benchmark: len(question_ids)
        for benchmark, question_ids in selection.ordered_questions
    }
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected test question counts: {actual}")
    if (len(selection.memgallery), len(selection.h2h), len(selection.wma)) != (4, 5, 8):
        raise ValueError("Unexpected test conversation counts")
    all_question_ids = [
        question_id
        for _, question_ids in selection.ordered_questions
        for question_id in question_ids
    ]
    if any(not question_id for question_id in all_question_ids):
        raise ValueError("Test split contains an empty question ID")
    if len(all_question_ids) != len(set(all_question_ids)):
        raise ValueError("Test split contains duplicate question IDs")
    return selection


def write_smoke_manifest(source_path: Path, destination: Path) -> Selection:
    payload = load_json(source_path)
    for dataset in payload.get("datasets", []):
        for split_name, split_payload in (dataset.get("splits") or {}).items():
            if split_name != SPLIT_NAME:
                split_payload["conversations"] = []
                split_payload["conversation_count"] = 0
                split_payload["question_count"] = 0
                continue
            rows = split_payload.get("conversations") or []
            selected_rows = rows[:1]
            for row in selected_rows:
                row["question_ids"] = list(row.get("question_ids") or [])[:1]
            split_payload["conversations"] = selected_rows
            split_payload["conversation_count"] = len(selected_rows)
            split_payload["question_count"] = sum(
                len(row["question_ids"]) for row in selected_rows
            )
    payload["sha256"] = "derived-at-runtime; see file hash"
    write_json_atomic(destination, payload)
    selection = selection_from_manifest(destination)
    actual = {
        benchmark: len(selection.question_ids_for_benchmark(benchmark))
        for benchmark in EXPECTED_COUNTS
    }
    if actual != SMOKE_EXPECTED_COUNTS:
        raise ValueError(f"Unexpected smoke manifest counts: {actual}")
    return selection


def ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"Existing staged link has wrong target: {link}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace staged path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def stage_inputs(stage_root: Path, selection: Selection) -> dict[str, Path]:
    memgallery_source = WORKSPACE / "Mem-Gallery" / "benchmark" / "data"
    memgallery_stage = stage_root / "Mem-Gallery" / "data"
    ensure_link(memgallery_stage / "image", memgallery_source / "image")
    for source_id in selection.memgallery:
        ensure_link(
            memgallery_stage / "dialog" / f"{source_id}.json",
            memgallery_source / "dialog" / f"{source_id}.json",
        )

    h2h_source = WORKSPACE / "H2HMEM-main" / "dataset"
    h2h_stage = stage_root / "H2HMEM" / "dataset"
    for variant, source_id in selection.h2h:
        directory = "multi-party" if variant == "multiparty" else "dyadic"
        ensure_link(
            h2h_stage / directory / source_id,
            h2h_source / directory / source_id,
        )

    wma_source = WORKSPACE / "WorldMemArena" / "WorldMemArena" / "lifelong"
    wma_stage = stage_root / "WorldMemArena" / "lifelong"
    source_files: dict[str, Path] = {}
    for path in wma_source.rglob("*.json"):
        if path.stem in source_files:
            raise RuntimeError(f"Duplicate WMA sample stem: {path.stem}")
        source_files[path.stem] = path
    for source_id in selection.wma:
        source_path = source_files.get(source_id)
        if source_path is None:
            raise FileNotFoundError(f"Missing WMA test sample: {source_id}")
        ensure_link(wma_stage / source_path.relative_to(wma_source), source_path)
    for image_dir in wma_source.rglob("images"):
        if image_dir.is_dir():
            ensure_link(wma_stage / image_dir.relative_to(wma_source), image_dir)

    write_json_atomic(
        stage_root / "selection.json",
        {
            "split": SPLIT_NAME,
            "manifest": str(selection.manifest_path),
            "manifest_sha256": selection.manifest_sha256,
            "Mem-Gallery": list(selection.memgallery),
            "H2HMEM": [
                {"variant": variant, "source_id": source_id}
                for variant, source_id in selection.h2h
            ],
            "WorldMemArena": list(selection.wma),
            "ordered_question_ids": {
                benchmark: list(question_ids)
                for benchmark, question_ids in selection.ordered_questions
            },
            "expected_questions": EXPECTED_COUNTS,
        },
    )
    return {
        "Mem-Gallery": memgallery_stage,
        "H2HMEM": h2h_stage,
        "WorldMemArena": wma_stage,
    }


def check_services(endpoints: list[str], embedding_url: str, config: dict[str, Any]) -> None:
    if not endpoints:
        raise ValueError("At least one answer endpoint is required")
    answer_model = str(config["answer_model"])
    answer_api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    for endpoint in endpoints:
        payload = request_json(
            endpoint.rstrip("/") + "/models",
            api_key=answer_api_key,
        )
        models = {str(row.get("id")) for row in payload.get("data") or []}
        if answer_model not in models:
            raise RuntimeError(
                f"Answer model {answer_model!r} missing at {endpoint}: {sorted(models)}"
            )
    payload = request_json(
        embedding_url.rstrip("/") + "/embeddings",
        {"model": EMBEDDING_MODEL, "input": ["test-only matrix preflight"]},
    )
    rows = payload.get("data") or []
    dimension = len(rows[0].get("embedding") or []) if rows else 0
    if dimension != int(config["embedding_dim"]):
        raise RuntimeError(f"Embedding dimension is {dimension}, expected 2048")

    key_file = Path(str(config["judge_key_file"]))
    if not key_file.is_absolute():
        key_file = (ROOT / key_file).resolve()
    matches = re.findall(
        r"sk-or-v1-[A-Za-z0-9_-]+",
        key_file.read_text(encoding="utf-8") if key_file.is_file() else "",
    )
    if not matches:
        raise RuntimeError(f"No OpenRouter key found in {key_file}")
    judge_payload = request_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": str(config["judge_model"]),
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "max_tokens": 2,
        },
        api_key=matches[0],
        timeout=int(config["judge_timeout"]),
    )
    if not (judge_payload.get("choices") or []):
        raise RuntimeError("OpenRouter judge health check returned no choices")


def common_args(
    job: Job,
    result_dir: Path,
    endpoint: str,
    embedding_url: str,
    config: dict[str, Any],
    *,
    smoke: bool,
) -> list[str]:
    executor_max_tokens = int(config["executor_max_tokens"])
    if job.method == "M2A":
        executor_max_tokens = int(
            config.get("m2a_executor_max_tokens") or executor_max_tokens
        )
    elif job.method == "MMA":
        executor_max_tokens = int(
            config.get("mma_executor_max_tokens") or 4096
        )
    elif job.method == "M3-Agent-caption":
        executor_max_tokens = int(
            config.get("m3_executor_max_tokens") or 1024
        )
    elif job.method == "MIRIX":
        executor_max_tokens = int(
            config.get("mirix_executor_max_tokens") or 8192
        )
    arguments = [
        "--baseline", job.method,
        "--result-dir", str(result_dir),
        "--baseline-state-dir", str(result_dir / "memory" / "datasets"),
        "--sample-concurrency", "1" if smoke else str(config["sample_concurrency"]),
        "--answer-concurrency", "1" if smoke else str(config["answer_concurrency"]),
        "--checkpoint-every", str(config["checkpoint_every"]),
        "--answer-model", str(config["answer_model"]),
        "--answer-base-url", endpoint,
        "--answer-temperature", str(config["answer_temperature"]),
        "--num-predict", str(config["num_predict"]),
        "--executor-model", str(config["executor_model"]),
        "--executor-base-url", endpoint,
        "--executor-temperature", str(config["executor_temperature"]),
        "--executor-max-tokens", str(executor_max_tokens),
        "--executor-visual-input", str(config["executor_visual_input"]),
        "--embedding-model", str(config["embedding_model"]),
        "--embedding-base-url", embedding_url,
        "--embedding-dim", str(config["embedding_dim"]),
        "--top-k", str(config["top_k"]),
        "--request-timeout", str(config["request_timeout"]),
        "--retries", str(config["retries"]),
        "--efficiency-config", str(config["efficiency_config"]),
        "--resume",
    ]
    reasoning_effort = str(config.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        arguments.extend(["--reasoning-effort", reasoning_effort])
    if job.method == "M2A" and job.benchmark == "WorldMemArena":
        arguments.extend(
            [
                "--m2a-wma-rounds-per-ingest",
                str(int(config.get("m2a_wma_rounds_per_ingest") or 1)),
            ]
        )
    if job.method == "M2A" and bool(
        config.get("m2a_skip_failed_build_points", False)
    ):
        arguments.extend(
            [
                "--m2a-skip-failed-build-points",
                "--m2a-max-consecutive-failed-build-points",
                str(
                    int(
                        config.get(
                            "m2a_max_consecutive_failed_build_points", 10
                        )
                    )
                ),
            ]
        )
    if job.method == "MMA":
        arguments.extend(
            [
                "--mma-native-batch-size",
                str(int(config.get("mma_native_batch_size") or 20)),
            ]
        )
    if job.method == "M3-Agent-caption":
        reuse_roots = config.get("m3_reuse_state_roots") or {}
        reuse_root = str(reuse_roots.get(job.benchmark) or "").strip()
        if reuse_root:
            arguments.extend(["--m3-reuse-state-root", reuse_root])
    return arguments


def command_for(
    job: Job,
    result_dir: Path,
    endpoint: str,
    embedding_url: str,
    config: dict[str, Any],
    data_dirs: dict[str, Path],
    selection: Selection,
    *,
    smoke: bool = False,
) -> list[str]:
    common = common_args(
        job, result_dir, endpoint, embedding_url, config, smoke=smoke
    )
    manifest_path = selection.manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
    strict_selection = [
        "--split-manifest", str(manifest_path),
        "--split", SPLIT_NAME,
    ]
    if job.benchmark == "Mem-Gallery":
        command = [
            "-m", "benchmarks.memgallery_harness.eval_memgallery",
            *common,
            "--data-dir", str(data_dirs[job.benchmark]),
            "--all-datasets",
            *strict_selection,
        ]
        if job.method == "MIRIX" and bool(
            config.get("mirix_skip_failed_build_points", False)
        ):
            command.extend(
                [
                    "--mirix-skip-failed-build-points",
                    "--mirix-max-consecutive-failed-build-points",
                    str(
                        int(
                            config.get(
                                "mirix_max_consecutive_failed_build_points", 10
                            )
                        )
                    ),
                ]
            )
        retrieval_tokenizer = str(
            config.get("retrieval_memory_tokenizer") or ""
        ).strip()
        if retrieval_tokenizer:
            command.extend(["--retrieval-memory-tokenizer", retrieval_tokenizer])
        return command
    if job.benchmark == "WorldMemArena":
        command = [
            "-m", "benchmarks.wma_harness.eval_wma",
            *common,
            "--data-dir", str(data_dirs[job.benchmark]),
            *strict_selection,
        ]
        if job.method == "MIRIX":
            command.extend(
                [
                    "--sample-attempts",
                    str(int(config.get("wma_sample_attempts") or 3)),
                ]
            )
            if bool(config.get("mirix_skip_failed_build_points", False)):
                command.extend(
                    [
                        "--mirix-skip-failed-build-points",
                        "--mirix-max-consecutive-failed-build-points",
                        str(
                            int(
                                config.get(
                                    "mirix_max_consecutive_failed_build_points",
                                    10,
                                )
                            )
                        ),
                    ]
                )
        return command
    command = [
        "-m", "benchmarks.h2hmem_harness.eval_h2hmem",
        *common,
        "--data-dir", str(data_dirs[job.benchmark]),
        *strict_selection,
    ]
    if job.method == "MIRIX" and bool(
        config.get("mirix_skip_failed_build_points", False)
    ):
        command.extend(
            [
                "--mirix-skip-failed-build-points",
                "--mirix-max-consecutive-failed-build-points",
                str(
                    int(
                        config.get(
                            "mirix_max_consecutive_failed_build_points", 10
                        )
                    )
                ),
            ]
        )
    return command


def sample_keys(results: list[dict[str, Any]], benchmark: str) -> set[str]:
    if benchmark == "Mem-Gallery":
        return {str(row.get("dataset") or "") for row in results}
    if benchmark == "WorldMemArena":
        return {str(row.get("sample_id") or row.get("dataset") or "") for row in results}
    return {
        f"{row.get('variant')}:{row.get('conversation_id') or row.get('sample_id')}"
        for row in results
    }


def validate_run_manifest_selection(
    job: Job,
    manifest: dict[str, Any],
    selection: Selection,
) -> None:
    """Reject legacy conversation-only runs before accepting their metrics."""
    expected_question_ids = selection.question_ids_for_benchmark(job.benchmark)
    if manifest.get("selection_mode") != "strict_manifest":
        raise RuntimeError(
            f"{job.name}: run did not use strict question-level manifest selection"
        )
    if str(manifest.get("split") or "").casefold() != SPLIT_NAME.casefold():
        raise RuntimeError(
            f"{job.name}: run_manifest split is not {SPLIT_NAME}"
        )
    raw_manifest_path = str(manifest.get("split_manifest") or "")
    if not raw_manifest_path:
        raise RuntimeError(f"{job.name}: run_manifest has no split manifest")
    if Path(raw_manifest_path).expanduser().resolve() != selection.manifest_path.resolve():
        raise RuntimeError(f"{job.name}: run used a different split manifest")
    if manifest.get("split_manifest_sha256") != selection.manifest_sha256:
        raise RuntimeError(f"{job.name}: split manifest hash mismatch")
    if int(manifest.get("questions", -1)) != len(expected_question_ids):
        raise RuntimeError(f"{job.name}: run_manifest question count mismatch")
    actual_question_ids = tuple(
        str(value) for value in (manifest.get("ordered_question_ids") or ())
    )
    if actual_question_ids != expected_question_ids:
        raise RuntimeError(
            f"{job.name}: run_manifest question IDs do not exactly match manifest order"
        )
    expected_prompt_sha = {
        "Mem-Gallery": memgallery_prompt_sha256(),
        "H2HMEM": h2hmem_prompt_sha256(),
        "WorldMemArena": wma_prompt_sha256(),
    }.get(job.benchmark)
    actual_prompt_sha = manifest.get("prompt_sha256")
    if expected_prompt_sha and actual_prompt_sha != expected_prompt_sha:
        raise RuntimeError(f"{job.name}: answer prompt hash mismatch")


def validate_output(
    job: Job,
    result_dir: Path,
    selection: Selection,
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> None:
    results_path = result_dir / "results.json"
    trace_path = result_dir / "retrieval_trace.jsonl"
    pipeline_path = result_dir / "pipeline_qa.jsonl"
    snapshot_path = result_dir / "memory" / "memory_snapshot.jsonl"
    manifest_path = result_dir / "run_manifest.json"
    metrics_path = result_dir / "metrics.json"
    efficiency_path = result_dir / "efficiency_metrics.json"
    for path in (
        results_path,
        trace_path,
        pipeline_path,
        snapshot_path,
        manifest_path,
        metrics_path,
        efficiency_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"Missing output: {path}")
    manifest = load_json(manifest_path)
    validate_run_manifest_selection(job, manifest, selection)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pipeline_rows = [
        json.loads(line)
        for line in pipeline_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = len(selection.question_ids_for_benchmark(job.benchmark))
    if (
        len(results) != expected_count
        or len(traces) != expected_count
        or len(pipeline_rows) != expected_count
    ):
        raise RuntimeError(
            f"{job.name}: results/traces/pipeline="
            f"{len(results)}/{len(traces)}/{len(pipeline_rows)}, "
            f"expected {expected_count}"
        )
    errors = [row for row in results if row.get("error")]
    if errors:
        raise RuntimeError(f"{job.name}: {len(errors)} answer errors")
    oversized = [
        row for row in traces
        if not isinstance(row.get("top_k"), list)
        or len(row["top_k"]) > int(config["top_k"])
    ]
    if oversized:
        raise RuntimeError(f"{job.name}: malformed Top-{config['top_k']} traces")
    metrics = load_json(metrics_path)
    efficiency = load_json(efficiency_path)
    for metric_name in (
        "cost_mb",
        "cost_qa",
        "cost_total",
        "latency_mb",
        "latency_qa",
        "latency_total",
    ):
        metric = metrics.get(metric_name)
        if not isinstance(metric, dict) or not metric.get("available"):
            raise RuntimeError(
                f"{job.name}: required efficiency metric unavailable: {metric_name}"
            )
        if metric != efficiency.get(metric_name):
            raise RuntimeError(
                f"{job.name}: metrics.json and efficiency_metrics.json differ "
                f"for {metric_name}"
            )
    expected_question_ids = selection.question_ids_for_benchmark(job.benchmark)
    for label, rows in (
        ("results", results),
        ("retrieval_trace", traces),
        ("pipeline_qa", pipeline_rows),
    ):
        actual_question_ids = tuple(
            str(row.get("manifest_question_id") or "") for row in rows
        )
        if actual_question_ids != expected_question_ids:
            raise RuntimeError(
                f"{job.name}: {label} question IDs do not exactly match "
                "manifest order"
            )
    if not smoke:
        expected_samples = set(selection.for_benchmark(job.benchmark))
        actual_samples = sample_keys(results, job.benchmark)
        if actual_samples != expected_samples:
            raise RuntimeError(
                f"{job.name}: sample set mismatch: {sorted(actual_samples)}"
            )
    manifest.update(
        {
            "split": SPLIT_NAME,
            "split_manifest": str(selection.manifest_path),
            "split_manifest_sha256": selection.manifest_sha256,
            "selected_conversations": selection.for_benchmark(job.benchmark),
            "expected_questions": expected_count,
            "top_k": int(config["top_k"]),
            "smoke": smoke,
        }
    )
    write_json_atomic(manifest_path, manifest)


def run_process(
    command: list[str], log_path: Path, status: Status, section: str, key: str
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n=== START {now()} ===\n")
        log.write("COMMAND " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status.update(section, key, child_pid=process.pid)
        return_code = process.wait()
        log.write(f"=== EXIT {return_code} {now()} ===\n")
    return return_code


def run_smokes(
    endpoints: list[str],
    output_root: Path,
    embedding_url: str,
    config: dict[str, Any],
    data_dirs: dict[str, Path],
    selection: Selection,
    status: Status,
    smoke_jobs: list[Job] | None = None,
) -> None:
    if smoke_jobs is None:
        smoke_jobs = [
            Job(method, benchmark) for method, benchmark in SMOKE_JOB_ORDER
        ]
    pending: queue.Queue[Job] = queue.Queue()
    for job in smoke_jobs:
        pending.put(job)

    def worker(endpoint: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            key = job.name
            result_dir = output_root / "_smoke" / job.benchmark / job.method
            log_path = output_root / "_logs" / "smoke" / f"{job.name}.log"
            status.update(
                "smoke",
                key,
                status="running",
                endpoint=endpoint,
                started_at=now(),
                result_dir=str(result_dir),
            )
            command = [
                sys.executable,
                *command_for(
                    job,
                    result_dir,
                    endpoint,
                    embedding_url,
                    config,
                    data_dirs,
                    selection,
                    smoke=True,
                ),
            ]
            return_code = run_process(command, log_path, status, "smoke", key)
            try:
                if return_code:
                    raise RuntimeError(f"exit code {return_code}")
                validate_output(job, result_dir, selection, config, smoke=True)
            except Exception as exc:
                status.update(
                    "smoke", key, status="failed", finished_at=now(), error=str(exc)
                )
            else:
                status.update("smoke", key, status="completed", finished_at=now())
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(endpoint,)) for endpoint in endpoints]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [
        job.name
        for job in smoke_jobs
        if status.data.get("smoke", {}).get(job.name, {}).get("status") != "completed"
    ]
    if failed:
        raise RuntimeError(f"Smoke tests failed: {failed}")


def judge_worker(
    pending: queue.Queue[Job | None],
    output_root: Path,
    config: dict[str, Any],
    status: Status,
) -> None:
    key_file = Path(str(config["judge_key_file"]))
    if not key_file.is_absolute():
        key_file = (ROOT / key_file).resolve()
    while True:
        job = pending.get()
        if job is None:
            pending.task_done()
            return
        result_dir = formal_result_dir(output_root, job)
        log_path = run_artifact_root(output_root) / "_logs" / "judge" / f"{job.name}.log"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "judge_results_llm_parallel.py"),
            "--benchmark", BENCHMARK_ARGUMENT[job.benchmark],
            "--results", str(result_dir / "results.json"),
            "--out-dir", str(result_dir),
            "--key-file", str(key_file),
            "--model", str(config["judge_model"]),
            "--workers", str(config.get("judge_workers", 32)),
            "--timeout", str(config["judge_timeout"]),
            "--retries", str(config["retries"]),
            "--max-tokens", str(config["judge_max_tokens"]),
            "--checkpoint-every", "25",
            "--resume",
        ]
        status.update(
            "judges", job.name, status="running", started_at=now(), log=str(log_path)
        )
        return_code = run_process(command, log_path, status, "judges", job.name)
        metrics_path = result_dir / "llm_judge_metrics.json"
        complete = False
        if return_code == 0 and metrics_path.is_file():
            metrics = load_json(metrics_path)
            complete = (
                int(metrics.get("count", -1)) == EXPECTED_COUNTS[job.benchmark]
                and int(metrics.get("judge_errors", -1)) == 0
            )
        status.update(
            "judges",
            job.name,
            status="completed" if complete else "failed",
            finished_at=now(),
            return_code=return_code,
        )
        pending.task_done()


def judge_output_complete(job: Job, output_root: Path) -> bool:
    metrics_path = formal_result_dir(output_root, job) / "llm_judge_metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = load_json(metrics_path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        int(metrics.get("count", -1)) == EXPECTED_COUNTS[job.benchmark]
        and int(metrics.get("judge_errors", -1)) == 0
    )


def run_formal(
    endpoints: list[str],
    output_root: Path,
    embedding_url: str,
    config: dict[str, Any],
    data_dirs: dict[str, Path],
    selection: Selection,
    status: Status,
    jobs: list[Job] | None = None,
) -> None:
    max_job_attempts = formal_job_attempts(config)
    pending: queue.Queue[Job] = queue.Queue()
    if jobs is None:
        jobs = [Job(method, benchmark) for method, benchmark in JOB_ORDER]
    completed_jobs: list[Job] = []
    for priority, job in enumerate(jobs, start=1):
        result_dir = formal_result_dir(output_root, job)
        try:
            validate_output(job, result_dir, selection, config)
        except Exception:
            pending.put(job)
            status.update(
                "jobs",
                job.name,
                status="pending",
                priority=priority,
                method=job.method,
                benchmark=job.benchmark,
            )
        else:
            completed_jobs.append(job)
            status.update(
                "jobs",
                job.name,
                status="completed",
                priority=priority,
                method=job.method,
                benchmark=job.benchmark,
                result_dir=str(result_dir),
                resumed_existing=True,
            )
    judge_queue: queue.Queue[Job | None] = queue.Queue()
    judge_threads = [
        threading.Thread(
            target=judge_worker,
            args=(judge_queue, output_root, config, status),
            name=f"test-only-judge-{index + 1}",
        )
        for index in range(max(1, int(config.get("judge_job_concurrency", 1))))
    ]
    for judge_thread in judge_threads:
        judge_thread.start()
    for job in completed_jobs:
        if judge_output_complete(job, output_root):
            status.update(
                "judges", job.name, status="completed", resumed_existing=True
            )
        else:
            judge_queue.put(job)

    def worker(endpoint: str) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            result_dir = formal_result_dir(output_root, job)
            log_path = run_artifact_root(output_root) / "_logs" / "baseline" / f"{job.name}.log"
            succeeded = False
            error = ""
            for attempt in range(1, max_job_attempts + 1):
                status.update(
                    "jobs",
                    job.name,
                    status="running",
                    endpoint=endpoint,
                    attempt=attempt,
                    started_at=now(),
                    result_dir=str(result_dir),
                    log=str(log_path),
                )
                command = [
                    sys.executable,
                    *command_for(
                        job,
                        result_dir,
                        endpoint,
                        embedding_url,
                        config,
                        data_dirs,
                        selection,
                    ),
                ]
                return_code = run_process(
                    command, log_path, status, "jobs", job.name
                )
                try:
                    if return_code:
                        raise RuntimeError(f"exit code {return_code}")
                    validate_output(job, result_dir, selection, config)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    status.update("jobs", job.name, status="retrying", error=error)
                    if attempt < max_job_attempts:
                        time.sleep(60)
                else:
                    succeeded = True
                    break
            status.update(
                "jobs",
                job.name,
                status="completed" if succeeded else "failed",
                finished_at=now(),
                error="" if succeeded else error,
                child_pid=None,
            )
            if succeeded:
                judge_queue.put(job)
            pending.task_done()

    workers = [
        threading.Thread(target=worker, args=(endpoint,), name=f"gpu-worker-{index}")
        for index, endpoint in enumerate(endpoints, start=3)
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    for _ in judge_threads:
        judge_queue.put(None)
    judge_queue.join()
    for judge_thread in judge_threads:
        judge_thread.join()

    failed = [
        job.name
        for job in jobs
        if status.data["jobs"].get(job.name, {}).get("status") != "completed"
    ]
    judge_failed = [
        job.name
        for job in jobs
        if status.data["jobs"].get(job.name, {}).get("status") == "completed"
        and status.data["judges"].get(job.name, {}).get("status") != "completed"
    ]
    status.update_root(
        phase="complete" if not failed and not judge_failed else "incomplete",
        finished_at=now(),
        failed_jobs=failed,
        failed_judges=judge_failed,
    )
    if failed or judge_failed:
        raise RuntimeError(f"Incomplete jobs={failed}, judges={judge_failed}")


def main() -> None:
    global RUN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="Repeat for each OpenAI-compatible answer endpoint.",
    )
    parser.add_argument(
        "--embedding-base-url", default="http://127.0.0.1:8001/v1"
    )
    parser.add_argument(
        "--defaults", type=Path, default=ROOT / "configs" / "defaults.json"
    )
    parser.add_argument(
        "--efficiency-config",
        type=Path,
        default=configured_efficiency_config(),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=configured_split_manifest(),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument(
        "--run-id",
        help="Override the protocol Run-ID without changing the fixed split protocol.",
    )
    parser.add_argument(
        "--m3-reuse-run-id",
        default="",
        help=(
            "Reuse M3 memory graphs read-only from this prior Run-ID while "
            "writing QA and all new artifacts under --run-id."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=sorted({method for method, _benchmark in JOB_ORDER}),
        help="Run only this baseline. Repeat to select more than one.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=sorted(BENCHMARK_ARGUMENT),
        help="Run only this benchmark. Repeat to select more than one.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run configured smoke jobs, validate them, and stop before formal jobs.",
    )
    args = parser.parse_args()

    if args.skip_smoke and args.smoke_only:
        parser.error("--skip-smoke and --smoke-only cannot be used together")

    if args.run_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
            raise ValueError("--run-id must be a safe path component")
        RUN_ID = args.run_id
    if args.m3_reuse_run_id:
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", args.m3_reuse_run_id
        ):
            raise ValueError("--m3-reuse-run-id must be a safe path component")
        if args.m3_reuse_run_id == RUN_ID:
            raise ValueError("M3 reuse Run-ID must differ from the new --run-id")

    selected_baselines = set(args.baseline or ())
    selected_benchmarks = set(args.benchmark or ())
    jobs = [
        Job(method, benchmark)
        for method, benchmark in JOB_ORDER
        if (not selected_baselines or method in selected_baselines)
        and (not selected_benchmarks or benchmark in selected_benchmarks)
    ]
    if not jobs:
        raise ValueError("The baseline/benchmark filters selected no jobs")
    selected_job_keys = {(job.method, job.benchmark) for job in jobs}
    smoke_jobs = [
        Job(method, benchmark)
        for method, benchmark in SMOKE_JOB_ORDER
        if (method, benchmark) in selected_job_keys
    ]

    endpoints = args.endpoint or [
        "http://127.0.0.1:8013/v1",
        "http://127.0.0.1:8014/v1",
        "http://127.0.0.1:8015/v1",
    ]
    output_root = args.output_root.expanduser().resolve()
    run_root = run_artifact_root(output_root)
    config = load_json(args.defaults.expanduser().resolve())
    if args.m3_reuse_run_id:
        reuse_roots = {
            benchmark: (
                output_root
                / benchmark
                / "M3-Agent-caption"
                / args.m3_reuse_run_id
                / "memory"
                / "datasets"
            ).resolve()
            for benchmark in BENCHMARK_ARGUMENT
        }
        missing_reuse_roots = [
            str(path) for path in reuse_roots.values() if not path.is_dir()
        ]
        if missing_reuse_roots:
            raise FileNotFoundError(
                f"M3 reuse state roots do not exist: {missing_reuse_roots}"
            )
        config["m3_reuse_state_roots"] = {
            benchmark: str(path) for benchmark, path in reuse_roots.items()
        }
    config["top_k"] = int(PROTOCOL["top_k"])
    config["efficiency_config"] = str(
        args.efficiency_config.expanduser().resolve()
    )
    if str(config.get("judge_model")) != "openai/gpt-4o-mini":
        raise ValueError("This planned run requires judge_model=openai/gpt-4o-mini")
    selection = load_selection(args.split_manifest)
    status = Status(run_root / "status.json", endpoints, output_root)
    try:
        check_services(endpoints, args.embedding_base_url, config)
        data_dirs = stage_inputs(run_root / "_test_inputs", selection)
        smoke_selection = write_smoke_manifest(
            selection.manifest_path,
            run_root / "_preflight" / "smoke_split_manifest.json",
        )
        status.update_root(
            phase="smoke" if not args.skip_smoke else "formal",
            split=SPLIT_NAME,
            split_manifest=str(selection.manifest_path),
            split_manifest_sha256=selection.manifest_sha256,
            top_k=config["top_k"],
            run_id=RUN_ID,
            efficiency_config=config["efficiency_config"],
            judge_model=config["judge_model"],
            smoke_split_manifest=str(smoke_selection.manifest_path),
            job_order=[job.name for job in jobs],
            m3_reuse_run_id=args.m3_reuse_run_id,
            m3_reuse_state_roots=config.get("m3_reuse_state_roots", {}),
        )
        if not args.skip_smoke:
            if not smoke_jobs:
                raise RuntimeError(
                    "The selected jobs have no configured smoke test; use --skip-smoke "
                    "only after a separate smoke test has passed"
                )
            run_smokes(
                endpoints,
                run_root,
                args.embedding_base_url,
                config,
                data_dirs,
                smoke_selection,
                status,
                smoke_jobs,
            )
        if args.smoke_only:
            status.update_root(phase="smoke_completed", finished_at=now())
            return
        status.update_root(phase="formal")
        run_formal(
            endpoints,
            output_root,
            args.embedding_base_url,
            config,
            data_dirs,
            selection,
            status,
            jobs,
        )
    except Exception as exc:
        status.update_root(
            phase="incomplete",
            fatal_error=f"{type(exc).__name__}: {exc}",
            finished_at=now(),
        )
        raise


if __name__ == "__main__":
    main()
