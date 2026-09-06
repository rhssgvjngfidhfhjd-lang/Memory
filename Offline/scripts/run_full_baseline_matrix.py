"""Run HiveMem and all registered baselines on the three formal benchmarks.

Three fixed workers consume one inference endpoint each.  HiveMem occupies the
first three slots (one benchmark per endpoint); completed workers then consume
the remaining baseline jobs from a shared queue.  Each subprocess writes its
own log, while ``logs/full_matrix/status.json`` is the machine-readable source
of truth for progress.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
LOG_ROOT = ROOT / "logs" / "full_matrix"
STATUS_PATH = LOG_ROOT / "status.json"
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
BASELINES = (
    "AUGUSTUSMemory",
    "OmniSimpleMem",
    "M2A",
    "MIRIX",
    "MMA",
    "MemVerse",
    "M3-Agent-caption",
)
BENCHMARKS = ("Mem-Gallery", "WorldMemArena", "H2HMEM")
EXPECTED_RESULT_COUNTS = {
    "Mem-Gallery": 1711,
    "WorldMemArena": 2090,
    "H2HMEM": 2207,
}
INFERENCE_ENDPOINTS = {
    "http://127.0.0.1:8013/v1",
    "http://127.0.0.1:8014/v1",
    "http://127.0.0.1:8015/v1",
}
EMBEDDING_ENDPOINT = "http://127.0.0.1:8001/v1"
TOP_K = int(
    json.loads(
        (ROOT / "configs" / "defaults.json").read_text(encoding="utf-8")
    )["top_k"]
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    name: str
    benchmark: str
    method: str


class Status:
    def __init__(
        self,
        ports: list[str],
        embedding_base_url: str,
        *,
        resume_completed: bool,
    ) -> None:
        self.lock = threading.Lock()
        previous: dict[str, Any] = {}
        if STATUS_PATH.is_file():
            try:
                previous = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
        self.data: dict[str, Any] = {
            "runner_pid": os.getpid(),
            "started_at": now(),
            "updated_at": now(),
            "inference_ports": ports,
            "embedding_base_url": embedding_base_url,
            "jobs": (previous.get("jobs") or {}) if resume_completed else {},
        }
        for row in self.data["jobs"].values():
            if row.get("status") == "running":
                row["status"] = "pending"
                row.pop("child_pid", None)
        self.write()

    def completed(self, name: str) -> bool:
        with self.lock:
            return self.data["jobs"].get(name, {}).get("status") == "completed"

    def update(self, name: str, **values: Any) -> None:
        with self.lock:
            row = self.data["jobs"].setdefault(name, {})
            row.update(values)
            self.data["updated_at"] = now()
            self._write_unlocked()

    def heartbeat(self) -> None:
        with self.lock:
            self.data["updated_at"] = now()
            self._write_unlocked()

    def write(self) -> None:
        with self.lock:
            self._write_unlocked()

    def _write_unlocked(self) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = STATUS_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # On /mnt/<drive>, a Windows reader can briefly prevent NTFS rename.
        # Monitoring must never make the experiment subprocess look failed.
        for attempt in range(50):
            try:
                temporary.replace(STATUS_PATH)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)


def request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def check_endpoints(ports: list[str], embedding_base_url: str) -> None:
    for base_url in ports:
        payload = request_json(base_url.rstrip("/") + "/models")
        models = {str(row.get("id")) for row in payload.get("data") or []}
        if MODEL not in models:
            raise RuntimeError(f"{MODEL} is unavailable at {base_url}: {sorted(models)}")
    payload = request_json(
        embedding_base_url.rstrip("/") + "/embeddings",
        {"model": EMBEDDING_MODEL, "input": ["full matrix preflight"]},
    )
    rows = payload.get("data") or []
    dimension = len(rows[0].get("embedding") or []) if rows else 0
    if dimension != 2048:
        raise RuntimeError(
            f"embedding endpoint returned dimension {dimension}, expected 2048"
        )


def wait_for_inference(endpoint: str, job: Job, status: Status) -> None:
    """Do not hand a new job to a tunnel while its model server is saturated."""
    failures = 0
    while True:
        try:
            payload = request_json(endpoint.rstrip("/") + "/models")
            models = {str(row.get("id")) for row in payload.get("data") or []}
            if MODEL in models:
                return
            error = f"model unavailable: {sorted(models)}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        failures += 1
        status.update(
            job.name,
            status="waiting_endpoint",
            endpoint=endpoint,
            endpoint_failures=failures,
            endpoint_error=error,
        )
        time.sleep(30)


def validate_inputs() -> None:
    required = (
        ROOT / "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl",
        ROOT / "data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl",
        ROOT / "data/wma_qwen3_vl_embedding_2b/query_embeddings_lifelong",
        ROOT / "data/h2hmem/chunks_dyadic.jsonl",
        ROOT / "data/h2hmem/chunks_multiparty.jsonl",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing formal inputs: " + ", ".join(missing))
    for path in required[-2:]:
        first = json.loads(path.open(encoding="utf-8").readline())
        dataset = str((first.get("metadata") or {}).get("dataset") or "")
        if ":" in dataset:
            raise ValueError(f"Windows-unsafe H2HMem dataset name in {path}: {dataset}")


def common_eval_args(endpoint: str, embedding_base_url: str) -> list[str]:
    return [
        "--answer-model", MODEL,
        "--answer-base-url", endpoint,
        "--answer-temperature", "0",
        "--executor-model", MODEL,
        "--executor-base-url", endpoint,
        "--executor-temperature", "0",
        "--embedding-model", EMBEDDING_MODEL,
        "--embedding-base-url", embedding_base_url,
        "--embedding-dim", "2048",
        "--top-k", str(TOP_K),
        "--request-timeout", "180",
        "--retries", "2",
    ]


def build_args(chunks: str, output_root: Path, endpoint: str, embedding_base_url: str) -> list[str]:
    return [
        "-m", "hive_mem.build_memories",
        "--chunks", chunks,
        "--all-datasets",
        "--output-root", str(output_root),
        "--executor-base-url", endpoint,
        "--executor-model", MODEL,
        "--embedding-base-url", embedding_base_url,
        "--embedding-model", EMBEDDING_MODEL,
        "--embedding-dim", "2048",
        "--executor-timeout", "180",
        "--executor-retries", "2",
        "--executor-concurrency", "16",
    ]


def hive_commands(benchmark: str, endpoint: str, embedding_base_url: str) -> list[list[str]]:
    result_dir = OUTPUT_ROOT / benchmark / "HiveMem"
    memory_dir = result_dir / "memory"
    common = common_eval_args(endpoint, embedding_base_url)
    if benchmark == "Mem-Gallery":
        return [
            build_args(
                "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl",
                memory_dir,
                endpoint,
                embedding_base_url,
            ),
            [
                "-m", "benchmarks.memgallery_harness.eval_memgallery",
                "--baseline", "HiveMem",
                "--all-datasets",
                "--index-root", str(memory_dir),
                "--result-dir", str(result_dir),
                "--resume",
                "--exclude-categories", "",
                *common,
            ],
        ]
    if benchmark == "WorldMemArena":
        return [
            build_args(
                "data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl",
                memory_dir,
                endpoint,
                embedding_base_url,
            ),
            [
                "-m", "benchmarks.wma_harness.eval_wma",
                "--baseline", "HiveMem",
                "--index-root", str(memory_dir),
                "--query-embedding-dir",
                "data/wma_qwen3_vl_embedding_2b/query_embeddings_lifelong",
                "--result-dir", str(result_dir),
                "--resume",
                "--exclude-categories", "",
                *common,
            ],
        ]
    return [
        build_args(
            "data/h2hmem/chunks_dyadic.jsonl",
            memory_dir,
            endpoint,
            embedding_base_url,
        ),
        build_args(
            "data/h2hmem/chunks_multiparty.jsonl",
            memory_dir,
            endpoint,
            embedding_base_url,
        ),
        [
            "-m", "benchmarks.h2hmem_harness.eval_h2hmem",
            "--baseline", "HiveMem",
            "--variant", "all",
            "--index-root", str(memory_dir),
            "--result-dir", str(result_dir),
            *common,
        ],
    ]


def baseline_commands(
    benchmark: str,
    baseline: str,
    endpoint: str,
    embedding_base_url: str,
) -> list[list[str]]:
    result_dir = OUTPUT_ROOT / benchmark / baseline
    state_dir = result_dir / "memory" / "datasets"
    common = [
        "--baseline", baseline,
        "--result-dir", str(result_dir),
        "--baseline-state-dir", str(state_dir),
        "--resume",
        "--sample-concurrency", "4",
        "--answer-concurrency", "16",
        *common_eval_args(endpoint, embedding_base_url),
    ]
    if benchmark == "Mem-Gallery":
        module = "benchmarks.memgallery_harness.eval_memgallery"
        extra = ["--all-datasets", "--exclude-categories", ""]
    elif benchmark == "WorldMemArena":
        module = "benchmarks.wma_harness.eval_wma"
        extra = ["--exclude-categories", ""]
    else:
        module = "benchmarks.h2hmem_harness.eval_h2hmem"
        extra = ["--variant", "all"]
    return [["-m", module, *common, *extra]]


def commands_for(job: Job, endpoint: str, embedding_base_url: str) -> list[list[str]]:
    if job.method == "HiveMem":
        return hive_commands(job.benchmark, endpoint, embedding_base_url)
    return baseline_commands(job.benchmark, job.method, endpoint, embedding_base_url)


def validate_job_outputs(job: Job, result_dir: Path) -> None:
    required = (
        result_dir / "results.json",
        result_dir / "retrieval_trace.jsonl",
        result_dir / "memory" / "memory_snapshot.jsonl",
        result_dir / "run_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"{job.name} is missing required outputs: {missing}")
    results = json.loads(required[0].read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in required[1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshots = [
        json.loads(line)
        for line in required[2].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(required[3].read_text(encoding="utf-8"))
    expected_count = EXPECTED_RESULT_COUNTS[job.benchmark]
    if len(results) != expected_count:
        raise RuntimeError(
            f"{job.name} has {len(results)} results; expected {expected_count}"
        )
    if len(results) != len(traces):
        raise RuntimeError(
            f"{job.name} has {len(results)} results and {len(traces)} traces"
        )
    answer_errors = [row for row in results if row.get("error")]
    if answer_errors:
        raise RuntimeError(
            f"{job.name} has {len(answer_errors)} answer errors"
        )
    if not snapshots:
        raise RuntimeError(f"{job.name} produced an empty memory snapshot")
    malformed_snapshots = [
        row for row in snapshots
        if not all(str(row.get(key) or "").strip() for key in ("memory_id", "text", "backend_type"))
    ]
    if malformed_snapshots:
        raise RuntimeError(
            f"{job.name} has {len(malformed_snapshots)} malformed memory snapshot rows"
        )
    if job.benchmark == "Mem-Gallery":
        result_keys = Counter(
            (row.get("dataset"), row.get("question"), row.get("category"))
            for row in results
        )
        trace_keys = Counter(
            (row.get("dataset"), row.get("question"), row.get("category"))
            for row in traces
        )
        if result_keys != trace_keys:
            raise RuntimeError(f"{job.name} result/trace question sets differ")
        trace_ids = {
            (row.get("dataset"), row.get("qa_index")) for row in traces
        }
        if len(trace_ids) != expected_count:
            raise RuntimeError(f"{job.name} contains duplicate trace QA ids")
        datasets = {str(row.get("dataset") or "") for row in results}
        if "" in datasets or len(datasets) != 20:
            raise RuntimeError(
                f"{job.name} covers {len(datasets - {''})} Mem-Gallery datasets; expected 20"
            )
    else:
        result_ids = [str(row.get("query_id") or "") for row in results]
        trace_ids = [str(row.get("query_id") or "") for row in traces]
        if "" in result_ids or len(set(result_ids)) != expected_count:
            raise RuntimeError(f"{job.name} has missing or duplicate result query ids")
        if "" in trace_ids or len(set(trace_ids)) != expected_count:
            raise RuntimeError(f"{job.name} has missing or duplicate trace query ids")
        if set(result_ids) != set(trace_ids):
            raise RuntimeError(f"{job.name} result/trace query id sets differ")
    oversized_traces = [
        row for row in traces
        if not isinstance(row.get("top_k"), list) or len(row["top_k"]) > TOP_K
    ]
    if oversized_traces:
        raise RuntimeError(
            f"{job.name} has {len(oversized_traces)} malformed or oversized retrieval traces"
        )
    malformed_hits = [
        hit
        for row in traces
        for hit in row["top_k"]
        if not isinstance(hit, dict)
        or not str(hit.get("memory_id") or "").strip()
        or not str(hit.get("content") or "").strip()
        or not isinstance(hit.get("rank"), int)
    ]
    if malformed_hits:
        raise RuntimeError(
            f"{job.name} has {len(malformed_hits)} malformed retrieved memory rows"
        )
    if job.benchmark == "H2HMEM":
        variants = Counter(str(row.get("variant") or "") for row in results)
        expected_variants = Counter({"dyadic": 2017, "multiparty": 190})
        if variants != expected_variants:
            raise RuntimeError(
                f"{job.name} variant counts are {dict(variants)}; "
                f"expected {dict(expected_variants)}"
            )
    if job.method in {"MIRIX", "MMA"}:
        databases = sorted((result_dir / "memory" / "datasets").rglob("sqlite.db"))
        if not databases:
            raise RuntimeError(f"{job.name} produced no native SQLite databases")
        failed_tools: list[str] = []
        for database in databases:
            try:
                connection = sqlite3.connect(database)
                try:
                    contents = connection.execute(
                        "SELECT content FROM messages WHERE role = 'tool'"
                    ).fetchall()
                finally:
                    connection.close()
            except sqlite3.Error as exc:
                raise RuntimeError(
                    f"{job.name} cannot inspect native database {database}: {exc}"
                ) from exc
            for (content,) in contents:
                normalized = str(content or "").replace('\\"', '"')
                if (
                    '"status": "Failed"' in normalized
                    or '"message": "Error executing function' in normalized
                ):
                    failed_tools.append(str(database))
        if failed_tools:
            raise RuntimeError(
                f"{job.name} has {len(failed_tools)} failed native tool calls "
                f"across {len(set(failed_tools))} databases"
            )
    expected = {
        "answer_model": MODEL,
        "answer_temperature": 0.0,
        "executor_model": MODEL,
        "executor_temperature": 0.0,
        "executor_visual_input": "image",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_base_url": EMBEDDING_ENDPOINT,
        "embedding_dim": 2048,
        "top_k": TOP_K,
        "request_timeout": 180,
        "retries": 2,
    }
    recorded_config = (
        manifest.get("configuration")
        if isinstance(manifest.get("configuration"), dict)
        else manifest
    )
    mismatched = {
        key: {"expected": value, "actual": recorded_config.get(key)}
        for key, value in expected.items()
        if recorded_config.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"{job.name} manifest mismatch: {mismatched}")
    answer_endpoint = str(recorded_config.get("answer_base_url") or "")
    executor_endpoint = str(recorded_config.get("executor_base_url") or "")
    if answer_endpoint not in INFERENCE_ENDPOINTS:
        raise RuntimeError(
            f"{job.name} uses unexpected answer endpoint {answer_endpoint!r}"
        )
    if executor_endpoint != answer_endpoint:
        raise RuntimeError(
            f"{job.name} executor endpoint {executor_endpoint!r} does not match "
            f"answer endpoint {answer_endpoint!r}"
        )
    manifest_baseline = manifest.get("baseline")
    if isinstance(manifest_baseline, dict):
        manifest_baseline = manifest_baseline.get("name")
    if manifest_baseline != job.method:
        raise RuntimeError(
            f"{job.name} manifest baseline is {manifest_baseline!r}; "
            f"expected {job.method!r}"
        )
    if job.benchmark == "Mem-Gallery":
        if manifest.get("all_datasets") is not True or manifest.get("questions") != 1711:
            raise RuntimeError(f"{job.name} manifest does not describe the full Mem-Gallery run")
    elif job.benchmark == "WorldMemArena":
        data_dir = str(manifest.get("data_dir") or "")
        if Path(data_dir).name != "lifelong":
            raise RuntimeError(
                f"{job.name} data_dir is not the lifelong split: {data_dir!r}"
            )
        if manifest.get("samples") != 38 or manifest.get("questions") != 2090:
            raise RuntimeError(f"{job.name} manifest does not describe full WMA lifelong")
    else:
        if manifest.get("variants") != ["dyadic", "multiparty"]:
            raise RuntimeError(
                f"{job.name} manifest variants are {manifest.get('variants')!r}"
            )
        if manifest.get("questions") != 2207:
            raise RuntimeError(f"{job.name} manifest does not describe full H2HMEM")


def run_job(job: Job, endpoint: str, embedding_base_url: str, status: Status) -> bool:
    log_path = LOG_ROOT / f"{job.name}.log"
    result_dir = OUTPUT_ROOT / job.benchmark / job.method
    result_dir.mkdir(parents=True, exist_ok=True)
    status.update(
        job.name,
        benchmark=job.benchmark,
        method=job.method,
        endpoint=endpoint,
        status="running",
        started_at=now(),
        log=str(log_path),
        result_dir=str(result_dir),
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    commands = commands_for(job, endpoint, embedding_base_url)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n=== {job.name} start {now()} endpoint={endpoint} ===\n")
        for index, arguments in enumerate(commands, start=1):
            command = [sys.executable, *arguments]
            log.write(f"\n--- command {index}/{len(commands)}: {' '.join(command)} ---\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            status.update(
                job.name,
                command_index=index,
                command_count=len(commands),
                child_pid=process.pid,
            )
            return_code = process.wait()
            status.update(job.name, last_return_code=return_code)
            if return_code:
                log.write(f"\n=== {job.name} failed with exit code {return_code} at {now()} ===\n")
                status.update(
                    job.name,
                    status="failed",
                    finished_at=now(),
                    failed_command=command,
                )
                return False
        validate_job_outputs(job, result_dir)
        log.write(f"\n=== {job.name} completed {now()} ===\n")
    status.update(job.name, status="completed", finished_at=now(), child_pid=None)
    return True


def worker(
    endpoint: str,
    first_job: Job | None,
    pending: queue.Queue[Job],
    embedding_base_url: str,
    status: Status,
    hive_attempts: int,
    baseline_attempts: int,
) -> None:
    job = first_job
    while job is not None:
        completed_and_valid = False
        if status.completed(job.name):
            result_dir = OUTPUT_ROOT / job.benchmark / job.method
            try:
                validate_job_outputs(job, result_dir)
            except Exception as exc:
                status.update(
                    job.name,
                    status="pending",
                    resume_validation_error=f"{type(exc).__name__}: {exc}",
                )
            else:
                completed_and_valid = True
                status.update(
                    job.name,
                    skipped_at=now(),
                    resume_validation_error=None,
                )
        if not completed_and_valid:
            attempts = hive_attempts if job.method == "HiveMem" else baseline_attempts
            for attempt in range(1, attempts + 1):
                wait_for_inference(endpoint, job, status)
                status.update(job.name, attempt=attempt, max_attempts=attempts)
                try:
                    succeeded = run_job(job, endpoint, embedding_base_url, status)
                except Exception as exc:  # keep the endpoint queue alive after one bad job
                    succeeded = False
                    status.update(
                        job.name,
                        status="failed",
                        finished_at=now(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if succeeded:
                    break
                if attempt < attempts:
                    status.update(job.name, status="retrying", retry_after_seconds=60)
                    time.sleep(60)
        pending.task_done() if first_job is None else None
        first_job = None
        try:
            job = pending.get_nowait()
        except queue.Empty:
            job = None
    status.heartbeat()


def all_jobs() -> tuple[list[Job], list[Job]]:
    first = [
        Job("hivemem_memgallery", "Mem-Gallery", "HiveMem"),
        Job("hivemem_wma_lifelong", "WorldMemArena", "HiveMem"),
        Job("hivemem_h2hmem", "H2HMEM", "HiveMem"),
    ]
    ordered = [
        Job(
            f"{baseline.lower().replace('-', '_')}_{benchmark.lower().replace('-', '_')}",
            benchmark,
            baseline,
        )
        for benchmark in ("Mem-Gallery", "H2HMEM", "WorldMemArena")
        for baseline in BASELINES
    ]
    # Keep the fast/medium baselines flowing first.  OmniSimpleMem and MemVerse
    # have much heavier long-tail builds, so run them after the other methods.
    deferred_methods = {"OmniSimpleMem", "MemVerse"}
    rest = [job for job in ordered if job.method not in deferred_methods]
    rest.extend(job for job in ordered if job.method in deferred_methods)
    return first, rest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        action="append",
        default=[],
        help="OpenAI-compatible inference base URL; repeat exactly three times.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:8001/v1",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse jobs marked complete in the existing status file.",
    )
    parser.add_argument("--hive-attempts", type=int, default=10)
    parser.add_argument("--baseline-attempts", type=int, default=2)
    args = parser.parse_args()
    ports = args.port or [
        "http://127.0.0.1:8013/v1",
        "http://127.0.0.1:8014/v1",
        "http://127.0.0.1:8015/v1",
    ]
    if len(ports) != 3:
        parser.error("exactly three --port values are required")
    if args.hive_attempts < 1 or args.baseline_attempts < 1:
        parser.error("retry counts must be positive")
    validate_inputs()
    check_endpoints(ports, args.embedding_base_url)
    first, rest = all_jobs()
    if args.dry_run:
        for index, job in enumerate(first):
            print(f"{ports[index]} -> {job.name}")
        for job in rest:
            print(f"queue -> {job.name}")
        return

    status = Status(
        ports,
        args.embedding_base_url,
        resume_completed=args.resume_completed,
    )
    pending: queue.Queue[Job] = queue.Queue()
    for job in rest:
        pending.put(job)
        status.update(
            job.name,
            benchmark=job.benchmark,
            method=job.method,
            status=("completed" if status.completed(job.name) else "pending"),
        )
    for job in first:
        status.update(
            job.name,
            benchmark=job.benchmark,
            method=job.method,
            status=("completed" if status.completed(job.name) else "pending"),
        )

    threads = [
        threading.Thread(
            target=worker,
            name=f"full-matrix-{index + 1}",
            args=(
                endpoint,
                first[index],
                pending,
                args.embedding_base_url,
                status,
                args.hive_attempts,
                args.baseline_attempts,
            ),
        )
        for index, endpoint in enumerate(ports)
    ]
    for thread in threads:
        thread.start()

    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=1)
        status.heartbeat()
        time.sleep(29)
    status.heartbeat()
    expected = [*first, *rest]
    failed = [
        job.name
        for job in expected
        if status.data["jobs"].get(job.name, {}).get("status") != "completed"
    ]
    if failed:
        raise SystemExit(
            "full matrix incomplete; failed jobs: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
