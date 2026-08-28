"""Run HiveMem and all registered baselines on the three formal benchmarks.

Three fixed workers consume one inference endpoint each.  HiveMem occupies the
first three slots (one benchmark per endpoint); completed workers then consume
the remaining baseline jobs from a shared queue.  Each subprocess writes its
own log, while ``logs/full_matrix/status.json`` is the machine-readable source
of truth for progress.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import queue
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


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    name: str
    benchmark: str
    method: str


class Status:
    def __init__(self, ports: list[str], embedding_base_url: str) -> None:
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
            "jobs": previous.get("jobs") or {},
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
        ROOT / "data/wma_qwen3_vl_embedding_2b/lifelong_chunks.jsonl",
        ROOT / "data/wma_qwen3_vl_embedding_2b/lifelong_query_embeddings",
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
        "--answer-base-url", endpoint,
        "--executor-base-url", endpoint,
        "--embedding-base-url", embedding_base_url,
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
        "--embedding-base-url", embedding_base_url,
        "--executor-timeout", "180",
        "--executor-retries", "2",
        # Sixteen simultaneous multimodal generations saturated the forwarded
        # vLLM queues and made even /models miss its 10-second harness probe.
        # Four keeps each A100 busy without starving health checks.
        "--executor-concurrency", "4",
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
                *common,
            ],
        ]
    if benchmark == "WorldMemArena":
        return [
            build_args(
                "data/wma_qwen3_vl_embedding_2b/lifelong_chunks.jsonl",
                memory_dir,
                endpoint,
                embedding_base_url,
            ),
            [
                "-m", "benchmarks.wma_harness.eval_wma",
                "--baseline", "HiveMem",
                "--index-root", str(memory_dir),
                "--query-embedding-dir",
                "data/wma_qwen3_vl_embedding_2b/lifelong_query_embeddings",
                "--result-dir", str(result_dir),
                "--resume",
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
        *common_eval_args(endpoint, embedding_base_url),
    ]
    if benchmark == "Mem-Gallery":
        module = "benchmarks.memgallery_harness.eval_memgallery"
        extra = ["--all-datasets"]
    elif benchmark == "WorldMemArena":
        module = "benchmarks.wma_harness.eval_wma"
        extra = []
    else:
        module = "benchmarks.h2hmem_harness.eval_h2hmem"
        extra = ["--variant", "all"]
    return [["-m", module, *common, *extra]]


def commands_for(job: Job, endpoint: str, embedding_base_url: str) -> list[list[str]]:
    if job.method == "HiveMem":
        return hive_commands(job.benchmark, endpoint, embedding_base_url)
    return baseline_commands(job.benchmark, job.method, endpoint, embedding_base_url)


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
        if status.completed(job.name):
            status.update(job.name, skipped_at=now())
        else:
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
    rest = [
        Job(
            f"{baseline.lower().replace('-', '_')}_{benchmark.lower().replace('-', '_')}",
            benchmark,
            baseline,
        )
        for benchmark in ("Mem-Gallery", "H2HMEM", "WorldMemArena")
        for baseline in BASELINES
    ]
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
    parser.add_argument("--hive-attempts", type=int, default=10)
    parser.add_argument("--baseline-attempts", type=int, default=2)
    args = parser.parse_args()
    ports = args.port or [
        "http://127.0.0.1:28001/v1",
        "http://127.0.0.1:28002/v1",
        "http://127.0.0.1:28003/v1",
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

    status = Status(ports, args.embedding_base_url)
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


if __name__ == "__main__":
    main()
