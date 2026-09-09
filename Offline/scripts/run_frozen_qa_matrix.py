#!/usr/bin/env python3
"""Run the 21-baseline QA/Judge matrix from frozen 0907a retrieval artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = OFFLINE_ROOT / "outputs"
EXPECTED_QA = {"Mem-Gallery": 275, "H2HMEM": 360, "WorldMemArena": 440}
BENCHMARK_SLUG = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
}
JOB_ORDER = (
    ("M3-Agent-caption", "Mem-Gallery"),
    ("M2A", "Mem-Gallery"),
    ("AUGUSTUSMemory", "Mem-Gallery"),
    ("MMA", "Mem-Gallery"),
    ("MIRIX", "Mem-Gallery"),
    ("OmniSimpleMem", "Mem-Gallery"),
    ("M3-Agent-caption", "H2HMEM"),
    ("M2A", "H2HMEM"),
    ("AUGUSTUSMemory", "H2HMEM"),
    ("MMA", "H2HMEM"),
    ("MIRIX", "H2HMEM"),
    ("OmniSimpleMem", "H2HMEM"),
    ("M3-Agent-caption", "WorldMemArena"),
    ("M2A", "WorldMemArena"),
    ("AUGUSTUSMemory", "WorldMemArena"),
    ("MMA", "WorldMemArena"),
    ("MIRIX", "WorldMemArena"),
    ("OmniSimpleMem", "WorldMemArena"),
    ("MemVerse", "Mem-Gallery"),
    ("MemVerse", "H2HMEM"),
    ("MemVerse", "WorldMemArena"),
)


@dataclass(frozen=True)
class Job:
    baseline: str
    benchmark: str

    @property
    def key(self) -> str:
        return f"{self.benchmark}__{self.baseline}"


class Status:
    def __init__(self, path: Path, *, run_id: str, endpoints: list[str]):
        self.path = path
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "run_id": run_id,
            "mode": "qa_from_frozen_memory_and_retrieval",
            "started_at": now(),
            "updated_at": now(),
            "endpoints": endpoints,
            "jobs": {},
            "judges": {},
        }
        self._write()

    def update(self, section: str, key: str, **values: Any) -> None:
        with self.lock:
            self.payload.setdefault(section, {}).setdefault(key, {}).update(values)
            self.payload["updated_at"] = now()
            self._write()

    def root(self, **values: Any) -> None:
        with self.lock:
            self.payload.update(values)
            self.payload["updated_at"] = now()
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def qa_complete(result_dir: Path, benchmark: str) -> bool:
    path = result_dir / "results.json"
    manifest_path = result_dir / "run_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        return False
    try:
        rows = load_json(path)
        manifest = load_json(manifest_path)
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(rows, list)
        and len(rows) == EXPECTED_QA[benchmark]
        and not any(row.get("error") for row in rows)
        and manifest.get("execution_mode")
        == "qa_from_frozen_memory_and_retrieval"
    )


def judge_complete(result_dir: Path, benchmark: str) -> bool:
    path = result_dir / "llm_judge_metrics.json"
    if not path.is_file():
        return False
    try:
        metrics = load_json(path)
    except (OSError, ValueError, TypeError):
        return False
    expected = EXPECTED_QA[benchmark]
    return (
        int(metrics.get("count", -1)) == expected
        and int(metrics.get("valid_count", -1)) == expected
        and int(metrics.get("judge_errors", -1)) == 0
        and not bool(metrics.get("provisional", True))
    )


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n=== START {now()} ===\n")
        handle.write("COMMAND " + " ".join(command) + "\n")
        process = subprocess.run(
            command,
            cwd=OFFLINE_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.write(f"=== EXIT {process.returncode} {now()} ===\n")
        return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--source-run-id", default="0907a")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--efficiency-config",
        type=Path,
        default=OFFLINE_ROOT / "configs" / "model_efficiency.json",
    )
    parser.add_argument(
        "--judge-key-file",
        type=Path,
        default=OFFLINE_ROOT.parent / "Nvida_api" / "Openrouter_api",
    )
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--max-task-attempts", type=int, default=8)
    args = parser.parse_args()
    if args.run_id == args.source_run_id:
        parser.error("--run-id must differ from --source-run-id")
    if not args.endpoint:
        parser.error("At least one answer endpoint is required")

    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / "_runs" / args.run_id
    status = Status(run_root / "status.json", run_id=args.run_id, endpoints=args.endpoint)
    jobs = [Job(baseline, benchmark) for baseline, benchmark in JOB_ORDER]
    qa_queue: queue.Queue[Job] = queue.Queue()
    judge_queue: queue.Queue[Job | None] = queue.Queue()
    failed_jobs: list[str] = []
    failed_lock = threading.Lock()

    for priority, job in enumerate(jobs, start=1):
        source_dir = output_root / job.benchmark / job.baseline / args.source_run_id
        result_dir = output_root / job.benchmark / job.baseline / args.run_id
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source run: {source_dir}")
        if result_dir.exists() and not (
            (result_dir / ".checkpoint" / "qa_replay_manifest.json").is_file()
            or qa_complete(result_dir, job.benchmark)
        ):
            raise RuntimeError(
                f"Refusing to reuse an unrelated existing output directory: {result_dir}"
            )
        if qa_complete(result_dir, job.benchmark):
            status.update("jobs", job.key, status="completed", resumed=True)
            if judge_complete(result_dir, job.benchmark):
                status.update("judges", job.key, status="completed", resumed=True)
            else:
                judge_queue.put(job)
        else:
            qa_queue.put(job)
            status.update(
                "jobs",
                job.key,
                status="pending",
                priority=priority,
                source_dir=str(source_dir),
                result_dir=str(result_dir),
            )

    def qa_worker(endpoint: str) -> None:
        while True:
            try:
                job = qa_queue.get_nowait()
            except queue.Empty:
                return
            source_dir = output_root / job.benchmark / job.baseline / args.source_run_id
            result_dir = output_root / job.benchmark / job.baseline / args.run_id
            log_path = run_root / "logs" / "qa" / f"{job.key}.log"
            success = False
            for attempt in range(1, args.max_task_attempts + 1):
                status.update(
                    "jobs",
                    job.key,
                    status="running",
                    endpoint=endpoint,
                    attempt=attempt,
                    started_at=now(),
                )
                command = [
                    sys.executable,
                    str(OFFLINE_ROOT / "scripts" / "rerun_qa_from_frozen_retrieval.py"),
                    "--benchmark",
                    job.benchmark,
                    "--baseline",
                    job.baseline,
                    "--source-dir",
                    str(source_dir),
                    "--result-dir",
                    str(result_dir),
                    "--answer-base-url",
                    endpoint,
                    "--concurrency",
                    str(args.answer_concurrency),
                    "--efficiency-config",
                    str(args.efficiency_config),
                    "--resume",
                ]
                return_code = run_logged(command, log_path)
                if return_code == 0 and qa_complete(result_dir, job.benchmark):
                    success = True
                    break
                status.update(
                    "jobs",
                    job.key,
                    status="retrying",
                    return_code=return_code,
                    next_retry_seconds=60,
                )
                time.sleep(60)
            if success:
                status.update(
                    "jobs", job.key, status="completed", completed_at=now()
                )
                judge_queue.put(job)
            else:
                with failed_lock:
                    failed_jobs.append(job.key)
                status.update(
                    "jobs",
                    job.key,
                    status="failed",
                    completed_at=now(),
                )
            qa_queue.task_done()

    def judge_worker() -> None:
        while True:
            job = judge_queue.get()
            if job is None:
                judge_queue.task_done()
                return
            result_dir = output_root / job.benchmark / job.baseline / args.run_id
            log_path = run_root / "logs" / "judge" / f"{job.key}.log"
            success = False
            for attempt in range(1, args.max_task_attempts + 1):
                status.update(
                    "judges",
                    job.key,
                    status="running",
                    attempt=attempt,
                    started_at=now(),
                )
                command = [
                    sys.executable,
                    str(OFFLINE_ROOT / "scripts" / "judge_results_llm_parallel.py"),
                    "--benchmark",
                    BENCHMARK_SLUG[job.benchmark],
                    "--results",
                    str(result_dir / "results.json"),
                    "--out-dir",
                    str(result_dir),
                    "--key-file",
                    str(args.judge_key_file),
                    "--workers",
                    str(args.judge_workers),
                    "--resume",
                ]
                return_code = run_logged(command, log_path)
                if return_code == 0 and judge_complete(result_dir, job.benchmark):
                    success = True
                    break
                status.update(
                    "judges",
                    job.key,
                    status="retrying",
                    return_code=return_code,
                    next_retry_seconds=60,
                )
                time.sleep(60)
            if success:
                status.update(
                    "judges", job.key, status="completed", completed_at=now()
                )
            else:
                with failed_lock:
                    failed_jobs.append(f"judge:{job.key}")
                status.update(
                    "judges", job.key, status="failed", completed_at=now()
                )
            judge_queue.task_done()

    judge_thread = threading.Thread(target=judge_worker, daemon=False)
    judge_thread.start()
    qa_threads = [
        threading.Thread(target=qa_worker, args=(endpoint,), daemon=False)
        for endpoint in args.endpoint
    ]
    for thread in qa_threads:
        thread.start()
    for thread in qa_threads:
        thread.join()
    qa_queue.join()
    judge_queue.put(None)
    judge_queue.join()
    judge_thread.join()

    if failed_jobs:
        status.root(status="failed", failed_jobs=failed_jobs, completed_at=now())
        raise SystemExit(1)
    status.root(status="completed", completed_at=now())


if __name__ == "__main__":
    main()
