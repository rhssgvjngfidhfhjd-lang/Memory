#!/usr/bin/env python3
"""Keep the three remaining MemVerse MB-call measurements running in tmux."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "mb_call_rerun_20260904"
STATUS_DIR = ROOT / "logs" / "mb_call_supervisor"
EXPECTED_SAMPLES = {
    "Mem-Gallery": 20,
    "WorldMemArena": 38,
    "H2HMEM": 25,
}


@dataclass(frozen=True)
class Task:
    benchmark: str
    endpoint: str
    session: str


TASKS = (
    Task("WorldMemArena", "http://127.0.0.1:8013/v1", "mb_calls_gpu3_after_mirix"),
    Task("Mem-Gallery", "http://127.0.0.1:8014/v1", "mb_calls_gpu4_mma_memverse"),
    Task("H2HMEM", "http://127.0.0.1:8014/v1", "mb_calls_gpu4_mma_memverse"),
)
GPU5_WMA_HELPER = Task(
    "WorldMemArena", "http://127.0.0.1:8015/v1", "mb_calls_gpu5_memverse_wma"
)
GPU5_H2_HELPER = Task(
    "H2HMEM", "http://127.0.0.1:8015/v1", "mb_calls_gpu5_memverse_h2"
)
GPU5_HELPERS = (GPU5_WMA_HELPER, GPU5_H2_HELPER)
FORMAL_WMA_RESULT = ROOT / "outputs" / "WorldMemArena" / "MemVerse" / "results.json"
FORMAL_WMA_SESSION = "recover_memverse_wma"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def metrics_complete(path: Path, expected: int) -> bool:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        metrics.get("available")
        and metrics.get("num_samples") == expected
        and metrics.get("completed_samples") == expected
        and metrics.get("failed_samples") == 0
    )


def tmux_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def endpoint_ready(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(f"{endpoint.rstrip('/')}/models", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def task_command(
    task: Task,
    *,
    output_root: Path,
    embedding_base_url: str,
    sample_concurrency: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "measure_baseline_mb_calls.py"),
        "--baseline",
        "MemVerse",
        "--benchmark",
        task.benchmark,
        "--executor-base-url",
        task.endpoint,
        "--output-root",
        str(output_root),
        "--embedding-base-url",
        embedding_base_url,
        "--sample-concurrency",
        str(sample_concurrency),
        "--resume",
    ]


def start_task(
    task: Task,
    *,
    output_root: Path,
    embedding_base_url: str,
    sample_concurrency: int,
) -> None:
    log_path = STATUS_DIR / f"{task.benchmark}_MemVerse.log"
    command = task_command(
        task,
        output_root=output_root,
        embedding_base_url=embedding_base_url,
        sample_concurrency=sample_concurrency,
    )
    shell = (
        f"cd {shlex.quote(str(ROOT))} && "
        "export PYTHONPATH=src PYTHONUNBUFFERED=1 "
        f"TMPDIR={shlex.quote(str(ROOT / 'tmp'))} "
        f"SQLITE_TMPDIR={shlex.quote(str(ROOT / 'tmp'))} && "
        f"{' '.join(shlex.quote(value) for value in command)} "
        f">> {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", task.session, shell],
        check=True,
    )


def write_status(payload: dict[str, Any]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    path = STATUS_DIR / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--sample-concurrency", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.sample_concurrency < 1 or args.poll_seconds < 1:
        parser.error("concurrency and poll interval must be positive")

    output_root = Path(args.output_root)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    while True:
        complete = {
            task.benchmark: metrics_complete(
                output_root / task.benchmark / "MemVerse" / "metrics.json",
                EXPECTED_SAMPLES[task.benchmark],
            )
            for task in TASKS
        }
        launches: list[str] = []

        wma = TASKS[0]
        if not complete[wma.benchmark] and not tmux_exists(wma.session):
            if endpoint_ready(wma.endpoint):
                start_task(
                    wma,
                    output_root=output_root,
                    embedding_base_url=args.embedding_base_url,
                    sample_concurrency=args.sample_concurrency,
                )
                launches.append(wma.benchmark)

        # GPU4 is intentionally serial: finish Mem-Gallery before H2HMEM.
        gpu4_task = next(
            (task for task in TASKS[1:] if not complete[task.benchmark]),
            None,
        )
        if gpu4_task is not None and not tmux_exists(gpu4_task.session):
            if endpoint_ready(gpu4_task.endpoint):
                start_task(
                    gpu4_task,
                    output_root=output_root,
                    embedding_base_url=args.embedding_base_url,
                    sample_concurrency=args.sample_concurrency,
                )
                launches.append(gpu4_task.benchmark)

        # Once the formal WMA job releases GPU5, use it on the longest
        # remaining MB-call path first. Per-sample file locks in the
        # measurement runner let GPU5 share a benchmark with its primary
        # worker without rebuilding the same sample.
        gpu5_released = FORMAL_WMA_RESULT.is_file() and not tmux_exists(
            FORMAL_WMA_SESSION
        )
        gpu5_task = next(
            (task for task in GPU5_HELPERS if not complete[task.benchmark]),
            None,
        )
        if (
            gpu5_released
            and gpu5_task is not None
            and not any(tmux_exists(task.session) for task in GPU5_HELPERS)
            and endpoint_ready(gpu5_task.endpoint)
        ):
            start_task(
                gpu5_task,
                output_root=output_root,
                embedding_base_url=args.embedding_base_url,
                sample_concurrency=args.sample_concurrency,
            )
            launches.append(f"{gpu5_task.benchmark}@gpu5")

        write_status(
            {
                "updated_at": now(),
                "complete": complete,
                "launches": launches,
                "sessions": {
                    task.session: tmux_exists(task.session)
                    for task in (*TASKS, *GPU5_HELPERS)
                },
            }
        )
        if all(complete.values()):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
