#!/usr/bin/env python3
"""Run the 7-baseline x 3-benchmark MB-call measurement matrix."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from benchmarks.io_utils import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
MEASURE_SCRIPT = ROOT / "scripts" / "measure_baseline_mb_calls.py"
BASELINES = (
    "M2A",
    "M3-Agent-caption",
    "AUGUSTUSMemory",
    "MIRIX",
    "MMA",
    "OmniSimpleMem",
    "MemVerse",
)
BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena")


def _job_key(baseline: str, benchmark: str) -> str:
    return f"{benchmark}__{baseline}"


def _complete(output_root: Path, baseline: str, benchmark: str) -> bool:
    path = output_root / benchmark / baseline / "metrics.json"
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("available"))
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--baseline", action="append", choices=BASELINES, default=[])
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS, default=[])
    args = parser.parse_args()
    if args.sample_concurrency < 1:
        parser.error("--sample-concurrency must be positive")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    selected_baselines = tuple(args.baseline) or BASELINES
    selected_benchmarks = tuple(args.benchmark) or BENCHMARKS
    jobs = deque(
        (baseline, benchmark)
        for baseline in selected_baselines
        for benchmark in selected_benchmarks
        if not _complete(output_root, baseline, benchmark)
    )
    status_path = output_root / "status.json"
    lock = threading.Lock()
    status: dict[str, dict[str, Any]] = {}
    for baseline in selected_baselines:
        for benchmark in selected_benchmarks:
            key = _job_key(baseline, benchmark)
            status[key] = {
                "baseline": baseline,
                "benchmark": benchmark,
                "status": (
                    "completed" if _complete(output_root, baseline, benchmark) else "pending"
                ),
            }
    write_json_atomic(status_path, status)

    def update(key: str, **values: Any) -> None:
        with lock:
            status[key].update(values)
            status[key]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json_atomic(status_path, status)

    def worker(endpoint: str) -> None:
        while True:
            with lock:
                if not jobs:
                    return
                baseline, benchmark = jobs.popleft()
            key = _job_key(baseline, benchmark)
            log_path = log_root / f"{key}.log"
            command = [
                sys.executable,
                str(MEASURE_SCRIPT),
                "--baseline",
                baseline,
                "--benchmark",
                benchmark,
                "--output-root",
                str(output_root),
                "--executor-base-url",
                endpoint,
                "--embedding-base-url",
                args.embedding_base_url,
                "--sample-concurrency",
                str(args.sample_concurrency),
                "--resume",
            ]
            update(key, status="running", endpoint=endpoint, log_path=str(log_path))
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "src")
            env["PYTHONUNBUFFERED"] = "1"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"START endpoint={endpoint}\n"
                )
                log.flush()
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"EXIT code={completed.returncode}\n"
                )
            update(
                key,
                status="completed" if completed.returncode == 0 else "failed",
                returncode=completed.returncode,
            )

    threads = [
        threading.Thread(target=worker, args=(endpoint,), daemon=False)
        for endpoint in args.endpoint
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
