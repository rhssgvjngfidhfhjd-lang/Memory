#!/usr/bin/env python3
"""Run one new MIRIX benchmark on a delegated local vLLM endpoint.

The master five-benchmark runner only skips jobs whose public status is
``completed``.  While this delegated job is active, that public status acts as
a scheduling guard and ``delegated_status`` carries the real state.  This
prevents the already-running sequential process from launching a duplicate on
GPU4 after WorldMemArena finishes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import re
from typing import Any

try:
    from scripts.run_mirix_local_five_benchmarks import (
        EXPECTED_QA,
        NEW_HARNESSES,
        OUTPUT_ROOT,
        _environment,
        _judge,
        _run_new,
        _validate_generation,
        _write_json,
        check_local_services,
    )
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    # Direct execution adds this file's directory, rather than its parent, to
    # sys.path.  Keep both direct and package-style invocation supported.
    from run_mirix_local_five_benchmarks import (
        EXPECTED_QA,
        NEW_HARNESSES,
        OUTPUT_ROOT,
        _environment,
        _judge,
        _run_new,
        _validate_generation,
        _write_json,
        check_local_services,
    )


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run ID must be a safe path component")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _update_master(
    status_path: Path,
    *,
    benchmark: str,
    values: dict[str, Any],
) -> None:
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        key = f"formal:{benchmark}"
        status[key] = {**status.get(key, {}), **values}
        _write_json(status_path, status)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=_safe_run_id)
    parser.add_argument("--benchmark", required=True, choices=tuple(NEW_HARNESSES))
    parser.add_argument("--endpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_path = OUTPUT_ROOT / "_runs" / args.run_id / "five_benchmark_status.json"
    started_at = _now()
    # ``status=completed`` is deliberately the master scheduler guard.  The
    # actual state remains explicit and machine-readable in delegated_status.
    _update_master(
        status_path,
        benchmark=args.benchmark,
        values={
            "status": "completed",
            "delegated_status": "running",
            "delegated_endpoint": args.endpoint,
            "delegated_started_at": started_at,
            "delegated_finished_at": None,
            "started_at": started_at,
            "finished_at": None,
            "error": None,
            "scheduler_guard": "completed_while_delegated_running",
        },
    )
    try:
        check_local_services(args.endpoint)
        env = _environment()
        result_dir = _run_new(
            run_id=args.run_id,
            benchmark=args.benchmark,
            phase="formal",
            env=env,
            base_url=args.endpoint,
        )
        recovered = _validate_generation(
            result_dir, expected=EXPECTED_QA[args.benchmark]
        )
        _judge(
            run_id=args.run_id,
            benchmark=args.benchmark,
            result_dir=result_dir,
            env=env,
        )
    except Exception as exc:
        _update_master(
            status_path,
            benchmark=args.benchmark,
            values={
                "status": "failed",
                "delegated_status": "failed",
                "delegated_finished_at": _now(),
                "finished_at": _now(),
                "scheduler_guard": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finished_at = _now()
    _update_master(
        status_path,
        benchmark=args.benchmark,
        values={
            "status": "completed",
            "delegated_status": "completed",
            "delegated_finished_at": finished_at,
            "finished_at": finished_at,
            "scheduler_guard": None,
            "result_dir": str(result_dir),
            "expected_qa": EXPECTED_QA[args.benchmark],
            "answer_errors": 0,
            "recovered_truncations": recovered,
            "error": None,
        },
    )


if __name__ == "__main__":
    main()
