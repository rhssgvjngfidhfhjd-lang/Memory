#!/usr/bin/env python3
"""Keep at most two five-benchmark OpenRouter jobs active and gate formal on smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_memverse_openrouter_five_benchmarks.py"
ORDER = ("Mem-Gallery", "H2HMEM", "WorldMemArena", "MemEye", "MEMLENS")


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_phase(run_id: str, phase: str, status_path: Path, max_parallel: int) -> bool:
    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    log_root = status_path.parent / "_logs" / "scheduler"
    log_root.mkdir(parents=True, exist_ok=True)
    while True:
        for benchmark, (process, handle) in list(processes.items()):
            if process.poll() is None:
                continue
            handle.close()
            del processes[benchmark]

        status = _load(status_path)
        states = {
            benchmark: str(status.get(f"{phase}:{benchmark}", {}).get("status") or "pending")
            for benchmark in ORDER
        }
        if all(value in {"completed", "failed"} for value in states.values()):
            return all(value == "completed" for value in states.values())

        occupied = sum(value == "running" for value in states.values())
        capacity = max(max_parallel - occupied, 0)
        for benchmark in ORDER:
            if capacity <= 0:
                break
            if states[benchmark] != "pending":
                continue
            log_path = log_root / f"{phase}_{benchmark}.log"
            handle = log_path.open("a", encoding="utf-8", buffering=1)
            command = [
                sys.executable,
                str(RUNNER),
                "--run-id", run_id,
                "--benchmark", benchmark,
                f"--{phase}",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes[benchmark] = (process, handle)
            capacity -= 1
        time.sleep(20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    status_path = (
        ROOT / "outputs" / "_runs" / args.run_id / "five_benchmark_status.json"
    )
    smoke_ok = _run_phase(args.run_id, "smoke", status_path, args.max_parallel)
    if not smoke_ok:
        raise RuntimeError("one or more smoke jobs failed; formal jobs were not started")
    formal_ok = _run_phase(args.run_id, "formal", status_path, args.max_parallel)
    if not formal_ok:
        raise RuntimeError("one or more formal jobs failed; successful peers were preserved")


if __name__ == "__main__":
    main()
