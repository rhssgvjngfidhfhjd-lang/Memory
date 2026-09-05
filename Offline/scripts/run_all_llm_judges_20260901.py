#!/usr/bin/env python3
"""Run final LLM judges for every complete benchmark/method output.

HiveMem is deliberately scheduled first.  With ``--watch``, incomplete baseline
runs are picked up after their full ``results.json`` appears, so this process can
stay detached while the baseline supervisor finishes the remaining experiments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.benchmark_manifest import (  # noqa: E402
    BenchmarkExpectation,
    load_benchmark_manifest,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
JUDGE_SCRIPT = PROJECT_ROOT / "scripts" / "judge_results_llm_parallel.py"
PYTHON = Path("/data/haozhen/miniconda3/envs/pipeline_repro/bin/python")
KEY_FILE = Path("/data/haozhen/Memory/Nvida_api/gpt-4o-mini")
LOG_ROOT = PROJECT_ROOT / "logs" / "llm_judge_all_20260901"
STATUS_PATH = LOG_ROOT / "status.json"

BENCHMARKS = ("Mem-Gallery", "WorldMemArena", "H2HMEM")
DEFAULT_BENCHMARK_MANIFEST = PROJECT_ROOT / "configs" / "full_benchmark_manifest.json"
METHODS = (
    "HiveMem",
    "AUGUSTUSMemory",
    "OmniSimpleMem",
    "M2A",
    "MIRIX",
    "MMA",
    "MemVerse",
    "M3-Agent-caption",
)


def result_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return len(payload["results"])
    return 0


def is_judge_complete(result_dir: Path, expected: int) -> bool:
    path = result_dir / "llm_judge_metrics.json"
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metrics.get("count") == expected
        and metrics.get("valid_count") == expected
        and metrics.get("judge_errors") == 0
        and not metrics.get("provisional", True)
    )


def task_order() -> list[tuple[str, str]]:
    # Finish all HiveMem judges before spending quota on baselines.
    hive = [(benchmark, "HiveMem") for benchmark in BENCHMARKS]
    baselines = [
        (benchmark, method)
        for benchmark in BENCHMARKS
        for method in METHODS
        if method != "HiveMem"
    ]
    return hive + baselines


def write_status(state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def run_judge(
    benchmark: str,
    method: str,
    workers: int,
    state: dict,
    expectation: BenchmarkExpectation,
) -> bool:
    benchmark_arg = expectation.judge_name
    expected = expectation.question_count
    result_dir = OUTPUT_ROOT / benchmark / method
    results_path = result_dir / "results.json"
    log_path = LOG_ROOT / f"{benchmark}__{method}.log"
    command = [
        str(PYTHON),
        str(JUDGE_SCRIPT),
        "--benchmark", benchmark_arg,
        "--results", str(results_path),
        "--out-dir", str(result_dir),
        "--key-file", str(KEY_FILE),
        "--workers", str(workers),
        "--timeout", "60",
        "--retries", "3",
        "--checkpoint-every", "25",
        "--resume",
    ]
    key = f"{benchmark}/{method}"
    state["active"] = key
    state["tasks"][key] = {
        "status": "running",
        "expected": expected,
        "log": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_status(state)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== START {datetime.now().isoformat()} workers={workers} ===\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"=== EXIT {result.returncode} {datetime.now().isoformat()} ===\n")
    complete = result.returncode == 0 and is_judge_complete(result_dir, expected)
    state["tasks"][key].update(
        {
            "status": "complete" if complete else "failed",
            "returncode": result.returncode,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    state["active"] = None
    write_status(state)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK_MANIFEST,
        help="Full-benchmark completeness manifest.",
    )
    args = parser.parse_args()
    expectations = load_benchmark_manifest(
        args.benchmark_manifest,
        required_benchmarks=BENCHMARKS,
    )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state = {
        "workers": args.workers,
        "active": None,
        "tasks": {},
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
    }

    while True:
        pending_incomplete: list[str] = []
        made_progress = False
        for benchmark, method in task_order():
            expectation = expectations[benchmark]
            expected = expectation.question_count
            result_dir = OUTPUT_ROOT / benchmark / method
            results_path = result_dir / "results.json"
            key = f"{benchmark}/{method}"
            if is_judge_complete(result_dir, expected):
                state["tasks"].setdefault(key, {"status": "complete", "expected": expected})
                continue
            count = result_count(results_path)
            if count != expected:
                pending_incomplete.append(f"{key} ({count}/{expected})")
                state["tasks"][key] = {
                    "status": "waiting_for_results",
                    "count": count,
                    "expected": expected,
                }
                continue
            made_progress = True
            run_judge(benchmark, method, args.workers, state, expectation)

        state["waiting_for_results"] = pending_incomplete
        completed = sum(
            row.get("status") == "complete" for row in state["tasks"].values()
        )
        state["completed"] = completed
        state["expected_tasks"] = len(BENCHMARKS) * len(METHODS)
        write_status(state)
        if completed == state["expected_tasks"] or not args.watch:
            break
        if not made_progress:
            time.sleep(max(args.poll_seconds, 30))


if __name__ == "__main__":
    main()
