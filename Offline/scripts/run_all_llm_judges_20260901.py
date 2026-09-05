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
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
JUDGE_SCRIPT = PROJECT_ROOT / "scripts" / "judge_results_llm_parallel.py"
PYTHON = Path("/data/haozhen/miniconda3/envs/pipeline_repro/bin/python")
KEY_FILE = Path("/data/haozhen/Memory/Nvida_api/gpt-4o-mini")
LOG_ROOT = PROJECT_ROOT / "logs" / "llm_judge_all_20260901"
STATUS_PATH = LOG_ROOT / "status.json"

BENCHMARKS = {
    "Mem-Gallery": ("memgallery", 1711),
    "WorldMemArena": ("worldmemarena", 2090),
    "H2HMEM": ("h2hmem", 2207),
}
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


def task_order(methods: tuple[str, ...] = METHODS) -> list[tuple[str, str]]:
    # Finish all HiveMem judges before spending quota on baselines.
    hive = (
        [(benchmark, "HiveMem") for benchmark in BENCHMARKS]
        if "HiveMem" in methods
        else []
    )
    baselines = [
        (benchmark, method)
        for benchmark in BENCHMARKS
        for method in methods
        if method != "HiveMem"
    ]
    return hive + baselines


def write_status(state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def run_judge(benchmark: str, method: str, workers: int, state: dict) -> bool:
    benchmark_arg, expected = BENCHMARKS[benchmark]
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
        "--exclude-method",
        action="append",
        choices=METHODS,
        default=[],
        help="Method to omit entirely; may be passed more than once.",
    )
    args = parser.parse_args()
    excluded_methods = set(args.exclude_method)
    selected_methods = tuple(
        method for method in METHODS if method not in excluded_methods
    )
    if not selected_methods:
        parser.error("At least one method must remain after exclusions")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    state = {
        "workers": args.workers,
        "active": None,
        "tasks": {},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    while True:
        pending_incomplete: list[str] = []
        made_progress = False
        for benchmark, method in task_order(selected_methods):
            benchmark_arg, expected = BENCHMARKS[benchmark]
            del benchmark_arg
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
            run_judge(benchmark, method, args.workers, state)

        state["waiting_for_results"] = pending_incomplete
        completed = sum(
            row.get("status") == "complete" for row in state["tasks"].values()
        )
        state["completed"] = completed
        state["expected_tasks"] = len(BENCHMARKS) * len(selected_methods)
        write_status(state)
        if completed == state["expected_tasks"] or not args.watch:
            break
        if not made_progress:
            time.sleep(max(args.poll_seconds, 30))


if __name__ == "__main__":
    main()
