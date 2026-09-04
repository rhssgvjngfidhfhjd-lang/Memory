#!/usr/bin/env python3
"""Detached recovery supervisor for the remaining baseline jobs.

This one-off supervisor keeps the active recovery jobs in tmux and checks
progress every 30 minutes.  It avoids duplicating a job if a tmux session or a
final results.json already exists.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/haozhen/miniconda3/envs/pipeline_repro/bin/python")
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
EMBED_URL = "http://127.0.0.1:8001/v1"
LOG_DIR = ROOT / "logs" / "recovery_supervisor"
STATUS_PATH = LOG_DIR / "status_20260901.json"
CHECK_INTERVAL_SECONDS = 1800


@dataclass(frozen=True)
class Task:
    name: str
    session: str
    result_dir: Path
    log_path: Path
    command: list[str]
    deps: tuple[str, ...] = ()


def run(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def tmux_exists(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session]).returncode == 0


def shell_join(command: list[str], log_path: Path) -> str:
    exports = {
        "PYTHONPATH": "src",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(ROOT / "tmp"),
        "SQLITE_TMPDIR": str(ROOT / "tmp"),
    }
    prefix = " ".join(f"export {key}={shlex.quote(value)};" for key, value in exports.items())
    return (
        f"cd {shlex.quote(str(ROOT))}; "
        f"{prefix} "
        f"{' '.join(shlex.quote(part) for part in command)} "
        f">> {shlex.quote(str(log_path))} 2>&1"
    )


def start_task(task: Task) -> bool:
    if task.result_dir.joinpath("results.json").is_file() or tmux_exists(task.session):
        return False
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            task.session,
            shell_join(task.command, task.log_path),
        ],
        check=True,
    )
    return True


def common_args(endpoint: str) -> list[str]:
    return [
        "--answer-model",
        MODEL,
        "--answer-base-url",
        endpoint,
        "--answer-temperature",
        "0",
        "--executor-model",
        MODEL,
        "--executor-base-url",
        endpoint,
        "--executor-temperature",
        "0",
        "--embedding-model",
        EMBED_MODEL,
        "--embedding-base-url",
        EMBED_URL,
        "--embedding-dim",
        "2048",
        "--top-k",
        "5",
        "--request-timeout",
        "180",
        "--retries",
        "2",
    ]


def baseline_dirs(benchmark: str, method: str) -> tuple[Path, Path]:
    result = ROOT / "outputs" / benchmark / method
    return result, result / "memory" / "datasets"


def task_memverse_h2h() -> Task:
    result, state = baseline_dirs("H2HMEM", "MemVerse")
    return Task(
        name="memverse_h2hmem",
        session="recover_memverse_h2h",
        result_dir=result,
        log_path=ROOT / "logs/recovery_memverse/h2hmem_20260901.log",
        command=[
            "env",
            "MEMVERSE_REUSE_STATE=1",
            "BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT=1",
            str(PYTHON),
            "-m",
            "benchmarks.h2hmem_harness.eval_h2hmem",
            "--baseline",
            "MemVerse",
            "--variant",
            "all",
            "--result-dir",
            str(result),
            "--baseline-state-dir",
            str(state),
            "--resume",
            "--sample-concurrency",
            "4",
            "--answer-concurrency",
            "16",
            *common_args("http://127.0.0.1:8013/v1"),
        ],
    )


def task_memverse_memgallery() -> Task:
    result, state = baseline_dirs("Mem-Gallery", "MemVerse")
    return Task(
        name="memverse_mem_gallery",
        session="recover_memverse_memgallery",
        result_dir=result,
        log_path=ROOT / "logs/recovery_memverse/memgallery_20260901.log",
        command=[
            str(PYTHON),
            "-m",
            "benchmarks.memgallery_harness.eval_memgallery",
            "--baseline",
            "MemVerse",
            "--all-datasets",
            "--result-dir",
            str(result),
            "--baseline-state-dir",
            str(state),
            "--resume",
            "--sample-concurrency",
            "4",
            "--answer-concurrency",
            "16",
            "--exclude-categories",
            "",
            *common_args("http://127.0.0.1:8014/v1"),
        ],
        deps=("mirix_wma_done",),
    )


def task_memverse_wma() -> Task:
    result, state = baseline_dirs("WorldMemArena", "MemVerse")
    return Task(
        name="memverse_worldmemarena",
        session="recover_memverse_wma",
        result_dir=result,
        log_path=ROOT / "logs/recovery_memverse/wma_20260901.log",
        command=[
            "env",
            "MEMVERSE_REUSE_STATE=1",
            "BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT=1",
            str(PYTHON),
            "-m",
            "benchmarks.wma_harness.eval_wma",
            "--baseline",
            "MemVerse",
            "--result-dir",
            str(result),
            "--baseline-state-dir",
            str(state),
            "--resume",
            "--sample-concurrency",
            "4",
            "--answer-concurrency",
            "16",
            "--exclude-categories",
            "",
            *common_args("http://127.0.0.1:8015/v1"),
        ],
        deps=("memverse_mem_gallery",),
    )


def task_omni_h2h() -> Task:
    result, state = baseline_dirs("H2HMEM", "OmniSimpleMem")
    return Task(
        name="omni_h2hmem",
        session="recover_omni_h2h",
        result_dir=result,
        log_path=ROOT / "logs/recovery_omni/h2hmem_omni_20260901.log",
        command=[
            "env",
            "OMNI_SIMPLEMEM_REUSE_STATE=1",
            "BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT=1",
            str(PYTHON),
            "-m",
            "benchmarks.h2hmem_harness.eval_h2hmem",
            "--baseline",
            "OmniSimpleMem",
            "--variant",
            "all",
            "--result-dir",
            str(result),
            "--baseline-state-dir",
            str(state),
            "--resume",
            "--sample-concurrency",
            "1",
            "--answer-concurrency",
            "16",
            *common_args("http://127.0.0.1:8013/v1"),
        ],
    )


def task_omni_wma() -> Task:
    result, state = baseline_dirs("WorldMemArena", "OmniSimpleMem")
    return Task(
        name="omni_worldmemarena",
        session="recover_omni_wma_retry",
        result_dir=result,
        log_path=ROOT / "logs/recovery_omni/wma_omni_20260901.log",
        command=[
            "env",
            "BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT=1",
            str(PYTHON),
            "-m",
            "benchmarks.wma_harness.eval_wma",
            "--baseline",
            "OmniSimpleMem",
            "--result-dir",
            str(result),
            "--baseline-state-dir",
            str(state),
            "--resume",
            "--sample-concurrency",
            "4",
            "--answer-concurrency",
            "16",
            "--exclude-categories",
            "",
            *common_args("http://127.0.0.1:8013/v1"),
        ],
    )


def checkpoint_count(result_dir: Path) -> int | None:
    sample_dir = result_dir / ".checkpoint" / "samples"
    if not sample_dir.is_dir():
        return None
    return sum(1 for _ in sample_dir.glob("*.json"))


def task_done(name: str) -> bool:
    if name == "mirix_wma_done":
        return (ROOT / "outputs/WorldMemArena/MIRIX/results.json").is_file() and not tmux_exists(
            "recover_mirix_wma"
        )
    mapping = {
        "omni_h2hmem": ROOT / "outputs/H2HMEM/OmniSimpleMem/results.json",
        "memverse_h2hmem": ROOT / "outputs/H2HMEM/MemVerse/results.json",
        "memverse_mem_gallery": ROOT / "outputs/Mem-Gallery/MemVerse/results.json",
    }
    return mapping[name].is_file()


def gpu_snapshot() -> str:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return "\n".join(result.stdout.splitlines()[3:6])


def write_status(tasks: list[Task], launches: list[str]) -> None:
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu345": gpu_snapshot(),
        "launches": launches,
        "tasks": {
            task.name: {
                "session": task.session,
                "session_active": tmux_exists(task.session),
                "results": task.result_dir.joinpath("results.json").is_file(),
                "checkpoint_samples": checkpoint_count(task.result_dir),
                "log": str(task.log_path),
            }
            for task in tasks
        },
        "existing_recovery": {
            name: tmux_exists(name)
            for name in (
                "recover_mirix_wma",
                "recover_mma_wma",
                "recover_omni_remaining",
            )
        },
    }
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    with (LOG_DIR / "monitor_20260901.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, ensure_ascii=False) + "\n")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    tasks = [
        task_omni_h2h(),
        task_omni_wma(),
        task_memverse_h2h(),
        task_memverse_memgallery(),
        task_memverse_wma(),
    ]
    while True:
        launches: list[str] = []
        for task in tasks:
            if task.name.startswith("omni_") and tmux_exists("recover_omni_remaining"):
                continue
            if all(task_done(dep) for dep in task.deps):
                if start_task(task):
                    launches.append(task.name)
        write_status(tasks, launches)
        if all(task.result_dir.joinpath("results.json").is_file() for task in tasks):
            break
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
