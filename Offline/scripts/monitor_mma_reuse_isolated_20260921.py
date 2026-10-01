#!/usr/bin/env python3
"""Watch and safely resume the isolated MMA QA-only replays."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/data/haozhen/Memory-clean")
OFFLINE = ROOT / "Offline"
LAUNCHER = OFFLINE / "scripts/run_mma_reuse_isolated_20260921.sh"
STATUS_ROOT = OFFLINE / "outputs/_runs/mma_reuse_isolated_20260921"
STATUS_PATH = STATUS_ROOT / "watchdog_status.json"
LOG_PATH = STATUS_ROOT / "watchdog.log"
STATE_PATH = STATUS_ROOT / "watchdog_state.json"
QWEN_WAIT_LOG = STATUS_ROOT / "qwen_wait.log"

TASKS = {
    "gpt_memgallery": {
        "tmux": "mma_gpt_reuse_memgallery_0921",
        "command": ["gpt", "memgallery"],
        "result_dir": OFFLINE / "outputs/Mem-Gallery/MMA/mma_gpt5mini_reuse_isolated_20260921",
        "expected": 275,
        "eligible": "always",
    },
    "gpt_wma": {
        "tmux": "mma_gpt_reuse_wma_0921",
        "command": ["gpt", "wma"],
        "result_dir": OFFLINE / "outputs/WorldMemArena/MMA/mma_gpt5mini_reuse_isolated_20260921",
        "expected": 440,
        "eligible": "always",
    },
    "gpt_h2h": {
        "tmux": "mma_gpt_reuse_h2h_0921",
        "command": ["gpt", "h2hmem"],
        "result_dir": OFFLINE / "outputs/H2HMEM/MMA/mma_gpt5mini_reuse_isolated_20260921",
        "expected": 360,
        "eligible": "always",
    },
    "qwen_wma": {
        "tmux": "mma_qwen_reuse_wma_0921",
        "command": ["qwen", "wma"],
        "result_dir": OFFLINE / "outputs/WorldMemArena/MMA/mma_qwen_reuse_isolated_20260921",
        "expected": 440,
        "eligible": "qwen_released",
    },
    "qwen_memgallery": {
        "tmux": "mma_qwen_reuse_memgallery_0921",
        "command": ["qwen", "memgallery"],
        "result_dir": OFFLINE / "outputs/Mem-Gallery/MMA/mma_qwen_reuse_isolated_20260921",
        "expected": 275,
        "eligible": "qwen_released",
    },
    "qwen_h2h": {
        "tmux": "mma_qwen_reuse_h2h_0921",
        "command": ["qwen", "h2hmem"],
        "result_dir": OFFLINE / "outputs/H2HMEM/MMA/mma_qwen_reuse_isolated_20260921",
        "expected": 360,
        "eligible": "qwen_released",
    },
}

HARD_MARKERS = (
    "http 401", "http 403", "insufficient credits", "insufficient balance",
    "credit balance is too low", "DetachedInstanceError", "StaleDataError",
    "PendingRollbackError", "IntegrityError", "database disk image is malformed",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(message: str) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"tasks": {}, "log_offsets": {}}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def tmux_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def matching_pids(result_dir: Path) -> list[int]:
    found = []
    needle = str(result_dir).encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in command and b"benchmarks." in command and b"--resume" in command:
            found.append(int(entry.name))
    return sorted(found)


def trace_summary(result_dir: Path) -> tuple[int, dict[str, int], float]:
    phases: Counter[str] = Counter()
    calls = 0
    latest = 0.0
    for path in (result_dir / "call_traces").glob("*.jsonl"):
        latest = max(latest, path.stat().st_mtime)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            calls += 1
            phases[str(row.get("phase") or "unknown")] += 1
    return calls, dict(phases), latest


def checkpoint_progress(result_dir: Path) -> tuple[int, dict[str, int]]:
    total = 0
    samples = {}
    for path in (result_dir / ".checkpoint/native_samples").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        count = int(payload.get("completed_questions") or 0)
        samples[str(payload.get("sample_id") or path.stem)] = count
        total += count
    return total, samples


def final_count(result_dir: Path) -> int:
    path = result_dir / "results.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        rows = payload.get("results")
        return len(rows) if isinstance(rows, list) else 0
    return 0


def qwen_released() -> bool:
    try:
        return "scheduler_launched_all" in QWEN_WAIT_LOG.read_text(encoding="utf-8")
    except OSError:
        return False


def new_log_text(path: Path, offsets: dict[str, int]) -> str:
    key = str(path)
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    offset = min(int(offsets.get(key, size)), size)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        offsets[key] = handle.tell()
    return text


def failure_fingerprint(text: str) -> str:
    lines = [line for line in text.splitlines() if "[sample-error]" in line]
    source = lines[-1] if lines else "unexpected_exit"
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def start_task(task: dict) -> None:
    name = str(task["tmux"])
    command = " ".join(["bash", repr(str(LAUNCHER)), *map(str, task["command"])])
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", name, f"cd {ROOT!s} && {command}"],
        check=True,
    )


def main() -> None:
    state = load_state()
    task_state = state.setdefault("tasks", {})
    offsets = state.setdefault("log_offsets", {})
    # Existing logs are historical. Only hard-stop on text appended after this
    # watchdog begins, so earlier diagnosed canaries cannot poison a replay.
    for task in TASKS.values():
        path = Path(task["result_dir"]) / "run.log"
        offsets.setdefault(str(path), path.stat().st_size if path.exists() else 0)
    atomic_json(STATE_PATH, state)

    while True:
        released = qwen_released()
        report = {"updated_at": now(), "qwen_released": released, "tasks": {}}
        for key, task in TASKS.items():
            result_dir = Path(task["result_dir"])
            expected = int(task["expected"])
            local = task_state.setdefault(key, {"restarts": {}, "hard_stop": ""})
            log_text = new_log_text(result_dir / "run.log", offsets)
            lowered = log_text.lower()
            hard = next((marker for marker in HARD_MARKERS if marker.lower() in lowered), "")
            if hard:
                local["hard_stop"] = f"log marker: {hard}"
                append_log(f"HARD_STOP task={key} reason={local['hard_stop']}")

            calls, phases, latest = trace_summary(result_dir)
            checkpoint, samples = checkpoint_progress(result_dir)
            finals = final_count(result_dir)
            pids = matching_pids(result_dir)
            complete = finals == expected
            eligible = task["eligible"] == "always" or released
            status = "complete" if complete else ("running" if pids else "waiting")
            if local.get("hard_stop"):
                status = "hard_stopped"
            elif len(pids) > 1:
                local["hard_stop"] = f"duplicate processes: {pids}"
                status = "hard_stopped"
                append_log(f"HARD_STOP task={key} reason={local['hard_stop']}")
            elif eligible and not complete and not pids and not tmux_alive(str(task["tmux"])):
                fingerprint = failure_fingerprint(log_text)
                restarts = local.setdefault("restarts", {})
                count = int(restarts.get(fingerprint, 0))
                if count >= 3:
                    local["hard_stop"] = f"same failure reached 3 restarts: {fingerprint}"
                    status = "hard_stopped"
                    append_log(f"HARD_STOP task={key} reason={local['hard_stop']}")
                else:
                    start_task(task)
                    restarts[fingerprint] = count + 1
                    status = "restarted"
                    append_log(
                        f"RESTART task={key} attempt={count + 1}/3 "
                        f"checkpoint={checkpoint}/{expected} calls={calls}"
                    )
            report["tasks"][key] = {
                "status": status,
                "checkpoint_questions": checkpoint,
                "expected_questions": expected,
                "final_results": finals,
                "calls": calls,
                "phases": phases,
                "latest_trace_epoch": latest,
                "pids": pids,
                "samples": samples,
                "hard_stop": local.get("hard_stop", ""),
                "restart_counts": local.get("restarts", {}),
            }
        atomic_json(STATE_PATH, state)
        atomic_json(STATUS_PATH, report)
        append_log(
            "HEARTBEAT " + " ".join(
                f"{key}={value['status']}:{value['checkpoint_questions']}/"
                f"{value['expected_questions']}:{value['calls']}calls"
                for key, value in report["tasks"].items()
            )
        )
        time.sleep(60)


if __name__ == "__main__":
    main()
