#!/usr/bin/env python3
"""Continuously monitor one MMA formal run without aborting recoverable retries."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time
from typing import Any


TERMINAL_PHASES = {"complete", "incomplete"}
FINAL_ARTIFACTS = (
    "results.json",
    "retrieval_trace.jsonl",
    "run_manifest.json",
    "pipeline_qa.jsonl",
)
AUTH_STATUSES = {401, 403}
TRANSIENT_STATUSES = {429, 502, 503, 504}
BALANCE_MARKERS = (
    "insufficient credits",
    "insufficient balance",
    "credit balance is too low",
    "quota exceeded",
)
REQUIRED_RESUME_ENV_KEYS = ("OPENAI_API_KEY",)
UNSAFE_LOG_MARKERS = (
    "DetachedInstanceError",
    "ObjectDeletedError",
    "StaleDataError",
    "PendingRollbackError",
    "ResourceClosedError",
    "UnboundExecutionError",
    "DBAPIError",
    "DatabaseError",
    "IntegrityError",
    "OperationalError",
    "InternalError",
    "ProgrammingError",
    "InterfaceError",
    "StatementError",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{now()} {message}\n")


def descendants(pid: int) -> list[int]:
    parent_by_pid: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            parent_by_pid[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    found: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children = [child for child, value in parent_by_pid.items() if value == parent]
        found.extend(children)
        frontier.extend(children)
    return found


def stop_job(pid: int, run_id: str, log_path: Path, reason: str) -> None:
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        append_log(log_path, f"STOP_SKIPPED pid={pid} vanished reason={reason}")
        return
    if run_id.encode() not in command:
        append_log(
            log_path,
            f"STOP_SKIPPED pid={pid} command_does_not_match_run reason={reason}",
        )
        return
    targets = [*reversed(descendants(pid)), pid]
    append_log(log_path, f"STOP_JOB pid={pid} descendants={targets[:-1]} reason={reason}")
    for target in targets:
        try:
            os.kill(target, signal.SIGTERM)
        except ProcessLookupError:
            pass


def process_command(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def process_environment_value(pid: int, key: str) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError:
        return ""
    prefix = key.encode() + b"="
    for entry in raw.split(b"\0"):
        if entry.startswith(prefix):
            return entry[len(prefix) :].decode(errors="replace")
    return ""


def inherit_required_resume_environment(jobs: dict[str, dict[str, Any]]) -> list[tuple[str, int]]:
    """Copy missing API credentials from a still-live sibling formal job."""
    inherited: list[tuple[str, int]] = []
    candidate_pids = [
        int(job.get("child_pid") or 0)
        for job in jobs.values()
        if str(job.get("method")) == "MMA"
    ]
    for key in REQUIRED_RESUME_ENV_KEYS:
        if os.environ.get(key):
            continue
        for pid in candidate_pids:
            if pid <= 0 or not process_command(pid):
                continue
            value = process_environment_value(pid, key)
            if value:
                os.environ[key] = value
                inherited.append((key, pid))
                break
    return inherited


def command_matches_job(command: list[str], job: dict[str, Any]) -> bool:
    result_dir = str(job.get("result_dir") or "")
    return bool(
        command
        and result_dir
        and result_dir in command
        and "--resume" in command
        and "--baseline" in command
        and "MMA" in command
    )


def matching_job_pids(job: dict[str, Any]) -> list[int]:
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if command_matches_job(process_command(pid), job):
            matches.append(pid)
    return sorted(matches)


def final_artifacts_present(result_dir: Path) -> bool:
    return all((result_dir / name).is_file() for name in FINAL_ARTIFACTS)


def validate_resume_checkpoints(result_dir: Path) -> tuple[bool, str]:
    """Validate checkpoint metadata and its immutable SQLite snapshots read-only."""
    state_root = result_dir / "memory" / "datasets"
    if not state_root.is_dir():
        return False, f"state root is missing: {state_root}"
    manifests = sorted(state_root.rglob(".offline_mma_resume.json"))
    if not manifests:
        return False, f"no MMA session checkpoints under {state_root}"
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"unreadable checkpoint {manifest}: {exc}"
        snapshot_name = str(payload.get("sqlite_snapshot") or "")
        if (
            payload.get("version") != 1
            or not payload.get("sample_id")
            or not payload.get("signature")
            or not snapshot_name
            or Path(snapshot_name).name != snapshot_name
        ):
            return False, f"invalid checkpoint metadata: {manifest}"
        snapshot = manifest.parent / ".resume" / snapshot_name
        if not snapshot.is_file():
            return False, f"checkpoint snapshot is missing: {snapshot}"
        try:
            connection = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
            try:
                row = connection.execute("PRAGMA quick_check").fetchone()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            return False, f"checkpoint SQLite check failed for {snapshot}: {exc}"
        if not row or str(row[0]).lower() != "ok":
            return False, f"checkpoint SQLite is corrupt: {snapshot}: {row}"
    return True, f"validated {len(manifests)} MMA session checkpoints"


def write_status_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def launch_environment(offline_root: Path) -> dict[str, str]:
    """Recreate the import environment used by the formal Offline runner."""
    environment = os.environ.copy()
    source_root = str((offline_root / "src").resolve())
    existing = environment.get("PYTHONPATH", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root, *(entry for entry in entries if entry != source_root)]
    )
    return environment


def launch_job(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    offline_root = Path(__file__).resolve().parents[1]
    try:
        process = subprocess.Popen(
            command,
            cwd=offline_root,
            env=launch_environment(offline_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    return int(process.pid)


def trace_files(result_dir: Path) -> list[Path]:
    return sorted((result_dir / "call_traces").glob("*.jsonl"))


def read_new_jsonl(path: Path, offsets: dict[Path, int]) -> list[dict[str, Any]]:
    offset = offsets.get(path, 0)
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size < offset:
        offset = 0
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.endswith("\n"):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        offsets[path] = handle.tell()
    return rows


def read_new_text(path: Path, offsets: dict[Path, int]) -> str:
    offset = offsets.get(path, 0)
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        value = handle.read()
        offsets[path] = handle.tell()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--heartbeat", type=float, default=60.0)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume a stopped incomplete job after checkpoint validation.",
    )
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--restart-grace", type=float, default=20.0)
    parser.add_argument(
        "--start-at-end",
        action="store_true",
        help="Ignore trace/log history that predates this monitor process.",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    status_path = run_root / "status.json"
    monitor_log = run_root / "mma_live_monitor.log"
    trace_offsets: dict[Path, int] = {}
    baseline_log_offsets: dict[Path, int] = {}
    consecutive_failures: dict[tuple[str, str, str], int] = {}
    stopped_jobs: set[str] = set()
    last_heartbeat = 0.0
    initialized_at_end = not args.start_at_end
    saved_commands: dict[str, list[str]] = {}
    dead_since: dict[str, float] = {}
    environment_checked = False
    append_log(
        monitor_log,
        "MONITOR_START policy=memory_and_qa_bad_point_limit_10;"
        "truncation_warn_only;network_retry_limit_3;"
        f"stop_job_only start_at_end={args.start_at_end}",
    )

    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(args.interval)
            continue
        run_id = str(status.get("run_id") or run_root.name)
        jobs = dict(status.get("jobs") or {})
        if not environment_checked:
            for key, source_pid in inherit_required_resume_environment(jobs):
                append_log(
                    monitor_log,
                    f"INHERITED_REQUIRED_ENV key={key} source_pid={source_pid}",
                )
            environment_checked = True
        status_changed = False
        if not initialized_at_end:
            for job in jobs.values():
                if str(job.get("method")) != "MMA":
                    continue
                result_dir = Path(str(job.get("result_dir") or ""))
                for trace_path in trace_files(result_dir):
                    try:
                        trace_offsets[trace_path] = trace_path.stat().st_size
                    except OSError:
                        pass
                baseline_log = Path(str(job.get("log") or ""))
                if baseline_log.is_file():
                    baseline_log_offsets[baseline_log] = baseline_log.stat().st_size
            initialized_at_end = True
        trace_counts: dict[str, int] = {}
        checkpoint_counts: dict[str, int] = {}

        for job_name, job in jobs.items():
            if str(job.get("method")) != "MMA":
                continue
            configured_pid = int(job.get("child_pid") or 0)
            configured_command = process_command(configured_pid) if configured_pid else []
            if command_matches_job(configured_command, job):
                saved_commands[job_name] = configured_command
                if job.get("resume_command") != configured_command:
                    job["resume_command"] = configured_command
                    status_changed = True
            elif command_matches_job(list(job.get("resume_command") or []), job):
                saved_commands[job_name] = list(job["resume_command"])
            result_dir = Path(str(job.get("result_dir") or ""))
            if not result_dir.is_dir():
                continue
            trace_counts[job_name] = 0
            for trace_path in trace_files(result_dir):
                rows = read_new_jsonl(trace_path, trace_offsets)
                trace_counts[job_name] += sum(
                    1 for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                for row in rows:
                    sample = str(row.get("sample_id") or trace_path.stem)
                    phase = str(row.get("phase") or "unknown")
                    key = (job_name, sample, phase)
                    truncated = bool(row.get("truncated")) or (
                        int(row.get("completion_tokens") or 0) > 0
                        and int(row.get("completion_tokens") or 0)
                        >= int(row.get("max_output_tokens") or 10**18)
                    )
                    http_status = int(row.get("status") or 0)
                    error_text = str(row.get("error") or "").lower()
                    if http_status in AUTH_STATUSES or http_status == 402 or any(
                        marker in error_text for marker in BALANCE_MARKERS
                    ):
                        pid = int(job.get("child_pid") or 0)
                        reason = f"hard_stop_http_{http_status or 'balance'}"
                        if pid > 0:
                            stop_job(pid, run_id, monitor_log, reason)
                        job["status"] = reason
                        job["hard_stop_reason"] = error_text or reason
                        stopped_jobs.add(job_name)
                        status_changed = True
                        continue
                    failed = http_status in TRANSIENT_STATUSES or (
                        bool(row.get("failed"))
                        and any(value in error_text for value in ("timeout", "connection"))
                    )
                    if truncated or failed:
                        count = consecutive_failures.get(key, 0) + 1
                        consecutive_failures[key] = count
                        kind = "truncation" if truncated else f"http_{row.get('status')}"
                        append_log(
                            monitor_log,
                            f"WARNING job={job_name} sample={sample} phase={phase} "
                            f"kind={kind} consecutive={count}; allowing native retry",
                        )
                        if (
                            failed
                            and not truncated
                            and count >= 3
                            and job_name not in stopped_jobs
                        ):
                            pid = int(job.get("child_pid") or 0)
                            if pid > 0:
                                stop_job(
                                    pid,
                                    run_id,
                                    monitor_log,
                                    f"{kind} repeated {count} times without recovery",
                                )
                                job["status"] = "restart_pending_transient_error"
                                job["last_restart_reason"] = (
                                    f"{kind} repeated {count} times without recovery"
                                )
                                consecutive_failures.pop(key, None)
                                status_changed = True
                    elif bool(row.get("success")):
                        previous = consecutive_failures.pop(key, 0)
                        if previous:
                            append_log(
                                monitor_log,
                                f"RECOVERED job={job_name} sample={sample} phase={phase} "
                                f"after={previous}",
                            )

            state_root = result_dir / "memory" / "datasets"
            checkpoint_counts[job_name] = sum(
                1 for _ in state_root.rglob(".offline_mma_resume.json")
            ) if state_root.is_dir() else 0

            baseline_log = Path(str(job.get("log") or ""))
            if baseline_log.is_file() and job_name not in stopped_jobs:
                new_text = read_new_text(baseline_log, baseline_log_offsets)
                for line in new_text.splitlines():
                    if "[mma-bad-point]" in line:
                        append_log(
                            monitor_log,
                            f"BAD_POINT job={job_name} {line.split('[mma-bad-point]', 1)[1].strip()}",
                        )
                    elif "[mma-qa-bad-point]" in line:
                        append_log(
                            monitor_log,
                            f"QA_BAD_POINT job={job_name} {line.split('[mma-qa-bad-point]', 1)[1].strip()}",
                        )
                    elif "[sample-error]" in line:
                        append_log(
                            monitor_log,
                            f"SAMPLE_ERROR job={job_name} {line.split('[sample-error]', 1)[1].strip()}",
                        )
                unsafe = next(
                    (marker for marker in UNSAFE_LOG_MARKERS if marker in new_text),
                    "",
                )
                lower_new_text = new_text.lower()
                authentication_failed = (
                    "UNAUTHENTICATED" in new_text
                    or "Authentication failed" in new_text
                    or "http 401" in lower_new_text
                    or "http 403" in lower_new_text
                )
                balance_failed = any(
                    marker in lower_new_text for marker in BALANCE_MARKERS
                )
                if unsafe or authentication_failed or balance_failed:
                    pid = int(job.get("child_pid") or 0)
                    reason = (
                        f"unsafe_database_error={unsafe}"
                        if unsafe
                        else (
                            "authentication_failed"
                            if authentication_failed
                            else "balance_or_quota_exhausted"
                        )
                    )
                    if pid > 0:
                        stop_job(pid, run_id, monitor_log, reason)
                        stopped_jobs.add(job_name)
                    job["status"] = f"hard_stopped_{reason}"
                    job["hard_stop_reason"] = reason
                    status_changed = True

            if job_name in stopped_jobs or str(job.get("status") or "").startswith(
                "hard_stop"
            ):
                continue
            live_pids = matching_job_pids(job)
            if len(live_pids) > 1:
                reason = f"duplicate job processes detected: {live_pids}"
                for pid in live_pids:
                    stop_job(pid, run_id, monitor_log, reason)
                job["status"] = "hard_stopped_duplicate_processes"
                job["hard_stop_reason"] = reason
                stopped_jobs.add(job_name)
                status_changed = True
                continue
            if live_pids:
                live_pid = live_pids[0]
                dead_since.pop(job_name, None)
                if int(job.get("child_pid") or 0) != live_pid:
                    job["child_pid"] = live_pid
                    status_changed = True
                continue

            if final_artifacts_present(result_dir):
                dead_since.pop(job_name, None)
                if job.get("status") != "finished_unvalidated":
                    job["status"] = "finished_unvalidated"
                    job["child_pid"] = None
                    job["finished_at"] = now()
                    append_log(monitor_log, f"JOB_FINISHED job={job_name}; awaiting validation")
                    status_changed = True
                continue

            if not args.auto_resume:
                continue
            first_seen_dead = dead_since.setdefault(job_name, time.monotonic())
            if time.monotonic() - first_seen_dead < args.restart_grace:
                continue
            restart_count = int(job.get("automatic_restart_count") or 0)
            if restart_count >= args.max_restarts:
                reason = f"automatic restart limit reached: {restart_count}"
                job["status"] = "hard_stopped_restart_limit"
                job["hard_stop_reason"] = reason
                stopped_jobs.add(job_name)
                append_log(monitor_log, f"HARD_STOP job={job_name} reason={reason}")
                status_changed = True
                continue
            command = saved_commands.get(job_name) or []
            if not command_matches_job(command, job):
                reason = "original resumable command is unavailable"
                job["status"] = "hard_stopped_missing_resume_command"
                job["hard_stop_reason"] = reason
                stopped_jobs.add(job_name)
                append_log(monitor_log, f"HARD_STOP job={job_name} reason={reason}")
                status_changed = True
                continue
            checkpoints_ok, checkpoint_message = validate_resume_checkpoints(result_dir)
            if not checkpoints_ok:
                job["status"] = "hard_stopped_checkpoint_validation"
                job["hard_stop_reason"] = checkpoint_message
                stopped_jobs.add(job_name)
                append_log(
                    monitor_log,
                    f"HARD_STOP job={job_name} reason={checkpoint_message}",
                )
                status_changed = True
                continue
            new_pid = launch_job(command, Path(str(job.get("log") or "")))
            job["child_pid"] = new_pid
            job["status"] = "running"
            job["automatic_restart_count"] = restart_count + 1
            job["last_restarted_at"] = now()
            dead_since.pop(job_name, None)
            append_log(
                monitor_log,
                f"AUTO_RESUME job={job_name} pid={new_pid} attempt={restart_count + 1} "
                f"checkpoint={checkpoint_message}",
            )
            status_changed = True

        status["jobs"] = jobs
        if jobs and all(
            row.get("status") == "finished_unvalidated" for row in jobs.values()
        ) and status.get("phase") != "awaiting_validation":
            status["phase"] = "awaiting_validation"
            status_changed = True
        if status_changed:
            status["updated_at"] = now()
            write_status_atomic(status_path, status)

        clock = time.monotonic()
        if clock - last_heartbeat >= args.heartbeat:
            states = {name: row.get("status") for name, row in jobs.items()}
            append_log(
                monitor_log,
                f"HEARTBEAT phase={status.get('phase')} jobs={states} "
                f"traces={trace_counts} session_checkpoints={checkpoint_counts}",
            )
            last_heartbeat = clock
        if status.get("phase") in TERMINAL_PHASES:
            append_log(monitor_log, f"MONITOR_END phase={status.get('phase')}")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
