#!/usr/bin/env python3
"""Append a compact five-benchmark M2A health snapshot every five minutes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time


BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena", "MemEye", "MEMLENS")


def snapshot(output_root: Path, run_id: str) -> dict:
    run_root = output_root / "_runs" / run_id
    status_path = run_root / "five_benchmark_formal_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    process_count = int(
        subprocess.run(
            ["pgrep", "-fc", run_id], capture_output=True, text=True, check=False
        ).stdout.strip()
        or 0
    )
    rows = {}
    now = time.time()
    alerts = []
    for benchmark in BENCHMARKS:
        result_dir = output_root / benchmark / "M2A" / run_id
        calls = skipped = truncated = answer_calls = 0
        latest = 0.0
        traces = list((result_dir / "call_traces").glob("*.jsonl"))
        for trace in traces:
            latest = max(latest, trace.stat().st_mtime)
            for line in trace.open(errors="ignore"):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                calls += 1
                phase = str(item.get("phase") or item.get("stage") or "").lower()
                answer_calls += int("answer" in phase or phase == "qa")
                skipped += int(
                    phase == "build_fault"
                    or item.get("event") == "skipped_build_point"
                )
                truncated += int(
                    bool(item.get("truncated"))
                    or item.get("finish_reason") in {"length", "max_tokens"}
                )
        result_rows = None
        results_path = result_dir / "results.json"
        if results_path.is_file():
            try:
                results = json.loads(results_path.read_text())
                result_rows = len(
                    results if isinstance(results, list) else results.get("results", [])
                )
            except (json.JSONDecodeError, OSError):
                result_rows = "invalid"
        age = round(now - latest) if latest else None
        state = (status.get(benchmark) or {}).get("status", "missing")
        if state == "running" and age is not None and age > 900:
            alerts.append(f"{benchmark}: no trace write for {age}s")
        rows[benchmark] = {
            "status": state,
            "samples": len(traces),
            "calls": calls,
            "answer_calls": answer_calls,
            "skipped": skipped,
            "truncated": truncated,
            "results": result_rows,
            "last_write_age_s": age,
        }
    if process_count == 0 and any(row["status"] == "running" for row in rows.values()):
        alerts.append("status says running but no run process exists")
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "run_id": run_id,
        "process_count": process_count,
        "benchmarks": rows,
        "alerts": alerts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    while True:
        print(json.dumps(snapshot(args.output_root, args.run_id), ensure_ascii=False), flush=True)
        # Short sleeps keep the monitor responsive to termination signals.
        remaining = max(1, args.interval)
        while remaining:
            step = min(60, remaining)
            time.sleep(step)
            remaining -= step


if __name__ == "__main__":
    main()
