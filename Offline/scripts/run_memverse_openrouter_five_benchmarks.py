#!/usr/bin/env python3
"""Run one benchmark in the isolated MemVerse OpenRouter five-benchmark study.

The entrypoint deliberately runs one benchmark per process so callers can add
API concurrency gradually.  It never changes the generic matrix defaults.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT_ROOT = ROOT / "outputs"
KEY_FILE = WORKSPACE / "Nvida_api" / "Openrouter_api"
MODEL = "qwen/qwen3.5-9b"
BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BASE_URL = "http://127.0.0.1:8002/v1"
EMBEDDING_NATIVE_DIM = 1024
EMBEDDING_DIM = 2048
TOP_K = 5
JUDGE_MODEL = "openai/gpt-4o-mini"
STATUS_FILE_NAME = "five_benchmark_status.json"
BENCHMARKS = (
    "Mem-Gallery",
    "H2HMEM",
    "WorldMemArena",
    "MemEye",
    "MEMLENS",
)
EXPECTED_QA = {
    "Mem-Gallery": 275,
    "H2HMEM": 360,
    "WorldMemArena": 440,
    "MemEye": 371,
    "MEMLENS": 173,
}
SMOKE_QA = {
    "Mem-Gallery": 1,
    # The fixed matrix smoke covers one dyadic and one multiparty dialogue.
    "H2HMEM": 2,
    "WorldMemArena": 1,
    "MemEye": 1,
    "MEMLENS": 1,
}
JUDGE_NAMES = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
    "MemEye": "memeye",
    "MEMLENS": "memlens",
}
NEW_HARNESSES = {
    "MemEye": "benchmarks.memeye_harness.eval_memeye",
    "MEMLENS": "benchmarks.memlens_harness.eval_memlens",
}
SMOKE_SELECTIONS = {
    "MemEye": ["--sample-id", "Card_Playlog_Test", "--max-qa", "1"],
    "MEMLENS": ["--sample-id", "q_600a79aa", "--max-qa", "1"],
}


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run ID must be a safe path component")
    return value


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime(
        "memverse_openrouter_qwen35_9b_emb06_top5_test_%Y%m%d_%H%M%S"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _update_status(path: Path, key: str, values: dict[str, Any]) -> None:
    """Merge one job state without losing updates from parallel processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status = json.loads(path.read_text()) if path.is_file() else {}
        status[key] = {**status.get(key, {}), **values}
        _write_json(path, status)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _load_key() -> str:
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"sk-or-v1-[A-Za-z0-9_-]+", key):
        raise RuntimeError(f"No valid OpenRouter key in {KEY_FILE}")
    return key


def _derived_files(run_id: str) -> tuple[Path, Path]:
    run_root = OUTPUT_ROOT / "_runs" / run_id
    defaults = json.loads((ROOT / "configs" / "defaults.json").read_text())
    defaults.update(
        {
            "answer_model": MODEL,
            "answer_base_url": BASE_URL,
            "executor_model": MODEL,
            "executor_base_url": BASE_URL,
            # Qwen3.5 reasons by default on OpenRouter.  The repository's
            # fixed think=false setting maps to the provider's explicit none.
            "reasoning_effort": "none",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "embedding_native_dim": EMBEDDING_NATIVE_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
        }
    )
    defaults_path = run_root / "openrouter_qwen35_9b_defaults.json"
    _write_json(defaults_path, defaults)

    efficiency = json.loads(
        (ROOT / "configs" / "model_efficiency.json").read_text()
    )
    inherited_latency = dict(
        efficiency["models"]["qwen/qwen3-vl-8b-instruct"]["latency"]
    )
    inherited_latency["source"] = (
        "qwen3_vl_8b_latency_proxy_for_qwen3_5_9b_openrouter_2026-09-15"
    )
    efficiency["models"][MODEL] = {
        "latency": inherited_latency,
        "pricing": {
            "input_per_million_usd": 0.1,
            "output_per_million_usd": 0.15,
            "source": "openrouter_models_api_2026-09-15",
        },
    }
    efficiency_path = run_root / "openrouter_qwen35_9b_efficiency.json"
    _write_json(efficiency_path, efficiency)
    _write_json(
        run_root / "experiment_config.json",
        {
            "run_id": run_id,
            "baseline": "MemVerse",
            "benchmarks": list(BENCHMARKS),
            "answer_model": MODEL,
            "executor_model": MODEL,
            "base_url": BASE_URL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "embedding_native_dim": EMBEDDING_NATIVE_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "expected_qa": EXPECTED_QA,
            "defaults": str(defaults_path),
            "efficiency_config": str(efficiency_path),
        },
    )
    return defaults_path, efficiency_path


def _result_dir(run_id: str, benchmark: str, smoke: bool) -> Path:
    if smoke:
        return OUTPUT_ROOT / "_runs" / run_id / "_smoke" / benchmark / "MemVerse"
    return OUTPUT_ROOT / benchmark / "MemVerse" / run_id


def _run(command: list[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command exited {completed.returncode}; inspect {log_path}"
        )


def _run_existing(
    args: argparse.Namespace,
    defaults_path: Path,
    efficiency_path: Path,
    env: dict[str, str],
) -> Path:
    smoke_root = OUTPUT_ROOT / "_runs" / args.run_id / "_matrix_smoke" / args.benchmark
    output_root = smoke_root if args.smoke else OUTPUT_ROOT
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_memverse_embedding06_top5.py"),
        "--run-id", args.run_id,
        "--defaults", str(defaults_path),
        "--efficiency-config", str(efficiency_path),
        "--output-root", str(output_root),
        "--endpoint", BASE_URL,
        "--benchmark", args.benchmark,
    ]
    if args.smoke:
        command.append("--smoke-only")
    else:
        command.append("--skip-smoke")
    if args.resume:
        command.append("--reuse-state")
    _run(
        command,
        log_path=OUTPUT_ROOT / "_runs" / args.run_id / "_logs" / args.phase / f"{args.benchmark}.log",
        env=env,
    )
    if args.smoke:
        return (
            smoke_root
            / "_runs"
            / args.run_id
            / "_smoke"
            / args.benchmark
            / "MemVerse"
        )
    return _result_dir(args.run_id, args.benchmark, False)


def _run_new(
    args: argparse.Namespace,
    efficiency_path: Path,
    env: dict[str, str],
) -> Path:
    result_dir = _result_dir(args.run_id, args.benchmark, args.smoke)
    command = [
        sys.executable,
        "-m", NEW_HARNESSES[args.benchmark],
        "--result-dir", str(result_dir),
        "--baseline-state-dir", str(result_dir / "memory" / "datasets"),
        "--sample-concurrency", str(args.sample_concurrency),
        "--answer-concurrency", "1" if args.smoke else "16",
        "--checkpoint-every", "10",
        "--answer-model", MODEL,
        "--answer-base-url", BASE_URL,
        "--answer-temperature", "0.0",
        "--num-predict", "512",
        "--request-timeout", "180",
        "--retries", "2",
        "--no-think",
        "--reasoning-effort", "none",
        "--executor-model", MODEL,
        "--executor-base-url", BASE_URL,
        "--executor-temperature", "0.0",
        "--executor-max-tokens", "512",
        "--embedding-model", EMBEDDING_MODEL,
        "--embedding-base-url", EMBEDDING_BASE_URL,
        "--embedding-dim", str(EMBEDDING_DIM),
        "--top-k", str(TOP_K),
        "--efficiency-config", str(efficiency_path),
    ]
    if args.smoke:
        command.extend(SMOKE_SELECTIONS[args.benchmark])
    if args.resume:
        command.append("--resume")
    _run(
        command,
        log_path=OUTPUT_ROOT / "_runs" / args.run_id / "_logs" / args.phase / f"{args.benchmark}.log",
        env=env,
    )
    return result_dir


def _validate_generation(result_dir: Path, *, expected: int) -> None:
    required = (
        "results.json",
        "retrieval_trace.jsonl",
        "memory/memory_snapshot.jsonl",
        "run_manifest.json",
        "metrics.json",
        "call_trace.jsonl",
        "call_metrics.json",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"missing generation artifacts: {missing}")
    rows = json.loads((result_dir / "results.json").read_text())
    ids = [str(row.get("manifest_question_id") or row.get("query_id") or "") for row in rows]
    if len(rows) != expected or len(ids) != len(set(ids)) or not all(ids):
        raise RuntimeError(
            f"result ID/count mismatch: rows={len(rows)} unique={len(set(ids))} expected={expected}"
        )
    errors = [row for row in rows if row.get("error") or not str(row.get("system_answer") or "").strip()]
    if errors:
        raise RuntimeError(f"{len(errors)} empty/error answer(s)")
    traces = [
        json.loads(line)
        for line in (result_dir / "retrieval_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(traces) != expected or any(len(row.get("top_k") or []) > TOP_K for row in traces):
        raise RuntimeError("retrieval trace count or Top-K constraint failed")


def _judge(
    args: argparse.Namespace,
    result_dir: Path,
    env: dict[str, str],
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "judge_results_llm_parallel.py"),
        "--benchmark", JUDGE_NAMES[args.benchmark],
        "--results", str(result_dir / "results.json"),
        "--out-dir", str(result_dir),
        "--key-file", str(KEY_FILE),
        "--model", JUDGE_MODEL,
        "--workers", "1" if args.smoke else "32",
        "--timeout", "60",
        "--retries", "2",
        "--max-tokens", "512",
        "--checkpoint-every", "25",
        "--resume",
    ]
    _run(
        command,
        log_path=OUTPUT_ROOT / "_runs" / args.run_id / "_logs" / args.phase / f"{args.benchmark}.judge.log",
        env=env,
    )
    metrics = json.loads((result_dir / "llm_judge_metrics.json").read_text())
    expected = SMOKE_QA[args.benchmark] if args.smoke else EXPECTED_QA[args.benchmark]
    if (
        int(metrics.get("count", -1)) != expected
        or int(metrics.get("valid_count", -1)) != expected
        or int(metrics.get("judge_errors", -1)) != 0
        or bool(metrics.get("provisional", True))
    ):
        raise RuntimeError(f"Judge did not complete cleanly: {metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_safe_run_id, default=_default_run_id())
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="Per-benchmark sample concurrency for MemEye and MEMLENS.",
    )
    args = parser.parse_args()
    if args.sample_concurrency < 1:
        parser.error("--sample-concurrency must be positive")
    args.phase = "smoke" if args.smoke else "formal"
    return args


def main() -> None:
    args = parse_args()
    defaults_path, efficiency_path = _derived_files(args.run_id)
    env = os.environ.copy()
    key = _load_key()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OPENAI_API_KEY": key,
            "OPENROUTER_API_KEY": key,
        }
    )
    if args.resume:
        env["MEMVERSE_REUSE_STATE"] = "1"
    else:
        env.pop("MEMVERSE_REUSE_STATE", None)

    status_path = OUTPUT_ROOT / "_runs" / args.run_id / STATUS_FILE_NAME
    key_name = f"{args.phase}:{args.benchmark}"
    _update_status(
        status_path,
        key_name,
        {
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": None,
            "error": None,
        },
    )
    try:
        if args.benchmark in NEW_HARNESSES:
            result_dir = _run_new(args, efficiency_path, env)
        else:
            result_dir = _run_existing(args, defaults_path, efficiency_path, env)
        expected = SMOKE_QA[args.benchmark] if args.smoke else EXPECTED_QA[args.benchmark]
        _validate_generation(result_dir, expected=expected)
        _judge(args, result_dir, env)
    except Exception as exc:
        _update_status(
            status_path,
            key_name,
            {
            "status": "failed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    completed = {
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(),
            "result_dir": str(result_dir),
            "expected_qa": expected,
            "answer_errors": 0,
            "judge_errors": 0,
            "error": None,
        }
    _update_status(status_path, key_name, completed)
    print(json.dumps(completed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
