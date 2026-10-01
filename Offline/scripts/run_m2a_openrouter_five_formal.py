#!/usr/bin/env python3
"""Run the five formal M2A benchmarks with 16-way sample/answer concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUTPUT_ROOT = ROOT / "outputs"
DEFAULTS = WORKSPACE / "Nvida_api" / "defaults_qwen35_9b_m2a_1200_top5.json"
EFFICIENCY = WORKSPACE / "Nvida_api" / "config_qwen35_9b"
SPLIT_MANIFEST = ROOT / "configs" / "multimodal_split_manifest.json"
KEY_FILE = WORKSPACE / "Nvida_api" / "Openrouter_api"
MODEL = "qwen/qwen3.5-9b"
BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BASE_URL = "http://127.0.0.1:8002/v1"
EMBEDDING_DIM = 2048
TOP_K = 5
EXECUTOR_MAX_TOKENS = 1200
CONCURRENCY = 16
BENCHMARKS = (
    "Mem-Gallery",
    "H2HMEM",
    "WorldMemArena",
    "MemEye",
    "MEMLENS",
)
_STATUS_LOCK = threading.Lock()
EXPECTED_QA = {
    "Mem-Gallery": 275,
    "H2HMEM": 360,
    "WorldMemArena": 440,
    "MemEye": 371,
    "MEMLENS": 173,
}
NEW_HARNESSES = {
    "MemEye": "benchmarks.memeye_harness.eval_memeye",
    "MEMLENS": "benchmarks.memlens_harness.eval_memlens",
}
JUDGE_NAMES = {
    "MemEye": "memeye",
    "MEMLENS": "memlens",
}


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run ID must be a safe path component")
    return value


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime(
        "m2a_qwen35_9b_emb06_top5_1200_salvage_formal_%Y%m%d_%H%M%S"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _update_status(path: Path, benchmark: str, **values: Any) -> None:
    with _STATUS_LOCK:
        status = _load_json(path)
        status[benchmark] = {**status.get(benchmark, {}), **values}
        _write_json(path, status)


def _load_key() -> str:
    match = re.search(
        r"sk-or-v1-[A-Za-z0-9_-]+", KEY_FILE.read_text(encoding="utf-8")
    )
    if not match:
        raise RuntimeError(f"No valid OpenRouter key in {KEY_FILE}")
    return match.group(0)


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    key = _load_key()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "OPENAI_API_KEY": key,
            "OPENROUTER_API_KEY": key,
            "EMBEDDING_API_KEY": "EMPTY",
        }
    )
    return env


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


def _result_dir(run_id: str, benchmark: str) -> Path:
    return OUTPUT_ROOT / benchmark / "M2A" / run_id


def _run_matrix_benchmark(
    run_id: str, benchmark: str, env: dict[str, str], log_path: Path
) -> Path:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_test_baseline_matrix.py"),
        "--defaults", str(DEFAULTS),
        "--efficiency-config", str(EFFICIENCY),
        "--split-manifest", str(SPLIT_MANIFEST),
        "--output-root", str(OUTPUT_ROOT),
        "--run-id", run_id,
        "--status-file-name", f"matrix_formal_{benchmark.casefold().replace('-', '_')}.json",
        "--endpoint", BASE_URL,
        "--embedding-base-url", EMBEDDING_BASE_URL,
        "--top-k", str(TOP_K),
        "--baseline", "M2A",
        "--benchmark", benchmark,
        "--skip-smoke",
    ]
    _run(command, log_path=log_path, env=env)
    return _result_dir(run_id, benchmark)


def _run_new_benchmark(
    run_id: str, benchmark: str, env: dict[str, str], log_path: Path
) -> Path:
    result_dir = _result_dir(run_id, benchmark)
    command = [
        sys.executable,
        "-m", NEW_HARNESSES[benchmark],
        "--baseline", "M2A",
        "--result-dir", str(result_dir),
        "--baseline-state-dir", str(result_dir / "memory" / "datasets"),
        "--sample-concurrency", str(CONCURRENCY),
        "--answer-concurrency", str(CONCURRENCY),
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
        "--executor-max-tokens", str(EXECUTOR_MAX_TOKENS),
        "--m2a-salvage-truncated-updates",
        "--m2a-skip-failed-build-points",
        "--m2a-max-consecutive-failed-build-points", "1000000000",
        "--embedding-model", EMBEDDING_MODEL,
        "--embedding-base-url", EMBEDDING_BASE_URL,
        "--embedding-dim", str(EMBEDDING_DIM),
        "--top-k", str(TOP_K),
        "--efficiency-config", str(EFFICIENCY),
        "--resume",
    ]
    _run(command, log_path=log_path, env=env)
    return result_dir


def _judge_new(
    run_id: str, benchmark: str, result_dir: Path, env: dict[str, str]
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "judge_results_llm_parallel.py"),
        "--benchmark", JUDGE_NAMES[benchmark],
        "--results", str(result_dir / "results.json"),
        "--out-dir", str(result_dir),
        "--key-file", str(KEY_FILE),
        "--model", "openai/gpt-4o-mini",
        "--workers", "32",
        "--timeout", "60",
        "--retries", "2",
        "--max-tokens", "512",
        "--checkpoint-every", "25",
        "--resume",
    ]
    _run(
        command,
        log_path=(
            OUTPUT_ROOT / "_runs" / run_id / "_logs" / "formal" / f"{benchmark}.judge.log"
        ),
        env=env,
    )


def _validate(result_dir: Path, benchmark: str) -> None:
    results_path = result_dir / "results.json"
    manifest_path = result_dir / "run_manifest.json"
    if not results_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"missing final artifacts under {result_dir}")
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    expected = EXPECTED_QA[benchmark]
    if len(rows) != expected:
        raise RuntimeError(f"{benchmark}: results={len(rows)}, expected={expected}")
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"{benchmark}: {len(errors)} answer error(s)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest.get("configuration") or {}

    def value(key: str) -> Any:
        return manifest[key] if key in manifest else configuration.get(key)

    checks = {
        "top_k": TOP_K,
        "embedding_dim": EMBEDDING_DIM,
        "answer_model": MODEL,
        "executor_model": MODEL,
        "executor_max_tokens": EXECUTOR_MAX_TOKENS,
    }
    mismatches = {
        key: (value(key), expected_value)
        for key, expected_value in checks.items()
        if value(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"{benchmark}: manifest mismatch {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_safe_run_id, default=_default_run_id())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
        help="Benchmarks to run concurrently (each keeps its own concurrency setting).",
    )
    args = parser.parse_args()

    run_root = OUTPUT_ROOT / "_runs" / args.run_id
    status_path = run_root / "five_benchmark_formal_status.json"
    _write_json(
        run_root / "experiment_config.json",
        {
            "run_id": args.run_id,
            "baseline": "M2A",
            "phase": "formal",
            "benchmarks": list(BENCHMARKS),
            "answer_model": MODEL,
            "executor_model": MODEL,
            "executor_max_tokens": EXECUTOR_MAX_TOKENS,
            "salvage_truncated_updates": True,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_native_dim": 1024,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            "top_k": TOP_K,
            "sample_concurrency": CONCURRENCY,
            "answer_concurrency": CONCURRENCY,
            "wma_rounds_per_ingest": 4,
            "expected_qa": EXPECTED_QA,
        },
    )
    env = _environment()
    failures: list[str] = []

    def run_one(benchmark: str) -> None:
        previous = _load_json(status_path).get(benchmark, {})
        if args.resume and previous.get("status") == "completed":
            return
        _update_status(
            status_path,
            benchmark,
            status="running",
            started_at=datetime.now().astimezone().isoformat(),
            finished_at=None,
            error=None,
        )
        log_path = run_root / "_logs" / "formal" / f"{benchmark}.log"
        try:
            if benchmark in NEW_HARNESSES:
                result_dir = _run_new_benchmark(
                    args.run_id, benchmark, env, log_path
                )
                _validate(result_dir, benchmark)
                _judge_new(args.run_id, benchmark, result_dir, env)
            else:
                result_dir = _run_matrix_benchmark(
                    args.run_id, benchmark, env, log_path
                )
                _validate(result_dir, benchmark)
            _update_status(
                status_path,
                benchmark,
                status="completed",
                finished_at=datetime.now().astimezone().isoformat(),
                result_dir=str(result_dir),
                expected_qa=EXPECTED_QA[benchmark],
                error=None,
            )
        except Exception as exc:
            _update_status(
                status_path,
                benchmark,
                status="failed",
                finished_at=datetime.now().astimezone().isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    with ThreadPoolExecutor(max_workers=len(args.benchmarks)) as executor:
        futures = {executor.submit(run_one, b): b for b in args.benchmarks}
        for future in as_completed(futures):
            benchmark = futures[future]
            try:
                future.result()
            except Exception:
                failures.append(benchmark)
    if failures:
        raise RuntimeError(f"formal benchmark failures: {failures}")


if __name__ == "__main__":
    main()
