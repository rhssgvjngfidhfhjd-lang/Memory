#!/usr/bin/env python3
"""Run the local MIRIX Top-5/Embedding-0.6B five-benchmark experiment.

The five smoke jobs run sequentially on one vLLM endpoint. Formal generation
starts only after every selected smoke artifact passes validation, and formal
benchmarks also run sequentially so MIRIX instances never contend for GPU4.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
DEFAULTS_PATH = ROOT / "configs" / "defaults.json"
EFFICIENCY_PATH = ROOT / "configs" / "model_efficiency.json"
SPLIT_MANIFEST_PATH = ROOT / "configs" / "multimodal_split_manifest.json"
KEY_FILE = ROOT.parent / "Nvida_api" / "Openrouter_api"
MIRIX_PYTHON = ROOT / ".venvs" / "mirix" / "bin" / "python"

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_URL = "http://127.0.0.1:8014/v1"
MIN_MODEL_LEN = 131072
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_BASE_URL = "http://127.0.0.1:8002/v1"
EMBEDDING_NATIVE_DIM = 1024
EMBEDDING_DIM = 2048
TOP_K = 5
MIRIX_EXECUTOR_MAX_TOKENS = 8192
JUDGE_MODEL = "openai/gpt-4o-mini"

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
    "H2HMEM": 2,
    "WorldMemArena": 1,
    "MemEye": 1,
    "MEMLENS": 1,
}
NEW_HARNESSES = {
    "MemEye": "benchmarks.memeye_harness.eval_memeye",
    "MEMLENS": "benchmarks.memlens_harness.eval_memlens",
}
SMOKE_SELECTIONS = {
    "MemEye": ["--sample-id", "Card_Playlog_Test", "--max-qa", "1"],
    "MEMLENS": ["--sample-id", "q_600a79aa", "--max-qa", "1"],
}
JUDGE_NAMES = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
    "MemEye": "memeye",
    "MEMLENS": "memlens",
}


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run ID must be a safe path component")
    return value


def _default_run_id() -> str:
    return datetime.now().astimezone().strftime(
        "mirix_local_qwen3vl4b_emb06_top5_%Y%m%d_%H%M%S"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_runtime_values(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize flat and matrix-harness run-manifest schemas."""
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        configuration = {}

    baseline = manifest.get("baseline")
    if isinstance(baseline, dict):
        baseline = baseline.get("name")

    def value(key: str) -> Any:
        return manifest[key] if key in manifest else configuration.get(key)

    return {
        "baseline": baseline,
        "top_k": value("top_k"),
        "embedding_model": value("embedding_model"),
        "embedding_dim": value("embedding_dim"),
        "answer_model": value("answer_model"),
        "executor_model": value("executor_model"),
        "executor_max_tokens": value("executor_max_tokens"),
    }


def _recovered_truncation_count(calls: list[dict[str, Any]]) -> int:
    """Accept a truncated build request recovered by a later retry.

    MIRIX rejects a length-capped native tool envelope before executing it and
    retries the same memory agent.  The transport trace still retains that
    rejected HTTP 200 response.  Memory agents issue requests concurrently, so
    the recovery is not necessarily the next aggregate trace row and may also
    follow bounded timeout/truncation retries.  Require a later successful,
    non-truncated call for the same sample and phase; everything else remains
    a smoke-gate failure.
    """
    recovered = 0
    for index, call in enumerate(calls):
        if not bool(call.get("truncated")):
            continue
        recovery = next(
            (
                candidate
                for candidate in calls[index + 1 :]
                if candidate.get("sample_id") == call.get("sample_id")
                and candidate.get("phase") == call.get("phase")
                and bool(candidate.get("success"))
                and not bool(candidate.get("failed"))
                and not bool(candidate.get("truncated"))
            ),
            None,
        )
        if call.get("phase") == "memory_build" and recovery is not None:
            recovered += 1
            continue
        raise RuntimeError(
            "unrecovered model-response truncation at request "
            f"{call.get('request_id')!r}"
        )
    return recovered


def _update_status(path: Path, key: str, **values: Any) -> None:
    status = _load_json(path) if path.is_file() else {}
    status[key] = {**status.get(key, {}), **values}
    _write_json(path, status)


def build_experiment_defaults(source: Path) -> dict[str, Any]:
    config = _load_json(source)
    config.update(
        {
            "answer_model": MODEL,
            "answer_base_url": BASE_URL,
            "executor_model": MODEL,
            "executor_base_url": BASE_URL,
            "mirix_executor_max_tokens": MIRIX_EXECUTOR_MAX_TOKENS,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "embedding_native_dim": EMBEDDING_NATIVE_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            "query_embedding_dir": "data/qwen3_embedding_0_6b/query_embeddings",
            "top_k": TOP_K,
        }
    )
    return config


def _prepare_run_files(run_id: str) -> tuple[Path, Path]:
    run_root = OUTPUT_ROOT / "_runs" / run_id
    defaults_path = run_root / "mirix_local_embedding06_top5_defaults.json"
    config = build_experiment_defaults(DEFAULTS_PATH)
    _write_json(defaults_path, config)
    _write_json(
        run_root / "experiment_config.json",
        {
            "run_id": run_id,
            "baseline": "MIRIX",
            "benchmarks": list(BENCHMARKS),
            "answer_model": MODEL,
            "executor_model": MODEL,
            "base_url": BASE_URL,
            "minimum_vllm_model_len": MIN_MODEL_LEN,
            "mirix_executor_max_tokens": MIRIX_EXECUTOR_MAX_TOKENS,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_base_url": EMBEDDING_BASE_URL,
            "embedding_native_dim": EMBEDDING_NATIVE_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_dimension_adapter": "right_zero_pad_1024_to_2048",
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "expected_qa": EXPECTED_QA,
            "smoke_qa": SMOKE_QA,
            "defaults": str(defaults_path),
            "efficiency_config": str(EFFICIENCY_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH),
        },
    )
    return defaults_path, run_root / "five_benchmark_status.json"


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def check_local_services(base_url: str = BASE_URL) -> None:
    model_payload = _request_json(base_url.rstrip("/") + "/models")
    rows = [row for row in model_payload.get("data") or [] if row.get("id") == MODEL]
    if not rows:
        raise RuntimeError(f"{MODEL} is not served at {base_url}")
    actual_model_len = int(rows[0].get("max_model_len") or 0)
    if actual_model_len < MIN_MODEL_LEN:
        raise RuntimeError(
            f"vLLM max_model_len is {actual_model_len}, expected at least {MIN_MODEL_LEN}"
        )
    embedding_payload = _request_json(
        EMBEDDING_BASE_URL.rstrip("/") + "/embeddings",
        {"model": EMBEDDING_MODEL, "input": ["MIRIX five-benchmark preflight"]},
    )
    embedding_rows = embedding_payload.get("data") or []
    actual_dim = len(embedding_rows[0].get("embedding") or []) if embedding_rows else 0
    if actual_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"embedding dimension is {actual_dim}, expected {EMBEDDING_DIM}"
        )


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
            "MIRIX_PYTHON": str(MIRIX_PYTHON),
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


def _result_dir(run_id: str, benchmark: str, phase: str) -> Path:
    if phase == "smoke":
        return OUTPUT_ROOT / "_runs" / run_id / "_smoke" / benchmark / "MIRIX"
    return OUTPUT_ROOT / benchmark / "MIRIX" / run_id


def _run_existing(
    *,
    run_id: str,
    benchmark: str,
    phase: str,
    defaults_path: Path,
    env: dict[str, str],
) -> Path:
    if phase == "smoke":
        matrix_output = OUTPUT_ROOT / "_runs" / run_id / "_matrix_smoke" / benchmark
    else:
        matrix_output = OUTPUT_ROOT
    command = [
        str(MIRIX_PYTHON),
        str(ROOT / "scripts" / "run_test_baseline_matrix.py"),
        "--defaults", str(defaults_path),
        "--efficiency-config", str(EFFICIENCY_PATH),
        "--split-manifest", str(SPLIT_MANIFEST_PATH),
        "--output-root", str(matrix_output),
        "--run-id", run_id,
        "--status-file-name", f"matrix_{phase}_{_slug(benchmark)}.json",
        "--endpoint", BASE_URL,
        "--embedding-base-url", EMBEDDING_BASE_URL,
        "--top-k", str(TOP_K),
        "--baseline", "MIRIX",
        "--benchmark", benchmark,
        "--smoke-only" if phase == "smoke" else "--skip-smoke",
    ]
    _run(
        command,
        log_path=(
            OUTPUT_ROOT / "_runs" / run_id / "_logs" / phase / f"{benchmark}.log"
        ),
        env=env,
    )
    if phase == "smoke":
        return matrix_output / "_runs" / run_id / "_smoke" / benchmark / "MIRIX"
    return _result_dir(run_id, benchmark, phase)


def _run_new(
    *,
    run_id: str,
    benchmark: str,
    phase: str,
    env: dict[str, str],
    base_url: str = BASE_URL,
) -> Path:
    result_dir = _result_dir(run_id, benchmark, phase)
    smoke = phase == "smoke"
    command = [
        str(MIRIX_PYTHON),
        "-m", NEW_HARNESSES[benchmark],
        "--baseline", "MIRIX",
        "--result-dir", str(result_dir),
        "--baseline-state-dir", str(result_dir / "memory" / "datasets"),
        "--sample-concurrency", "1" if smoke else "4",
        "--answer-concurrency", "1" if smoke else "16",
        "--checkpoint-every", "10",
        "--answer-model", MODEL,
        "--answer-base-url", base_url,
        "--answer-temperature", "0.0",
        "--num-predict", "512",
        "--no-think",
        "--executor-model", MODEL,
        "--executor-base-url", base_url,
        "--executor-temperature", "0.0",
        "--executor-max-tokens", str(MIRIX_EXECUTOR_MAX_TOKENS),
        "--embedding-model", EMBEDDING_MODEL,
        "--embedding-base-url", EMBEDDING_BASE_URL,
        "--embedding-dim", str(EMBEDDING_DIM),
        "--top-k", str(TOP_K),
        "--request-timeout", "180",
        "--retries", "2",
        "--efficiency-config", str(EFFICIENCY_PATH),
        "--mirix-skip-failed-build-points",
        "--mirix-max-consecutive-failed-build-points", "10",
        "--resume",
    ]
    if smoke:
        command.extend(SMOKE_SELECTIONS[benchmark])
    _run(
        command,
        log_path=(
            OUTPUT_ROOT / "_runs" / run_id / "_logs" / phase / f"{benchmark}.log"
        ),
        env=env,
    )
    return result_dir


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_generation(result_dir: Path, *, expected: int) -> int:
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
    results = _load_json(result_dir / "results.json")
    ids = [
        str(row.get("manifest_question_id") or row.get("query_id") or "")
        for row in results
    ]
    if len(results) != expected or len(ids) != len(set(ids)) or not all(ids):
        raise RuntimeError(
            f"result ID/count mismatch: rows={len(results)} "
            f"unique={len(set(ids))} expected={expected}"
        )
    errors = [
        row
        for row in results
        if row.get("error") or not str(row.get("system_answer") or "").strip()
    ]
    if errors:
        raise RuntimeError(f"{len(errors)} empty/error answer(s)")

    retrievals = _load_jsonl(result_dir / "retrieval_trace.jsonl")
    top_lengths = [
        len(row.get("top_k") or row.get("retrieval_top_k") or [])
        for row in retrievals
    ]
    if len(retrievals) != expected or any(length > TOP_K for length in top_lengths):
        raise RuntimeError(
            f"retrieval count/Top-K failed: rows={len(retrievals)} lengths={top_lengths}"
        )

    manifest = _load_json(result_dir / "run_manifest.json")
    expected_manifest = {
        "baseline": "MIRIX",
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "answer_model": MODEL,
        "executor_model": MODEL,
        "executor_max_tokens": MIRIX_EXECUTOR_MAX_TOKENS,
    }
    actual_manifest = _manifest_runtime_values(manifest)
    mismatches = {
        key: (actual_manifest.get(key), value)
        for key, value in expected_manifest.items()
        if actual_manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"run manifest configuration mismatch: {mismatches}")

    calls = _load_jsonl(result_dir / "call_trace.jsonl")
    recovered_truncations = _recovered_truncation_count(calls)
    context_errors = []
    for path in (result_dir / "call_traces" / "response_bodies").rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "maximum model length" in text or "maximum context length" in text:
            context_errors.append(str(path))
    if context_errors:
        raise RuntimeError(f"vLLM context-limit errors: {context_errors[:3]}")
    return recovered_truncations


def _judge(
    *, run_id: str, benchmark: str, result_dir: Path, env: dict[str, str]
) -> None:
    command = [
        str(MIRIX_PYTHON),
        str(ROOT / "scripts" / "judge_results_llm_parallel.py"),
        "--benchmark", JUDGE_NAMES[benchmark],
        "--results", str(result_dir / "results.json"),
        "--out-dir", str(result_dir),
        "--key-file", str(KEY_FILE),
        "--model", JUDGE_MODEL,
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
    metrics = _load_json(result_dir / "llm_judge_metrics.json")
    expected = EXPECTED_QA[benchmark]
    if (
        int(metrics.get("count", -1)) != expected
        or int(metrics.get("valid_count", -1)) != expected
        or int(metrics.get("judge_errors", -1)) != 0
        or bool(metrics.get("provisional", True))
    ):
        raise RuntimeError(f"Judge did not complete cleanly: {metrics}")


def _run_job(
    *,
    run_id: str,
    benchmark: str,
    phase: str,
    defaults_path: Path,
    status_path: Path,
    env: dict[str, str],
) -> None:
    key = f"{phase}:{benchmark}"
    _update_status(
        status_path,
        key,
        status="running",
        started_at=datetime.now().astimezone().isoformat(),
        finished_at=None,
        error=None,
    )
    try:
        if benchmark in NEW_HARNESSES:
            result_dir = _run_new(
                run_id=run_id, benchmark=benchmark, phase=phase, env=env
            )
        else:
            result_dir = _run_existing(
                run_id=run_id,
                benchmark=benchmark,
                phase=phase,
                defaults_path=defaults_path,
                env=env,
            )
        expected = SMOKE_QA[benchmark] if phase == "smoke" else EXPECTED_QA[benchmark]
        recovered_truncations = _validate_generation(result_dir, expected=expected)
        if phase == "formal" and benchmark in NEW_HARNESSES:
            _judge(
                run_id=run_id,
                benchmark=benchmark,
                result_dir=result_dir,
                env=env,
            )
    except Exception as exc:
        _update_status(
            status_path,
            key,
            status="failed",
            finished_at=datetime.now().astimezone().isoformat(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _update_status(
        status_path,
        key,
        status="completed",
        finished_at=datetime.now().astimezone().isoformat(),
        result_dir=str(result_dir),
        expected_qa=expected,
        answer_errors=0,
        recovered_truncations=recovered_truncations,
        error=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=_safe_run_id, default=_default_run_id())
    parser.add_argument(
        "--benchmark", action="append", choices=BENCHMARKS, default=[]
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--formal-only", action="store_true")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run jobs even when the status file already marks them completed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults_path, status_path = _prepare_run_files(args.run_id)
    env = _environment()
    check_local_services()
    _run(
        [str(MIRIX_PYTHON), str(ROOT / "scripts" / "prepare_mirix_v011.py")],
        log_path=OUTPUT_ROOT / "_runs" / args.run_id / "_logs" / "preflight.log",
        env=env,
    )
    benchmarks = tuple(args.benchmark or BENCHMARKS)
    phases = ("formal",) if args.formal_only else ("smoke",)
    if not args.smoke_only and not args.formal_only:
        phases = ("smoke", "formal")
    for phase in phases:
        for benchmark in benchmarks:
            key = f"{phase}:{benchmark}"
            status = _load_json(status_path) if status_path.is_file() else {}
            if (
                not args.rerun_completed
                and status.get(key, {}).get("status") == "completed"
            ):
                continue
            _run_job(
                run_id=args.run_id,
                benchmark=benchmark,
                phase=phase,
                defaults_path=defaults_path,
                status_path=status_path,
                env=env,
            )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "benchmarks": list(benchmarks),
                "phases": list(phases),
                "status_file": str(status_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
