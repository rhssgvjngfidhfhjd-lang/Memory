#!/usr/bin/env python3
"""Rebuild baseline memories while measuring exact executor HTTP attempts.

This runner intentionally performs only the memory-build phase.  Each sample
gets its own local reverse proxy, so concurrent workers cannot mix call counts.
Existing benchmark results and memory banks are never modified.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import fcntl
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from benchmarks.baseline_runtime import canonical_name, create_adapter
from benchmarks.io_utils import write_json_atomic
from embedding.chunk_builder import (
    Chunk,
    build_chunks_from_data,
    build_h2h_chunks_from_directory,
    build_wma_chunks_from_data,
    iter_h2h_session_files,
    iter_wma_sample_files,
)


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = OFFLINE_ROOT.parent
DEFAULT_MEMGALLERY = WORKSPACE_ROOT / "Mem-Gallery" / "benchmark" / "data"
DEFAULT_H2HMEM = WORKSPACE_ROOT / "H2HMEM-main" / "dataset"
DEFAULT_WMA = WORKSPACE_ROOT / "WorldMemArena" / "WorldMemArena" / "lifelong"
COUNTED_PATH_SUFFIXES = ("/chat/completions", "/completions", "/responses")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class SampleSpec:
    benchmark: str
    sample_id: str
    state_parts: tuple[str, ...]
    source_path: Path
    data_dir: Path
    variant: str = ""

    def load_chunks(self) -> list[Chunk]:
        if self.benchmark == "Mem-Gallery":
            payload = json.loads(self.source_path.read_text(encoding="utf-8"))
            return build_chunks_from_data(payload, self.data_dir, self.source_path.stem)
        if self.benchmark == "H2HMEM":
            return build_h2h_chunks_from_directory(
                self.data_dir,
                variant=self.variant,
                conversation_ids={self.source_path.name},
            )
        if self.benchmark == "WorldMemArena":
            payload = json.loads(self.source_path.read_text(encoding="utf-8"))
            return build_wma_chunks_from_data(
                payload,
                self.data_dir,
                sample_path=self.source_path,
            )
        raise ValueError(f"Unsupported benchmark: {self.benchmark}")


class CallRecorder:
    def __init__(self, sample_id: str, trace_path: Path) -> None:
        self.sample_id = sample_id
        self.trace_path = trace_path
        self._lock = threading.Lock()
        self._next_id = 0
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def append(self, row: dict[str, Any]) -> None:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    def rows(self) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        result = []
        with self.trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    result.append(json.loads(line))
        return result


class CountingProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        target_base_url: str,
        recorder: CallRecorder,
        upstream_timeout: float,
    ) -> None:
        target = urlsplit(target_base_url)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError(f"Invalid executor base URL: {target_base_url}")
        self.target_scheme = target.scheme
        self.target_host = target.hostname
        self.target_port = target.port or (443 if target.scheme == "https" else 80)
        self.recorder = recorder
        self.upstream_timeout = upstream_timeout
        super().__init__(("127.0.0.1", 0), CountingProxyHandler)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class CountingProxyHandler(BaseHTTPRequestHandler):
    server: CountingProxyServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._forward(count_call=False)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        self._forward(count_call=path.endswith(COUNTED_PATH_SUFFIXES))

    def log_message(self, _format: str, *args: Any) -> None:
        del args

    def _forward(self, *, count_call: bool) -> None:
        started = time.time()
        request_id = self.server.recorder.next_id() if count_call else 0
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        status = 502
        response_body = b""
        response_headers: list[tuple[str, str]] = []
        error = ""
        try:
            connection_cls = (
                http.client.HTTPSConnection
                if self.server.target_scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_cls(
                self.server.target_host,
                self.server.target_port,
                timeout=self.server.upstream_timeout,
            )
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            response_body = response.read()
            response_headers = list(response.getheaders())
            connection.close()
        except Exception as exc:  # The client sees a retryable 502.
            error = f"{type(exc).__name__}: {exc}"
            response_body = json.dumps(
                {"error": {"message": error, "type": "mb_call_proxy_error"}}
            ).encode("utf-8")

        self.send_response(status)
        for key, value in response_headers:
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-length":
                self.send_header(key, value)
        if not any(key.lower() == "content-type" for key, _ in response_headers):
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

        if count_call:
            usage = _response_usage(response_body)
            self.server.recorder.append(
                {
                    "version": 1,
                    "sample_id": self.server.recorder.sample_id,
                    "request_id": request_id,
                    "method": self.command,
                    "path": urlsplit(self.path).path,
                    "status": status,
                    "failed": not 200 <= status < 300,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "started_at": started,
                    "finished_at": time.time(),
                    "duration_seconds": time.time() - started,
                    "error": error,
                }
            )


class CountingProxy:
    def __init__(
        self,
        target_base_url: str,
        recorder: CallRecorder,
        upstream_timeout: float,
    ) -> None:
        self.server = CountingProxyServer(target_base_url, recorder, upstream_timeout)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> CountingProxyServer:
        self.thread.start()
        return self.server

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _response_usage(body: bytes) -> dict[str, int | None]:
    try:
        payload = json.loads(body)
        usage = payload.get("usage") or {}
        return {
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def _artifact_name(sample_id: str) -> str:
    slug = "".join(value if value.isalnum() or value in "._-" else "_" for value in sample_id)
    slug = slug.strip("._") or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96]}-{digest}.json"


def _artifact_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _completed_artifact(path: Path) -> dict[str, Any] | None:
    payload = _artifact_payload(path)
    return payload if payload is not None and payload.get("status") == "completed" else None


def _try_sample_lock(path: Path):
    """Claim one sample without blocking another worker or GPU."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _sample_specs(
    benchmark: str,
    data_dir: Path,
    *,
    variant: str,
    selected: set[str],
) -> list[SampleSpec]:
    if benchmark == "Mem-Gallery":
        paths = sorted((data_dir / "dialog").glob("*.json"))
        return [
            SampleSpec(benchmark, path.stem, (path.stem,), path, data_dir)
            for path in paths
            if not selected or path.stem in selected
        ]
    if benchmark == "H2HMEM":
        if (data_dir / "dataset").is_dir():
            data_dir = data_dir / "dataset"
        variants = ("dyadic", "multiparty") if variant == "all" else (variant,)
        specs = []
        for current_variant in variants:
            names = list(
                dict.fromkeys(
                    path.parents[2].name
                    for path in iter_h2h_session_files(data_dir, variant=current_variant)
                )
            )
            for name in names:
                sample_id = f"{current_variant}/{name}"
                if selected and sample_id not in selected and name not in selected:
                    continue
                conversation_dir = data_dir / (
                    "multi-party" if current_variant == "multiparty" else current_variant
                ) / name
                specs.append(
                    SampleSpec(
                        benchmark,
                        sample_id,
                        (current_variant, name),
                        conversation_dir,
                        data_dir,
                        current_variant,
                    )
                )
        return specs
    if benchmark == "WorldMemArena":
        paths = iter_wma_sample_files(data_dir)
        return [
            SampleSpec(benchmark, path.stem, (path.stem,), path, data_dir)
            for path in paths
            if not selected or path.stem in selected
        ]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known_usage = [row for row in rows if row.get("total_tokens") is not None]
    return {
        "total_calls": len(rows),
        "failed_calls": sum(bool(row.get("failed")) for row in rows),
        "successful_calls": sum(not bool(row.get("failed")) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in known_usage),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in known_usage
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in known_usage),
        "usage_available_calls": len(known_usage),
        "usage_missing_calls": len(rows) - len(known_usage),
    }


def _run_sample(
    spec: SampleSpec,
    *,
    baseline: str,
    job_root: Path,
    executor_base_url: str,
    executor_model: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_dim: int,
    request_timeout: int,
    retries: int,
    max_chunks: int,
    resume: bool,
) -> dict[str, Any]:
    artifact_path = job_root / "samples" / _artifact_name(spec.sample_id)
    if resume and (payload := _completed_artifact(artifact_path)) is not None:
        print(f"[resume] {spec.sample_id}", flush=True)
        return payload

    lock_path = job_root / "locks" / f"{artifact_path.name}.lock"
    lock_handle = _try_sample_lock(lock_path)
    if lock_handle is None:
        print(f"[busy] {spec.sample_id}", flush=True)
        return {
            "sample_id": spec.sample_id,
            "status": "running_elsewhere",
            "error": "sample is claimed by another MB-call worker",
        }
    try:
        # Another worker may have completed the sample before this lock was acquired.
        if resume and (payload := _completed_artifact(artifact_path)) is not None:
            print(f"[resume] {spec.sample_id}", flush=True)
            return payload
        return _run_sample_claimed(
            spec,
            baseline=baseline,
            job_root=job_root,
            executor_base_url=executor_base_url,
            executor_model=executor_model,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            request_timeout=request_timeout,
            retries=retries,
            max_chunks=max_chunks,
            resume=resume,
        )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def _run_sample_claimed(
    spec: SampleSpec,
    *,
    baseline: str,
    job_root: Path,
    executor_base_url: str,
    executor_model: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_dim: int,
    request_timeout: int,
    retries: int,
    max_chunks: int,
    resume: bool,
) -> dict[str, Any]:
    artifact_path = job_root / "samples" / _artifact_name(spec.sample_id)
    if resume and (payload := _completed_artifact(artifact_path)) is not None:
        print(f"[resume] {spec.sample_id}", flush=True)
        return payload

    state_dir = job_root / "state" / Path(*spec.state_parts)
    trace_path = job_root / "traces" / _artifact_name(spec.sample_id).replace(".json", ".jsonl")
    if state_dir.exists():
        shutil.rmtree(state_dir)
    trace_path.unlink(missing_ok=True)
    artifact_path.unlink(missing_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    recorder = CallRecorder(spec.sample_id, trace_path)
    adapter = None
    chunk_count = 0
    sessions: set[str] = set()
    error = ""
    try:
        with CountingProxy(
            executor_base_url,
            recorder,
            upstream_timeout=max(request_timeout + 30, 210),
        ) as proxy:
            config = {
                "top_k": 5,
                "embedding_dim": embedding_dim,
                "embedding_model": embedding_model,
                "embedding_base_url": embedding_base_url,
                "executor_model": executor_model,
                "executor_base_url": proxy.endpoint,
                "executor_temperature": 0.0,
                "executor_visual_input": "image",
                "answer_model": executor_model,
                "answer_base_url": proxy.endpoint,
                "answer_temperature": 0.0,
                "request_timeout": request_timeout,
                "retries": retries,
                "num_predict": 512,
            }
            adapter = create_adapter(baseline, config_overrides=config)
            adapter.reset(spec.sample_id, state_dir)
            chunks = spec.load_chunks()
            if max_chunks:
                chunks = chunks[:max_chunks]
            current_session = ""
            for chunk in chunks:
                session_id = str(chunk.metadata.get("session_id") or "")
                if current_session and session_id != current_session:
                    adapter.end_session(current_session)
                adapter.ingest(chunk)
                chunk_count += 1
                current_session = session_id
                if session_id:
                    sessions.add(session_id)
            if current_session:
                adapter.end_session(current_session)
        status = "completed"
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                if not error:
                    status = "failed"
                    error = f"close {type(exc).__name__}: {exc}"

    calls = _summarize_trace(recorder.rows())
    payload = {
        "version": 1,
        "benchmark": spec.benchmark,
        "baseline": baseline,
        "sample_id": spec.sample_id,
        "status": status,
        "error": error,
        "chunk_count": chunk_count,
        "session_count": len(sessions),
        "duration_seconds": time.time() - started,
        "trace_path": str(trace_path.resolve()),
        **calls,
    }
    write_json_atomic(artifact_path, payload)
    print(
        f"[{status}] {spec.sample_id}: chunks={chunk_count} "
        f"calls={calls['total_calls']} failed={calls['failed_calls']}",
        flush=True,
    )
    return payload


def _aggregate(
    benchmark: str,
    baseline: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    total_calls = sum(int(row.get("total_calls") or 0) for row in completed)
    failed_calls = sum(int(row.get("failed_calls") or 0) for row in completed)
    num_samples = len(rows)
    return {
        "version": 1,
        "benchmark": benchmark,
        "baseline": baseline,
        "definition": (
            "Actual executor HTTP attempts during reset/ingest/end_session only; "
            "embedding, retrieval, QA, and judge calls are excluded."
        ),
        "available": not failed and len(completed) == num_samples,
        "num_samples": num_samples,
        "completed_samples": len(completed),
        "failed_samples": len(failed),
        "total_calls": total_calls,
        "failed_calls": failed_calls,
        "successful_calls": total_calls - failed_calls,
        "mean_per_sample": total_calls / num_samples if num_samples and not failed else None,
        "formula": (
            f"{total_calls} / {num_samples} = {total_calls / num_samples:.12g}"
            if num_samples and not failed
            else None
        ),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in completed),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in completed
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in completed),
        "usage_missing_calls": sum(
            int(row.get("usage_missing_calls") or 0) for row in completed
        ),
        "errors": [
            {"sample_id": row.get("sample_id"), "error": row.get("error")}
            for row in failed
        ],
        "samples": rows,
    }


def _ordered_rows(
    specs: list[SampleSpec],
    job_root: Path,
    fallback: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer durable per-sample artifacts over process-local status rows."""

    rows = []
    for spec in specs:
        artifact = job_root / "samples" / _artifact_name(spec.sample_id)
        row = _artifact_payload(artifact) or fallback.get(spec.sample_id)
        if row is not None:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("Mem-Gallery", "H2HMEM", "WorldMemArena"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--variant", choices=("dyadic", "multiparty", "all"), default="all")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--executor-base-url", required=True)
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_samples < 0 or args.max_chunks < 0 or args.sample_concurrency < 1:
        parser.error("Limits must be non-negative and concurrency must be positive")

    baseline = canonical_name(args.baseline)
    default_dirs = {
        "Mem-Gallery": DEFAULT_MEMGALLERY,
        "H2HMEM": DEFAULT_H2HMEM,
        "WorldMemArena": DEFAULT_WMA,
    }
    data_dir = Path(args.data_dir) if args.data_dir else default_dirs[args.benchmark]
    specs = _sample_specs(
        args.benchmark,
        data_dir,
        variant=args.variant,
        selected=set(args.sample_id),
    )
    if args.max_samples:
        specs = specs[: args.max_samples]
    if not specs:
        raise FileNotFoundError("No matching benchmark samples")

    job_root = Path(args.output_root) / args.benchmark / baseline
    job_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        job_root / "run_config.json",
        {
            "benchmark": args.benchmark,
            "baseline": baseline,
            "data_dir": str(data_dir.resolve()),
            "executor_base_url": args.executor_base_url,
            "executor_model": args.executor_model,
            "embedding_base_url": args.embedding_base_url,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "request_timeout": args.request_timeout,
            "retries": args.retries,
            "sample_concurrency": args.sample_concurrency,
            "max_samples": args.max_samples,
            "max_chunks": args.max_chunks,
            "samples": [spec.sample_id for spec in specs],
        },
    )
    os.environ["MEMVERSE_REUSE_STATE"] = "0"
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.sample_concurrency) as pool:
        futures = {
            pool.submit(
                _run_sample,
                spec,
                baseline=baseline,
                job_root=job_root,
                executor_base_url=args.executor_base_url,
                executor_model=args.executor_model,
                embedding_base_url=args.embedding_base_url,
                embedding_model=args.embedding_model,
                embedding_dim=args.embedding_dim,
                request_timeout=args.request_timeout,
                retries=args.retries,
                max_chunks=args.max_chunks,
                resume=args.resume,
            ): spec.sample_id
            for spec in specs
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                results[sample_id] = future.result()
            except Exception as exc:
                results[sample_id] = {
                    "sample_id": sample_id,
                    "status": "failed",
                    "error": f"runner {type(exc).__name__}: {exc}",
                }
            ordered = _ordered_rows(specs, job_root, results)
            write_json_atomic(job_root / "progress.json", ordered)

    ordered = _ordered_rows(specs, job_root, results)
    metrics = _aggregate(args.benchmark, baseline, ordered)
    write_json_atomic(job_root / "metrics.json", metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "samples"}, indent=2))
    if not metrics["available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
