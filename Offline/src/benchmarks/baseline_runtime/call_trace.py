from __future__ import annotations

from contextlib import contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator
from urllib.parse import urlsplit


TRACE_VERSION = 2
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


def trace_filename(sample_id: str) -> str:
    """Return a stable, filesystem-safe name for one sample trace."""
    slug = "".join(
        value if value.isalnum() or value in "._-" else "_" for value in sample_id
    ).strip("._") or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96]}-{digest}.jsonl"


class CallRecorder:
    """Thread-safe recorder used by one sample-local counting proxy."""

    def __init__(
        self,
        *,
        trace_path: Path,
        baseline: str,
        benchmark: str,
        sample_id: str,
        reset: bool = False,
    ) -> None:
        self.trace_path = trace_path
        self.baseline = baseline
        self.benchmark = benchmark
        self.sample_id = sample_id
        self._lock = threading.Lock()
        self._next_id = 0
        self._phase = "memory_build"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            trace_path.unlink(missing_ok=True)

    @property
    def phase_name(self) -> str:
        with self._lock:
            return self._phase

    @contextmanager
    def phase(self, value: str) -> Iterator[None]:
        with self._lock:
            previous = self._phase
            self._phase = value
        try:
            yield
        finally:
            with self._lock:
                self._phase = previous

    def next_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def append(self, row: dict[str, Any]) -> None:
        payload = {
            "trace_version": TRACE_VERSION,
            "baseline": self.baseline,
            "benchmark": self.benchmark,
            "sample_id": self.sample_id,
            **row,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")


class _CountingProxyServer(ThreadingHTTPServer):
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
        self.target_prefix = target.path.rstrip("/")
        self.recorder = recorder
        self.upstream_timeout = upstream_timeout
        super().__init__(("127.0.0.1", 0), _CountingProxyHandler)

    @property
    def endpoint(self) -> str:
        prefix = self.target_prefix or "/v1"
        return f"http://127.0.0.1:{self.server_address[1]}{prefix}"


class _CountingProxyHandler(BaseHTTPRequestHandler):
    server: _CountingProxyServer
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
        phase = self.server.recorder.phase_name
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
        except Exception as exc:  # Return a retryable response to the native client.
            error = f"{type(exc).__name__}: {exc}"
            response_body = json.dumps(
                {"error": {"message": error, "type": "call_trace_proxy_error"}}
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
            finished = time.time()
            usage = _response_usage(response_body)
            request_payload = _request_metadata(body)
            self.server.recorder.append(
                {
                    "request_id": request_id,
                    "phase": phase,
                    "service": "llm",
                    "method": self.command,
                    "path": urlsplit(self.path).path,
                    "model": request_payload.get("model", ""),
                    "status": status,
                    "success": 200 <= status < 300,
                    "failed": not 200 <= status < 300,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "image_count": _request_image_count(request_payload),
                    "started_at": started,
                    "finished_at": finished,
                    "duration_seconds": finished - started,
                    "error": error,
                }
            )


class CountingProxy:
    """Route one adapter's executor traffic through a local counting proxy."""

    def __init__(
        self,
        target_base_url: str,
        recorder: CallRecorder,
        upstream_timeout: float,
    ) -> None:
        self.server = _CountingProxyServer(target_base_url, recorder, upstream_timeout)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _CountingProxyServer:
        self.thread.start()
        return self.server

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def load_call_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize_call_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    num_samples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("phase") == phase]
    total = len(selected)
    failed = sum(bool(row.get("failed")) for row in selected)
    mean = total / num_samples if num_samples else None
    return {
        "total_calls": total,
        "failed_calls": failed,
        "successful_calls": total - failed,
        "num_samples": num_samples,
        "mean_per_sample": mean,
        "formula": f"{total} / {num_samples} = {mean:.12g}" if num_samples else None,
        "aggregation": f"{phase}_calls_divided_by_samples",
        "available": True,
    }


def _request_metadata(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _request_image_count(payload: dict[str, Any]) -> int:
    """Count images in OpenAI-compatible multimodal request content."""
    count = 0
    for message in payload.get("messages") or ():
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = str(part.get("type") or "").casefold()
                if kind in {"image", "image_url", "input_image"}:
                    count += 1
        images = message.get("images")
        if isinstance(images, list):
            count += len(images)
    return count


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
