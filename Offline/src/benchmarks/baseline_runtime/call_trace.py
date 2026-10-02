from __future__ import annotations

import ast
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


TRACE_VERSION = 3
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
        self._scope: dict[str, Any] = {}
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            trace_path.unlink(missing_ok=True)
        elif trace_path.is_file():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    request_id = int(json.loads(line).get("request_id") or 0)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                self._next_id = max(self._next_id, request_id)

    @property
    def phase_name(self) -> str:
        with self._lock:
            return self._phase

    def context_snapshot(self) -> tuple[str, dict[str, Any]]:
        """Snapshot the phase and logical QA ownership for one HTTP call."""
        with self._lock:
            return self._phase, dict(self._scope)

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

    @contextmanager
    def scope(self, **values: Any) -> Iterator[None]:
        """Attach stable query/operation metadata to calls made in this block."""
        with self._lock:
            previous = self._scope
            self._scope = {**previous, **values}
        try:
            yield
        finally:
            with self._lock:
                self._scope = previous

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

    def capture_response_body(self, request_id: int, body: bytes) -> Path:
        """Persist a failed/truncated provider response for diagnosis."""
        destination = (
            self.trace_path.parent
            / "response_bodies"
            / self.trace_path.stem
            / f"request_{request_id:06d}.response.json"
        )
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
        return destination


class _CountingProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        target_base_url: str,
        recorder: CallRecorder,
        upstream_timeout: float,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        qa_max_output_tokens: int | None = None,
        reasoning_effort: str = "",
        qa_upstream_timeout: float | None = None,
    ) -> None:
        target = urlsplit(target_base_url)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError(f"Invalid executor base URL: {target_base_url}")
        self.target_scheme = target.scheme
        self.target_host = target.hostname
        self.target_port = target.port or (443 if target.scheme == "https" else 80)
        self.target_prefix = target.path.rstrip("/")
        self.m2a_qwen_vllm_compat = (
            recorder.baseline == "M2A"
            and target.scheme == "http"
            and target.hostname in {"127.0.0.1", "localhost", "::1"}
        )
        self.recorder = recorder
        self.upstream_timeout = upstream_timeout
        self.qa_upstream_timeout = qa_upstream_timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.qa_max_output_tokens = qa_max_output_tokens
        self.reasoning_effort = str(reasoning_effort).strip()
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
        phase, logical_scope = self.server.recorder.context_snapshot()
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request_path = urlsplit(self.path).path.rstrip("/")
        if count_call and request_path.endswith("/chat/completions"):
            phase_cap = self.server.max_output_tokens
            # Retrieval is part of the baseline executor and needs the same
            # generation budget as memory construction.  Only the final
            # benchmark answer is governed by the smaller QA budget.
            if phase == "qa":
                phase_cap = (
                    self.server.qa_max_output_tokens
                    if self.server.qa_max_output_tokens is not None
                    else phase_cap
                )
            body = _normalize_chat_completion_request(
                body,
                max_output_tokens=phase_cap,
                temperature=self.server.temperature,
                reasoning_effort=self.server.reasoning_effort,
            )
            if self.server.m2a_qwen_vllm_compat:
                body = _normalize_m2a_qwen_vllm_request(body, phase=phase)
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
            upstream_timeout = (
                self.server.qa_upstream_timeout
                if phase == "qa" and self.server.qa_upstream_timeout is not None
                else self.server.upstream_timeout
            )
            connection = connection_cls(
                self.server.target_host,
                self.server.target_port,
                timeout=upstream_timeout,
            )
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            response_body = response.read()
            response_headers = list(response.getheaders())
            connection.close()
            if (
                count_call
                and request_path.endswith("/chat/completions")
                and self.server.m2a_qwen_vllm_compat
            ):
                response_body = _normalize_m2a_qwen_vllm_response(
                    response_body,
                    request_body=body,
                )
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
            finish_reason, native_finish_reason = _response_finish_reasons(
                response_body
            )
            request_payload = _request_metadata(body)
            truncated = finish_reason in {"length", "max_tokens"} or (
                native_finish_reason
                in {"length", "max_tokens", "max_output_tokens"}
            )
            response_capture: dict[str, Any] = {}
            if truncated or not 200 <= status < 300:
                response_path = self.server.recorder.capture_response_body(
                    request_id, response_body
                )
                response_capture = {
                    "response_body_path": str(response_path),
                    "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
                    "response_body_bytes": len(response_body),
                }
            self.server.recorder.append(
                {
                    "request_id": request_id,
                    "phase": phase,
                    **logical_scope,
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
                    "finish_reason": finish_reason,
                    "native_finish_reason": native_finish_reason,
                    "truncated": truncated,
                    "image_count": _request_image_count(request_payload),
                    "max_output_tokens": request_payload.get("max_tokens"),
                    "temperature": request_payload.get("temperature"),
                    "started_at": started,
                    "finished_at": finished,
                    "duration_seconds": finished - started,
                    "upstream_timeout_seconds": upstream_timeout,
                    "error": error,
                    **response_capture,
                }
            )


class CountingProxy:
    """Route one adapter's executor traffic through a local counting proxy."""

    def __init__(
        self,
        target_base_url: str,
        recorder: CallRecorder,
        upstream_timeout: float,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        qa_max_output_tokens: int | None = None,
        reasoning_effort: str = "",
        qa_upstream_timeout: float | None = None,
    ) -> None:
        self.server = _CountingProxyServer(
            target_base_url,
            recorder,
            upstream_timeout,
            max_output_tokens,
            temperature,
            qa_max_output_tokens,
            reasoning_effort,
            qa_upstream_timeout,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _CountingProxyServer:
        self.thread.start()
        return self.server

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _normalize_chat_completion_request(
    body: bytes,
    *,
    max_output_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str = "",
) -> bytes:
    """Enforce experiment generation settings at the executor boundary."""
    if max_output_tokens is None and temperature is None and not reasoning_effort:
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    if max_output_tokens is not None:
        candidates = [int(max_output_tokens)]
        for key in ("max_tokens", "max_completion_tokens"):
            value = payload.get(key)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        payload["max_tokens"] = min(candidates)
        # vLLM accepts max_tokens for both normal and native-tool requests.
        payload.pop("max_completion_tokens", None)
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if reasoning_effort:
        payload["reasoning"] = {"effort": str(reasoning_effort)}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _request_tool_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in payload.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _has_tool_history(payload: dict[str, Any]) -> bool:
    for message in payload.get("messages") or ():
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
    return False


def _named_tool_choice(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


def _normalize_m2a_qwen_vllm_request(body: bytes, *, phase: str) -> bytes:
    """Apply local Qwen/vLLM compatibility without changing remote API calls.

    Qwen3-VL is unreliable with unconstrained ``auto`` selection for the first
    retrieval hop. vLLM turns a named tool choice into schema-constrained JSON,
    so require only the two retrieval calls that the harness already mandates:
    ChatAgent -> query_memory and the MemoryManager's first semantic search.
    Later MemoryManager turns stay ``auto`` so the original agent can stop.
    """

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    model = str(payload.get("model") or "").casefold()
    if "qwen3-vl" not in model or phase != "retrieval":
        return body

    tool_names = _request_tool_names(payload)
    forced_name = ""
    if tool_names == ["query_memory"] and not _has_tool_history(payload):
        forced_name = "query_memory"
    elif (
        "search_semantic_memories" in tool_names
        and not _has_tool_history(payload)
    ):
        forced_name = "search_semantic_memories"
    if not forced_name:
        return body

    payload["tool_choice"] = _named_tool_choice(forced_name)
    payload["parallel_tool_calls"] = False
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _literal_python_tool_call(
    content: str,
    *,
    allowed_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    """Parse one complete ``name(key=value)`` call without executing it."""

    try:
        expression = ast.parse(content.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id not in allowed_names
        or expression.args
    ):
        return None
    arguments: dict[str, Any] = {}
    try:
        for keyword in expression.keywords:
            if keyword.arg is None or keyword.arg in arguments:
                return None
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return expression.func.id, arguments


def _normalize_m2a_qwen_vllm_response(
    body: bytes,
    *,
    request_body: bytes,
) -> bytes:
    """Promote a complete bare Qwen function call to OpenAI tool-call form.

    This intentionally refuses partial or length-truncated output. It exists
    for Qwen's occasional valid ``query_memory(text=...)`` response without the
    XML envelope requested by its chat template.
    """

    try:
        payload = json.loads(body)
        request = json.loads(request_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict) or not isinstance(request, dict):
        return body
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return body
    choice = choices[0]
    if str(choice.get("finish_reason") or "").casefold() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }:
        return body
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        return body
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return body

    parsed = _literal_python_tool_call(
        content,
        allowed_names=set(_request_tool_names(request)),
    )
    if parsed is None:
        return body
    name, arguments = parsed
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    message["content"] = None
    message["tool_calls"] = [
        {
            "id": f"call_qwen_{digest}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
    ]
    choice["finish_reason"] = "tool_calls"
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


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


def _response_finish_reasons(body: bytes) -> tuple[str, str]:
    try:
        payload = json.loads(body)
        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        return (
            str(choice.get("finish_reason") or ""),
            str(choice.get("native_finish_reason") or ""),
        )
    except (AttributeError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return "", ""
