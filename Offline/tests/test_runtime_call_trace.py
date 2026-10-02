from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request

import pytest

from benchmarks.baseline_runtime.call_trace import (
    CallRecorder,
    CountingProxy,
    _normalize_m2a_qwen_vllm_request,
    _normalize_m2a_qwen_vllm_response,
)
from benchmarks.memgallery_harness.runner.metrics import write_runtime_call_metrics
from benchmarks.memgallery_harness.runner.metrics import merge_llm_judge_metrics
from scripts.judge_results_llm_parallel import summarize as summarize_judge
from scripts.rerun_qa_from_frozen_retrieval import write_frozen_nonanswer_trace


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    last_payload = None
    finish_reason = "stop"
    native_finish_reason = "stop"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length)
        type(self).last_payload = json.loads(raw_body)
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": type(self).finish_reason,
                        "native_finish_reason": type(self).native_finish_reason,
                        "message": {"content": "ok"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        del args


def _post(url: str, image_count: int = 0) -> None:
    content = [{"type": "text", "text": "test"}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}}
        for index in range(image_count)
    ]
    body = json.dumps(
        {"model": "fake", "messages": [{"role": "user", "content": content}]}
    ).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_m2a_local_qwen_forces_only_mandatory_first_retrieval_hops():
    chat_request = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "messages": [{"role": "user", "content": "question"}],
        "tools": [_tool("query_memory")],
        "tool_choice": "auto",
    }
    normalized = json.loads(
        _normalize_m2a_qwen_vllm_request(
            json.dumps(chat_request).encode(), phase="retrieval"
        )
    )
    assert normalized["tool_choice"] == {
        "type": "function",
        "function": {"name": "query_memory"},
    }
    assert normalized["parallel_tool_calls"] is False

    chat_request["messages"].append(
        {"role": "tool", "tool_call_id": "call-0", "content": "memory"}
    )
    later_chat = _normalize_m2a_qwen_vllm_request(
        json.dumps(chat_request).encode(), phase="retrieval"
    )
    assert later_chat == json.dumps(chat_request).encode()

    manager_request = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "messages": [{"role": "user", "content": "memory query"}],
        "tools": [
            _tool("search_semantic_memories"),
            _tool("fetch_raw_messages"),
        ],
        "tool_choice": "auto",
    }
    normalized = json.loads(
        _normalize_m2a_qwen_vllm_request(
            json.dumps(manager_request).encode(), phase="retrieval"
        )
    )
    assert normalized["tool_choice"]["function"]["name"] == "search_semantic_memories"

    manager_request["messages"].append(
        {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    )
    later = _normalize_m2a_qwen_vllm_request(
        json.dumps(manager_request).encode(), phase="retrieval"
    )
    assert later == json.dumps(manager_request).encode()


def test_m2a_qwen_request_compat_does_not_change_api_or_build_requests():
    request = {
        "model": "openai/gpt-5-mini",
        "messages": [{"role": "user", "content": "question"}],
        "tools": [_tool("query_memory")],
        "tool_choice": "auto",
    }
    encoded = json.dumps(request).encode()
    assert _normalize_m2a_qwen_vllm_request(encoded, phase="retrieval") == encoded

    request["model"] = "Qwen/Qwen3-VL-4B-Instruct"
    encoded = json.dumps(request).encode()
    assert _normalize_m2a_qwen_vllm_request(encoded, phase="memory_build") == encoded


def test_m2a_local_qwen_promotes_only_complete_bare_tool_call():
    request = json.dumps(
        {
            "model": "Qwen/Qwen3-VL-4B-Instruct",
            "tools": [_tool("query_memory")],
        }
    ).encode()
    response = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": 'query_memory(text="Find the trip", image=None)'
                    },
                }
            ]
        }
    ).encode()
    normalized = json.loads(
        _normalize_m2a_qwen_vllm_response(response, request_body=request)
    )
    choice = normalized["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "query_memory"
    assert json.loads(call["function"]["arguments"]) == {
        "text": "Find the trip",
        "image": None,
    }

    truncated = response.replace(b'"stop"', b'"length"')
    assert (
        _normalize_m2a_qwen_vllm_response(truncated, request_body=request)
        == truncated
    )


def test_runtime_proxy_records_build_and_retrieval_calls(tmp_path: Path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "sample.jsonl"
    recorder = CallRecorder(
        trace_path=trace_path,
        baseline="MIRIX",
        benchmark="Mem-Gallery",
        sample_id="sample",
        reset=True,
    )
    try:
        target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        with CountingProxy(target, recorder, 5) as proxy:
            with recorder.phase("memory_build"):
                _post(f"{proxy.endpoint}/chat/completions", image_count=2)
            with recorder.phase("retrieval"):
                _post(f"{proxy.endpoint}/chat/completions", image_count=1)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    results = [
        {
            "dataset": "sample",
            "answer_attempts": 2,
            "answer_failed_attempts": 1,
        }
    ]
    calls = write_runtime_call_metrics(
        [trace_path],
        tmp_path / "result",
        results,
        sample_id_field="dataset",
        sample_ids=["sample"],
    )
    assert calls["memory_bank"]["total_calls"] == 1
    assert calls["retrieval"]["total_calls"] == 1
    assert calls["answer"]["total_calls"] == 2
    assert calls["qa"]["retrieval_calls"] == 1
    assert calls["qa"]["answer_calls"] == 2
    assert calls["qa"]["total_calls"] == 3
    assert calls["total"]["total_calls"] == 4
    rows = [
        json.loads(line)
        for line in (tmp_path / "result" / "call_trace.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows].count("memory_build") == 1
    assert [row["phase"] for row in rows].count("retrieval") == 1
    assert [row["phase"] for row in rows].count("qa") == 2
    native_rows = [
        row for row in rows if row["phase"] in {"memory_build", "retrieval"}
    ]
    assert all(row["finish_reason"] == "stop" for row in native_rows)
    assert all(row["native_finish_reason"] == "stop" for row in native_rows)
    assert not any(row["truncated"] for row in native_rows)
    assert all(
        row["total_tokens"] == 5
        for row in rows
        if row["phase"] in {"memory_build", "retrieval"}
    )
    assert [
        row["image_count"]
        for row in rows
        if row["phase"] in {"memory_build", "retrieval"}
    ] == [2, 1]


def test_m3_frozen_replay_keeps_retrieval_and_rejects_qa_phase_contamination(
    tmp_path: Path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run_manifest.json").write_text(
        json.dumps({"baseline": "M3-Agent-caption"}), encoding="utf-8"
    )
    (source / "results.json").write_text(
        json.dumps([{"answer_attempts": 1}]), encoding="utf-8"
    )
    rows = [
        {"phase": "memory_build"},
        {"phase": "retrieval"},
        {"phase": "qa"},
        {"phase": "judge"},
    ]
    (source / "call_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    target = write_frozen_nonanswer_trace(source, tmp_path / "valid")
    retained = [json.loads(line) for line in target.read_text().splitlines()]
    assert [row["phase"] for row in retained] == ["memory_build", "retrieval"]

    rows.append({"phase": "qa"})
    (source / "call_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Retrieval/Control calls may be mislabelled"):
        write_frozen_nonanswer_trace(source, tmp_path / "invalid")


def test_runtime_proxy_enforces_configured_output_cap(tmp_path: Path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "sample.jsonl"
    recorder = CallRecorder(
        trace_path=trace_path,
        baseline="MIRIX",
        benchmark="WorldMemArena",
        sample_id="sample",
        reset=True,
    )
    try:
        target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        with CountingProxy(
            target,
            recorder,
            5,
            max_output_tokens=8192,
            temperature=0.0,
            qa_max_output_tokens=512,
            qa_upstream_timeout=2,
        ) as proxy:
            with recorder.phase("memory_build"):
                _post(f"{proxy.endpoint}/chat/completions")
            with recorder.phase("retrieval"):
                _post(f"{proxy.endpoint}/chat/completions")
            with recorder.phase("qa"):
                _post(f"{proxy.endpoint}/chat/completions")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    assert _UpstreamHandler.last_payload["max_tokens"] == 512
    assert _UpstreamHandler.last_payload["temperature"] == 0.0
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["memory_build", "retrieval", "qa"]
    assert [row["max_output_tokens"] for row in rows] == [8192, 8192, 512]
    assert [row["upstream_timeout_seconds"] for row in rows] == [5, 5, 2]
    assert all(row["temperature"] == 0.0 for row in rows)


def test_runtime_proxy_marks_native_max_output_tokens_as_truncated(tmp_path: Path):
    class MaxOutputHandler(_UpstreamHandler):
        finish_reason = "tool_calls"
        native_finish_reason = "max_output_tokens"

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), MaxOutputHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "sample.jsonl"
    recorder = CallRecorder(
        trace_path=trace_path,
        baseline="MMA",
        benchmark="Mem-Gallery",
        sample_id="sample",
        reset=True,
    )
    try:
        target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        with CountingProxy(target, recorder, 5) as proxy:
            with recorder.phase("memory_build"):
                _post(f"{proxy.endpoint}/chat/completions")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    row = json.loads(trace_path.read_text().splitlines()[0])
    assert row["finish_reason"] == "tool_calls"
    assert row["native_finish_reason"] == "max_output_tokens"
    assert row["truncated"] is True
    response_path = Path(row["response_body_path"])
    assert response_path.is_file()
    assert row["response_body_bytes"] == response_path.stat().st_size
    assert len(row["response_body_sha256"]) == 64
    captured = json.loads(response_path.read_text())
    assert captured["choices"][0]["message"]["content"] == "ok"


def test_runtime_proxy_prefers_native_qa_calls_over_logical_answer_attempt(tmp_path: Path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    trace_path = tmp_path / "sample.jsonl"
    recorder = CallRecorder(
        trace_path=trace_path,
        baseline="MIRIX",
        benchmark="Mem-Gallery",
        sample_id="sample",
        reset=True,
    )
    try:
        target = f"http://127.0.0.1:{upstream.server_address[1]}/v1"
        with CountingProxy(target, recorder, 5) as proxy:
            with recorder.phase("memory_build"):
                _post(f"{proxy.endpoint}/chat/completions")
            with recorder.phase("qa"):
                _post(f"{proxy.endpoint}/chat/completions")
                _post(f"{proxy.endpoint}/chat/completions")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)

    calls = write_runtime_call_metrics(
        [trace_path],
        tmp_path / "result",
        [
            {
                "dataset": "sample",
                "answer_attempts": 1,
                "answer_failed_attempts": 0,
            }
        ],
        sample_id_field="dataset",
        sample_ids=["sample"],
    )

    assert calls["qa"]["total_calls"] == 2
    assert calls["total"]["total_calls"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "result" / "call_trace.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows].count("qa") == 2


def test_judge_attempts_are_merged_into_canonical_calls():
    judge = summarize_judge(
        [
            {
                "label": "correct",
                "score": 1.0,
                "judge_attempts": 2,
                "judge_failed_attempts": 1,
            }
        ],
        "fake-judge",
        expected_count=1,
    )
    merged = merge_llm_judge_metrics(
        {"f1": 0.5, "em": 0.5, "calls": {"qa": {"total_calls": 1}}},
        judge,
    )
    assert judge["calls"] == {
        "total_calls": 2,
        "failed_calls": 1,
        "successful_calls": 1,
        "available": True,
    }
    assert merged["calls"]["judge"] == judge["calls"]
