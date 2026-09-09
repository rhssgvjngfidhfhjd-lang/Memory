from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request

from benchmarks.baseline_runtime.call_trace import CallRecorder, CountingProxy
from benchmarks.memgallery_harness.runner.metrics import write_runtime_call_metrics
from benchmarks.memgallery_harness.runner.metrics import merge_llm_judge_metrics
from scripts.judge_results_llm_parallel import summarize as summarize_judge


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
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
    assert calls["qa"]["total_calls"] == 2
    assert calls["total"]["total_calls"] == 3
    rows = [
        json.loads(line)
        for line in (tmp_path / "result" / "call_trace.jsonl").read_text().splitlines()
    ]
    assert [row["phase"] for row in rows].count("memory_build") == 1
    assert [row["phase"] for row in rows].count("retrieval") == 1
    assert [row["phase"] for row in rows].count("qa") == 2
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
