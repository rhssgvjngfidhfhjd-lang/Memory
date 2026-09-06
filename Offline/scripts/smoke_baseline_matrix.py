"""Run a minimal real protocol cycle for all 24 benchmark/method groups."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.baseline_runtime import BASELINE_NAMES, create_adapter  # noqa: E402
from benchmarks.baseline_runtime.openai_compat import embed_texts  # noqa: E402
from benchmarks.baseline_runtime.protocol import (  # noqa: E402
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import (  # noqa: E402
    write_json_atomic,
    write_jsonl_atomic,
    write_text_atomic,
)
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient  # noqa: E402
from embedding.chunk_builder import (  # noqa: E402
    Chunk,
    build_chunks_from_data,
    build_h2h_chunks_from_data,
    build_wma_chunks_from_data,
    iter_h2h_session_files,
    iter_wma_sample_files,
)


METHODS = tuple(BASELINE_NAMES)
BENCHMARKS = ("Mem-Gallery", "WorldMemArena", "H2HMEM")
SYSTEM_PROMPT = (
    "Answer using only the retrieved memory. Return only a concise answer; "
    "say unknown when the memory does not contain the answer."
)
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class SmokeCase:
    name: str
    chunk: Chunk
    question: str
    category: str

    @property
    def sample_id(self) -> str:
        return str(self.chunk.metadata.get("dataset") or self.name)


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def check_endpoints(answer_urls: list[str], embedding_url: str, embedding_model: str, dim: int) -> None:
    if len(answer_urls) != 3:
        raise ValueError(f"exactly three answer endpoints are required, got {len(answer_urls)}")
    for base_url in answer_urls:
        payload = _json_request(base_url.rstrip("/") + "/models")
        models = {str(row.get("id")) for row in payload.get("data") or []}
        if "Qwen/Qwen3-VL-4B-Instruct" not in models:
            raise RuntimeError(f"answer model unavailable at {base_url}: {sorted(models)}")
    endpoint = embedding_url.rstrip("/") + "/embeddings"
    payload = _json_request(endpoint, {"model": embedding_model, "input": ["smoke test"]})
    rows = payload.get("data") or []
    actual_dim = len(rows[0].get("embedding") or []) if rows else 0
    if actual_dim != dim:
        raise RuntimeError(f"embedding dimension mismatch at {endpoint}: expected {dim}, got {actual_dim}")


def _memgallery_cases() -> list[SmokeCase]:
    data_dir = WORKSPACE / "Mem-Gallery" / "benchmark" / "data"
    paths = sorted(path for path in (data_dir / "dialog").glob("*.json") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No Mem-Gallery data under {data_dir}")
    path = paths[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunk = build_chunks_from_data(payload, data_dir, path.stem)[0]
    qa = (payload.get("human-annotated QAs") or [])[0]
    return [SmokeCase("default", chunk, str(qa.get("question") or ""), str(qa.get("point") or ""))]


def _wma_cases() -> list[SmokeCase]:
    data_dir = WORKSPACE / "WorldMemArena" / "WorldMemArena" / "lifelong"
    path = iter_wma_sample_files(data_dir)[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunk = build_wma_chunks_from_data(payload, path.parent, sample_path=path)[0]
    qa = next(
        qa
        for checkpoint in payload.get("qa_checkpoints") or []
        for qa in checkpoint.get("questions") or []
    )
    return [
        SmokeCase(
            "lifelong",
            chunk,
            str(qa.get("question") or ""),
            str(qa.get("question_type_abbrev") or ""),
        )
    ]


def _h2hmem_case(variant: str) -> SmokeCase:
    data_dir = WORKSPACE / "H2HMEM-main" / "dataset"
    path = iter_h2h_session_files(data_dir, variant=variant)[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    conversation_id = path.parents[2].name
    chunk = build_h2h_chunks_from_data(
        payload,
        session_path=path,
        variant=variant,
        conversation_id=conversation_id,
    )[0]
    question = (json.loads(path.with_name("questions.json").read_text(encoding="utf-8")).get("questions") or [])[0]
    question_type = question.get("question_type") or {}
    return SmokeCase(
        variant,
        chunk,
        str((question.get("question") or {}).get("text") or ""),
        str(question_type.get("sub_type") or question_type.get("main_type") or ""),
    )


def fixtures() -> dict[str, list[SmokeCase]]:
    return {
        "Mem-Gallery": _memgallery_cases(),
        "WorldMemArena": _wma_cases(),
        "H2HMEM": [_h2hmem_case("dyadic"), _h2hmem_case("multiparty")],
    }


def _config(answer_url: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "answer_model": args.answer_model,
        "answer_base_url": answer_url,
        "answer_temperature": 0.0,
        "executor_model": args.answer_model,
        "executor_base_url": answer_url,
        "executor_temperature": 0.0,
        "executor_visual_input": "image",
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_dim": args.embedding_dim,
        "top_k": args.top_k,
        "request_timeout": args.timeout,
        "retries": args.retries,
    }


def _build_hivemem(
    cases: list[SmokeCase],
    result_dir: Path,
    answer_url: str,
    args: argparse.Namespace,
) -> None:
    chunks_path = result_dir / "inputs" / "chunks.jsonl"
    write_jsonl_atomic(chunks_path, [case.chunk.to_dict() for case in cases])
    command = [
        sys.executable,
        "-m",
        "hive_mem.build_memories",
        "--chunks", str(chunks_path),
        "--all-datasets",
        "--output-root", str(result_dir / "memory"),
        "--executor-model", args.answer_model,
        "--executor-base-url", answer_url,
        "--executor-timeout", str(args.timeout),
        "--executor-retries", str(args.retries),
        "--executor-concurrency", "1",
        "--embedding-model", args.embedding_model,
        "--embedding-base-url", args.embedding_base_url,
        "--embedding-dim", str(args.embedding_dim),
        "--no-resume",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(600, args.timeout * max(1, len(cases)) * 2),
        check=False,
    )
    write_text_atomic(result_dir / "memory" / "build_smoke.log", completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"HiveMem build failed with exit code {completed.returncode}")


def run_one(
    benchmark: str,
    method: str,
    cases: list[SmokeCase],
    answer_url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_dir = Path(args.output_root) / benchmark / method
    config = _config(answer_url, args)
    record: dict[str, Any] = {
        "benchmark": benchmark,
        "method": method,
        "answer_base_url": answer_url,
        "embedding_base_url": args.embedding_base_url,
        "status": "failed",
        "phase": "init",
    }
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    started = time.time()
    try:
        if method == "HiveMem":
            record["phase"] = "build"
            _build_hivemem(cases, result_dir, answer_url, args)
            config["index_root"] = str(result_dir / "memory")
        for case in cases:
            adapter = None
            try:
                adapter = create_adapter(method, config_overrides=config)
                state_dir = result_dir / "memory" / "datasets" / case.sample_id
                adapter.reset(case.sample_id, state_dir)
                if method != "HiveMem":
                    record["phase"] = f"ingest:{case.name}"
                    adapter.ingest(case.chunk)
                    adapter.end_session(str(case.chunk.metadata.get("session_id") or case.name))
                record["phase"] = f"retrieve:{case.name}"
                query_vector = (
                    embed_texts([case.question], config)[0]
                    if method == "HiveMem"
                    else None
                )
                query_id = f"smoke:{benchmark}:{method}:{case.name}"
                retrieval = adapter.retrieve(
                    RetrievalRequest(
                        query_id=query_id,
                        text=case.question,
                        category=case.category,
                        top_k=args.top_k,
                        query_vector=query_vector,
                    )
                )
                memory_items = result_context_items(retrieval)
                if not memory_items:
                    raise RuntimeError(f"{case.name} retrieval returned no memories")
                record["phase"] = f"answer:{case.name}"
                client = VLMAnswerClient(
                    model=args.answer_model,
                    base_url=answer_url,
                    temperature=0.0,
                    num_predict=64,
                    timeout=args.timeout,
                    retries=args.retries,
                    think=False,
                )
                response = client.answer_with_usage(
                    system_prompt=SYSTEM_PROMPT,
                    memory_items=memory_items,
                    question_prompt=case.question,
                    category="VR",
                )
                if not response.text.strip():
                    raise RuntimeError(f"{case.name} answer model returned an empty answer")
                trace_rows = result_trace_rows(retrieval)
                results.append(
                    {
                        "query_id": query_id,
                        "benchmark": benchmark,
                        "method": method,
                        "variant": case.name,
                        "question": case.question,
                        "answer": response.text,
                        "retrieved_count": len(trace_rows),
                    }
                )
                traces.append(
                    {
                        "query_id": query_id,
                        "benchmark": benchmark,
                        "method": method,
                        "variant": case.name,
                        "retrieved": trace_rows,
                        "adapter_trace": retrieval.trace,
                    }
                )
                snapshots.extend(row.to_dict() for row in adapter.snapshot())
            finally:
                if adapter is not None:
                    adapter.close()
        record.update(
            {
                "status": "passed",
                "phase": "complete",
                "answers": len(results),
                "snapshot_count": len(snapshots),
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        record["seconds"] = time.time() - started
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(result_dir / "results.json", results)
        write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", traces)
        write_jsonl_atomic(result_dir / "memory" / "memory_snapshot.jsonl", snapshots)
        write_json_atomic(
            result_dir / "run_manifest.json",
            {
                **config,
                "benchmark": benchmark,
                "method": method,
                "smoke": True,
                "variants": [case.name for case in cases],
                "status": record["status"],
                "results": str(result_dir / "results.json"),
                "retrieval_trace": str(result_dir / "retrieval_trace.jsonl"),
                "memory_snapshot": str(result_dir / "memory" / "memory_snapshot.jsonl"),
                "native_memory_root": str(result_dir / "memory" / "datasets"),
            },
        )
        write_json_atomic(result_dir / "smoke_result.json", record)
        with _PRINT_LOCK:
            print(
                f"[{record['status'].upper():6}] {benchmark:14} {method:18} "
                f"phase={record['phase']} seconds={record['seconds']:.1f}",
                flush=True,
            )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answer-base-url",
        action="append",
        required=True,
        help="Repeat exactly three times; jobs are assigned round-robin.",
    )
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", default="outputs/_smoke")
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS, default=[])
    parser.add_argument("--baseline", action="append", choices=METHODS, default=[])
    args = parser.parse_args()
    check_endpoints(args.answer_base_url, args.embedding_base_url, args.embedding_model, args.embedding_dim)
    selected_benchmarks = tuple(args.benchmark) if args.benchmark else BENCHMARKS
    selected_methods = tuple(args.baseline) if args.baseline else METHODS
    source = fixtures()
    jobs = [
        (benchmark, method, source[benchmark], args.answer_base_url[index % 3])
        for index, (benchmark, method) in enumerate(
            (benchmark, method)
            for benchmark in selected_benchmarks
            for method in selected_methods
        )
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, benchmark, method, cases, answer_url, args): (benchmark, method)
            for benchmark, method, cases, answer_url in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (BENCHMARKS.index(row["benchmark"]), METHODS.index(row["method"])))
    passed = sum(row["status"] == "passed" for row in results)
    report = {"passed": passed, "total": len(results), "all_passed": passed == len(results), "results": results}
    write_json_atomic(Path(args.output_root) / "matrix.json", report)
    print(f"smoke matrix: {passed}/{len(results)} passed", flush=True)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
