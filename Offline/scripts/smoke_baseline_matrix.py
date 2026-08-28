"""Run one real ingest/retrieve/answer cycle for every benchmark/baseline pair."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
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
from benchmarks.baseline_runtime.protocol import (  # noqa: E402
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import write_json_atomic  # noqa: E402
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient  # noqa: E402
from embedding.chunk_builder import (  # noqa: E402
    build_chunks_from_data,
    build_h2h_chunks_from_data,
    build_wma_chunks_from_data,
    iter_h2h_session_files,
    iter_wma_sample_files,
)


BASELINES = tuple(name for name in BASELINE_NAMES if name != "HiveMem")
BENCHMARKS = ("Mem-Gallery", "WorldMemArena", "H2HMEM")
SYSTEM_PROMPT = (
    "Answer using only the retrieved memory. Return only a concise answer; "
    "say unknown when the memory does not contain the answer."
)
_PRINT_LOCK = threading.Lock()


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def check_endpoints(answer_urls: list[str], embedding_url: str, embedding_model: str, dim: int) -> None:
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


def _first_memgallery() -> tuple[Any, str, str]:
    data_dir = WORKSPACE / "Mem-Gallery" / "benchmark" / "data"
    paths = sorted(path for path in (data_dir / "dialog").glob("*.json") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No Mem-Gallery data under {data_dir}")
    path = paths[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = build_chunks_from_data(payload, data_dir, path.stem)
    qa = (payload.get("human-annotated QAs") or [])[0]
    return chunks[0], str(qa.get("question") or ""), str(qa.get("point") or "")


def _first_wma() -> tuple[Any, str, str]:
    data_dir = WORKSPACE / "WorldMemArena" / "WorldMemArena" / "lifelong"
    path = iter_wma_sample_files(data_dir)[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunk = build_wma_chunks_from_data(payload, path.parent, sample_path=path)[0]
    qa = next(
        qa
        for checkpoint in payload.get("qa_checkpoints") or []
        for qa in checkpoint.get("questions") or []
    )
    return chunk, str(qa.get("question") or ""), str(qa.get("question_type_abbrev") or "")


def _first_h2hmem() -> tuple[Any, str, str]:
    data_dir = WORKSPACE / "H2HMEM-main" / "dataset"
    path = iter_h2h_session_files(data_dir, variant="dyadic")[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    conversation_id = path.parents[2].name
    chunk = build_h2h_chunks_from_data(
        payload,
        session_path=path,
        variant="dyadic",
        conversation_id=conversation_id,
    )[0]
    question_path = path.with_name("questions.json")
    question = (json.loads(question_path.read_text(encoding="utf-8")).get("questions") or [])[0]
    question_type = question.get("question_type") or {}
    return (
        chunk,
        str((question.get("question") or {}).get("text") or ""),
        str(question_type.get("sub_type") or question_type.get("main_type") or ""),
    )


def fixtures() -> dict[str, tuple[Any, str, str]]:
    return {
        "Mem-Gallery": _first_memgallery(),
        "WorldMemArena": _first_wma(),
        "H2HMEM": _first_h2hmem(),
    }


def run_one(
    benchmark: str,
    baseline: str,
    fixture: tuple[Any, str, str],
    answer_url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    chunk, question, category = fixture
    result_dir = Path(args.output_root) / benchmark / baseline
    state_dir = result_dir / "memory" / "datasets" / "smoke"
    config = {
        "answer_model": args.answer_model,
        "answer_base_url": answer_url,
        "executor_model": args.answer_model,
        "executor_base_url": answer_url,
        "executor_temperature": 0.0,
        "executor_visual_input": "image",
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_dim": args.embedding_dim,
        "top_k": args.top_k,
        "request_timeout": args.timeout,
        "retries": 0,
    }
    record: dict[str, Any] = {
        "benchmark": benchmark,
        "baseline": baseline,
        "answer_base_url": answer_url,
        "embedding_base_url": args.embedding_base_url,
        "status": "failed",
        "phase": "init",
    }
    adapter = None
    started = time.time()
    try:
        adapter = create_adapter(baseline, config_overrides=config)
        adapter.reset(f"smoke-{benchmark}", state_dir)
        record["phase"] = "ingest"
        adapter.ingest(chunk)
        adapter.end_session(str(chunk.metadata.get("session_id") or "smoke"))
        record["phase"] = "retrieve"
        retrieval = adapter.retrieve(
            RetrievalRequest(
                query_id=f"smoke:{benchmark}:{baseline}",
                text=question,
                category=category,
                top_k=args.top_k,
            )
        )
        memory_items = result_context_items(retrieval)
        if not memory_items:
            raise RuntimeError("retrieval returned no memories")
        record["phase"] = "answer"
        client = VLMAnswerClient(
            model=args.answer_model,
            base_url=answer_url,
            temperature=0.0,
            num_predict=64,
            timeout=args.timeout,
            retries=0,
            think=False,
        )
        response = client.answer_with_usage(
            system_prompt=SYSTEM_PROMPT,
            memory_items=memory_items,
            question_prompt=question,
            category="VR",
        )
        if not response.text.strip():
            raise RuntimeError("answer model returned an empty answer")
        record.update(
            {
                "status": "passed",
                "phase": "complete",
                "answer": response.text,
                "retrieved": result_trace_rows(retrieval),
                "snapshot_count": len(adapter.snapshot()),
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                record.setdefault("close_error", f"{type(exc).__name__}: {exc}")
        record["seconds"] = time.time() - started
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(result_dir / "smoke_result.json", record)
        with _PRINT_LOCK:
            print(
                f"[{record['status'].upper():6}] {benchmark:14} {baseline:18} "
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
        help="Repeat four times; jobs are assigned round-robin.",
    )
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", default="outputs/_smoke")
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=BENCHMARKS,
        default=[],
        help="Limit the matrix; repeat for multiple benchmarks.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=BASELINES,
        default=[],
        help="Limit the matrix; repeat for multiple baselines.",
    )
    args = parser.parse_args()
    check_endpoints(
        args.answer_base_url,
        args.embedding_base_url,
        args.embedding_model,
        args.embedding_dim,
    )
    selected_benchmarks = tuple(args.benchmark) if args.benchmark else BENCHMARKS
    selected_baselines = tuple(args.baseline) if args.baseline else BASELINES
    source = fixtures()
    jobs = [
        (benchmark, baseline, source[benchmark], args.answer_base_url[index % len(args.answer_base_url)])
        for index, (benchmark, baseline) in enumerate(
            (benchmark, baseline)
            for benchmark in selected_benchmarks
            for baseline in selected_baselines
        )
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, benchmark, baseline, fixture, answer_url, args): (benchmark, baseline)
            for benchmark, baseline, fixture, answer_url in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (BENCHMARKS.index(row["benchmark"]), BASELINES.index(row["baseline"])))
    passed = sum(row["status"] == "passed" for row in results)
    report = {"passed": passed, "total": len(results), "all_passed": passed == len(results), "results": results}
    write_json_atomic(Path(args.output_root) / "matrix.json", report)
    print(f"smoke matrix: {passed}/{len(results)} passed", flush=True)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
