#!/usr/bin/env python3
"""Re-answer a completed baseline run without rebuilding or retrieving again."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

OFFLINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFLINE_ROOT / "src"))

from benchmarks.h2hmem_harness.eval_h2hmem import answer_conversation_job
from benchmarks.h2hmem_harness.prompts import (
    PROMPT_SOURCE as H2_PROMPT_SOURCE,
    PROMPT_VERSION as H2_PROMPT_VERSION,
    prompt_sha256 as h2_prompt_sha256,
)
from benchmarks.io_utils import sha256_file, write_json_atomic, write_jsonl_atomic
from benchmarks.memgallery_harness.eval_memgallery import answer_dataset_job
from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient as MemGalleryAnswerClient,
)
from benchmarks.memgallery_harness.runner.metrics import (
    add_memory_metrics,
    add_retrieval_memory_tokens,
    summarize_results as summarize_memory_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.memgallery_harness.runner.prompts import prompt_manifest
from benchmarks.wma_harness.eval_wma import (
    answer_job as answer_wma_job,
    to_pipeline_qa_record,
)
from benchmarks.wma_harness.runner.answer_client import (
    VLMAnswerClient as WMAAnswerClient,
)
from benchmarks.wma_harness.runner.metrics import summarize_results as summarize_wma_results
from benchmarks.wma_harness.runner.prompts import (
    PROMPT_SOURCE as WMA_PROMPT_SOURCE,
    PROMPT_VERSION as WMA_PROMPT_VERSION,
    prompt_sha256 as wma_prompt_sha256,
)


BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena")
EXPECTED_QA = {"Mem-Gallery": 275, "H2HMEM": 360, "WorldMemArena": 440}
BENCHMARK_SLUG = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_metadata(benchmark: str) -> dict[str, Any]:
    if benchmark == "H2HMEM":
        return {
            "prompt_version": H2_PROMPT_VERSION,
            "prompt_source": H2_PROMPT_SOURCE,
            "prompt_sha256": h2_prompt_sha256(),
        }
    if benchmark == "WorldMemArena":
        return {
            "prompt_version": WMA_PROMPT_VERSION,
            "prompt_source": WMA_PROMPT_SOURCE,
            "prompt_sha256": wma_prompt_sha256(),
        }
    return prompt_manifest()


def prepared_jobs_path(source_dir: Path, benchmark: str) -> Path:
    if benchmark == "WorldMemArena":
        path = source_dir / ".checkpoint" / "prepared_qa.jsonl"
    else:
        path = source_dir / "pipeline_qa.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen prepared QA artifact not found: {path}")
    return path


def answer_components(
    benchmark: str,
) -> tuple[type[MemGalleryAnswerClient], Callable[..., tuple[dict, dict]]]:
    if benchmark == "WorldMemArena":
        return WMAAnswerClient, answer_wma_job
    if benchmark == "H2HMEM":
        return MemGalleryAnswerClient, answer_conversation_job

    def answer_memgallery(client, job):
        return answer_dataset_job(client, job, allow_answer_errors=False)

    return MemGalleryAnswerClient, answer_memgallery


def metric_rows(
    benchmark: str, results: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str, list[str]]:
    if benchmark == "H2HMEM":
        normalized = []
        for row in results:
            item = dict(row)
            item["_metric_sample_id"] = (
                f"{row.get('variant', '')}/{row.get('conversation_id', '')}"
            )
            normalized.append(item)
        sample_field = "_metric_sample_id"
    elif benchmark == "WorldMemArena":
        normalized = results
        sample_field = "sample_id"
    else:
        normalized = results
        sample_field = "dataset"
    sample_ids = sorted(
        {
            str(row.get(sample_field) or "").strip()
            for row in normalized
            if str(row.get(sample_field) or "").strip()
        }
    )
    return normalized, sample_field, sample_ids


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def write_frozen_nonanswer_trace(source_dir: Path, result_dir: Path) -> Path:
    source = source_dir / "call_trace.jsonl"
    target = result_dir / ".checkpoint" / "frozen_mb_retrieval_call_trace.jsonl"
    rows = []
    if source.is_file():
        rows = [
            row
            for row in read_jsonl(source)
            if str(row.get("phase") or "") in {"memory_build", "retrieval"}
        ]
    write_jsonl_atomic(target, rows)
    return target


def build_pipeline_rows(
    benchmark: str,
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if benchmark == "WorldMemArena":
        return [
            to_pipeline_qa_record(result, trace)
            for result, trace in zip(results, traces)
        ]
    if benchmark == "H2HMEM":
        output = []
        for job in jobs:
            row = dict(job)
            # This field in the reused 0907a artifact contains the retired
            # generic prompt and is not used by the new answer path.
            row.pop("question_prompt", None)
            row.update(prompt_metadata(benchmark))
            output.append(row)
        return output
    if benchmark == "Mem-Gallery":
        output = []
        for job in jobs:
            row = dict(job)
            row.pop("question_prompt", None)
            row.pop("system_prompt", None)
            row.update(prompt_metadata(benchmark))
            output.append(row)
        return output
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse a frozen baseline memory/retrieval artifact and rerun QA only."
    )
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--answer-base-url", required=True)
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--efficiency-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "model_efficiency.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    if source_dir == result_dir:
        parser.error("--result-dir must differ from --source-dir")
    source_results = source_dir / "results.json"
    source_trace = source_dir / "retrieval_trace.jsonl"
    source_manifest_path = source_dir / "run_manifest.json"
    for path in (source_results, source_trace, source_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact not found: {path}")
    jobs_path = prepared_jobs_path(source_dir, args.benchmark)
    jobs = read_jsonl(jobs_path)
    frozen_traces = read_jsonl(source_trace)
    old_results = json.loads(source_results.read_text(encoding="utf-8"))
    if not isinstance(old_results, list):
        raise TypeError(f"Expected a JSON list: {source_results}")
    if not (len(jobs) == len(frozen_traces) == len(old_results)):
        raise RuntimeError(
            "Frozen QA artifacts disagree in length: "
            f"jobs={len(jobs)}, traces={len(frozen_traces)}, results={len(old_results)}"
        )
    expected = EXPECTED_QA[args.benchmark]
    if len(jobs) != expected:
        raise RuntimeError(
            f"{args.benchmark}/{args.baseline}: expected {expected} frozen questions, "
            f"found {len(jobs)}"
        )
    job_ids = [str(job.get("query_id") or job.get("uid") or "") for job in jobs]
    frozen_ids = [str(row.get("query_id") or "") for row in frozen_traces]
    old_result_ids = [str(row.get("query_id") or row.get("uid") or "") for row in old_results]
    if job_ids != frozen_ids or job_ids != old_result_ids or not all(job_ids):
        raise RuntimeError("Frozen job/result/retrieval IDs or order do not match")
    if args.limit:
        jobs = jobs[: args.limit]
        frozen_traces = frozen_traces[: args.limit]
        job_ids = job_ids[: args.limit]

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = result_dir / ".checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "mode": "qa_from_frozen_retrieval_v1",
        "benchmark": args.benchmark,
        "baseline": args.baseline,
        "source_dir": str(source_dir),
        "jobs_sha256": sha256_file(jobs_path),
        "retrieval_trace_sha256": sha256_file(source_trace),
        "prompt": prompt_metadata(args.benchmark),
        "answer_model": args.answer_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "limit": args.limit,
    }
    checkpoint_manifest = checkpoint_dir / "qa_replay_manifest.json"
    checkpoint_results = checkpoint_dir / "results.json"
    checkpoint_traces = checkpoint_dir / "retrieval_trace.jsonl"
    results_by_id: dict[str, dict[str, Any]] = {}
    traces_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_manifest.is_file():
        saved = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError(
                f"QA replay checkpoint signature mismatch: {checkpoint_manifest}"
            )
        if checkpoint_results.is_file():
            results_by_id = {
                str(row.get("query_id") or row.get("uid") or ""): row
                for row in json.loads(checkpoint_results.read_text(encoding="utf-8"))
            }
        if checkpoint_traces.is_file():
            traces_by_id = {
                str(row.get("query_id") or ""): row
                for row in read_jsonl(checkpoint_traces)
            }

    def save_checkpoint() -> None:
        completed = [query_id for query_id in job_ids if query_id in results_by_id]
        write_json_atomic(
            checkpoint_results, [results_by_id[query_id] for query_id in completed]
        )
        write_jsonl_atomic(
            checkpoint_traces,
            [traces_by_id[query_id] for query_id in completed if query_id in traces_by_id],
        )
        write_json_atomic(
            checkpoint_manifest,
            {
                "signature": signature,
                "completed": len(completed),
                "expected": len(job_ids),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    client_cls, answer_function = answer_components(args.benchmark)
    client = client_cls(
        model=args.answer_model,
        base_url=args.answer_base_url,
        temperature=args.temperature,
        num_predict=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        think=False,
    )
    pending = [
        job
        for job in jobs
        if str(job.get("query_id") or job.get("uid") or "") not in results_by_id
        or results_by_id[str(job.get("query_id") or job.get("uid") or "")].get("error")
    ]
    if pending:
        client.assert_model_available()
    failures: list[str] = []
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(answer_function, client, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            query_id = str(job.get("query_id") or job.get("uid") or "")
            try:
                result, trace = future.result()
            except Exception as exc:
                failures.append(f"{query_id}: {type(exc).__name__}: {exc}")
                print(f"[failed] {failures[-1]}", flush=True)
                continue
            result["query_id"] = query_id
            trace["query_id"] = query_id
            results_by_id[query_id] = result
            traces_by_id[query_id] = trace
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0
            print(
                f"[{len(results_by_id)}/{len(job_ids)}] "
                f"{args.benchmark}/{args.baseline} {query_id} "
                f"error={str(result.get('error') or '')[:100]!r}",
                flush=True,
            )
    save_checkpoint()
    if failures:
        raise RuntimeError(f"{len(failures)} answer task(s) failed; first={failures[0]}")
    missing = [query_id for query_id in job_ids if query_id not in results_by_id]
    errored = [
        query_id for query_id in job_ids if results_by_id.get(query_id, {}).get("error")
    ]
    if missing or errored:
        raise RuntimeError(
            f"QA replay incomplete: missing={len(missing)}, answer_errors={len(errored)}"
        )

    results = [results_by_id[query_id] for query_id in job_ids]
    traces = [traces_by_id[query_id] for query_id in job_ids]
    write_json_atomic(result_dir / "results.json", results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", traces)
    write_jsonl_atomic(
        result_dir / "pipeline_qa.jsonl",
        build_pipeline_rows(args.benchmark, jobs, results, traces),
    )
    copy_if_present(
        source_dir / "memory" / "memory_snapshot.jsonl",
        result_dir / "memory" / "memory_snapshot.jsonl",
    )
    for filename in ("memory_metrics.json", "retrieval_memory_token.json"):
        copy_if_present(source_dir / filename, result_dir / filename)

    normalized_results, sample_field, sample_ids = metric_rows(args.benchmark, results)
    frozen_call_trace = write_frozen_nonanswer_trace(source_dir, result_dir)
    calls = write_runtime_call_metrics(
        [frozen_call_trace],
        result_dir,
        normalized_results,
        sample_id_field=sample_field,
        sample_ids=sample_ids,
    )
    summary = (
        summarize_wma_results(results, k=args.top_k)
        if args.benchmark == "WorldMemArena"
        else summarize_memory_results(results, k=args.top_k)
    )
    memory_metrics_path = result_dir / "memory_metrics.json"
    if memory_metrics_path.is_file():
        summary = add_memory_metrics(
            summary,
            json.loads(memory_metrics_path.read_text(encoding="utf-8")),
        )
    retrieval_metrics_path = result_dir / "retrieval_memory_token.json"
    if retrieval_metrics_path.is_file():
        summary = add_retrieval_memory_tokens(
            summary,
            json.loads(retrieval_metrics_path.read_text(encoding="utf-8")),
        )
    summary["calls"] = calls
    efficiency = write_efficiency_metrics(
        result_dir,
        normalized_results,
        sample_id_field=sample_field,
        sample_ids=sample_ids,
        model=args.answer_model,
        config_path=args.efficiency_config,
    )
    for key in (
        "cost_mb",
        "cost_qa",
        "cost_total",
        "latency_mb",
        "latency_qa",
        "latency_total",
    ):
        summary[key] = efficiency[key]
    write_json_atomic(result_dir / "metrics.json", summary)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **source_manifest,
        "benchmark": args.benchmark,
        "baseline_name": args.baseline,
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "answer_temperature": args.temperature,
        "num_predict": args.max_tokens,
        "top_k": args.top_k,
        "questions": len(results),
        "completed": len(results),
        "answer_errors": 0,
        "memory_snapshot": str(result_dir / "memory" / "memory_snapshot.jsonl"),
        "execution_mode": "qa_from_frozen_memory_and_retrieval",
        "reused_memory_bank_from": str(source_dir),
        "reused_retrieval_trace_from": str(source_trace),
        "frozen_retrieval_trace_sha256": sha256_file(source_trace),
        "run_signature": signature,
        **prompt_metadata(args.benchmark),
    }
    write_json_atomic(result_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "benchmark": args.benchmark,
                "baseline": args.baseline,
                "questions": len(results),
                "result_dir": str(result_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
