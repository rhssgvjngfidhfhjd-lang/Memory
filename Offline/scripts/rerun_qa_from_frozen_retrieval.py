#!/usr/bin/env python3
"""Re-answer a completed baseline run without rebuilding or retrieving again."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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
    _cost_from_inference_aggregate,
    _latency_from_inference_aggregate,
    _sum_modeled_latencies,
    _sum_priced_costs,
    add_memory_metrics,
    add_retrieval_memory_tokens,
    merge_existing_llm_judge_metrics,
    summarize_results as summarize_memory_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.memgallery_harness.runner.prompts import prompt_manifest
from benchmarks.memeye_harness.prompts import prompt_manifest as memeye_prompt_manifest
from benchmarks.memlens_harness.prompts import prompt_manifest as memlens_prompt_manifest
from benchmarks.multimodal_dataset_harness.runner import (
    _answer_job as answer_multimodal_job,
)
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


BENCHMARKS = ("Mem-Gallery", "H2HMEM", "WorldMemArena", "MemEye", "MEMLENS")
EXPECTED_QA = {
    "Mem-Gallery": 275,
    "H2HMEM": 360,
    "WorldMemArena": 440,
    "MemEye": 371,
    "MEMLENS": 173,
}
BENCHMARK_SLUG = {
    "Mem-Gallery": "memgallery",
    "H2HMEM": "h2hmem",
    "WorldMemArena": "worldmemarena",
    "MemEye": "memeye",
    "MEMLENS": "memlens",
}
ISOLATED_QA_VERSION = "isolated-final-answer-v1"
STRICT_SHORT_ANSWER_INSTRUCTION = """Return the shortest direct answer only.
Do not explain or provide supporting context.
For every yes/no question, output exactly "Yes." or "No."
For entity, date, number, location, or image-ID questions,
return only the requested value."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_metadata(
    benchmark: str, *, strict_short_answer: bool = False
) -> dict[str, Any]:
    if benchmark == "MemEye":
        return memeye_prompt_manifest()
    if benchmark == "MEMLENS":
        return memlens_prompt_manifest()
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
    metadata = prompt_manifest()
    if not strict_short_answer:
        return metadata
    payload = {
        "base_prompt_sha256": metadata["prompt_sha256"],
        "strict_instruction": STRICT_SHORT_ANSWER_INSTRUCTION,
        "answer_tag_scope": "text_inside_required_answer_block",
    }
    return {
        "prompt_version": f"{metadata['prompt_version']}+strict-short-answer-v1",
        "prompt_source": f"{metadata['prompt_source']}+{Path(__file__).name}",
        "prompt_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "base_prompt_sha256": metadata["prompt_sha256"],
        "strict_answer_instruction": STRICT_SHORT_ANSWER_INSTRUCTION,
    }


def transform_answer_messages(
    messages: list[dict[str, Any]], *, strict_short_answer: bool
) -> list[dict[str, Any]]:
    transformed = [dict(message) for message in messages]
    if not strict_short_answer:
        return transformed
    instruction = (
        "For the text inside the required <answer>...</answer> block:\n"
        + STRICT_SHORT_ANSWER_INSTRUCTION
    )
    if transformed and transformed[0].get("role") == "system":
        transformed[0]["content"] = (
            f"{transformed[0].get('content', '')}\n\n{instruction}"
        )
    user_indices = [
        index
        for index, message in enumerate(transformed)
        if message.get("role") == "user"
    ]
    if not user_indices:
        raise ValueError("Isolated QA prompt has no user message")
    user_index = user_indices[-1]
    transformed[user_index]["content"] = (
        f"{transformed[user_index].get('content', '')}"
        f"\n\nAnswer-style requirement:\n{instruction}"
    )
    return transformed


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
    *,
    isolate_final_answer: bool = False,
    strict_short_answer: bool = False,
    attach_retrieved_images: bool = False,
) -> tuple[type[MemGalleryAnswerClient], Callable[..., tuple[dict, dict]]]:
    if benchmark == "WorldMemArena":
        client_cls = WMAAnswerClient
        answer_function = answer_wma_job
    elif benchmark == "H2HMEM":
        client_cls = MemGalleryAnswerClient
        answer_function = answer_conversation_job
    elif benchmark in {"MemEye", "MEMLENS"}:
        client_cls = MemGalleryAnswerClient

        def answer_multimodal(client, job):
            return answer_multimodal_job(
                client,
                job,
                benchmark=benchmark,
                allow_answer_errors=False,
                attach_retrieved_images=attach_retrieved_images,
            )

        answer_function = answer_multimodal
    else:
        client_cls = MemGalleryAnswerClient

        def answer_memgallery(client, job):
            return answer_dataset_job(client, job, allow_answer_errors=False)

        answer_function = answer_memgallery

    if not isolate_final_answer and not strict_short_answer:
        return client_cls, answer_function

    def answer_isolated(client, job):
        isolated_job = dict(job)
        if isolate_final_answer:
            isolated_job.pop("native_answer", None)
        result, trace = answer_function(client, isolated_job)
        if strict_short_answer and trace.get("answer_prompt_messages"):
            trace["answer_prompt_messages"] = transform_answer_messages(
                trace["answer_prompt_messages"], strict_short_answer=True
            )
        if isolate_final_answer:
            result["native_answer_trace"] = {}
        return result, trace

    return client_cls, answer_isolated


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
    elif benchmark in {"WorldMemArena", "MemEye", "MEMLENS"}:
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


def preserve_frozen_memory_efficiency(
    source_dir: Path,
    efficiency: dict[str, Any],
) -> dict[str, Any]:
    """Keep build cost/latency on the model profile that built the bank.

    A frozen-memory replay may use a different final-answer model.  Repricing
    the copied memory-build trace with that answer model's profile makes mixed
    runs (for example, Qwen3.5-9B build + Qwen3-VL-4B answer) incorrect.
    """
    source_path = source_dir / "efficiency_metrics.json"
    if not source_path.is_file():
        source_path = source_dir / "metrics.json"
    if not source_path.is_file():
        return efficiency

    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not all(isinstance(source.get(key), dict) for key in ("cost_mb", "latency_mb")):
        return efficiency

    combined = dict(efficiency)
    combined["cost_mb"] = dict(source["cost_mb"])
    combined["latency_mb"] = dict(source["latency_mb"])

    source_components = source.get("components") or {}
    source_profiles = source.get("profiles") or {}
    source_profile = source.get("profile") or {}
    source_component_efficiency = source.get("component_efficiency") or {}
    current_component_efficiency = efficiency.get("component_efficiency") or {}
    source_retrieval = source_component_efficiency.get("retrieval") or {}
    if not source_retrieval and isinstance(source_components.get("retrieval"), dict):
        retrieval_profile = source_profiles.get("retrieval") or source_profile
        if retrieval_profile:
            source_retrieval = {
                "model": retrieval_profile.get("model"),
                "cost": _cost_from_inference_aggregate(
                    source_components["retrieval"],
                    retrieval_profile,
                    aggregation="sum_retrieval_cost_divided_by_samples",
                ),
                "latency": _latency_from_inference_aggregate(
                    source_components["retrieval"],
                    retrieval_profile,
                    aggregation="sum_retrieval_latency_divided_by_samples",
                ),
            }
    current_answer = current_component_efficiency.get("answer") or {}
    if source_retrieval and current_answer:
        component_efficiency = dict(current_component_efficiency)
        component_efficiency["retrieval"] = source_retrieval
        combined["component_efficiency"] = component_efficiency

        current_profiles = dict(efficiency.get("profiles") or {})
        if source_profiles.get("retrieval"):
            current_profiles["retrieval"] = source_profiles["retrieval"]
        elif source_profile:
            current_profiles["retrieval"] = source_profile
        combined["profiles"] = current_profiles

        combined["cost_qa"] = _sum_priced_costs(
            [source_retrieval["cost"], current_answer["cost"]],
            num_samples=int(efficiency["cost_qa"].get("num_samples") or 0),
            aggregation="sum_retrieval_answer_cost_divided_by_samples",
            source="frozen_retrieval_plus_replayed_answer",
        )
        combined["latency_qa"] = _sum_modeled_latencies(
            [source_retrieval["latency"], current_answer["latency"]],
            num_samples=int(efficiency["latency_qa"].get("num_samples") or 0),
            denominator_unit=str(
                efficiency["latency_qa"].get("denominator_unit") or "QA"
            ),
            aggregation="sum_retrieval_answer_latency_divided_by_queries",
            source="frozen_retrieval_plus_replayed_answer",
        )

    cost_mb = combined["cost_mb"]
    cost_qa = combined["cost_qa"]
    if cost_mb.get("available") and cost_qa.get("available"):
        cost_sum = float(cost_mb["cost_sum_usd"]) + float(cost_qa["cost_sum_usd"])
        num_samples = int(cost_mb["num_samples"])
        mean = cost_sum / num_samples
        combined["cost_total"] = {
            "input_tokens": int(cost_mb["input_tokens"]) + int(cost_qa["input_tokens"]),
            "output_tokens": int(cost_mb["output_tokens"]) + int(cost_qa["output_tokens"]),
            "cost_sum_usd": cost_sum,
            "cost_sum": cost_sum,
            "num_samples": num_samples,
            "mean_per_sample_usd": mean,
            "mean_per_sample": mean,
            "formula": (
                f"({cost_mb['cost_sum_usd']:.12g} + {cost_qa['cost_sum_usd']:.12g}) "
                f"/ {num_samples} = {mean:.12g} USD/sample"
            ),
            "aggregation": "sum_phase_priced_total_cost_divided_by_samples",
            "pricing": "per_phase_model_profiles",
            "available": True,
        }

    latency_mb = combined["latency_mb"]
    latency_qa = combined["latency_qa"]
    if latency_mb.get("available") and latency_qa.get("available"):
        latency_sum = float(latency_mb["latency_sum_seconds"]) + float(
            latency_qa["latency_sum_seconds"]
        )
        num_samples = int(latency_mb["num_samples"])
        mean = latency_sum / num_samples
        combined["latency_total"] = {
            "latency_sum_seconds": latency_sum,
            "num_samples": num_samples,
            "denominator_unit": "sample",
            "mean_per_sample_seconds": mean,
            "formula": (
                f"({latency_mb['latency_sum_seconds']:.12g} + "
                f"{latency_qa['latency_sum_seconds']:.12g}) / {num_samples} "
                f"= {mean:.12g} seconds/sample"
            ),
            "aggregation": "sum_phase_modeled_latency_divided_by_samples",
            "profiles": "per_phase_model_profiles",
            "available": True,
        }
    return combined


def write_frozen_nonanswer_trace(source_dir: Path, result_dir: Path) -> Path:
    source = source_dir / "call_trace.jsonl"
    target = result_dir / ".checkpoint" / "frozen_mb_retrieval_call_trace.jsonl"
    rows = []
    if source.is_file():
        source_rows = read_jsonl(source)
        rows = [
            row
            for row in source_rows
            if str(row.get("phase") or "") in {"memory_build", "retrieval"}
        ]
        manifest_path = source_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        if str(manifest.get("baseline") or "") == "M3-Agent-caption":
            retrieval_rows = [
                row for row in source_rows
                if str(row.get("phase") or "") == "retrieval"
            ]
            if not retrieval_rows:
                raise RuntimeError(
                    "M3 frozen replay has no phase=retrieval calls; refusing to "
                    "silently discard potentially mislabelled Control-Agent calls."
                )
            results_path = source_dir / "results.json"
            if results_path.is_file():
                results = json.loads(results_path.read_text(encoding="utf-8"))
                expected_answer_calls = sum(
                    int(row.get("answer_attempts") or 0) for row in results
                )
                traced_answer_calls = sum(
                    str(row.get("phase") or "") == "qa" for row in source_rows
                )
                if traced_answer_calls != expected_answer_calls:
                    raise RuntimeError(
                        "M3 phase audit failed before isolated replay: "
                        f"phase=qa calls={traced_answer_calls}, "
                        f"final-answer attempts={expected_answer_calls}. "
                        "Retrieval/Control calls may be mislabelled as qa."
                    )
    write_jsonl_atomic(target, rows)
    return target


def build_pipeline_rows(
    benchmark: str,
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    isolate_final_answer: bool = False,
    prompt_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prompt_info = prompt_info or prompt_metadata(benchmark)
    if benchmark == "WorldMemArena":
        output = [
            to_pipeline_qa_record(result, trace)
            for result, trace in zip(results, traces)
        ]
        if isolate_final_answer:
            for row in output:
                row.pop("native_answer", None)
        return output
    if benchmark in {"H2HMEM", "MemEye", "MEMLENS"}:
        output = []
        for job in jobs:
            row = dict(job)
            # This field in the reused 0907a artifact contains the retired
            # generic prompt and is not used by the new answer path.
            row.pop("question_prompt", None)
            if isolate_final_answer:
                row.pop("native_answer", None)
            row.update(prompt_info)
            output.append(row)
        return output
    if benchmark == "Mem-Gallery":
        output = []
        for job in jobs:
            row = dict(job)
            row.pop("question_prompt", None)
            row.pop("system_prompt", None)
            if isolate_final_answer:
                row.pop("native_answer", None)
            row.update(prompt_info)
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
    parser.add_argument("--answer-api-key-file", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="")
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
    parser.add_argument(
        "--isolate-final-answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Discard any baseline-native answer and answer only from the frozen "
            "retrieved memories with the shared benchmark answer client."
        ),
    )
    parser.add_argument(
        "--strict-short-answer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the benchmark-independent shortest-direct-answer contract.",
    )
    parser.add_argument(
        "--openrouter-json-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use OpenRouter JSON Schema output from the first attempt.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.strict_short_answer and args.benchmark != "Mem-Gallery":
        parser.error("--strict-short-answer is currently validated for Mem-Gallery only")
    if args.strict_short_answer and not args.isolate_final_answer:
        parser.error("--strict-short-answer requires --isolate-final-answer")

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
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    attach_retrieved_images = bool(source_manifest.get("attach_retrieved_images", False))
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
    effective_prompt = prompt_metadata(
        args.benchmark, strict_short_answer=args.strict_short_answer
    )
    signature = {
        "mode": (
            ISOLATED_QA_VERSION
            if args.isolate_final_answer
            else "qa_from_frozen_retrieval_v1"
        ),
        "benchmark": args.benchmark,
        "baseline": args.baseline,
        "source_dir": str(source_dir),
        "jobs_sha256": sha256_file(jobs_path),
        "retrieval_trace_sha256": sha256_file(source_trace),
        "prompt": effective_prompt,
        "answer_model": args.answer_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "isolate_final_answer": args.isolate_final_answer,
        "strict_short_answer": args.strict_short_answer,
        "openrouter_json_first": args.openrouter_json_first,
        "attach_retrieved_images": attach_retrieved_images,
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

    client_cls, answer_function = answer_components(
        args.benchmark,
        isolate_final_answer=args.isolate_final_answer,
        strict_short_answer=args.strict_short_answer,
        attach_retrieved_images=attach_retrieved_images,
    )
    answer_api_key = "EMPTY"
    if args.answer_api_key_file:
        answer_api_key = (
            args.answer_api_key_file.expanduser().resolve().read_text(encoding="utf-8").strip()
        )
        if not answer_api_key:
            raise RuntimeError(f"Empty answer API key file: {args.answer_api_key_file}")
    client = client_cls(
        model=args.answer_model,
        base_url=args.answer_base_url,
        api_key=answer_api_key,
        temperature=args.temperature,
        num_predict=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        think=False,
        reasoning_effort=args.reasoning_effort,
    )
    if args.strict_short_answer:
        original_answer_messages = client.answer_messages_with_usage

        def answer_messages_with_strict_prompt(**kwargs):
            kwargs["messages"] = transform_answer_messages(
                kwargs["messages"], strict_short_answer=True
            )
            return original_answer_messages(**kwargs)

        client.answer_messages_with_usage = answer_messages_with_strict_prompt
    if args.openrouter_json_first:
        original_prebuilt_transport = client._answer_prebuilt_openai_compatible

        def answer_prebuilt_with_json_schema(**kwargs):
            kwargs["structured_json"] = True
            return original_prebuilt_transport(**kwargs)

        client._answer_prebuilt_openai_compatible = answer_prebuilt_with_json_schema
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
        build_pipeline_rows(
            args.benchmark,
            jobs,
            results,
            traces,
            isolate_final_answer=args.isolate_final_answer,
            prompt_info=effective_prompt,
        ),
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
    efficiency = preserve_frozen_memory_efficiency(source_dir, efficiency)
    write_json_atomic(result_dir / "efficiency_metrics.json", efficiency)
    for key in (
        "cost_mb",
        "cost_qa",
        "cost_total",
        "latency_mb",
        "latency_qa",
        "latency_total",
    ):
        summary[key] = efficiency[key]
    summary = merge_existing_llm_judge_metrics(summary, result_dir)
    write_json_atomic(result_dir / "metrics.json", summary)

    manifest = {
        **source_manifest,
        "benchmark": args.benchmark,
        "baseline_name": args.baseline,
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "answer_temperature": args.temperature,
        "num_predict": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "top_k": args.top_k,
        "questions": len(results),
        "completed": len(results),
        "answer_errors": 0,
        "memory_snapshot": str(result_dir / "memory" / "memory_snapshot.jsonl"),
        "execution_mode": (
            "isolated_qa_from_frozen_memory_and_retrieval"
            if args.isolate_final_answer
            else "qa_from_frozen_memory_and_retrieval"
        ),
        "final_answer_isolated_from_baseline_agent": args.isolate_final_answer,
        "native_answer_reused": not args.isolate_final_answer,
        "strict_short_answer": args.strict_short_answer,
        "openrouter_json_first": args.openrouter_json_first,
        "reused_memory_bank_from": str(source_dir),
        "reused_retrieval_trace_from": str(source_trace),
        "frozen_retrieval_trace_sha256": sha256_file(source_trace),
        "run_signature": signature,
        **effective_prompt,
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
