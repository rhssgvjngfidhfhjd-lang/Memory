"""Mem-Gallery-style execution and artifacts for newly added datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from types import ModuleType
from typing import Any, Iterable

from benchmarks.baseline_runtime import (
    NativeAnswerRequest,
    RetrievalResult,
    baseline_metadata,
    canonical_name,
    create_adapter,
)
from benchmarks.baseline_runtime.adapters.m3_agent import m3_conformance_manifest
from benchmarks.baseline_runtime.build_fault_policy import ConsecutiveBuildFaultPolicy
from benchmarks.baseline_runtime.call_trace import (
    CallRecorder,
    CountingProxy,
    TRACE_VERSION,
    trace_filename,
)
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.m3_inputs import (
    build_m3_chunks_from_round_chunks,
    m3_input_manifest,
)
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
)
from benchmarks.baseline_runtime.protocol import (
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.memgallery_harness.runner.metrics import (
    add_memory_metrics,
    merge_existing_llm_judge_metrics,
    summarize_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
    write_snapshot_memory_metrics,
)
from embedding.chunk_builder import Chunk
from benchmarks.zero_hit import evidence_with_zero_hit_marker


SUPPORTED_BASELINES = frozenset(
    {"MemVerse", "M3-Agent-caption", "MIRIX", "M2A"}
)


@dataclass
class HarnessQuestion:
    question_id: str
    question: str
    answer: str
    category: str
    clue_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    visible_session_ids: list[str] = field(default_factory=list)
    query_image: dict[str, str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessSample:
    sample_id: str
    source_name: str
    source_path: Path
    chunks: list[Chunk]
    questions: list[HarnessQuestion]


@lru_cache(maxsize=2)
def _prompt_module(benchmark: str) -> ModuleType:
    normalized = str(benchmark or "").strip().casefold()
    if normalized == "memeye":
        from benchmarks.memeye_harness import prompts

        return prompts
    if normalized == "memlens":
        from benchmarks.memlens_harness import prompts

        return prompts
    raise ValueError(f"no QA prompt module registered for benchmark {benchmark!r}")


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_data_dir: Path,
) -> None:
    parser.add_argument("--baseline", default="MemVerse")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument(
        "--m3-reuse-state-root",
        default="",
        help=(
            "Read an existing M3 per-sample memory graph from this root while "
            "writing retrieval traces and conformance artifacts to the new run."
        ),
    )
    parser.add_argument("--sample-concurrency", type=int, default=1)
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--m3-control-rounds",
        type=int,
        default=5,
        help="M3 Control rounds; 1-5, where 5 is the official default.",
    )
    parser.add_argument(
        "--m3-search-top-k",
        type=int,
        default=2,
        help="M3 ordinary per-round search width; 1-2, where 2 is the official default.",
    )
    parser.add_argument(
        "--m3-retrieval-threshold",
        type=float,
        default=0.5,
        help="M3 ordinary-search similarity threshold in [0, 1]; official default is 0.5.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset mapping and images without creating an adapter or calling a model.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-answer-errors", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument(
        "--attach-retrieved-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Attach retrieved source images to final QA. The default is enabled for "
            "M3-Agent and disabled for MemVerse, M2A, and MIRIX; either choice can be explicit."
        ),
    )
    parser.add_argument(
        "--answer-base-url",
        default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1",
    )
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default=os.getenv("OPENAI_API_KEY") or "EMPTY")
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--executor-temperature", type=float, default=0.0)
    parser.add_argument("--executor-max-tokens", type=int, default=512)
    parser.add_argument(
        "--m2a-salvage-truncated-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--m2a-skip-failed-build-points",
        action="store_true",
        help="Audit and skip every failed M2A build point, then continue.",
    )
    parser.add_argument(
        "--m2a-max-consecutive-failed-build-points",
        type=int,
        default=1_000_000_000,
    )
    parser.add_argument(
        "--mirix-skip-failed-build-points",
        action="store_true",
        help=(
            "Reject and audit an isolated failed MIRIX build point, then continue "
            "unless the consecutive-failure limit is reached."
        ),
    )
    parser.add_argument(
        "--mirix-max-consecutive-failed-build-points",
        type=int,
        default=10,
    )
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--cost-mb-input-price", type=float, default=None)
    parser.add_argument("--cost-mb-output-price", type=float, default=None)
    parser.add_argument("--efficiency-config", default="configs/model_efficiency.json")


def validate_common_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        args.baseline = canonical_name(args.baseline)
    except KeyError as exc:
        parser.error(str(exc))
    if args.baseline not in SUPPORTED_BASELINES:
        parser.error(
            "the MEMLENS/MemEye interfaces support only: "
            + ", ".join(sorted(SUPPORTED_BASELINES))
        )
    if args.attach_retrieved_images is None:
        args.attach_retrieved_images = args.baseline == "M3-Agent-caption"
    if not args.validate_only and not str(args.result_dir).strip():
        parser.error("--result-dir is required unless --validate-only is used")
    positive = {
        "top-k": args.top_k,
        "sample-concurrency": args.sample_concurrency,
        "answer-concurrency": args.answer_concurrency,
        "checkpoint-every": args.checkpoint_every,
        "request-timeout": args.request_timeout,
        "num-predict": args.num_predict,
        "executor-max-tokens": args.executor_max_tokens,
        "embedding-dim": args.embedding_dim,
        "m3-control-rounds": args.m3_control_rounds,
        "m3-search-top-k": args.m3_search_top_k,
    }
    bad = [name for name, value in positive.items() if int(value) < 1]
    if bad:
        parser.error(f"positive values required for: {', '.join(bad)}")
    if args.max_qa < 0 or args.retries < 0:
        parser.error("--max-qa and --retries cannot be negative")
    if args.mirix_max_consecutive_failed_build_points < 1:
        parser.error("--mirix-max-consecutive-failed-build-points must be positive")
    if args.m2a_max_consecutive_failed_build_points < 1:
        parser.error("--m2a-max-consecutive-failed-build-points must be positive")
    if args.m3_control_rounds > 5:
        parser.error("--m3-control-rounds cannot exceed the official default of 5")
    if args.m3_search_top_k > 2:
        parser.error("--m3-search-top-k cannot exceed the official default of 2")
    if not 0.0 <= args.m3_retrieval_threshold <= 1.0:
        parser.error("--m3-retrieval-threshold must be between 0 and 1")


def select_samples(
    samples: list[HarnessSample],
    *,
    sample_ids: Iterable[str] = (),
    question_ids: Iterable[str] = (),
    max_qa: int = 0,
) -> list[HarnessSample]:
    requested_samples = {str(value) for value in sample_ids if str(value)}
    available_samples = {sample.sample_id for sample in samples}
    missing_samples = sorted(requested_samples - available_samples)
    if missing_samples:
        raise KeyError(f"unknown sample IDs: {missing_samples}")
    selected = [
        sample for sample in samples
        if not requested_samples or sample.sample_id in requested_samples
    ]
    requested_questions = {str(value) for value in question_ids if str(value)}
    available_questions = {
        question.question_id for sample in selected for question in sample.questions
    }
    missing_questions = sorted(requested_questions - available_questions)
    if missing_questions:
        raise KeyError(f"unknown question IDs in selected samples: {missing_questions}")

    remaining = max_qa
    output: list[HarnessSample] = []
    for sample in selected:
        questions = [
            question for question in sample.questions
            if not requested_questions or question.question_id in requested_questions
        ]
        if max_qa:
            questions = questions[:remaining]
            remaining -= len(questions)
        if questions:
            output.append(
                HarnessSample(
                    sample_id=sample.sample_id,
                    source_name=sample.source_name,
                    source_path=sample.source_path,
                    chunks=sample.chunks,
                    questions=questions,
                )
            )
        if max_qa and remaining <= 0:
            break
    return output


def validate_samples(
    samples: list[HarnessSample], *, require_image_captions: bool = True
) -> dict[str, Any]:
    if not samples:
        raise ValueError("dataset selection contains no samples")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs")
    query_ids: set[str] = set()
    missing_images: list[str] = []
    missing_caption_chunks: list[str] = []
    missing_clues: list[str] = []
    chunk_count = 0
    question_count = 0
    for sample in samples:
        if not sample.chunks:
            raise ValueError(f"sample {sample.sample_id!r} has no memory chunks")
        if not sample.questions:
            raise ValueError(f"sample {sample.sample_id!r} has no questions")
        dialogue_ids = {
            str(chunk.metadata.get("dialogue_id") or "") for chunk in sample.chunks
        }
        chunk_ids = [chunk.chunk_id for chunk in sample.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"sample {sample.sample_id!r} has duplicate chunk IDs")
        for chunk in sample.chunks:
            for raw_path in chunk.images:
                if not Path(raw_path).is_file():
                    missing_images.append(str(raw_path))
            captions = [
                str(value).strip()
                for value in chunk.metadata.get("image_captions") or []
                if str(value).strip()
            ]
            if chunk.images and len(captions) < len(chunk.images):
                missing_caption_chunks.append(chunk.chunk_id)
        for question in sample.questions:
            query_id = f"{sample.sample_id}:{question.question_id}"
            if query_id in query_ids:
                raise ValueError(f"duplicate sample/question pair: {query_id}")
            query_ids.add(query_id)
            if not question.question.strip() or not question.answer.strip():
                raise ValueError(f"empty question or answer at {query_id}")
            if question.query_image and not Path(question.query_image["path"]).is_file():
                missing_images.append(str(question.query_image["path"]))
            unknown = sorted(set(question.clue_ids) - dialogue_ids)
            if unknown:
                missing_clues.append(f"{query_id}: {unknown}")
        chunk_count += len(sample.chunks)
        question_count += len(sample.questions)
    if missing_images:
        unique = list(dict.fromkeys(missing_images))
        raise FileNotFoundError(
            f"{len(unique)} referenced image file(s) are missing; first paths: {unique[:10]}"
        )
    if require_image_captions and missing_caption_chunks:
        raise ValueError(
            "MemVerse caption memory requires one caption per memory image; "
            f"first affected chunks: {missing_caption_chunks[:10]}"
        )
    if missing_clues:
        raise ValueError(
            f"evidence clues do not map to memory chunks; first entries: {missing_clues[:10]}"
        )
    return {
        "samples": len(samples),
        "questions": question_count,
        "chunks": chunk_count,
        "memory_images": sum(len(chunk.images) for sample in samples for chunk in sample.chunks),
        "query_images": sum(
            int(question.query_image is not None)
            for sample in samples for question in sample.questions
        ),
    }


def _memory_chunks_for_baseline(
    chunks: list[Chunk], *, baseline: str, benchmark: str
) -> list[Chunk]:
    if baseline == "M3-Agent-caption":
        return build_m3_chunks_from_round_chunks(
            chunks, benchmark=benchmark.casefold()
        )
    return chunks


def _validate_m3_inputs(samples: list[HarnessSample], benchmark: str) -> int:
    count = 0
    for sample in samples:
        converted = _memory_chunks_for_baseline(
            sample.chunks, baseline="M3-Agent-caption", benchmark=benchmark
        )
        if len(converted) != len(sample.chunks):
            raise ValueError(f"M3 conversion changed chunk count for {sample.sample_id!r}")
        for chunk in converted:
            observation = chunk.metadata.get("m3_observation")
            if not isinstance(observation, dict):
                raise ValueError(f"M3 observation is missing for {chunk.chunk_id}")
            if observation.get("input_mode") != "dialogue_round_as_clip":
                raise ValueError(f"invalid M3 input mode for {chunk.chunk_id}")
            if not observation.get("turns"):
                raise ValueError(f"M3 observation has no source turns: {chunk.chunk_id}")
            count += 1
    return count


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "answer_temperature": args.answer_temperature,
        "num_predict": args.num_predict,
        "think": args.think,
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_temperature": args.executor_temperature,
        "executor_max_tokens": args.executor_max_tokens,
        "m2a_salvage_truncated_updates": args.m2a_salvage_truncated_updates,
        "m2a_skip_failed_build_points": args.m2a_skip_failed_build_points,
        "m2a_max_consecutive_failed_build_points": (
            args.m2a_max_consecutive_failed_build_points
        ),
        "executor_native_tool_calls": args.baseline == "MIRIX",
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_dim": args.embedding_dim,
        "top_k": args.top_k,
        "request_timeout": args.request_timeout,
        "retries": args.retries,
        "reasoning_effort": args.reasoning_effort,
        "mirix_skip_failed_build_points": args.mirix_skip_failed_build_points,
        "mirix_max_consecutive_failed_build_points": (
            args.mirix_max_consecutive_failed_build_points
        ),
        "m3_reuse_state_root": args.m3_reuse_state_root,
        "m3_control_rounds": args.m3_control_rounds,
        "m3_search_top_k": args.m3_search_top_k,
        "m3_retrieval_threshold": args.m3_retrieval_threshold,
    }


def _sample_signature(
    args: argparse.Namespace,
    benchmark: str,
    sample: HarnessSample,
    memory_chunks: list[Chunk],
) -> str:
    prompt_module = _prompt_module(benchmark)
    memory_images = [Path(path) for chunk in memory_chunks for path in chunk.images]
    return signature_digest(
        {
            "benchmark": benchmark,
            "baseline": args.baseline,
            "source": _cached_manifest([sample.source_path]),
            "question_ids": [question.question_id for question in sample.questions],
            "memory_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "images": list(chunk.images),
                    "metadata": chunk.metadata,
                }
                for chunk in memory_chunks
            ],
            "memory_images": _cached_manifest(memory_images),
            "m3_input": (
                m3_input_manifest(benchmark)
                if args.baseline == "M3-Agent-caption"
                else None
            ),
            "runtime": _runtime_config(args),
            **prompt_module.prompt_manifest(),
        }
    )


@lru_cache(maxsize=64)
def _cached_file(path: str) -> tuple[str, str]:
    manifest = file_manifest([path])
    if not manifest:
        raise FileNotFoundError(path)
    return next(iter(manifest.items()))


def _cached_manifest(paths: Iterable[Path]) -> dict[str, str]:
    resolved = sorted({str(Path(path).expanduser().resolve()) for path in paths})
    return dict(_cached_file(path) for path in resolved)


def _run_input_paths(
    samples: list[HarnessSample], *, include_memory_images: bool
) -> list[Path]:
    paths = {sample.source_path.resolve() for sample in samples}
    for sample in samples:
        for question in sample.questions:
            if question.query_image:
                paths.add(Path(question.query_image["path"]).resolve())
        if include_memory_images:
            for chunk in sample.chunks:
                paths.update(Path(value).resolve() for value in chunk.images)
    return sorted(paths)


def _prepare_sample(
    *,
    args: argparse.Namespace,
    benchmark: str,
    sample: HarnessSample,
    output_layout: BaselineOutputLayout,
    state_root: Path,
) -> dict[str, Any]:
    prompt_module = _prompt_module(benchmark)
    memory_chunks = _memory_chunks_for_baseline(
        sample.chunks, baseline=args.baseline, benchmark=benchmark
    )
    sample_signature = _sample_signature(args, benchmark, sample, memory_chunks)
    resumed_native_artifact: dict[str, Any] | None = None
    if args.resume:
        cached = load_sample_artifact(
            output_layout.sample_checkpoint_dir,
            sample.sample_id,
            signature=sample_signature,
        )
        if cached is not None:
            failed_native_ids = {
                str(job.get("query_id") or "")
                for job in cached.get("jobs") or []
                if str((job.get("native_answer") or {}).get("error") or "").strip()
            }
            if (
                args.baseline != "MIRIX"
                or (bool(cached.get("complete", True)) and not failed_native_ids)
            ):
                print(
                    f"[resume] {sample.sample_id}: "
                    f"{len(cached.get('jobs') or [])} prepared question(s)",
                    flush=True,
                )
                return cached
            resumed_native_artifact = {
                **cached,
                "jobs": [
                    job for job in cached.get("jobs") or []
                    if str(job.get("query_id") or "") not in failed_native_ids
                ],
            }
            print(
                f"[resume] {sample.sample_id}: "
                f"{len(resumed_native_artifact['jobs'])} completed native answer(s), "
                f"{len(failed_native_ids)} failed answer(s) queued for retry",
                flush=True,
            )

    trace_path = Path(args.result_dir) / "call_traces" / trace_filename(sample.sample_id)
    recorder = CallRecorder(
        trace_path=trace_path,
        baseline=args.baseline,
        benchmark=benchmark,
        sample_id=sample.sample_id,
        reset=not args.resume,
    )
    config = _runtime_config(args)
    reuse_root = str(config.get("m3_reuse_state_root") or "").strip()
    if args.baseline == "M3-Agent-caption" and reuse_root:
        config["m3_reuse_sample_state"] = str(
            Path(reuse_root).expanduser().resolve() / sample.sample_id
        )
    # The API key is runtime-only and deliberately excluded from signatures
    # and manifests. It lets a direct CLI invocation configure the isolated
    # MIRIX worker without requiring a process-global environment mutation.
    config["executor_api_key"] = args.answer_api_key
    if args.baseline == "MIRIX":
        config["mirix_resume_enabled"] = args.resume
        config["mirix_resume_signature"] = sample_signature
    proxy_context = CountingProxy(
        args.executor_base_url,
        recorder,
        args.request_timeout,
        max_output_tokens=args.executor_max_tokens,
        temperature=args.executor_temperature,
        qa_max_output_tokens=args.num_predict,
        reasoning_effort=args.reasoning_effort,
    )
    adapter = None
    snapshots: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    build_fault_policy = ConsecutiveBuildFaultPolicy(
        baseline=args.baseline,
        benchmark=benchmark,
        enabled=(
            (args.baseline == "MIRIX" and args.mirix_skip_failed_build_points)
            or (args.baseline == "M2A" and args.m2a_skip_failed_build_points)
        ),
        maximum=(
            args.m2a_max_consecutive_failed_build_points
            if args.baseline == "M2A"
            else args.mirix_max_consecutive_failed_build_points
        ),
        recorder=recorder,
        fail_open=(args.baseline == "M2A"),
    )
    if resumed_native_artifact is not None:
        build_fault_policy.failures.extend(
            resumed_native_artifact.get("build_failures") or []
        )
    completed_native_jobs = {
        str(job.get("query_id") or ""): job
        for job in (resumed_native_artifact or {}).get("jobs") or []
        if str(job.get("query_id") or "")
    }

    def save_native_progress() -> None:
        if args.baseline != "MIRIX":
            return
        save_sample_artifact(
            output_layout.sample_checkpoint_dir,
            sample.sample_id,
            signature=sample_signature,
            artifact={
                "sample_id": sample.sample_id,
                "jobs": list(jobs),
                "snapshots": [],
                "call_trace_path": str(trace_path),
                "build_failures": list(build_fault_policy.failures),
                "complete": False,
            },
        )

    with proxy_context as proxy:
        config["executor_base_url"] = proxy.endpoint
        adapter = create_adapter(args.baseline, config_overrides=config)
        try:
            adapter.reset(sample.sample_id, state_root / sample.sample_id)
            chunks_to_ingest = memory_chunks
            if args.baseline == "MIRIX" and args.resume:
                chunks_to_ingest = adapter.filter_completed_session_chunks(
                    memory_chunks
                )
            current_session = ""
            with recorder.phase("memory_build"):
                for chunk in chunks_to_ingest:
                    session_id = str(chunk.metadata.get("session_id") or "")
                    if (
                        args.baseline != "MIRIX"
                        and current_session
                        and session_id != current_session
                    ):
                        try:
                            adapter.end_session(current_session)
                        except Exception as exc:
                            build_fault_policy.handle(
                                exc,
                                chunk=None,
                                point_kind="end_session",
                                session_id=current_session,
                            )
                        else:
                            build_fault_policy.success()
                    try:
                        adapter.ingest(chunk)
                    except Exception as exc:
                        build_fault_policy.handle(
                            exc,
                            chunk=chunk,
                            point_kind="ingest",
                            session_id=session_id,
                        )
                        current_session = session_id
                        continue
                    build_fault_policy.success()
                    current_session = session_id
                if current_session:
                    try:
                        adapter.end_session(current_session)
                    except Exception as exc:
                        build_fault_policy.handle(
                            exc,
                            chunk=None,
                            point_kind="end_session",
                            session_id=current_session,
                        )
                    else:
                        build_fault_policy.success()
            for question in sample.questions:
                query_id = f"{benchmark.casefold()}:{sample.sample_id}:{question.question_id}"
                if query_id in completed_native_jobs:
                    jobs.append(completed_native_jobs[query_id])
                    continue
                retrieval_error = ""
                try:
                    with recorder.phase("retrieval"):
                        retrieval = adapter.retrieve(
                            RetrievalRequest(
                                query_id=query_id,
                                text=(
                                    question.question
                                    if args.baseline == "MIRIX"
                                    else (
                                        f"[{question.category}] {question.question}"
                                        if question.category
                                        else question.question
                                    )
                                ),
                                category=question.category,
                                top_k=args.top_k,
                                query_image=(
                                    question.query_image.get("path")
                                    if question.query_image else None
                                ),
                                # ``session_ids`` in both source datasets annotate
                                # gold evidence. They must never constrain retrieval.
                                visible_session_ids=tuple(question.visible_session_ids),
                            )
                        )
                except Exception as exc:
                    if not (args.baseline == "MIRIX" and args.allow_answer_errors):
                        raise
                    retrieval_error = str(exc)
                    retrieval = RetrievalResult(
                        trace={
                            "failed": True,
                            "stage": "retrieval",
                            "error": retrieval_error,
                        }
                    )
                memory_items = result_context_items(retrieval)
                trace_rows = result_trace_rows(retrieval)
                native_answer = None
                if args.baseline == "MIRIX" and retrieval_error:
                    native_answer = {
                        "text": "",
                        "error": retrieval_error,
                        "usage": None,
                        "attempts": args.retries + 1,
                        "failed_attempts": args.retries + 1,
                        "image_count": 1 if question.query_image else 0,
                        "trace": {
                            "failed": True,
                            "stage": "retrieval",
                            "error": retrieval_error,
                        },
                    }
                elif args.baseline == "MIRIX":
                    native_evidence, _ = build_retrieved_memory_evidence(
                        memory_items, question.category
                    )
                    native_messages = prompt_module.build_answer_messages(
                        question=question.question,
                        question_type=question.category,
                        memory_evidence=native_evidence,
                        query_images=query_image_prompt_metadata(
                            question.query_image
                        ),
                        question_date=str(
                            question.metadata.get("question_date") or ""
                        ),
                        allow_empty_evidence=True,
                    )
                    try:
                        with recorder.phase("qa"):
                            native_result = adapter.answer_with_memory(
                                NativeAnswerRequest(
                                    query_id=query_id,
                                    messages=native_messages,
                                    retrieval=retrieval,
                                    query_image=(
                                        question.query_image.get("path")
                                        if question.query_image
                                        else None
                                    ),
                                    top_k=args.top_k,
                                )
                            )
                    except Exception as exc:
                        if not args.allow_answer_errors:
                            raise
                        error = str(exc)
                        native_answer = {
                            "text": "",
                            "error": error,
                            "usage": None,
                            "attempts": args.retries + 1,
                            "failed_attempts": args.retries + 1,
                            "image_count": 1 if question.query_image else 0,
                            "trace": {"failed": True, "stage": "qa", "error": error},
                        }
                    else:
                        if native_result.retrieval is not None:
                            retrieval = native_result.retrieval
                            memory_items = result_context_items(retrieval)
                            trace_rows = result_trace_rows(retrieval)
                        native_answer = native_result.to_dict()
                groups = [list(row.get("source_dialogue_ids") or []) for row in trace_rows]
                jobs.append(
                    {
                        "query_id": query_id,
                        "manifest_question_id": query_id,
                        "sample_id": sample.sample_id,
                        "dataset": sample.sample_id,
                        "source_name": sample.source_name,
                        "question_id": question.question_id,
                        "question": question.question,
                        "category": question.category,
                        "session_id": list(question.session_ids),
                        "original_answer": question.answer,
                        "clue": list(question.clue_ids),
                        "query_image": question.query_image,
                        "retrieved_ids": list(
                            dict.fromkeys(value for group in groups for value in group)
                        ),
                        "retrieved_source_groups": groups,
                        "memory_items": memory_items,
                        "retrieval_top_k": trace_rows,
                        "retrieval_method_trace": dict(retrieval.trace),
                        "question_metadata": dict(question.metadata),
                        "native_answer": native_answer,
                    }
                )
                save_native_progress()
            snapshots = [row.to_dict() for row in adapter.snapshot()]
            for row in snapshots:
                metadata = dict(row.get("metadata") or {})
                metadata.update({"dataset": sample.sample_id, "source_name": sample.source_name})
                row["metadata"] = metadata
        finally:
            if adapter is not None:
                adapter.close()
    artifact = {
        "sample_id": sample.sample_id,
        "jobs": jobs,
        "snapshots": snapshots,
        "call_trace_path": str(trace_path),
        "build_failures": list(build_fault_policy.failures),
        "build_failure_policy": {
            "enabled": build_fault_policy.enabled,
            "max_consecutive_failed_build_points": build_fault_policy.maximum,
            "successful_point_resets_consecutive_count": True,
            "incomplete_tool_arguments_are_executed": False,
        },
        "complete": True,
    }
    save_sample_artifact(
        output_layout.sample_checkpoint_dir,
        sample.sample_id,
        signature=sample_signature,
        artifact=artifact,
    )
    print(f"[prepared] {sample.sample_id}: {len(jobs)} question(s)", flush=True)
    return artifact


def _answer_job(
    client: VLMAnswerClient,
    job: dict[str, Any],
    *,
    benchmark: str,
    allow_answer_errors: bool,
    attach_retrieved_images: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_module = _prompt_module(benchmark)
    transport_category = "VS" if attach_retrieved_images else job["category"]
    evidence, _ = build_retrieved_memory_evidence(job["memory_items"], transport_category)
    prompt_evidence, zero_hit_prompt_marker_used = evidence_with_zero_hit_marker(
        evidence
    )
    native = job.get("native_answer")
    messages = prompt_module.build_answer_messages(
        question=job["question"],
        question_type=job["category"],
        memory_evidence=prompt_evidence,
        query_images=query_image_prompt_metadata(job.get("query_image")),
        question_date=str(
            (job.get("question_metadata") or {}).get("question_date") or ""
        ),
        allow_empty_evidence=native is not None,
    )
    raw_answer = ""
    usage = None
    attempts = 0
    failed_attempts = 0
    image_count = 0
    response = None
    try:
        if native is not None:
            usage = native.get("usage")
            attempts = int(native.get("attempts") or 1)
            failed_attempts = int(native.get("failed_attempts") or 0)
            image_count = int(native.get("image_count") or 0)
            if native.get("error"):
                raise RuntimeError(str(native["error"]))
            raw_answer = str(native.get("text") or "")
        else:
            response = client.answer_messages_with_usage(
                messages=messages,
                memory_items=job["memory_items"],
                query_image=job.get("query_image"),
                category=transport_category,
            )
            raw_answer = response.text
            usage = response.usage
            attempts = response.attempts
            failed_attempts = response.failed_attempts
            image_count = response.image_count
        answer = prompt_module.parse_answer_response(raw_answer)
        error = ""
    except Exception as exc:
        if not allow_answer_errors:
            raise RuntimeError(f"answer failed for {job['query_id']}: {exc}") from exc
        answer, error = "", str(exc)
        if response is not None:
            usage = response.usage
            attempts = response.attempts
            failed_attempts = min(attempts, response.failed_attempts + 1)
            image_count = response.image_count
        elif native is None:
            attempts = attempts or client.retries + 1
            failed_attempts = attempts
            image_count = client.count_answer_images(
                job["memory_items"],
                query_image=job.get("query_image"),
                category=transport_category,
            )
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], transport_category
    )
    omitted = {
        "query_image",
        "memory_items",
        "retrieval_top_k",
        "retrieval_method_trace",
        "native_answer",
    }
    result = {key: value for key, value in job.items() if key not in omitted}
    result.update(
        {
            "system_answer": answer,
            "answer_raw_response": raw_answer,
            "error": error,
            "answer_token_usage": usage,
            "answer_attempts": attempts,
            "answer_failed_attempts": failed_attempts,
            "answer_image_count": image_count,
            "native_answer_trace": dict((native or {}).get("trace") or {}),
            "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    trace = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "sample_id": job["sample_id"],
        "source_name": job["source_name"],
        "question_id": job["question_id"],
        "question": job["question"],
        "category": job["category"],
        "clue": job["clue"],
        "top_k": job["retrieval_top_k"],
        "memory_context": memory_context,
        "answer_prompt_messages": messages,
        "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
        "retrieval_method_trace": dict(job.get("retrieval_method_trace") or {}),
        "attach_retrieved_images": attach_retrieved_images,
        "native_answer_trace": dict((native or {}).get("trace") or {}),
    }
    return result, trace


def run_harness(
    *,
    args: argparse.Namespace,
    benchmark: str,
    samples: list[HarnessSample],
    source_paths: Iterable[Path],
) -> dict[str, Any]:
    prompt_module = _prompt_module(benchmark)
    source_paths = list(source_paths)
    validation = validate_samples(
        samples, require_image_captions=args.baseline == "MemVerse"
    )
    if args.baseline == "M3-Agent-caption":
        validation["m3_observations"] = _validate_m3_inputs(samples, benchmark)
    if args.validate_only:
        payload = {"benchmark": benchmark, "baseline": args.baseline, **validation}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_layout = BaselineOutputLayout(result_dir)
    state_root = output_layout.state_root(args.baseline_state_dir)
    state_root.mkdir(parents=True, exist_ok=True)
    signature = {
        "benchmark": benchmark,
        "baseline": args.baseline,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {
                "answer_api_key", "result_dir", "resume", "sample_concurrency",
                "answer_concurrency", "checkpoint_every", "allow_answer_errors",
                "skip_model_check", "validate_only",
            }
        },
        "inputs": _cached_manifest(
            _run_input_paths(
                samples,
                include_memory_images=(
                    args.attach_retrieved_images
                    or args.baseline in {"M3-Agent-caption", "MIRIX"}
                ),
            )
        ),
        "question_ids": [
            f"{sample.sample_id}:{question.question_id}"
            for sample in samples for question in sample.questions
        ],
        **prompt_module.prompt_manifest(),
        "call_trace_version": TRACE_VERSION,
    }
    signature_hash = signature_digest(signature)

    def prepare(sample: HarnessSample) -> dict[str, Any]:
        return _prepare_sample(
            args=args,
            benchmark=benchmark,
            sample=sample,
            output_layout=output_layout,
            state_root=state_root,
        )

    artifacts = parallel_map_ordered(
        samples,
        prepare,
        max_workers=args.sample_concurrency,
        item_key=lambda sample: sample.sample_id,
    )
    jobs = [job for artifact in artifacts for job in artifact["jobs"]]
    write_jsonl_atomic(output_layout.pipeline_qa, jobs)

    checkpoint_manifest = output_layout.checkpoint_dir / "manifest.json"
    checkpoint_results = output_layout.checkpoint_dir / "results.json"
    checkpoint_traces = output_layout.checkpoint_dir / "retrieval_trace.jsonl"
    results_by_id: dict[str, dict[str, Any]] = {}
    traces_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_manifest.is_file():
        saved = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if saved.get("signature_hash") == signature_hash:
            if checkpoint_results.is_file():
                results_by_id = {
                    row["query_id"]: row
                    for row in json.loads(checkpoint_results.read_text(encoding="utf-8"))
                }
            if checkpoint_traces.is_file():
                traces_by_id = {
                    row["query_id"]: row
                    for row in (
                        json.loads(line)
                        for line in checkpoint_traces.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                }
            print(f"[resume] loaded {len(results_by_id)} answer(s)", flush=True)

    ordered_ids = [job["query_id"] for job in jobs]

    def save_checkpoint() -> None:
        completed = [query_id for query_id in ordered_ids if query_id in results_by_id]
        write_json_atomic(checkpoint_results, [results_by_id[value] for value in completed])
        write_jsonl_atomic(
            checkpoint_traces,
            [traces_by_id[value] for value in completed if value in traces_by_id],
        )
        write_json_atomic(
            checkpoint_manifest,
            {
                "signature_hash": signature_hash,
                "completed": len(completed),
                "expected": len(ordered_ids),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    pending = [
        job for job in jobs
        if job["query_id"] not in results_by_id
        or job["query_id"] not in traces_by_id
        or (results_by_id[job["query_id"]].get("error") and not args.allow_answer_errors)
        or (
            results_by_id[job["query_id"]].get("error")
            and job.get("native_answer") is not None
            and not (job.get("native_answer") or {}).get("error")
        )
    ]
    client = VLMAnswerClient(
        model=args.answer_model,
        base_url=args.answer_base_url,
        api_key=args.answer_api_key,
        temperature=args.answer_temperature,
        num_predict=args.num_predict,
        timeout=args.request_timeout,
        retries=args.retries,
        think=args.think,
        reasoning_effort=args.reasoning_effort,
        backend="openai",
    )
    requires_external_answers = any(
        job.get("native_answer") is None for job in pending
    )
    if requires_external_answers and not args.skip_model_check:
        client.assert_model_available()
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
        futures = {
            pool.submit(
                _answer_job,
                client,
                job,
                benchmark=benchmark,
                allow_answer_errors=args.allow_answer_errors,
                attach_retrieved_images=args.attach_retrieved_images,
            ): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            result, trace = future.result()
            results_by_id[job["query_id"]] = result
            traces_by_id[job["query_id"]] = trace
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0
            print(
                f"[{len(results_by_id)}/{len(jobs)}] {job['sample_id']} "
                f"{job['question_id']} error={result['error'][:80]!r}",
                flush=True,
            )
    save_checkpoint()
    all_results = [results_by_id[value] for value in ordered_ids]
    all_traces = [traces_by_id[value] for value in ordered_ids]
    write_json_atomic(result_dir / "results.json", all_results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", all_traces)
    snapshots = [row for artifact in artifacts for row in artifact.get("snapshots") or []]
    write_jsonl_atomic(output_layout.snapshot, snapshots)

    public_args = {key: value for key, value in vars(args).items() if key != "answer_api_key"}
    manifest = {
        **public_args,
        **prompt_module.prompt_manifest(),
        "benchmark": benchmark,
        "questions": len(all_results),
        "samples": len(samples),
        "answer_errors": sum(bool(row.get("error")) for row in all_results),
        "baseline_runtime": baseline_metadata(args.baseline),
        "run_signature": signature,
        "run_signature_sha256": signature_hash,
        "selection_mode": (
            "explicit" if args.sample_id or args.question_id or args.max_qa else "full"
        ),
        "chunk_input": {
            "builder": (
                "benchmarks.baseline_runtime.m3_inputs."
                "build_m3_chunks_from_round_chunks"
                if args.baseline == "M3-Agent-caption"
                else "dataset-specific source-to-Chunk adapter"
            ),
            "shared_fixed_chunks": False,
            "source_files": _cached_manifest(source_paths),
            "chunk_count": validation["chunks"],
            "memory_image_count": validation["memory_images"],
        },
        "caption_memory": (
            args.baseline == "MemVerse" and not args.attach_retrieved_images
        ),
        "build_failures": [
            row
            for artifact in artifacts
            for row in artifact.get("build_failures") or []
        ],
    }
    if args.baseline == "MIRIX":
        manifest["mirix_native_execution"] = {
            "memory_build": "native_six_agent_absorption",
            "retrieval": f"native_chat_agent_global_top_k_{args.top_k}",
            "global_top_k": args.top_k,
            "global_top_k_scope": "automatic_prefetch_plus_explicit_tools",
            "answer": "native_chat_agent",
            "generic_answer_client_fallback": False,
            "per_qa_sample_checkpoint": True,
            "resume": bool(args.resume),
            "build_failure_policy": {
                "enabled": bool(args.mirix_skip_failed_build_points),
                "max_consecutive_failed_build_points": (
                    args.mirix_max_consecutive_failed_build_points
                ),
            },
        }
    if args.baseline == "M3-Agent-caption":
        manifest["m3_input"] = m3_input_manifest(benchmark)
        manifest["m3_conformance"] = m3_conformance_manifest(
            benchmark.casefold(),
            answer_prompt_sha256=prompt_module.prompt_manifest()["prompt_sha256"],
            handoff_top_k=args.top_k,
            native_search_top_k=args.m3_search_top_k,
            native_round_limit=args.m3_control_rounds,
            retrieval_threshold=args.m3_retrieval_threshold,
        )
    write_json_atomic(result_dir / "run_manifest.json", manifest)
    if manifest["answer_errors"] and not args.allow_answer_errors:
        raise RuntimeError(
            f"{manifest['answer_errors']}/{len(all_results)} answers failed; metrics not written"
        )

    summary = summarize_results(all_results, k=args.top_k)
    sample_ids = [sample.sample_id for sample in samples]
    summary["calls"] = write_runtime_call_metrics(
        [artifact["call_trace_path"] for artifact in artifacts],
        result_dir,
        all_results,
        sample_id_field="sample_id",
        sample_ids=sample_ids,
    )
    memory_metrics = write_snapshot_memory_metrics(
        snapshots,
        result_dir,
        sample_ids=sample_ids,
        cost_mb_input_price=args.cost_mb_input_price,
        cost_mb_output_price=args.cost_mb_output_price,
    )
    summary = add_memory_metrics(summary, memory_metrics)
    efficiency = write_efficiency_metrics(
        result_dir,
        all_results,
        sample_id_field="sample_id",
        sample_ids=sample_ids,
        model=(
            args.executor_model if args.baseline == "MIRIX" else args.answer_model
        ),
        config_path=args.efficiency_config,
        memory_build_model=args.executor_model,
        retrieval_model=args.executor_model,
        answer_model=(
            args.executor_model if args.baseline == "MIRIX" else args.answer_model
        ),
    )
    summary.update(
        {
            key: efficiency[key]
            for key in (
                "cost_mb", "cost_qa", "cost_total",
                "latency_mb", "latency_qa", "latency_total",
            )
        }
    )
    summary = merge_existing_llm_judge_metrics(summary, result_dir)
    write_json_atomic(result_dir / "metrics.json", summary)
    return manifest
