from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.memgallery_harness.runner.prompts import (
    build_answer_messages,
    parse_answer_response,
    prompt_manifest,
    prompt_sha256,
    resolve_question_image,
)
from benchmarks.memgallery_harness.retrieval.query_embedding_cache import QueryEmbeddingCache, make_query_id

from benchmarks.memgallery_harness.runner.metrics import (
    add_memory_metrics,
    calculate_cost_mb,
    calculate_cost_qa,
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    merge_existing_llm_judge_metrics,
    summarize_results,
    write_memory_metrics,
    write_snapshot_memory_metrics,
    write_retrieval_memory_token,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.baseline_runtime.call_trace import (
    CallRecorder,
    CountingProxy,
    TRACE_VERSION,
    trace_filename,
)
from benchmarks.baseline_runtime.build_fault_policy import (
    is_non_skippable_build_failure,
)
from benchmarks.baseline_runtime.adapters.m2a import m2a_conformance_manifest
from benchmarks.baseline_runtime.adapters.m3_agent import m3_conformance_manifest
from benchmarks.baseline_runtime.adapters.mma_original import (
    is_mma_consecutive_bad_point_error,
    mma_conformance_manifest,
)
from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
    validated_paired_resume_signatures,
    validated_qa_only_resume_signatures,
)
from benchmarks.baseline_runtime.output_layout import (
    BaselineOutputLayout,
    load_hivemem_snapshot,
)
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.baseline_runtime.protocol import (
    NativeAnswerRequest,
    RetrievalRequest,
    RetrievalResult,
    result_context_items,
    result_trace_rows,
)
from benchmarks.question_filter import is_excluded_category, parse_excluded_categories
from benchmarks.zero_hit import evidence_with_zero_hit_marker
from benchmarks.fixed_chunks import chunk_source_manifest, memgallery_chunks
from benchmarks.baseline_runtime.omni_inputs import (
    build_omni_memgallery_chunks,
    omni_conformance_manifest,
    omni_input_manifest,
)
from benchmarks.baseline_runtime.m3_inputs import (
    build_m3_memgallery_chunks,
    m3_input_manifest,
)
from embedding.chunk_builder import build_chunks_from_data
from evidence_policy.split_manifest import SplitManifestIndex, normalize_split_name
from typing import Any, Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMGALLERY_DATA_DIR = WORKSPACE_ROOT / "Mem-Gallery" / "benchmark" / "data"


def memgallery_manifest_question_id(dataset_name: str, qa_index: int) -> str:
    """Canonical ID used by multimodal_split_manifest.json."""
    return f"{dataset_name}_q{qa_index - 1:04d}"


def select_diagnostic_chunks(
    chunks: list[Any],
    *,
    chunk_limit: int = 0,
    dialogue_ids: tuple[str, ...] = (),
) -> list[Any]:
    """Select an explicitly non-scoring integration subset in source order."""
    if chunk_limit and dialogue_ids:
        raise ValueError(
            "diagnostic smoke must use either a chunk prefix or explicit dialogue IDs, not both"
        )
    if chunk_limit:
        return chunks[:chunk_limit]
    if not dialogue_ids:
        return chunks

    requested = set(dialogue_ids)
    available = {
        str(chunk.metadata.get("dialogue_id") or "")
        for chunk in chunks
    }
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"diagnostic smoke references missing dialogue IDs: {missing}"
        )
    return [
        chunk
        for chunk in chunks
        if str(chunk.metadata.get("dialogue_id") or "") in requested
    ]


def validate_diagnostic_qa_coverage(
    selected_qas: list[tuple[str, int, dict[str, Any]]],
    chunks: list[Any],
) -> None:
    """Reject a diagnostic QA whose annotated clues were not built into memory."""
    built_ids = {
        str(chunk.metadata.get("dialogue_id") or "")
        for chunk in chunks
        if str(chunk.metadata.get("dialogue_id") or "")
    }
    failures = []
    for _, qa_index, qa in selected_qas:
        clues = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in (qa.get("clue") or [])
                if str(value).strip()
            )
        )
        if not clues:
            failures.append(f"QA {qa_index}: no annotated clue IDs")
            continue
        missing = sorted(set(clues) - built_ids)
        if missing:
            failures.append(f"QA {qa_index}: missing {missing}")
    if failures:
        raise ValueError(
            "diagnostic smoke QA/chunk mismatch before model calls: "
            + "; ".join(failures)
        )


def _is_non_skippable_build_failure(exc: Exception) -> bool:
    """Identify global failures that cannot be repaired by skipping one point."""
    return is_non_skippable_build_failure(exc)


_HARD_STOP_NATIVE_QA_MARKERS = (
    "unauthenticated",
    "unauthorized",
    "status 401",
    "status code: 401",
    "status 403",
    "status code: 403",
    "invalid api key",
    "insufficient_quota",
    "insufficient quota",
    "insufficient credit",
    "insufficient balance",
)


def _is_hard_stop_native_qa_job(job: dict[str, Any]) -> bool:
    """Do not treat an infrastructure hard stop as a completed baseline QA."""
    answer = job.get("native_answer") or {}
    error = str(answer.get("error") or "").casefold()
    return bool(error) and any(
        marker in error for marker in _HARD_STOP_NATIVE_QA_MARKERS
    )


def _is_failed_native_qa_job(job: dict[str, Any]) -> bool:
    """Retry only infrastructure hard stops, never baseline QA bad points."""
    return _is_hard_stop_native_qa_job(job)


def _build_failure_row(
    *,
    exc: Exception,
    chunk: Any | None,
    point_kind: str,
    session_id: str,
    consecutive: int,
    maximum: int,
) -> dict[str, Any]:
    metadata = dict(getattr(chunk, "metadata", None) or {})
    return {
        "phase": "build_fault",
        "service": "runtime",
        "event": "skipped_build_point",
        "failed": True,
        "success": False,
        "point_kind": point_kind,
        "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
        "dialogue_id": str(metadata.get("dialogue_id") or ""),
        "session_id": session_id,
        "consecutive_failed_build_points": consecutive,
        "max_consecutive_failed_build_points": maximum,
        "error_type": type(exc).__name__,
        "error": str(exc)[:2000],
    }


def prepare_dataset_jobs(
    dataset_path: Path,
    data_dir: Path,
    index_root: Path,
    query_cache: QueryEmbeddingCache | None,
    *,
    top_k: int = 7,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
    graph_options: dict | None = None,
    baseline: str = "HiveMem",
    state_root: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    excluded_categories: frozenset[str] = frozenset(),
    ordered_question_ids: tuple[str, ...] | None = None,
    call_recorder: CallRecorder | None = None,
    completed_jobs: dict[str, dict[str, Any]] | None = None,
    on_qa_completed: Callable[[dict[str, Any]], None] | None = None,
    allow_native_qa_errors: bool = False,
) -> dict[str, Any]:
    baseline = canonical_name(baseline)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset_path.stem
    profile = dataset.get("character_profile") or {}
    speaker_a = f"user ({profile.get('name')})" if profile.get("name") else "user"
    overrides = dict(config_overrides or {})
    overrides.update(
        {
            "top_k": top_k,
            "index_root": str(index_root),
            "graph_options": graph_options,
        }
    )
    reuse_root = str(overrides.get("m3_reuse_state_root") or "").strip()
    if baseline == "M3-Agent-caption" and reuse_root:
        overrides["m3_reuse_sample_state"] = str(Path(reuse_root) / dataset_name)
    native_adapter = create_adapter(baseline, config_overrides=overrides)
    sample_state = (state_root or Path("outputs") / "memory" / "datasets") / dataset_name
    qa_pairs = dataset.get("human-annotated QAs", [])
    qa_end = qa_end or len(qa_pairs)
    indexed_qas = [
        (memgallery_manifest_question_id(dataset_name, qa_index), qa_index, qa)
        for qa_index, qa in enumerate(qa_pairs, start=1)
    ]
    if ordered_question_ids is not None:
        by_manifest_id = {row[0]: row for row in indexed_qas}
        missing = [
            question_id
            for question_id in ordered_question_ids
            if question_id not in by_manifest_id
        ]
        if missing:
            raise KeyError(
                f"Mem-Gallery manifest references {len(missing)} missing "
                f"question(s) for {dataset_name}: {missing[:5]}"
            )
        selected_qas = [by_manifest_id[value] for value in ordered_question_ids]
        excluded = len(indexed_qas) - len(selected_qas)
    else:
        selected_qas = [
            row for row in indexed_qas if qa_start <= row[1] <= qa_end
        ]
        excluded = 0
    jobs: list[dict[str, Any]] = []
    completed_jobs = dict(completed_jobs or {})
    processed = 0
    diagnostic_selection: dict[str, Any] = {}
    build_failures: list[dict[str, Any]] = []
    build_policy_prefix = {
        "M2A": "m2a",
        "MIRIX": "mirix",
        "MMA": "mma",
    }.get(baseline, "")
    skip_failed_build_points = bool(
        baseline == "MMA"
        or (
            build_policy_prefix
            and overrides.get(
                f"{build_policy_prefix}_skip_failed_build_points", False
            )
        )
    )
    max_consecutive_build_failures = max(
        1,
        int(
            overrides.get(
                f"{build_policy_prefix}_max_consecutive_failed_build_points"
            )
            or 10
        ),
    )
    consecutive_build_failures = 0

    def handle_build_failure(
        exc: Exception,
        *,
        chunk: Any | None,
        point_kind: str,
        session_id: str,
    ) -> None:
        nonlocal consecutive_build_failures
        if not skip_failed_build_points or (
            baseline != "M2A" and _is_non_skippable_build_failure(exc)
        ):
            raise exc
        consecutive_build_failures += 1
        row = _build_failure_row(
            exc=exc,
            chunk=chunk,
            point_kind=point_kind,
            session_id=session_id,
            consecutive=consecutive_build_failures,
            maximum=max_consecutive_build_failures,
        )
        build_failures.append(row)
        if call_recorder is not None:
            call_recorder.append(row)
        if consecutive_build_failures >= max_consecutive_build_failures:
            raise RuntimeError(
                f"Mem-Gallery stopped after {consecutive_build_failures} "
                f"consecutive failed {baseline} build points"
            ) from exc

    try:
        native_adapter.reset(dataset_name, sample_state)
        with (
            call_recorder.phase("memory_build")
            if call_recorder is not None
            else nullcontext()
        ):
            if baseline != "HiveMem":
                if baseline == "M2A":
                    chunks = build_chunks_from_data(dataset, data_dir, dataset_name)
                elif baseline in {"OmniSimpleMem", "MMA"}:
                    chunks = build_omni_memgallery_chunks(
                        dataset, data_dir, dataset_name
                    )
                elif baseline == "M3-Agent-caption":
                    chunks = build_m3_memgallery_chunks(
                        dataset, data_dir, dataset_name
                    )
                else:
                    chunks = memgallery_chunks(overrides, dataset_name)
                integration_limit = int(overrides.get("integration_chunk_limit") or 0)
                integration_dialogue_ids = tuple(
                    str(value)
                    for value in (overrides.get("integration_dialogue_ids") or ())
                )
                chunks = select_diagnostic_chunks(
                    chunks,
                    chunk_limit=integration_limit,
                    dialogue_ids=integration_dialogue_ids,
                )
                if baseline == "MMA" or (
                    baseline == "MIRIX"
                    and bool(overrides.get("mirix_resume_enabled", False))
                ):
                    chunks = native_adapter.filter_completed_session_chunks(chunks)
                if integration_limit or integration_dialogue_ids:
                    validate_diagnostic_qa_coverage(selected_qas, chunks)
                    diagnostic_selection = {
                        "diagnostic_smoke": True,
                        "formal_evaluation": False,
                        "selection": (
                            "explicit_dialogue_ids"
                            if integration_dialogue_ids
                            else "chunk_prefix"
                        ),
                        "requested_dialogue_ids": list(integration_dialogue_ids),
                        "selected_dialogue_ids": [
                            str(chunk.metadata.get("dialogue_id") or "")
                            for chunk in chunks
                        ],
                    }
                current_session = ""
                for chunk in chunks:
                    session_id = str(chunk.metadata.get("session_id") or "")
                    if (
                        baseline != "MIRIX"
                        and current_session
                        and session_id != current_session
                    ):
                        try:
                            native_adapter.end_session(current_session)
                        except Exception as exc:
                            handle_build_failure(
                                exc,
                                chunk=None,
                                point_kind="end_session",
                                session_id=current_session,
                            )
                        else:
                            consecutive_build_failures = 0
                    try:
                        native_adapter.ingest(chunk)
                    except Exception as exc:
                        handle_build_failure(
                            exc,
                            chunk=chunk,
                            point_kind="ingest",
                            session_id=session_id,
                        )
                        current_session = session_id
                        continue
                    consecutive_build_failures = 0
                    current_session = session_id
                if current_session:
                    try:
                        native_adapter.end_session(current_session)
                    except Exception as exc:
                        handle_build_failure(
                            exc,
                            chunk=None,
                            point_kind="end_session",
                            session_id=current_session,
                        )
                    else:
                        consecutive_build_failures = 0
        terminal_sample_error = ""
        for manifest_question_id, qa_index, qa in selected_qas:
            category = str(qa.get("point", ""))
            if ordered_question_ids is None and is_excluded_category(
                category, excluded_categories
            ):
                excluded += 1
                continue
            if ordered_question_ids is None and max_qa and processed >= max_qa:
                break
            processed += 1
            question = str(qa.get("question", ""))
            query_image = resolve_question_image(data_dir, qa)
            query_id = make_query_id(
                dataset_name=dataset_name,
                qa_index=qa_index,
                category=category,
                question=question,
                query_image=query_image,
            )
            if manifest_question_id in completed_jobs:
                completed_job = completed_jobs[manifest_question_id]
                jobs.append(completed_job)
                completed_error = str(
                    (completed_job.get("native_answer") or {}).get("error") or ""
                )
                if (
                    baseline == "MMA"
                    and completed_error
                    and is_mma_consecutive_bad_point_error(
                        RuntimeError(completed_error)
                    )
                ):
                    terminal_sample_error = completed_error
                continue
            skipped_after_terminal = bool(terminal_sample_error)
            query_vector = (
                query_cache.get(
                    dataset_name=dataset_name,
                    qa_index=qa_index,
                    category=category,
                    question=question,
                    query_image=query_image,
                )
                if query_cache is not None and not skipped_after_terminal
                else None
            )
            if (
                baseline == "HiveMem"
                and query_vector is None
                and not skipped_after_terminal
            ):
                raise KeyError(f"Missing cached query embedding: {query_id}")
            retrieval_error = ""
            if skipped_after_terminal:
                retrieval_error = terminal_sample_error
                retrieval = RetrievalResult(
                    trace={
                        "failed": True,
                        "stage": "sample_terminal_skip",
                        "error": retrieval_error,
                        "skipped_after_consecutive_bad_points": True,
                    }
                )
            else:
                try:
                    with (
                        call_recorder.scope(
                            query_id=query_id,
                            chain_id=query_id,
                            operation="native_retrieve",
                            attempt=1,
                        )
                        if call_recorder is not None
                        else nullcontext()
                    ), (
                        call_recorder.phase("retrieval")
                        if call_recorder is not None
                        else nullcontext()
                    ):
                        retrieval = native_adapter.retrieve(
                            RetrievalRequest(
                                query_id=query_id,
                                text=(
                                    question
                                    if baseline in {"MIRIX", "MMA", "M2A"}
                                    else f"[{category}] {question}"
                                ),
                                category=category,
                                top_k=top_k,
                                query_image=(
                                    str(query_image.get("path") or "")
                                    if isinstance(query_image, dict)
                                    else None
                                ),
                                query_vector=query_vector,
                            )
                        )
                except Exception as exc:
                    terminal_retrieval = (
                        allow_native_qa_errors
                        and baseline == "MMA"
                        and is_mma_consecutive_bad_point_error(exc)
                    )
                    if not (
                        allow_native_qa_errors
                        and (baseline == "MIRIX" or terminal_retrieval)
                    ):
                        raise
                    retrieval_error = str(exc)
                    if terminal_retrieval:
                        terminal_sample_error = retrieval_error
                    retrieval = RetrievalResult(
                        trace={
                            "failed": True,
                            "stage": "retrieval",
                            "error": retrieval_error,
                            "sample_terminal": terminal_retrieval,
                        }
                    )
            memory_items = result_context_items(retrieval)
            trace_rows = result_trace_rows(retrieval)
            native_answer = None
            terminal_native_error: Exception | None = None
            if retrieval_error and (
                baseline == "MIRIX" or bool(terminal_sample_error)
            ):
                native_answer = {
                    "text": "",
                    "error": retrieval_error,
                    "usage": None,
                    "attempts": int(overrides.get("retries") or 0) + 1,
                    "failed_attempts": int(overrides.get("retries") or 0) + 1,
                    "image_count": 1 if query_image else 0,
                    "trace": {
                        "failed": True,
                        "stage": "retrieval",
                        "error": retrieval_error,
                    },
                }
            elif baseline in {"MIRIX", "MMA"}:
                native_evidence, _ = build_retrieved_memory_evidence(
                    memory_items, category
                )
                native_messages = build_answer_messages(
                    question=question,
                    question_type=category,
                    memory_evidence=native_evidence,
                    query_images=query_image_prompt_metadata(query_image),
                    allow_empty_evidence=True,
                )
                with (
                    call_recorder.scope(
                        query_id=query_id,
                        chain_id=query_id,
                        operation="native_answer",
                        attempt=1,
                    )
                    if call_recorder is not None
                    else nullcontext()
                ), (
                    call_recorder.phase("qa")
                    if call_recorder is not None
                    else nullcontext()
                ):
                    try:
                        native_result = native_adapter.answer_with_memory(
                            NativeAnswerRequest(
                                query_id=query_id,
                                messages=native_messages,
                                retrieval=retrieval,
                                query_image=(
                                    str(query_image.get("path") or "")
                                    if isinstance(query_image, dict)
                                    else None
                                ),
                                top_k=top_k,
                            )
                        )
                    except Exception as exc:
                        if not allow_native_qa_errors:
                            raise
                        if baseline == "MMA" and is_mma_consecutive_bad_point_error(exc):
                            terminal_native_error = exc
                            terminal_sample_error = str(exc)
                        error = str(exc)
                        native_answer = {
                            "text": "",
                            "error": error,
                            "usage": None,
                            "attempts": int(overrides.get("retries") or 0) + 1,
                            "failed_attempts": int(overrides.get("retries") or 0) + 1,
                            "image_count": 1 if query_image else 0,
                            "trace": {"failed": True, "error": error},
                        }
                    else:
                        if native_result.retrieval is not None:
                            retrieval = native_result.retrieval
                            memory_items = result_context_items(retrieval)
                            trace_rows = result_trace_rows(retrieval)
                        native_answer = native_result.to_dict()
            retrieved_groups = [row["source_dialogue_ids"] for row in trace_rows]
            retrieved_ids = list(
                dict.fromkeys(source for group in retrieved_groups for source in group)
            )
            clue = qa.get("clue", []) if isinstance(qa.get("clue", []), list) else []
            job = {
                    "query_id": query_id,
                    "manifest_question_id": manifest_question_id,
                    "sample_id": profile.get("name", dataset_name),
                    "dataset": dataset_name,
                    "session_id": qa.get("session_id", ""),
                    "qa_index": qa_index,
                    "question": question,
                    "category": category,
                    "speaker_a": speaker_a,
                    "query_image": query_image,
                    "original_answer": qa.get("answer", ""),
                    "retrieved_ids": retrieved_ids,
                    "retrieved_source_groups": retrieved_groups,
                    "clue": clue,
                    "memory_items": memory_items,
                    "retrieval_top_k": trace_rows,
                    "retrieval_method_trace": dict(retrieval.trace),
                    "native_answer": native_answer,
                    "sample_terminal_error": terminal_sample_error,
                    "skipped_after_consecutive_bad_points": skipped_after_terminal,
                }
            jobs.append(job)
            if on_qa_completed is not None:
                if not job.get("native_answer"):
                    raise RuntimeError(
                        "durable native QA checkpoint requires a native answer"
                    )
                on_qa_completed(job)
        snapshots = (
            [row.to_dict() for row in native_adapter.snapshot()]
            if baseline != "HiveMem"
            else []
        )
    finally:
        native_adapter.close()
    return {
        "sample_id": dataset_name,
        "jobs": jobs,
        "snapshots": snapshots,
        "eligible_questions": processed,
        "excluded_questions": excluded,
        "diagnostic_selection": diagnostic_selection,
        "build_failures": build_failures,
        "build_failure_policy": {
            "enabled": skip_failed_build_points,
            "max_consecutive_failed_build_points": max_consecutive_build_failures,
            "successful_point_resets_consecutive_count": True,
            "incomplete_tool_arguments_are_executed": False,
        },
    }


def answer_dataset_job(
    client: VLMAnswerClient,
    job: dict[str, Any],
    *,
    allow_answer_errors: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, _ = build_retrieved_memory_evidence(
        job["memory_items"], job["category"]
    )
    prompt_evidence, zero_hit_prompt_marker_used = evidence_with_zero_hit_marker(
        evidence
    )
    messages = build_answer_messages(
        question=str(job.get("question") or ""),
        question_type=str(job.get("category") or ""),
        memory_evidence=prompt_evidence,
        query_images=query_image_prompt_metadata(job.get("query_image")),
    )
    raw_answer = ""
    response = None
    native = job.get("native_answer")
    usage = None
    attempts = 0
    failed_attempts = 0
    image_count = 0
    try:
        if native:
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
                category=job["category"],
            )
            raw_answer = response.text
            usage = response.usage
            attempts = response.attempts
            failed_attempts = response.failed_attempts
            image_count = response.image_count
        answer, error = parse_answer_response(raw_answer), ""
    except Exception as exc:
        if not allow_answer_errors:
            raise RuntimeError(
                f"Answer request failed for {job['dataset']} QA {job['qa_index']}: {exc}"
            ) from exc
        answer, error = "", str(exc)
        if response is not None:
            usage = response.usage
            attempts = response.attempts
            failed_attempts = min(attempts, response.failed_attempts + 1)
            image_count = response.image_count
        elif not native:
            usage = None
            attempts = client.retries + 1
            failed_attempts = attempts
            image_count = client.count_answer_images(
                job["memory_items"],
                query_image=job.get("query_image"),
                category=job["category"],
            )
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], job["category"]
    )
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "qa_index", "question_prompt", "system_prompt",
            "query_image", "memory_items", "retrieval_top_k",
            "retrieval_method_trace", "native_answer",
        }
    }
    result.update(
        {
            "system_answer": answer,
            "answer_raw_response": raw_answer,
            "error": error,
            "answer_token_usage": usage,
            "answer_attempts": attempts,
            "answer_failed_attempts": failed_attempts,
            "answer_image_count": image_count,
            "native_answer_trace": dict(
                (job.get("native_answer") or {}).get("trace") or {}
            ),
            "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    trace = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "dataset": job["dataset"],
        "qa_index": job["qa_index"],
        "question": job["question"],
        "category": job["category"],
        "clue": job["clue"],
        "top_k": job["retrieval_top_k"],
        "memory_context": memory_context,
        "answer_prompt_messages": messages,
        "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
        "retrieval_method_trace": dict(job.get("retrieval_method_trace") or {}),
        "m2a_trace": dict(
            (job.get("retrieval_method_trace") or {}).get("m2a_trace") or {}
        ),
    }
    return result, trace


def run_dataset(
    dataset_path: Path,
    data_dir: Path,
    index_root: Path,
    client: VLMAnswerClient,
    query_cache: QueryEmbeddingCache | None,
    *,
    top_k: int = 7,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
    graph_options: dict | None = None,
    baseline: str = "HiveMem",
    state_root: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    memory_snapshots: list[dict[str, Any]] | None = None,
    allow_answer_errors: bool = True,
    excluded_categories: frozenset[str] = frozenset(),
    ordered_question_ids: tuple[str, ...] | None = None,
    qa_stats: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Compatibility wrapper for a single dataset; the full runner uses two phases."""
    artifact = prepare_dataset_jobs(
        dataset_path, data_dir, index_root, query_cache,
        top_k=top_k, max_qa=max_qa, qa_start=qa_start, qa_end=qa_end,
        graph_options=graph_options, baseline=baseline,
        state_root=state_root, config_overrides=config_overrides,
        excluded_categories=excluded_categories,
        ordered_question_ids=ordered_question_ids,
    )
    pairs = [
        answer_dataset_job(client, job, allow_answer_errors=allow_answer_errors)
        for job in artifact["jobs"]
    ]
    if memory_snapshots is not None:
        memory_snapshots.extend(artifact["snapshots"])
    if qa_stats is not None:
        qa_stats.update(
            eligible_questions=artifact["eligible_questions"],
            excluded_questions=artifact["excluded_questions"],
        )
    return [row[0] for row in pairs], [row[1] for row in pairs]


def _checkpoint_signature(
    args: argparse.Namespace,
    dataset_paths: list[Path],
) -> dict[str, Any]:
    ignored = {
        "resume",
        "sample_concurrency",
        "answer_concurrency",
        "checkpoint_every",
        "allow_answer_errors",
        "memory_tokenizer",
        "retrieval_memory_tokenizer",
        "result_dir",
        "answer_api_key",
    }
    input_paths: list[Path] = list(dataset_paths)
    if args.split_manifest:
        input_paths.append(Path(args.split_manifest))
    if args.baseline == "HiveMem":
        query_root = Path(args.query_embedding_dir)
        input_paths.extend(
            [
                query_root / "vectors.npy",
                query_root / "metadata.jsonl",
                query_root / "manifest.json",
            ]
        )
        index_root = Path(args.index_root)
        input_paths.append(index_root / "build_manifest.json")
        for dataset_path in dataset_paths:
            bank = index_root / "datasets" / dataset_path.stem
            input_paths.extend(
                [
                    bank / "memories.jsonl",
                    bank / "text_vectors.npy",
                    bank / "image_vectors.npy",
                    bank / "image_mask.npy",
                ]
            )
    return {
        "arguments": {
            key: value for key, value in vars(args).items() if key not in ignored
        },
        "inputs": file_manifest(input_paths),
        **prompt_manifest(),
        "call_trace_version": TRACE_VERSION,
    }


def _checkpoint_signatures_match(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    """Accept checkpoints made before API keys were removed from signatures."""
    normalized = dict(saved)
    arguments = dict(normalized.get("arguments") or {})
    arguments.pop("answer_api_key", None)
    normalized["arguments"] = arguments
    return normalized == current


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory baseline on Mem-Gallery.")
    parser.add_argument("--baseline", default="HiveMem")
    parser.add_argument("--data-name", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--data-dir", default=str(DEFAULT_MEMGALLERY_DATA_DIR))
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--split", default="")
    parser.add_argument(
        "--index-root",
        default="",
        help="HiveMem run directory containing datasets/.",
    )
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument(
        "--m3-reuse-state-root",
        default="",
        help="Read-only M3 memory/datasets root from a completed prior run.",
    )
    parser.add_argument("--query-embedding-dir", default="data/qwen3_vl_embedding_2b/query_embeddings")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument(
        "--mirix-multimodal-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For MIRIX only, embed each native memory field with the source "
            "batch images and embed visual questions with their query image."
        ),
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--qa-start", type=int, default=1)
    parser.add_argument("--qa-end", type=int, default=0)
    parser.add_argument(
        "--integration-chunk-limit",
        type=int,
        default=0,
        help="Explicit integration-test-only prefix of the immutable chunk JSONL.",
    )
    parser.add_argument(
        "--integration-dialogue-ids",
        default="",
        help=(
            "Explicit comma-separated dialogue IDs for a non-scoring diagnostic smoke; "
            "cannot be combined with --integration-chunk-limit or a split manifest."
        ),
    )
    parser.add_argument(
        "--qa-indices",
        default="",
        help="Explicit comma-separated 1-based QA indices for integration tests.",
    )
    parser.add_argument(
        "--exclude-categories",
        default="AR",
        help="Comma-separated QA categories to skip before embedding, retrieval, and answering.",
    )
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--answer-api-key", default=os.getenv("OPENAI_API_KEY") or "EMPTY"
    )
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--cost-mb-input-price", type=float, default=None)
    parser.add_argument("--cost-mb-output-price", type=float, default=None)
    parser.add_argument("--cost-qa-input-price", type=float, default=None)
    parser.add_argument("--cost-qa-output-price", type=float, default=None)
    parser.add_argument("--efficiency-config", default="configs/model_efficiency.json")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--qa-request-timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--executor-temperature", type=float, default=0.0)
    parser.add_argument("--executor-max-tokens", type=int, default=512)
    parser.add_argument("--executor-hard-max-tokens", type=int, default=0)
    parser.add_argument("--mma-native-batch-size", type=int, default=20)
    parser.add_argument(
        "--mirix-skip-failed-build-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject and audit a failed MIRIX build point, then continue unless "
            "the configured consecutive-failure threshold is reached."
        ),
    )
    parser.add_argument(
        "--mirix-max-consecutive-failed-build-points",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--m2a-skip-failed-build-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Audit and skip an isolated M2A memory-build point; stop when the "
            "configured consecutive-failure threshold is reached."
        ),
    )
    parser.add_argument(
        "--m2a-max-consecutive-failed-build-points",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--m2a-salvage-truncated-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--executor-visual-input", choices=("image", "caption"), default="image")
    parser.add_argument(
        "--allow-answer-errors",
        action="store_true",
        help="Write metrics even when one or more VLM answer requests failed.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from the dataset-level checkpoint under RESULT_DIR/.checkpoint.",
    )
    parser.add_argument(
        "--graph-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Graph-expanded retrieval: keep vector top-k and append graph neighbours",
    )
    parser.add_argument("--seed-k", type=int, default=0, help="Seed count for expansion (0 = top_k)")
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--expand-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expand-related", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--related-types", default="",
                        help="Comma-separated edge types to expand via links.related (empty = all)")
    parser.add_argument("--graph-mode", default="append", choices=["rerank", "append"],
                        help="rerank: neighbours compete for top_k; append: vector top_k kept, neighbours appended")
    parser.add_argument("--append-k", type=int, default=2)
    parser.add_argument("--graph-categories", default="",
                        help="Comma-separated categories that use graph retrieval (empty = all)")
    parser.add_argument("--expand-entity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expand-attribute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--df-max", type=float, default=0.3)
    parser.add_argument("--df-stop", type=float, default=0.5)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--degree-cap", type=int, default=10)
    parser.add_argument(
        "--memory-tokenizer",
        default="",
        help="Tokenizer used only to backfill build tokens for historical traces without usage.",
    )
    parser.add_argument(
        "--retrieval-memory-tokenizer",
        default="",
        help="Tokenizer for retrieved-memory text; defaults to --answer-model.",
    )
    from hive_mem.build_memories import apply_config_defaults
    apply_config_defaults(parser)
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be provided together")
    manifest_index = (
        SplitManifestIndex(args.split_manifest) if args.split_manifest else None
    )
    manifest_split = normalize_split_name(args.split) if args.split else ""
    try:
        integration_qa_indices = tuple(
            int(value.strip())
            for value in str(args.qa_indices).split(",")
            if value.strip()
        )
    except ValueError:
        parser.error("--qa-indices must be comma-separated positive integers")
    if any(value < 1 for value in integration_qa_indices):
        parser.error("--qa-indices must contain only positive integers")
    integration_dialogue_ids = tuple(
        dict.fromkeys(
            value.strip()
            for value in str(args.integration_dialogue_ids).split(",")
            if value.strip()
        )
    )
    if args.integration_chunk_limit and integration_dialogue_ids:
        parser.error(
            "--integration-chunk-limit and --integration-dialogue-ids are mutually exclusive"
        )
    if (args.integration_chunk_limit or integration_dialogue_ids) and not integration_qa_indices:
        parser.error("partial diagnostic chunk selection requires --qa-indices")
    if (args.integration_chunk_limit or integration_dialogue_ids or integration_qa_indices) and (
        manifest_index is not None or args.all_datasets
    ):
        parser.error("integration selection requires one --data-name and no split manifest")
    if manifest_index is not None and (
        args.max_qa or args.qa_start != 1 or args.qa_end
    ):
        parser.error(
            "QA ranges/limits cannot be combined with strict manifest selection"
        )
    excluded_categories = (
        frozenset()
        if manifest_index is not None
        else parse_excluded_categories(args.exclude_categories)
    )
    try:
        args.baseline = canonical_name(args.baseline)
    except KeyError as exc:
        parser.error(str(exc))
    if args.baseline == "HiveMem" and not args.index_root:
        parser.error("--index-root is required when --baseline=HiveMem")
    if (
        args.top_k < 1
        or args.max_qa < 0
        or args.qa_start < 1
        or args.qa_end < 0
        or args.integration_chunk_limit < 0
    ):
        parser.error("--top-k and --qa-start must be positive; QA limits cannot be negative")
    if args.sample_concurrency < 1 or args.answer_concurrency < 1 or args.checkpoint_every < 1:
        parser.error("Sample/answer concurrency and checkpoint interval must be positive")
    if args.qa_end and args.qa_end < args.qa_start:
        parser.error("--qa-end must be 0 or at least --qa-start")
    if (
        args.retries < 0
        or args.request_timeout <= 0
        or args.qa_request_timeout <= 0
        or args.executor_hard_max_tokens < 0
        or args.num_predict < 1
        or args.mirix_max_consecutive_failed_build_points < 1
        or args.m2a_max_consecutive_failed_build_points < 1
    ):
        parser.error("Invalid answer retry, timeout, or token limit")

    graph_options: dict | bool = False
    if args.graph_retrieval:
        graph_options = {
            "seed_k": args.seed_k,
            "expansion_bonus": args.expansion_bonus,
            "expand_temporal": args.expand_temporal,
            "expand_related": args.expand_related,
            "expand_entity": args.expand_entity,
            "expand_attribute": args.expand_attribute,
            "related_types": (
                {t.strip() for t in args.related_types.split(",") if t.strip()}
                if args.related_types else None
            ),
            "df_max": args.df_max,
            "df_stop": args.df_stop,
            "min_shared": args.min_shared,
            "degree_cap": args.degree_cap,
            "mode": args.graph_mode,
            "append_k": args.append_k,
            "categories": (
                {c.strip() for c in args.graph_categories.split(",") if c.strip()}
                if args.graph_categories else None
            ),
        }

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
    query_cache = (
        QueryEmbeddingCache(
            args.query_embedding_dir,
            expected_dim=args.embedding_dim,
            expected_model=args.embedding_model,
        )
        if args.baseline == "HiveMem"
        and args.query_embedding_dir
        and Path(args.query_embedding_dir).exists()
        else None
    )
    data_dir = Path(args.data_dir)
    ordered_ids_by_dataset: dict[str, tuple[str, ...]] = {}
    if manifest_index is not None:
        manifest_rows = manifest_index.conversations(
            manifest_split, data_source="mem_gallery"
        )
        paths = [data_dir / "dialog" / f"{row.source_id}.json" for row in manifest_rows]
        ordered_ids_by_dataset = {
            row.source_id: row.question_ids for row in manifest_rows
        }
    else:
        paths = (
            sorted((data_dir / "dialog").glob("*.json"))
            if args.all_datasets
            else [data_dir / "dialog" / f"{args.data_name}.json"]
        )
        if integration_qa_indices:
            ordered_ids_by_dataset[args.data_name] = tuple(
                memgallery_manifest_question_id(args.data_name, index)
                for index in integration_qa_indices
            )
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"MemGallery dataset file(s) not found: {missing_paths}")
    if not paths:
        raise FileNotFoundError(f"No MemGallery datasets found under {data_dir / 'dialog'}")
    source_questions = 0
    source_excluded_questions = 0
    for path in paths:
        source_qas = json.loads(path.read_text(encoding="utf-8")).get(
            "human-annotated QAs", []
        ) or []
        source_questions += len(source_qas)
        source_excluded_questions += sum(
            is_excluded_category(qa.get("point", ""), excluded_categories)
            for qa in source_qas
        )
    result_dir = Path(args.result_dir)
    output_layout = BaselineOutputLayout(result_dir)
    baseline_state_root = output_layout.state_root(args.baseline_state_dir)
    checkpoint_dir = output_layout.checkpoint_dir
    checkpoint_results = checkpoint_dir / "results.json"
    checkpoint_traces = checkpoint_dir / "retrieval_trace.jsonl"
    checkpoint_manifest = checkpoint_dir / "manifest.json"
    signature = _checkpoint_signature(args, paths)
    computed_sample_signature = signature_digest(signature)
    qa_only_reuse = os.getenv("MMA_QA_ONLY_REUSE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    stored_sample_signatures = (
        (validated_qa_only_resume_signatures if qa_only_reuse else validated_paired_resume_signatures)(
            (
                path.stem,
                baseline_state_root / path.stem / (
                    ".mma_reuse_provenance.json"
                    if qa_only_reuse else ".offline_mma_resume.json"
                ),
                checkpoint_dir
                / "native_samples"
                / Path(trace_filename(path.stem)).with_suffix(".json"),
            )
            for path in paths
        )
        if args.baseline == "MMA" and args.resume
        else ()
    )
    compatible_sample_signatures = tuple(
        dict.fromkeys((*stored_sample_signatures, computed_sample_signature))
    )
    sample_signature = compatible_sample_signatures[0]

    def prepare(path: Path) -> dict[str, Any]:
        if args.resume:
            cached = load_sample_artifact(
                output_layout.sample_checkpoint_dir,
                path.stem,
                signature=sample_signature,
            )
            if cached is not None:
                print(f"[resume] skip prepared dataset: {path.stem}", flush=True)
                return cached
        native_progress_path = (
            checkpoint_dir
            / "native_samples"
            / Path(trace_filename(path.stem)).with_suffix(".json")
        )
        native_progress_jobs: list[dict[str, Any]] = []
        invalidated_hard_stop_jobs: list[dict[str, Any]] = []
        if args.baseline in {"MMA", "MIRIX"} and args.resume and native_progress_path.is_file():
            try:
                native_progress = json.loads(
                    native_progress_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                native_progress = {}
            if (
                native_progress.get("version") == 1
                and native_progress.get("sample_id") == path.stem
                and native_progress.get("signature")
                in compatible_sample_signatures
            ):
                loaded_jobs = list(native_progress.get("jobs") or [])
                invalidated_hard_stop_jobs = [
                    *list(native_progress.get("invalidated_hard_stop_jobs") or []),
                    *[
                        row
                        for row in loaded_jobs
                        if _is_hard_stop_native_qa_job(row)
                    ],
                ]
                native_progress_jobs = [
                    row
                    for row in loaded_jobs
                    if not _is_failed_native_qa_job(row)
                ]

        def checkpoint_native_qa(job: dict[str, Any]) -> None:
            manifest_id = str(job.get("manifest_question_id") or "")
            if any(
                str(row.get("manifest_question_id") or "") == manifest_id
                for row in native_progress_jobs
            ):
                return
            native_progress_jobs.append(job)
            if native_progress_path.is_file():
                try:
                    existing = json.loads(
                        native_progress_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing = {}
                existing_count = sum(
                    not _is_failed_native_qa_job(row)
                    for row in (existing.get("jobs") or [])
                )
                if existing_count > len(native_progress_jobs):
                    raise RuntimeError(
                        "refusing to regress native QA checkpoint from "
                        f"{existing_count} to {len(native_progress_jobs)} questions"
                    )
            write_json_atomic(
                native_progress_path,
                {
                    "version": 1,
                    "sample_id": path.stem,
                    "signature": sample_signature,
                    "status": "running",
                    "completed_questions": len(native_progress_jobs),
                    "jobs": native_progress_jobs,
                    "invalidated_hard_stop_jobs": invalidated_hard_stop_jobs,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        config_overrides = {
                "index_root": args.index_root,
                "graph_options": graph_options,
                "answer_model": args.answer_model,
                "answer_base_url": args.answer_base_url,
                "answer_temperature": args.answer_temperature,
                "executor_model": args.executor_model,
                "executor_base_url": args.executor_base_url,
                "executor_temperature": args.executor_temperature,
                "executor_max_tokens": args.executor_max_tokens,
                "mirix_executor_retry_max_tokens": (
                    args.executor_hard_max_tokens or args.executor_max_tokens
                ),
                "mma_native_batch_size": args.mma_native_batch_size,
                "mirix_skip_failed_build_points": args.mirix_skip_failed_build_points,
                "mirix_max_consecutive_failed_build_points": (
                    args.mirix_max_consecutive_failed_build_points
                ),
                "m2a_skip_failed_build_points": args.m2a_skip_failed_build_points,
                "m2a_max_consecutive_failed_build_points": (
                    args.m2a_max_consecutive_failed_build_points
                ),
                "m2a_salvage_truncated_updates": (
                    args.m2a_salvage_truncated_updates
                ),
                "executor_visual_input": args.executor_visual_input,
                "executor_native_tool_calls": args.baseline in {"MIRIX", "MMA"},
                "integration_chunk_limit": args.integration_chunk_limit,
                "integration_dialogue_ids": integration_dialogue_ids,
                "embedding_model": args.embedding_model,
                "embedding_base_url": args.embedding_base_url,
                "embedding_dim": args.embedding_dim,
                "mirix_multimodal_embedding": args.mirix_multimodal_embedding,
                "top_k": args.top_k,
                "request_timeout": args.request_timeout,
                "retries": args.retries,
                "reasoning_effort": args.reasoning_effort,
            "m3_reuse_state_root": args.m3_reuse_state_root,
            "mma_resume_enabled": args.resume,
            "mma_resume_signature": sample_signature,
            "mma_resume_compatible_signatures": list(
                compatible_sample_signatures
            ),
            "mirix_resume_enabled": args.resume,
            "mirix_resume_signature": sample_signature,
            }
        call_trace_path = result_dir / "call_traces" / trace_filename(path.stem)
        recorder = None
        proxy_context = nullcontext(None)
        if args.baseline != "HiveMem":
            recorder = CallRecorder(
                trace_path=call_trace_path,
                baseline=args.baseline,
                benchmark="Mem-Gallery",
                sample_id=path.stem,
                # Preserve billable retry calls even when native state must be
                # rebuilt from the sample checkpoint during a resumed run.
                reset=not args.resume,
            )
            proxy_context = CountingProxy(
                args.executor_base_url,
                recorder,
                args.request_timeout,
                max_output_tokens=(
                    args.executor_hard_max_tokens or args.executor_max_tokens
                ),
                temperature=args.executor_temperature,
                qa_max_output_tokens=args.num_predict,
                reasoning_effort=args.reasoning_effort,
                qa_upstream_timeout=args.qa_request_timeout,
            )
        with proxy_context as proxy:
            if proxy is not None:
                config_overrides["executor_base_url"] = proxy.endpoint
            artifact = prepare_dataset_jobs(
                path,
                data_dir,
                Path(args.index_root) if args.index_root else Path(),
                query_cache,
                top_k=args.top_k,
                max_qa=args.max_qa,
                qa_start=args.qa_start,
                qa_end=args.qa_end,
                graph_options=graph_options,
                baseline=args.baseline,
                state_root=baseline_state_root,
                config_overrides=config_overrides,
                excluded_categories=excluded_categories,
                ordered_question_ids=ordered_ids_by_dataset.get(path.stem),
                call_recorder=recorder,
                completed_jobs={
                    str(row.get("manifest_question_id") or ""): row
                    for row in native_progress_jobs
                    if row.get("manifest_question_id")
                },
                on_qa_completed=(
                    checkpoint_native_qa
                    if args.baseline in {"MMA", "MIRIX"}
                    else None
                ),
                allow_native_qa_errors=args.allow_answer_errors,
            )
        if recorder is not None:
            artifact["call_trace_path"] = str(call_trace_path)
        save_sample_artifact(
            output_layout.sample_checkpoint_dir,
            path.stem,
            signature=sample_signature,
            artifact=artifact,
        )
        if args.baseline in {"MMA", "MIRIX"}:
            write_json_atomic(
                native_progress_path,
                {
                    "version": 1,
                    "sample_id": path.stem,
                    "signature": sample_signature,
                    "status": "completed",
                    "completed_questions": len(artifact.get("jobs") or []),
                    "jobs": list(artifact.get("jobs") or []),
                    "invalidated_hard_stop_jobs": invalidated_hard_stop_jobs,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        print(
            f"[prepared] {path.stem}: {len(artifact['jobs'])} question(s)",
            flush=True,
        )
        return artifact

    artifacts = parallel_map_ordered(
        paths,
        prepare,
        max_workers=args.sample_concurrency,
        item_key=lambda path: path.stem,
    )
    jobs = [job for artifact in artifacts for job in artifact["jobs"]]
    expected_manifest_question_ids = (
        manifest_index.ordered_question_ids(
            manifest_split, data_source="mem_gallery"
        )
        if manifest_index is not None
        else None
    )
    if expected_manifest_question_ids is not None:
        actual_manifest_question_ids = tuple(
            str(job.get("manifest_question_id") or "") for job in jobs
        )
        if actual_manifest_question_ids != expected_manifest_question_ids:
            raise RuntimeError(
                "Mem-Gallery prepared jobs do not exactly match manifest question order"
            )
    write_jsonl_atomic(output_layout.pipeline_qa, jobs)
    memory_snapshots = [
        row for artifact in artifacts for row in artifact.get("snapshots", [])
    ]
    diagnostic_selections = [
        dict(artifact.get("diagnostic_selection") or {})
        for artifact in artifacts
        if artifact.get("diagnostic_selection")
    ]
    excluded_questions = sum(
        int(artifact.get("excluded_questions", 0)) for artifact in artifacts
    )

    results_by_id: dict[str, dict[str, Any]] = {}
    traces_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_manifest.is_file():
        saved = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if _checkpoint_signatures_match(saved.get("signature") or {}, signature):
            if checkpoint_results.is_file() and checkpoint_traces.is_file():
                results_by_id = {
                    str(row.get("query_id") or ""): row
                    for row in json.loads(checkpoint_results.read_text(encoding="utf-8"))
                    if row.get("query_id")
                }
                traces_by_id = {
                    str(row.get("query_id") or ""): row
                    for row in (
                        json.loads(line)
                        for line in checkpoint_traces.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                    if row.get("query_id")
                }
                print(f"[resume] loaded {len(results_by_id)} answer(s)", flush=True)

    job_ids = [str(job["query_id"]) for job in jobs]

    def save_checkpoint() -> None:
        completed = [query_id for query_id in job_ids if query_id in results_by_id]
        write_json_atomic(checkpoint_results, [results_by_id[key] for key in completed])
        write_jsonl_atomic(
            checkpoint_traces,
            [traces_by_id[key] for key in completed if key in traces_by_id],
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

    completed_answers = {
        query_id
        for query_id, row in results_by_id.items()
        if query_id in traces_by_id
        and (args.baseline == "MMA" or not row.get("error"))
    }
    pending = [job for job in jobs if job["query_id"] not in completed_answers]
    if pending:
        client.assert_model_available()
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
        futures = {
            pool.submit(
                answer_dataset_job,
                client,
                job,
                allow_answer_errors=args.allow_answer_errors,
            ): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            result, trace = future.result()
            query_id = str(job["query_id"])
            result["query_id"] = query_id
            trace["query_id"] = query_id
            results_by_id[query_id] = result
            traces_by_id[query_id] = trace
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0
            print(
                f"[{len(results_by_id)}/{len(jobs)}] {job['dataset']} "
                f"QA {job['qa_index']} error={result['error'][:80]!r}",
                flush=True,
            )
    save_checkpoint()
    all_results = [results_by_id[query_id] for query_id in job_ids]
    all_traces = [traces_by_id[query_id] for query_id in job_ids]
    if expected_manifest_question_ids is not None:
        for label, rows in (("results", all_results), ("retrieval traces", all_traces)):
            actual = tuple(str(row.get("manifest_question_id") or "") for row in rows)
            if actual != expected_manifest_question_ids:
                raise RuntimeError(
                    f"Mem-Gallery {label} do not exactly match manifest question order"
                )

    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", all_results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", all_traces)
    if args.baseline == "HiveMem":
        memory_snapshots = load_hivemem_snapshot(
            args.index_root,
            (path.stem for path in paths),
        )
    write_jsonl_atomic(output_layout.snapshot, memory_snapshots)
    answer_errors = sum(bool(row.get("error")) for row in all_results)
    public_args = {
        key: value for key, value in vars(args).items() if key != "answer_api_key"
    }
    if args.baseline != "HiveMem":
        public_args["baseline_state_dir"] = str(baseline_state_root)
    public_args["memory_snapshot"] = str(output_layout.snapshot)
    manifest = public_args | prompt_manifest() | {
        "questions": len(all_results),
        "excluded_categories": sorted(excluded_categories),
        "excluded_questions": excluded_questions,
        "source_questions": source_questions,
        "source_excluded_questions": source_excluded_questions,
        "source_eligible_questions": source_questions - source_excluded_questions,
        "answer_errors": answer_errors,
        "baseline_runtime": baseline_metadata(args.baseline),
        "run_signature": signature,
        "selection_mode": (
            "strict_manifest"
            if manifest_index is not None
            else "diagnostic_smoke"
            if args.integration_chunk_limit or integration_dialogue_ids
            else "integration_explicit"
            if integration_qa_indices
            else "legacy"
        ),
        "diagnostic_smoke": bool(diagnostic_selections),
        "formal_evaluation": not bool(diagnostic_selections),
        "diagnostic_selections": diagnostic_selections,
        "split_manifest_sha256": (
            manifest_index.file_sha256 if manifest_index is not None else ""
        ),
        "ordered_question_ids": list(expected_manifest_question_ids or ()),
        "chunk_input": (
            omni_input_manifest("memgallery")
            if args.baseline in {"OmniSimpleMem", "MMA"}
            else m3_input_manifest("memgallery")
            if args.baseline == "M3-Agent-caption"
            else chunk_source_manifest({}, "memgallery")
        ),
    }
    if args.baseline == "M2A":
        manifest["m2a_conformance"] = m2a_conformance_manifest(
            answer_prompt_sha256=prompt_sha256()
        )
    if args.baseline == "MMA":
        manifest["mma_conformance"] = mma_conformance_manifest(
            answer_prompt_sha256=prompt_sha256()
        )
    if args.baseline == "OmniSimpleMem":
        manifest["omni_conformance"] = omni_conformance_manifest("memgallery")
    if args.baseline == "M3-Agent-caption":
        manifest["m3_conformance"] = m3_conformance_manifest(
            "memgallery",
            answer_prompt_sha256=prompt_sha256(),
            handoff_top_k=args.top_k,
        )
    write_json_atomic(result_dir / "run_manifest.json", manifest)
    if answer_errors and not args.allow_answer_errors:
        (result_dir / "metrics.json").unlink(missing_ok=True)
        raise RuntimeError(
            f"{answer_errors}/{len(all_results)} answer requests failed; "
            f"partial results were saved under {result_dir}, but metrics were not written"
        )
    effective_top_k = (
        args.top_k
        if args.baseline == "M3-Agent-caption"
        else args.top_k + args.append_k
        if args.graph_retrieval and args.graph_mode == "append"
        else args.top_k
    )
    summary = summarize_results(all_results, k=effective_top_k)
    evaluated_sample_ids = sorted(
        {str(row.get("dataset") or "").strip() for row in all_results}
        - {""}
    )
    if args.baseline == "HiveMem":
        summary["calls"] = combine_call_metrics(
            calculate_calls_mb(Path(args.index_root), evaluated_sample_ids),
            calculate_calls_qa(all_results, sample_id_field="dataset"),
        )
    else:
        summary["calls"] = write_runtime_call_metrics(
            [
                artifact["call_trace_path"]
                for artifact in artifacts
                if artifact.get("call_trace_path")
            ],
            result_dir,
            all_results,
            sample_id_field="dataset",
            sample_ids=evaluated_sample_ids,
        )
    try:
        write_retrieval_memory_token(
            result_dir,
            tokenizer_name=args.retrieval_memory_tokenizer or args.answer_model,
        )
    except (OSError, KeyError, ValueError) as exc:
        print(f"retrieval memory token metrics unavailable: {exc}", flush=True)
    try:
        if args.baseline == "HiveMem":
            memory_metrics = write_memory_metrics(
                Path(args.index_root),
                result_dir,
                tokenizer_name=args.memory_tokenizer,
                sample_ids=evaluated_sample_ids,
                cost_mb_input_price=args.cost_mb_input_price,
                cost_mb_output_price=args.cost_mb_output_price,
            )
        else:
            memory_metrics = write_snapshot_memory_metrics(
                memory_snapshots,
                result_dir,
                sample_ids=evaluated_sample_ids,
                cost_mb_input_price=args.cost_mb_input_price,
                cost_mb_output_price=args.cost_mb_output_price,
            )
        summary = add_memory_metrics(summary, memory_metrics)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # Legacy baseline traces (including A-Mem) do not always contain the
        # tokenizer metadata needed for an honest memory-token estimate.  QA
        # results are still complete and should not prevent the judge stage.
        print(f"memory metrics unavailable: {exc}", flush=True)
        if args.baseline == "HiveMem":
            summary["cost_mb"] = calculate_cost_mb(
                Path(args.index_root),
                evaluated_sample_ids,
                input_price=args.cost_mb_input_price,
                output_price=args.cost_mb_output_price,
            )
    efficiency = write_efficiency_metrics(
        result_dir,
        all_results,
        sample_id_field="dataset",
        sample_ids=evaluated_sample_ids,
        model=args.answer_model,
        config_path=args.efficiency_config,
        memory_build_model=(
            args.executor_model
            if args.baseline == "M3-Agent-caption"
            else args.answer_model
        ),
        retrieval_model=(
            args.executor_model
            if args.baseline == "M3-Agent-caption"
            else args.answer_model
        ),
        answer_model=args.answer_model,
        hivemem_index_root=(
            Path(args.index_root) if args.baseline == "HiveMem" else None
        ),
    )
    summary.update(
        {
            key: efficiency[key]
            for key in (
                "cost_mb",
                "cost_qa",
                "cost_total",
                "latency_mb",
                "latency_qa",
                "latency_total",
            )
        }
    )
    summary = merge_existing_llm_judge_metrics(summary, result_dir)
    write_json_atomic(result_dir / "metrics.json", summary)


if __name__ == "__main__":
    main()
