from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
    validated_paired_resume_signatures,
    validated_qa_only_resume_signatures,
)
from benchmarks.baseline_runtime.openai_compat import embed_texts
from benchmarks.baseline_runtime.call_trace import (
    CallRecorder,
    CountingProxy,
    TRACE_VERSION,
    trace_filename,
)
from benchmarks.baseline_runtime.build_fault_policy import (
    ConsecutiveBuildFaultPolicy,
)
from benchmarks.baseline_runtime.adapters.m2a import m2a_conformance_manifest
from benchmarks.baseline_runtime.adapters.m3_agent import m3_conformance_manifest
from benchmarks.baseline_runtime.adapters.mma_original import (
    is_mma_consecutive_bad_point_error,
    mma_conformance_manifest,
)
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.protocol import (
    NativeAnswerRequest,
    RetrievalRequest,
    RetrievalResult,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from benchmarks.memgallery_harness.runner.answer_client import (
    build_retrieved_memory_context,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.h2hmem_harness.prompts import (
    PROMPT_SOURCE,
    PROMPT_VERSION,
    build_answer_messages,
    parse_answer_response,
    prompt_sha256,
)
from benchmarks.memgallery_harness.runner.metrics import (
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    merge_existing_llm_judge_metrics,
    summarize_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.fixed_chunks import chunk_source_manifest, h2hmem_chunks
from benchmarks.baseline_runtime.omni_inputs import (
    build_omni_h2h_chunks_from_directory,
    omni_conformance_manifest,
    omni_input_manifest,
)
from benchmarks.baseline_runtime.m3_inputs import (
    build_m3_h2h_chunks_from_directory,
    m3_input_manifest,
)
from embedding.chunk_builder import (
    build_h2h_chunks_from_directory,
    iter_h2h_session_files,
)
from hive_mem.build_memories import apply_config_defaults
from evidence_policy.split_manifest import SplitManifestIndex, normalize_split_name
from benchmarks.zero_hit import evidence_with_zero_hit_marker


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_H2HMEM_DATA_DIR = WORKSPACE_ROOT / "H2HMEM-main" / "dataset"


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

_GLOBAL_SAMPLE_FAILURE_MARKERS = _HARD_STOP_NATIVE_QA_MARKERS + (
    "no space left on device",
    "database or disk is full",
    "input file is damaged",
    "input data is damaged",
)


def _is_global_sample_failure(exc: Exception) -> bool:
    """Distinguish run-wide blockers from one bad baseline sample."""
    text = str(exc).casefold()
    return any(marker in text for marker in _GLOBAL_SAMPLE_FAILURE_MARKERS)


def run_conversation_retry_queue(
    specs: list[tuple[str, str, int, tuple[str, ...] | None]],
    worker: Callable[[tuple[str, str, int, tuple[str, ...] | None]], dict[str, Any]],
    *,
    max_attempts: int,
    status_path: Path,
    on_skipped: Callable[
        [tuple[str, str, int, tuple[str, ...] | None], str], dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """Retry failed conversations at the sample boundary without stopping H2HMem.

    A MIRIX worker timeout terminates the native subprocess, so the current
    build point cannot safely continue in-process.  The durable MIRIX/sample
    checkpoints still make a whole-sample retry safe.  After the bounded retry
    budget, materialize explicit skipped QA rows and continue the benchmark.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    status: dict[str, Any] = {"version": 1, "samples": {}}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if loaded.get("version") == 1 and isinstance(loaded.get("samples"), dict):
                status = loaded
        except (OSError, json.JSONDecodeError):
            pass
    samples = status["samples"]
    for key, value in list(samples.items()):
        if isinstance(value, dict) and value.get("state") == "running":
            attempts = int(value.get("attempts") or 0)
            errors = list(value.get("errors") or [])
            errors.append(
                {
                    "attempt": attempts,
                    "error_type": "InterruptedRun",
                    "error": "previous harness stopped before recording sample outcome",
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            samples[key] = {
                **value,
                "state": "pending_retry",
                # A supervisor takeover resumes the same checkpointed attempt.
                "attempts": max(0, attempts - 1),
                "errors": errors,
            }

    def key_for(spec: tuple[str, str, int, tuple[str, ...] | None]) -> str:
        return f"{spec[0]}/{spec[1]}"

    def save_status() -> None:
        counts = {
            state: sum(
                row.get("state") == state
                for row in samples.values()
                if isinstance(row, dict)
            )
            for state in ("completed", "running", "pending_retry", "skipped")
        }
        status.update(
            {
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "counts": counts,
                "max_attempts_per_sample": max_attempts,
            }
        )
        write_json_atomic(status_path, status)

    pending = deque(specs)
    artifacts: dict[str, dict[str, Any]] = {}
    while pending:
        spec = pending.popleft()
        key = key_for(spec)
        row = dict(samples.get(key) or {})
        attempts = int(row.get("attempts") or 0)
        if row.get("state") == "skipped" and attempts >= max_attempts:
            error = str((row.get("errors") or [{}])[-1].get("error") or "")
            artifacts[key] = on_skipped(spec, error)
            continue
        if row.get("state") == "completed":
            try:
                artifacts[key] = worker(spec)
                continue
            except Exception:
                row["state"] = "pending_retry"
        attempts += 1
        errors = list(row.get("errors") or [])
        samples[key] = {
            **row,
            "state": "running",
            "attempts": attempts,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "errors": errors,
        }
        save_status()
        try:
            artifact = worker(spec)
        except Exception as exc:
            errors.append(
                {
                    "attempt": attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            if _is_global_sample_failure(exc):
                samples[key].update(state="blocked_global", errors=errors)
                save_status()
                raise
            state = "pending_retry" if attempts < max_attempts else "skipped"
            samples[key].update(state=state, errors=errors)
            save_status()
            print(
                f"[sample-{state}] {key} attempt={attempts}/{max_attempts}: {exc}",
                flush=True,
            )
            if state == "pending_retry":
                pending.append(spec)
            else:
                artifacts[key] = on_skipped(spec, str(exc))
        else:
            artifacts[key] = artifact
            samples[key].update(
                state="completed",
                completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_status()
    return [artifacts[key_for(spec)] for spec in specs]


def _is_failed_native_qa_job(job: dict[str, Any]) -> bool:
    """Retry only infrastructure hard stops, never baseline QA bad points."""
    answer = job.get("native_answer") or {}
    if not isinstance(answer, dict):
        return True
    error = str(answer.get("error") or "").casefold()
    return bool(error) and any(
        marker in error for marker in _HARD_STOP_NATIVE_QA_MARKERS
    )


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple(
        int(value) if value.isdigit() else value
        for part in path.parts
        for value in re.split(r"(\d+)", part.casefold())
    )


def _question_files(conversation_dir: Path) -> list[Path]:
    return sorted(
        conversation_dir.glob("scenes/session*/questions.json"),
        key=_natural_key,
    )


def _question_image(question_file: Path, raw: Any) -> dict[str, str] | None:
    value = str(raw or "").strip()
    if not value:
        return None
    scenes_dir = question_file.parents[2] / "scenes"
    if "/" in value or "\\" in value:
        session_name, filename = re.split(r"[/\\]", value, maxsplit=1)
        path = scenes_dir / session_name / "image" / filename
    else:
        path = question_file.parent / "image" / value
    if not path.is_file():
        raise FileNotFoundError(f"H2HMem question image not found: {path}")
    return {"path": str(path.resolve()), "img_id": value}


def _question_rows(conversation_dir: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for path in _question_files(conversation_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            (path, index, question)
            for index, question in enumerate(payload.get("questions") or [], start=1)
            if question.get("validated", True)
        )
    return rows


def h2hmem_manifest_question_id(
    variant: str,
    conversation_id: str,
    session_id: str,
    qa_index: int,
    qa: dict[str, Any],
) -> str:
    """Canonical H2HMem question ID used by the split manifest."""
    original_id = str(
        qa.get("question_id") or qa.get("original_question_id") or ""
    ).strip()
    code = original_id if original_id else f"Q{qa_index:03d}"
    return f"h2hmem:{variant}:{conversation_id}:{session_id}:{code}"


def prepare_conversation_jobs(
    *,
    data_dir: Path,
    variant: str,
    conversation_id: str,
    baseline: str,
    state_root: Path,
    config: dict[str, Any],
    max_qa: int = 0,
    ordered_question_ids: tuple[str, ...] | None = None,
    call_recorder: CallRecorder | None = None,
    completed_jobs: dict[str, dict[str, Any]] | None = None,
    on_qa_completed: Callable[[dict[str, Any]], None] | None = None,
    allow_native_qa_errors: bool = False,
) -> dict[str, Any]:
    variant_dir = "multi-party" if variant == "multiparty" else variant
    conversation_dir = data_dir / variant_dir / conversation_id
    adapter_config = dict(config)
    reuse_root = str(adapter_config.get("m3_reuse_state_root") or "").strip()
    if baseline == "M3-Agent-caption" and reuse_root:
        adapter_config["m3_reuse_sample_state"] = str(
            Path(reuse_root) / variant / conversation_id
        )
    adapter = create_adapter(baseline, config_overrides=adapter_config)
    sample_id = f"{variant}_{conversation_id}"
    build_fault_policy = ConsecutiveBuildFaultPolicy(
        baseline=baseline,
        benchmark="H2HMEM",
        enabled=(
            baseline == "MMA"
            or (
                baseline == "M2A"
                and bool(config.get("m2a_skip_failed_build_points", False))
            )
            or (
                baseline == "MIRIX"
                and bool(config.get("mirix_skip_failed_build_points", False))
            )
        ),
        maximum=int(
            config.get(
                (
                    "mirix_max_consecutive_failed_build_points"
                    if baseline == "MIRIX"
                    else "m2a_max_consecutive_failed_build_points"
                )
            )
            or 10
        ),
        recorder=call_recorder,
        fail_open=(baseline == "M2A"),
    )
    try:
        adapter.reset(sample_id, state_root / variant / conversation_id)
        with (
            call_recorder.phase("memory_build")
            if call_recorder is not None
            else nullcontext()
        ):
            if baseline != "HiveMem":
                if baseline == "M2A":
                    chunks = build_h2h_chunks_from_directory(
                        data_dir,
                        variant=variant,
                        conversation_ids={conversation_id},
                    )
                elif baseline in {"OmniSimpleMem", "MMA"}:
                    chunks = build_omni_h2h_chunks_from_directory(
                        data_dir,
                        variant=variant,
                        conversation_id=conversation_id,
                    )
                elif baseline == "M3-Agent-caption":
                    chunks = build_m3_h2h_chunks_from_directory(
                        data_dir,
                        variant=variant,
                        conversation_id=conversation_id,
                    )
                else:
                    chunks = h2hmem_chunks(
                        config,
                        variant=variant,
                        conversation_id=conversation_id,
                    )
                if baseline == "MMA" or (
                    baseline == "MIRIX"
                    and bool(config.get("mirix_resume_enabled", False))
                ):
                    chunks = adapter.filter_completed_session_chunks(chunks)
                current_session = ""
                for chunk in chunks:
                    session_id = str(chunk.metadata.get("session_id") or "")
                    if (
                        baseline != "MIRIX"
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

        jobs: list[dict[str, Any]] = []
        completed_jobs = dict(completed_jobs or {})
        indexed_questions = []
        for question_file, qa_index, qa in _question_rows(conversation_dir):
            session_id = question_file.parent.name
            manifest_question_id = h2hmem_manifest_question_id(
                variant, conversation_id, session_id, qa_index, qa
            )
            indexed_questions.append(
                (manifest_question_id, question_file, qa_index, qa)
            )
        if ordered_question_ids is not None:
            by_manifest_id = {row[0]: row for row in indexed_questions}
            missing = [
                question_id
                for question_id in ordered_question_ids
                if question_id not in by_manifest_id
            ]
            if missing:
                raise KeyError(
                    f"H2HMem manifest references {len(missing)} missing question(s) "
                    f"for {variant}/{conversation_id}: {missing[:5]}"
                )
            selected_questions = [
                by_manifest_id[question_id] for question_id in ordered_question_ids
            ]
        else:
            selected_questions = indexed_questions

        terminal_sample_error = ""
        for manifest_question_id, question_file, qa_index, qa in selected_questions:
            if ordered_question_ids is None and max_qa and len(jobs) >= max_qa:
                break
            question_data = qa.get("question") or {}
            question = str(question_data.get("text") or "")
            question_type = qa.get("question_type") or {}
            category = str(question_type.get("sub_type") or question_type.get("main_type") or "")
            session_id = question_file.parent.name
            question_id = str(
                qa.get("question_id")
                or qa.get("original_question_id")
                or f"{conversation_id}:{session_id}:{qa_index}"
            )
            query_id = f"h2hmem:{variant}:{conversation_id}:{question_id}"
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
                embed_texts([question], config)[0]
                if baseline == "HiveMem" and not skipped_after_terminal
                else None
            )
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
                        retrieval = adapter.retrieve(
                            RetrievalRequest(
                                query_id=query_id,
                                text=question,
                                category=category,
                                top_k=int(config["top_k"]),
                                query_image=(
                                    str(_question_image(question_file, question_data.get("image"))["path"])
                                    if question_data.get("image")
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
            query_image = _question_image(question_file, question_data.get("image"))
            native_answer = None
            terminal_native_error: Exception | None = None
            if retrieval_error and (
                baseline == "MIRIX" or bool(terminal_sample_error)
            ):
                native_answer = {
                    "text": "",
                    "error": retrieval_error,
                    "usage": None,
                    "attempts": int(config.get("retries") or 0) + 1,
                    "failed_attempts": int(config.get("retries") or 0) + 1,
                    "image_count": 1 if query_image else 0,
                    "trace": {
                        "failed": True,
                        "stage": "retrieval",
                        "error": retrieval_error,
                    },
                }
            elif baseline in {"MIRIX", "MMA"}:
                native_evidence, _ = build_retrieved_memory_evidence(
                    memory_items, category="VR"
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
                        native_result = adapter.answer_with_memory(
                            NativeAnswerRequest(
                                query_id=query_id,
                                messages=native_messages,
                                retrieval=retrieval,
                                query_image=(
                                    str(query_image.get("path") or "")
                                    if isinstance(query_image, dict)
                                    else None
                                ),
                                top_k=int(config["top_k"]),
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
                            "attempts": int(config.get("retries") or 0) + 1,
                            "failed_attempts": int(config.get("retries") or 0) + 1,
                            "image_count": 1 if query_image else 0,
                            "trace": {"failed": True, "error": error},
                        }
                    else:
                        if native_result.retrieval is not None:
                            retrieval = native_result.retrieval
                            memory_items = result_context_items(retrieval)
                            trace_rows = result_trace_rows(retrieval)
                        native_answer = native_result.to_dict()
            job = {
                "uid": query_id,
                "query_id": query_id,
                "manifest_question_id": manifest_question_id,
                "question_id": question_id,
                "sample_id": conversation_id,
                "conversation_id": conversation_id,
                "dialogue_name": conversation_id,
                "variant": variant,
                "session_id": session_id,
                "question": question,
                "question_text": question,
                "question_image": str(question_data.get("image") or ""),
                "question_type": question_type,
                "category": category,
                "difficulty": qa.get("difficulty", ""),
                "original_answer": qa.get("original_answer", ""),
                "answer_session": qa.get("answer_session") or [],
                "query_image_payload": query_image,
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
        snapshots = [row.to_dict() for row in adapter.snapshot()]
        return {
            "sample_id": sample_id,
            "variant": variant,
            "conversation_id": conversation_id,
            "jobs": jobs,
            "snapshots": snapshots,
            "build_failures": build_fault_policy.failures,
        }
    finally:
        adapter.close()


def materialize_skipped_conversation(
    *,
    data_dir: Path,
    spec: tuple[str, str, int, tuple[str, ...] | None],
    error: str,
    existing_jobs: list[dict[str, Any]],
    call_trace_path: Path,
) -> dict[str, Any]:
    """Preserve completed native QA and explicitly mark the rest as skipped."""
    variant, conversation_id, quota, ordered_question_ids = spec
    variant_dir = "multi-party" if variant == "multiparty" else variant
    conversation_dir = data_dir / variant_dir / conversation_id
    indexed = []
    for question_file, qa_index, qa in _question_rows(conversation_dir):
        manifest_question_id = h2hmem_manifest_question_id(
            variant, conversation_id, question_file.parent.name, qa_index, qa
        )
        indexed.append((manifest_question_id, question_file, qa_index, qa))
    if ordered_question_ids is not None:
        by_id = {row[0]: row for row in indexed}
        missing = [value for value in ordered_question_ids if value not in by_id]
        if missing:
            raise KeyError(
                f"H2HMem manifest references {len(missing)} missing question(s) "
                f"for {variant}/{conversation_id}: {missing[:5]}"
            )
        selected = [by_id[value] for value in ordered_question_ids]
    else:
        selected = indexed[:quota] if quota else indexed

    jobs_by_id = {
        str(row.get("manifest_question_id") or ""): row
        for row in existing_jobs
        if row.get("manifest_question_id")
    }
    skip_error = f"H2HMem sample skipped after bounded retries: {error}"[:4000]
    jobs: list[dict[str, Any]] = []
    for manifest_question_id, question_file, qa_index, qa in selected:
        if manifest_question_id in jobs_by_id:
            jobs.append(jobs_by_id[manifest_question_id])
            continue
        question_data = qa.get("question") or {}
        question_type = qa.get("question_type") or {}
        session_id = question_file.parent.name
        question_id = str(
            qa.get("question_id")
            or qa.get("original_question_id")
            or f"{conversation_id}:{session_id}:{qa_index}"
        )
        query_id = f"h2hmem:{variant}:{conversation_id}:{question_id}"
        query_image = _question_image(question_file, question_data.get("image"))
        jobs.append(
            {
                "uid": query_id,
                "query_id": query_id,
                "manifest_question_id": manifest_question_id,
                "question_id": question_id,
                "sample_id": conversation_id,
                "conversation_id": conversation_id,
                "dialogue_name": conversation_id,
                "variant": variant,
                "session_id": session_id,
                "question": str(question_data.get("text") or ""),
                "question_text": str(question_data.get("text") or ""),
                "question_image": str(question_data.get("image") or ""),
                "question_type": question_type,
                "category": str(
                    question_type.get("sub_type")
                    or question_type.get("main_type")
                    or ""
                ),
                "difficulty": qa.get("difficulty", ""),
                "original_answer": qa.get("original_answer", ""),
                "answer_session": qa.get("answer_session") or [],
                "query_image_payload": query_image,
                "memory_items": [],
                "retrieval_top_k": [],
                "retrieval_method_trace": {
                    "failed": True,
                    "stage": "sample_skipped",
                    "error": skip_error,
                },
                "native_answer": {
                    "text": "",
                    "error": skip_error,
                    "usage": None,
                    "attempts": 0,
                    "failed_attempts": 0,
                    "image_count": 1 if query_image else 0,
                    "trace": {
                        "failed": True,
                        "stage": "sample_skipped",
                        "error": skip_error,
                    },
                },
                "sample_terminal_error": skip_error,
                "sample_skipped": True,
                "sample_skip_error": skip_error,
                "skipped_after_consecutive_bad_points": True,
            }
        )
    return {
        "sample_id": f"{variant}_{conversation_id}",
        "variant": variant,
        "conversation_id": conversation_id,
        "jobs": jobs,
        "snapshots": [],
        "build_failures": [],
        "call_trace_path": str(call_trace_path),
        "skipped": True,
        "skip_error": skip_error,
    }


def answer_conversation_job(
    client: VLMAnswerClient,
    job: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], category="VR"
    )
    evidence, _ = build_retrieved_memory_evidence(
        job["memory_items"], category="VR"
    )
    prompt_evidence, zero_hit_prompt_marker_used = evidence_with_zero_hit_marker(
        evidence
    )
    messages = build_answer_messages(
        question=job["question"],
        question_type=job["category"],
        memory_evidence=prompt_evidence,
        query_images=query_image_prompt_metadata(job.get("query_image_payload")),
    )
    raw_answer = ""
    response = None
    native = job.get("native_answer")
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
                query_image=job.get("query_image_payload"),
                category="VR",
            )
            raw_answer = response.text
            usage, attempts = response.usage, response.attempts
            failed_attempts = response.failed_attempts
            image_count = response.image_count
        answer = parse_answer_response(raw_answer)
        error = ""
    except Exception as exc:
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
                query_image=job.get("query_image_payload"),
                category="VR",
            )
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "question_prompt", "query_image_payload", "memory_items", "retrieval_top_k",
            "retrieval_method_trace", "native_answer",
        }
    }
    result.update(
        {
            "system_answer": answer,
            "answer_raw_response": raw_answer,
            "retrieved_ids": [row["memory_id"] for row in job["retrieval_top_k"]],
            "retrieved_source_groups": [
                row["source_dialogue_ids"] for row in job["retrieval_top_k"]
            ],
            "error": error,
            "answer_seconds": time.time() - started,
            "answer_token_usage": usage,
            "answer_attempts": attempts,
            "answer_failed_attempts": failed_attempts,
            "answer_image_count": image_count,
            "native_answer_trace": dict(
                (job.get("native_answer") or {}).get("trace") or {}
            ),
            "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
        }
    )
    trace = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "conversation_id": job["conversation_id"],
        "session_id": job["session_id"],
        "question": job["question"],
        "category": job["category"],
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


def run_conversation(
    *,
    data_dir: Path,
    variant: str,
    conversation_id: str,
    baseline: str,
    state_root: Path,
    client: VLMAnswerClient,
    config: dict[str, Any],
    max_qa: int = 0,
    ordered_question_ids: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper for one conversation."""
    artifact = prepare_conversation_jobs(
        data_dir=data_dir,
        variant=variant,
        conversation_id=conversation_id,
        baseline=baseline,
        state_root=state_root,
        config=config,
        max_qa=max_qa,
        ordered_question_ids=ordered_question_ids,
    )
    pairs = [answer_conversation_job(client, job) for job in artifact["jobs"]]
    return [x[0] for x in pairs], [x[1] for x in pairs], artifact["snapshots"]


def _mma_resume_signature_digests(
    args: argparse.Namespace, signature: dict[str, Any]
) -> tuple[str, ...]:
    """Accept the pre-normalization digest for execution-only M2A flags."""
    primary = signature_digest(signature)
    if args.baseline != "MMA":
        return (primary,)
    legacy_arguments = dict(signature["arguments"])
    for key in (
        "m2a_skip_failed_build_points",
        "m2a_max_consecutive_failed_build_points",
    ):
        legacy_arguments[key] = getattr(args, key)
    legacy = signature_digest({**signature, "arguments": legacy_arguments})
    return tuple(dict.fromkeys((primary, legacy)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory baseline on H2HMem.")
    parser.add_argument("--baseline", default="HiveMem")
    parser.add_argument("--data-dir", default=str(DEFAULT_H2HMEM_DATA_DIR))
    parser.add_argument("--variant", choices=("dyadic", "multiparty", "all"), default="all")
    parser.add_argument("--conversation-id", action="append", default=[])
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--index-root", default="")
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument(
        "--m3-reuse-state-root",
        default="",
        help="Read-only M3 memory/datasets root from a completed prior run.",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument(
        "--sample-max-attempts",
        type=int,
        default=3,
        help=(
            "For MIRIX, retry a failed conversation from its checkpoint this many "
            "times before emitting explicit skipped QA rows and continuing."
        ),
    )
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--graph-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-k", type=int, default=0)
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--graph-mode", choices=("rerank", "append"), default="append")
    parser.add_argument("--append-k", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--answer-api-key", default=os.getenv("OPENAI_API_KEY") or "EMPTY"
    )
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=512)
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
    parser.add_argument(
        "--allow-answer-errors",
        action="store_true",
        help=(
            "After native answer retries are exhausted, checkpoint an empty "
            "answer with the error and continue."
        ),
    )
    parser.add_argument(
        "--mirix-skip-failed-build-points",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--mma-native-batch-size", type=int, default=20)
    parser.add_argument("--executor-visual-input", choices=("image", "caption"), default="image")
    parser.add_argument("--efficiency-config", default="configs/model_efficiency.json")
    parser.add_argument("--skip-model-check", action="store_true")
    apply_config_defaults(
        parser,
        allowed_keys={
            "answer_base_url", "answer_model", "answer_api_key",
            "answer_temperature", "num_predict", "request_timeout",
            "qa_request_timeout", "retries",
            "think", "reasoning_effort", "top_k", "embedding_dim", "embedding_model",
            "embedding_base_url", "executor_model", "executor_base_url",
            "executor_temperature", "executor_max_tokens",
            "executor_hard_max_tokens", "executor_visual_input",
            "allow_answer_errors",
            "sample_concurrency", "answer_concurrency", "checkpoint_every",
            "graph_retrieval", "graph_mode", "append_k", "seed_k",
            "expansion_bonus",
            "efficiency_config",
            "m3_reuse_state_root",
            "mirix_skip_failed_build_points",
            "mirix_max_consecutive_failed_build_points",
            "m2a_skip_failed_build_points",
            "m2a_max_consecutive_failed_build_points",
        },
    )
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be provided together")
    manifest_index = (
        SplitManifestIndex(args.split_manifest) if args.split_manifest else None
    )
    manifest_split = normalize_split_name(args.split) if args.split else ""
    if manifest_index is not None and args.max_qa:
        parser.error("--max-qa cannot be combined with strict manifest selection")
    args.baseline = canonical_name(args.baseline)
    if args.baseline == "HiveMem" and not args.index_root:
        parser.error("--index-root is required when --baseline=HiveMem")
    if (
        args.sample_concurrency < 1
        or args.sample_max_attempts < 1
        or args.answer_concurrency < 1
        or args.checkpoint_every < 1
        or args.request_timeout <= 0
        or args.qa_request_timeout <= 0
        or args.executor_hard_max_tokens < 0
        or args.mirix_max_consecutive_failed_build_points < 1
        or args.m2a_max_consecutive_failed_build_points < 1
    ):
        parser.error("Sample/answer concurrency and checkpoint interval must be positive")

    data_dir = Path(args.data_dir)
    if (data_dir / "dataset").is_dir():
        data_dir = data_dir / "dataset"
    manifest_rows = []
    if manifest_index is not None:
        if args.conversation_id or args.variant != "all":
            parser.error(
                "--variant/--conversation-id cannot be combined with strict manifest selection"
            )
        for data_source in manifest_index.data_sources:
            if data_source in {"h2hmem_dyadic", "h2hmem_multiparty"}:
                manifest_rows.extend(
                    manifest_index.conversations(
                        manifest_split, data_source=data_source
                    )
                )
        variants = tuple(dict.fromkeys(row.variant for row in manifest_rows))
        conversations = {
            variant: [
                row.source_id for row in manifest_rows if row.variant == variant
            ]
            for variant in variants
        }
    else:
        variants = ("dyadic", "multiparty") if args.variant == "all" else (args.variant,)
        selected = set(args.conversation_id)
        conversations = {}
        for variant in variants:
            conversations[variant] = list(
                dict.fromkeys(
                    path.parents[2].name
                    for path in iter_h2h_session_files(data_dir, variant=variant)
                    if not selected or path.parents[2].name in selected
                )
            )
            if not conversations[variant]:
                raise FileNotFoundError(f"No H2HMem conversations selected for {variant}")

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
    )
    if not args.skip_model_check:
        client.assert_model_available()

    result_dir = Path(args.result_dir)
    layout = BaselineOutputLayout(result_dir)
    state_root = layout.state_root(args.baseline_state_dir)
    graph_options = (
        {
            "seed_k": args.seed_k,
            "expansion_bonus": args.expansion_bonus,
            "mode": args.graph_mode,
            "append_k": args.append_k,
        }
        if args.graph_retrieval else None
    )
    config = {
        "top_k": args.top_k,
        "index_root": args.index_root,
        "graph_options": (
            {
                "seed_k": args.seed_k,
                "expansion_bonus": args.expansion_bonus,
                "mode": args.graph_mode,
                "append_k": args.append_k,
            }
            if args.graph_retrieval
            else False
        ),
        "embedding_dim": args.embedding_dim,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "answer_temperature": args.answer_temperature,
        "num_predict": args.num_predict,
        "think": args.think,
        "reasoning_effort": args.reasoning_effort,
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_temperature": args.executor_temperature,
        "executor_max_tokens": args.executor_max_tokens,
        "mirix_executor_retry_max_tokens": (
            args.executor_hard_max_tokens or args.executor_max_tokens
        ),
        "mma_native_batch_size": args.mma_native_batch_size,
        "executor_visual_input": args.executor_visual_input,
        "executor_native_tool_calls": args.baseline in {"MIRIX", "MMA"},
        "request_timeout": args.request_timeout,
        "retries": args.retries,
        "efficiency_config": args.efficiency_config,
        "m3_reuse_state_root": args.m3_reuse_state_root,
        "mirix_skip_failed_build_points": args.mirix_skip_failed_build_points,
        "mirix_max_consecutive_failed_build_points": (
            args.mirix_max_consecutive_failed_build_points
        ),
        "m2a_skip_failed_build_points": args.m2a_skip_failed_build_points,
        "m2a_max_consecutive_failed_build_points": (
            args.m2a_max_consecutive_failed_build_points
        ),
        "m2a_salvage_truncated_updates": args.m2a_salvage_truncated_updates,
    }
    sample_specs: list[tuple[str, str, int, tuple[str, ...] | None]] = []
    remaining_limit = args.max_qa
    ordered_rows = (
        [(row.variant, row.source_id, row.question_ids) for row in manifest_rows]
        if manifest_index is not None
        else [
            (variant, conversation_id, None)
            for variant in variants
            for conversation_id in conversations[variant]
        ]
    )
    for variant, conversation_id, ordered_question_ids in ordered_rows:
        if args.max_qa and remaining_limit <= 0:
            break
        conversation_dir = data_dir / (
            "multi-party" if variant == "multiparty" else variant
        ) / conversation_id
        question_count = len(_question_rows(conversation_dir))
        quota = min(remaining_limit, question_count) if args.max_qa else 0
        sample_specs.append(
            (variant, conversation_id, quota, ordered_question_ids)
        )
        if args.max_qa:
            remaining_limit -= quota

    signature = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {
                "answer_api_key", "sample_concurrency", "answer_concurrency",
                "checkpoint_every", "resume", "skip_model_check", "result_dir",
                "allow_answer_errors",
                "sample_max_attempts",
                # Execution-only fault policy. Excluding these two additions
                # preserves compatibility with checkpoints created before the
                # MIRIX policy was exposed by this harness.
                "mirix_skip_failed_build_points",
                "mirix_max_consecutive_failed_build_points",
                # These execution-only M2A controls do not affect MMA state.
                # Keeping them in the shared signature made an MMA resume
                # incompatible merely because the M2A CLI gained new flags.
                "m2a_skip_failed_build_points",
                "m2a_max_consecutive_failed_build_points",
            }
        },
        "inputs": file_manifest(
            path
            for variant, conversation_id, _, _ in sample_specs
            for path in sorted(
                (
                    data_dir
                    / ("multi-party" if variant == "multiparty" else variant)
                    / conversation_id
                ).rglob("*.json")
            )
        ) | (
            file_manifest([Path(args.split_manifest)])
            if args.split_manifest else {}
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_source": PROMPT_SOURCE,
        "prompt_sha256": prompt_sha256(),
        "call_trace_version": TRACE_VERSION,
    }
    computed_sample_signatures = _mma_resume_signature_digests(args, signature)
    qa_only_reuse = os.getenv("MMA_QA_ONLY_REUSE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    stored_sample_signatures = (
        (validated_qa_only_resume_signatures if qa_only_reuse else validated_paired_resume_signatures)(
            (
                f"{variant}/{conversation_id}",
                state_root
                / variant
                / conversation_id
                / (
                    ".mma_reuse_provenance.json"
                    if qa_only_reuse else ".offline_mma_resume.json"
                ),
                layout.checkpoint_dir
                / "native_samples"
                / Path(
                    trace_filename(f"{variant}/{conversation_id}")
                ).with_suffix(".json"),
            )
            for variant, conversation_id, _, _ in sample_specs
        )
        if args.baseline == "MMA" and args.resume
        else ()
    )
    compatible_sample_signatures = tuple(
        dict.fromkeys((*stored_sample_signatures, *computed_sample_signatures))
    )
    sample_signature = compatible_sample_signatures[0]

    def prepare(
        spec: tuple[str, str, int, tuple[str, ...] | None]
    ) -> dict[str, Any]:
        variant, conversation_id, quota, ordered_question_ids = spec
        sample_id = f"{variant}/{conversation_id}"
        if args.resume:
            cached = next(
                (
                    artifact
                    for compatible_signature in compatible_sample_signatures
                    if (
                        artifact := load_sample_artifact(
                            layout.sample_checkpoint_dir,
                            sample_id,
                            signature=compatible_signature,
                        )
                    )
                    is not None
                ),
                None,
            )
            if cached is not None:
                print(f"[resume] skip prepared conversation: {sample_id}", flush=True)
                return cached
        native_progress_path = (
            layout.checkpoint_dir
            / "native_samples"
            / Path(trace_filename(sample_id)).with_suffix(".json")
        )
        native_progress_jobs: list[dict[str, Any]] = []
        if args.baseline in {"MMA", "MIRIX"} and args.resume and native_progress_path.is_file():
            try:
                native_progress = json.loads(
                    native_progress_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                native_progress = {}
            if (
                native_progress.get("version") == 1
                and native_progress.get("sample_id") == sample_id
                and native_progress.get("signature")
                in compatible_sample_signatures
            ):
                native_progress_jobs = [
                    row
                    for row in (native_progress.get("jobs") or [])
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
                    "sample_id": sample_id,
                    "signature": sample_signature,
                    "status": "running",
                    "completed_questions": len(native_progress_jobs),
                    "jobs": native_progress_jobs,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        call_trace_path = layout.root / "call_traces" / trace_filename(sample_id)
        recorder = None
        proxy_context = nullcontext(None)
        sample_config = dict(config)
        sample_config["mma_resume_enabled"] = args.resume
        sample_config["mma_resume_signature"] = sample_signature
        sample_config["mma_resume_compatible_signatures"] = list(
            compatible_sample_signatures
        )
        sample_config["mirix_resume_enabled"] = args.resume
        sample_config["mirix_resume_signature"] = sample_signature
        if args.baseline != "HiveMem":
            recorder = CallRecorder(
                trace_path=call_trace_path,
                baseline=args.baseline,
                benchmark="H2HMEM",
                sample_id=sample_id,
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
                sample_config["executor_base_url"] = proxy.endpoint
            artifact = prepare_conversation_jobs(
                data_dir=data_dir,
                variant=variant,
                conversation_id=conversation_id,
                baseline=args.baseline,
                state_root=state_root,
                config=sample_config,
                max_qa=quota,
                ordered_question_ids=ordered_question_ids,
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
            layout.sample_checkpoint_dir,
            sample_id,
            signature=sample_signature,
            artifact=artifact,
        )
        if args.baseline in {"MMA", "MIRIX"}:
            write_json_atomic(
                native_progress_path,
                {
                    "version": 1,
                    "sample_id": sample_id,
                    "signature": sample_signature,
                    "status": "completed",
                    "completed_questions": len(artifact.get("jobs") or []),
                    "jobs": list(artifact.get("jobs") or []),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        print(f"[prepared] {sample_id}: {len(artifact['jobs'])} question(s)", flush=True)
        return artifact

    if args.baseline == "MIRIX":
        def materialize_skipped(
            spec: tuple[str, str, int, tuple[str, ...] | None], error: str
        ) -> dict[str, Any]:
            variant, conversation_id, _, _ = spec
            sample_id = f"{variant}/{conversation_id}"
            native_progress_path = (
                layout.checkpoint_dir
                / "native_samples"
                / Path(trace_filename(sample_id)).with_suffix(".json")
            )
            existing_jobs: list[dict[str, Any]] = []
            if native_progress_path.is_file():
                try:
                    progress = json.loads(
                        native_progress_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    progress = {}
                existing_jobs = list(progress.get("jobs") or [])
            call_trace_path = layout.root / "call_traces" / trace_filename(sample_id)
            artifact = materialize_skipped_conversation(
                data_dir=data_dir,
                spec=spec,
                error=error,
                existing_jobs=existing_jobs,
                call_trace_path=call_trace_path,
            )
            save_sample_artifact(
                layout.sample_checkpoint_dir,
                sample_id,
                signature=sample_signature,
                artifact=artifact,
            )
            write_json_atomic(
                native_progress_path,
                {
                    "version": 1,
                    "sample_id": sample_id,
                    "signature": sample_signature,
                    "status": "skipped",
                    "completed_questions": len(artifact["jobs"]),
                    "jobs": list(artifact["jobs"]),
                    "skip_error": artifact["skip_error"],
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            return artifact

        artifacts = run_conversation_retry_queue(
            sample_specs,
            prepare,
            max_attempts=args.sample_max_attempts,
            status_path=layout.checkpoint_dir / "sample_status.json",
            on_skipped=materialize_skipped,
        )
    else:
        artifacts = parallel_map_ordered(
            sample_specs,
            prepare,
            max_workers=args.sample_concurrency,
            item_key=lambda spec: f"{spec[0]}/{spec[1]}",
        )
    jobs = [job for artifact in artifacts for job in artifact["jobs"]]
    expected_manifest_question_ids = (
        tuple(
            question_id
            for row in manifest_rows
            for question_id in row.question_ids
        )
        if manifest_index is not None
        else None
    )
    if expected_manifest_question_ids is not None:
        actual = tuple(str(job.get("manifest_question_id") or "") for job in jobs)
        if actual != expected_manifest_question_ids:
            raise RuntimeError(
                "H2HMem prepared jobs do not exactly match manifest question order"
            )
    snapshots = [row for artifact in artifacts for row in artifact["snapshots"]]
    write_jsonl_atomic(layout.pipeline_qa, jobs)

    checkpoint_results = layout.checkpoint_dir / "results.json"
    checkpoint_traces = layout.checkpoint_dir / "retrieval_trace.jsonl"
    checkpoint_manifest = layout.checkpoint_dir / "manifest.json"
    results_by_id: dict[str, dict[str, Any]] = {}
    traces_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_manifest.is_file():
        saved = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            if checkpoint_results.is_file() and checkpoint_traces.is_file():
                results_by_id = {
                    str(row["query_id"]): row
                    for row in json.loads(checkpoint_results.read_text(encoding="utf-8"))
                    if row.get("query_id")
                }
                traces_by_id = {
                    str(row["query_id"]): row
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
        completed = [key for key in job_ids if key in results_by_id]
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
        and (
            args.baseline == "MMA"
            or row.get("sample_skipped")
            or not row.get("error")
        )
    }
    pending = [job for job in jobs if job["query_id"] not in completed_answers]
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
        futures = {
            pool.submit(answer_conversation_job, client, job): job for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            result, trace = future.result()
            query_id = str(job["query_id"])
            results_by_id[query_id] = result
            traces_by_id[query_id] = trace
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0
            print(
                f"[{len(results_by_id)}/{len(jobs)}] {query_id} "
                f"error={result['error'][:80]!r}",
                flush=True,
            )
    save_checkpoint()
    results = [results_by_id[key] for key in job_ids]
    traces = [traces_by_id[key] for key in job_ids]
    if expected_manifest_question_ids is not None:
        for label, rows in (("results", results), ("retrieval traces", traces)):
            actual = tuple(str(row.get("manifest_question_id") or "") for row in rows)
            if actual != expected_manifest_question_ids:
                raise RuntimeError(
                    f"H2HMem {label} do not exactly match manifest question order"
                )

    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", results)
    effective_top_k = (
        args.top_k
        if args.baseline == "M3-Agent-caption"
        else args.top_k + args.append_k
        if args.graph_retrieval and args.graph_mode == "append"
        else args.top_k
    )
    summary = summarize_results(results, k=effective_top_k)
    metric_results = []
    for row in results:
        normalized = dict(row)
        normalized["_metric_sample_id"] = (
            f"{row.get('variant', '')}/{row.get('conversation_id', '')}"
        )
        metric_results.append(normalized)
    evaluated_sample_ids = sorted(
        {
            str(row.get("_metric_sample_id") or "")
            for row in metric_results
            if row.get("_metric_sample_id")
        }
    )
    if args.baseline == "HiveMem":
        summary["calls"] = combine_call_metrics(
            calculate_calls_mb(Path(args.index_root), evaluated_sample_ids),
            calculate_calls_qa(metric_results, sample_id_field="_metric_sample_id"),
        )
    else:
        summary["calls"] = write_runtime_call_metrics(
            [
                artifact["call_trace_path"]
                for artifact in artifacts
                if artifact.get("call_trace_path")
            ],
            result_dir,
            metric_results,
            sample_id_field="_metric_sample_id",
            sample_ids=evaluated_sample_ids,
        )
    efficiency = write_efficiency_metrics(
        result_dir,
        metric_results,
        sample_id_field="_metric_sample_id",
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
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", traces)
    write_jsonl_atomic(layout.snapshot, snapshots)
    for variant in variants:
        filename = "prediction_multi_party.json" if variant == "multiparty" else "prediction_dyadic.json"
        write_json_atomic(
            result_dir / filename,
            {"predictions": [row for row in results if row["variant"] == variant]},
        )
    run_manifest = {
            "benchmark": "H2HMEM",
            "baseline": baseline_metadata(args.baseline),
            "variants": list(variants),
            "conversations": conversations,
            "questions": len(results),
            "configuration": {
                **config,
                "sample_concurrency": args.sample_concurrency,
                "sample_max_attempts": args.sample_max_attempts,
                "answer_concurrency": args.answer_concurrency,
                "checkpoint_every": args.checkpoint_every,
                "prompt_version": PROMPT_VERSION,
                "prompt_source": PROMPT_SOURCE,
                "prompt_sha256": prompt_sha256(),
            },
            "memory_snapshot": str(layout.snapshot),
            "selection_mode": (
                "strict_manifest" if manifest_index is not None else "legacy"
            ),
            "split": manifest_split,
            "split_manifest": str(manifest_index.path) if manifest_index else "",
            "split_manifest_sha256": (
                manifest_index.file_sha256 if manifest_index else ""
            ),
            "ordered_question_ids": list(expected_manifest_question_ids or ()),
            "chunk_inputs": [
                (
                    omni_input_manifest(f"h2hmem_{variant}")
                    if args.baseline in {"OmniSimpleMem", "MMA"}
                    else m3_input_manifest(f"h2hmem_{variant}")
                    if args.baseline == "M3-Agent-caption"
                    else chunk_source_manifest({}, f"h2hmem_{variant}")
                )
                for variant in variants
            ],
            "prompt_version": PROMPT_VERSION,
            "prompt_source": PROMPT_SOURCE,
            "prompt_sha256": prompt_sha256(),
        }
    if args.baseline == "M2A":
        run_manifest["m2a_conformance"] = m2a_conformance_manifest(
            answer_prompt_sha256=prompt_sha256()
        )
    if args.baseline == "MMA":
        run_manifest["mma_conformance"] = mma_conformance_manifest(
            answer_prompt_sha256=prompt_sha256()
        )
    if args.baseline == "OmniSimpleMem":
        run_manifest["omni_conformance"] = omni_conformance_manifest("h2hmem")
    if args.baseline == "M3-Agent-caption":
        run_manifest["m3_conformance"] = m3_conformance_manifest(
            "h2hmem",
            answer_prompt_sha256=prompt_sha256(),
            handoff_top_k=args.top_k,
        )
    write_json_atomic(result_dir / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
