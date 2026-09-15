from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from benchmarks.wma_harness.retrieval.query_embedding_cache import (
    QueryEmbeddingCache,
    build_gold_evidence_map,
    make_query_id,
    session_ids,
    visible_sessions_for_checkpoint,
)
from benchmarks.wma_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
    build_retrieved_memory_evidence,
)
from benchmarks.memgallery_harness.runner.metrics import (
    add_memory_metrics,
    calculate_cost_mb,
    calculate_cost_qa,
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    merge_existing_llm_judge_metrics,
    write_memory_metrics,
    write_snapshot_memory_metrics,
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
    ConsecutiveBuildFaultPolicy,
)
from benchmarks.baseline_runtime.adapters.m2a import m2a_conformance_manifest
from benchmarks.baseline_runtime.adapters.m3_agent import m3_conformance_manifest
from benchmarks.baseline_runtime.adapters.mma_original import (
    is_mma_consecutive_bad_point_error,
    mma_conformance_manifest,
)
from benchmarks.wma_harness.runner.metrics import summarize_results
from benchmarks.wma_harness.runner.prompts import (
    PROMPT_SOURCE,
    PROMPT_VERSION,
    build_answer_messages,
    parse_answer_response,
    prompt_sha256,
)
from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
)
from benchmarks.baseline_runtime.output_layout import (
    BaselineOutputLayout,
    load_hivemem_snapshot,
)
from benchmarks.baseline_runtime.protocol import (
    NativeAnswerRequest,
    RetrievalRequest,
    RetrievalResult,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import (
    file_manifest,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from benchmarks.question_filter import is_excluded_category, parse_excluded_categories
from benchmarks.fixed_chunks import chunk_source_manifest, wma_chunks
from benchmarks.baseline_runtime.omni_inputs import (
    build_omni_wma_chunks_from_data,
    omni_conformance_manifest,
    omni_input_manifest,
)
from benchmarks.baseline_runtime.m3_inputs import (
    build_m3_wma_chunks_from_data,
    m3_input_manifest,
)
from embedding.chunk_builder import (
    batch_wma_rounds_for_m2a,
    build_wma_chunks_from_data,
    iter_wma_sample_files,
)
from evidence_policy.split_manifest import SplitManifestIndex, normalize_split_name
from hive_mem.prefix_graph import (
    PREFIX_GRAPH_SCHEMA_VERSION,
    materialize_prefix_graph,
)
from hive_mem.output_layout import DatasetLayout


VISUAL_CATEGORIES = {"VFR", "VS", "VU", "CMR", ""}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WMA_DATA_DIR = Path(
    os.getenv(
        "WMA_DATA_DIR",
        PROJECT_ROOT / "WorldMemArena" / "WorldMemArena" / "lifelong",
    )
)


def _is_global_sample_failure(exc: Exception) -> bool:
    text = str(exc).casefold()
    return any(
        marker in text
        for marker in (
            "status 401",
            "status code: 401",
            "status 403",
            "status code: 403",
            "unauthorized",
            "authentication",
            "invalid api key",
            "insufficient_quota",
            "insufficient quota",
            "insufficient credit",
            "insufficient balance",
        )
    )


def run_sample_retry_queue(
    paths: list[Path],
    worker: Callable[[Path], dict[str, Any]],
    *,
    max_attempts: int,
    status_path: Path,
) -> list[dict[str, Any]]:
    """Run isolated samples once, then append failures to a bounded retry queue."""
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
    # A persisted "running" row means the previous harness disappeared
    # before it could record an outcome. Treat it as retryable on takeover;
    # otherwise status counts can claim multiple samples are running.
    for key, value in list(samples.items()):
        if isinstance(value, dict) and value.get("state") == "running":
            interrupted_attempt = int(value.get("attempts") or 0)
            errors = list(value.get("errors") or [])
            errors.append(
                {
                    "attempt": interrupted_attempt,
                    "error_type": "InterruptedRun",
                    "error": "previous harness stopped before recording sample outcome",
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            samples[key] = {
                **value,
                "state": "pending_retry",
                # Restarting the harness resumes the same checkpointed attempt;
                # it is not a new baseline sample failure.
                "attempts": max(0, interrupted_attempt - 1),
                "errors": errors,
            }
    pending = deque(paths)
    completed: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

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

    while pending:
        path = pending.popleft()
        key = path.stem
        row = dict(samples.get(key) or {})
        attempts = int(row.get("attempts") or 0)
        if row.get("state") == "skipped" and attempts >= max_attempts:
            skipped.append(key)
            continue
        if row.get("state") == "completed":
            # The worker's resume path loads the already-written sample artifact.
            try:
                completed[key] = worker(path)
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
            artifact = worker(path)
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
                pending.append(path)
            else:
                skipped.append(key)
        else:
            completed[key] = artifact
            samples[key].update(
                state="completed",
                completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            save_status()

    if skipped:
        raise RuntimeError(
            f"WMA skipped {len(skipped)} sample(s) after {max_attempts} attempts: "
            f"{', '.join(skipped)}; see {status_path}"
        )
    return [completed[path.stem] for path in paths]


def wma_manifest_question_id(
    sample_id: str, checkpoint_id: str, qa_index: int
) -> str:
    """Canonical WMA question ID used by the split manifest."""
    return f"{sample_id}:{checkpoint_id}:Q{qa_index:03d}"


def _with_manifest_question_id(job: dict[str, Any]) -> dict[str, Any]:
    """Backfill canonical IDs in sample checkpoints written before manifests."""
    if job.get("manifest_question_id"):
        return job
    qa_index = job.get("qa_index")
    if qa_index is None:
        raise KeyError(
            "WMA job lacks both manifest_question_id and qa_index: "
            f"{job.get('query_id', '<unknown>')}"
        )
    normalized = dict(job)
    normalized["manifest_question_id"] = wma_manifest_question_id(
        str(job["sample_id"]),
        str(job["checkpoint_id"]),
        int(qa_index),
    )
    return normalized


def _order_wma_jobs(
    jobs: list[dict[str, Any]],
    ordered_question_ids: tuple[str, ...] | None,
    *,
    sample_id: str,
) -> list[dict[str, Any]]:
    if ordered_question_ids is None:
        return jobs
    by_manifest_id = {str(row["manifest_question_id"]): row for row in jobs}
    missing = [value for value in ordered_question_ids if value not in by_manifest_id]
    if missing:
        raise KeyError(
            f"WMA manifest references {len(missing)} missing question(s) for "
            f"{sample_id}: {missing[:5]}"
        )
    return [by_manifest_id[value] for value in ordered_question_ids]


def prepare_sample_jobs(
    sample_path: Path,
    index_root: Path,
    query_cache: QueryEmbeddingCache,
    *,
    top_k: int,
    graph_options: dict[str, Any] | bool | None,
    prefix_graph_root: Path | None = None,
    excluded_categories: frozenset[str] = frozenset(),
    ordered_question_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_id = str(payload["sample_id"])
    ordered_sessions = session_ids(payload)
    gold_points = build_gold_evidence_map(payload)
    jobs: list[dict[str, Any]] = []
    selected_question_ids = (
        set(ordered_question_ids) if ordered_question_ids is not None else None
    )
    remaining_question_ids = (
        set(selected_question_ids) if selected_question_ids is not None else None
    )
    prefix_graph_root = prefix_graph_root or index_root / ".prefix_graphs"
    for checkpoint in payload.get("qa_checkpoints", []) or []:
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        if (
            not checkpoint_id
            or Path(checkpoint_id).name != checkpoint_id
            or checkpoint_id in {".", ".."}
        ):
            raise ValueError(f"Invalid WMA checkpoint id: {checkpoint_id!r}")
        covered_sessions = [str(value) for value in checkpoint.get("covered_sessions", [])]
        visible_sessions = visible_sessions_for_checkpoint(
            ordered_sessions, covered_sessions
        )
        checkpoint_index_root = index_root
        prefix_manifest = ""
        if graph_options is not False:
            checkpoint_index_root = (
                prefix_graph_root / sample_id / checkpoint_id
            )
            materialize_prefix_graph(
                index_root / "datasets" / sample_id,
                checkpoint_index_root,
                sample_id=sample_id,
                checkpoint_id=checkpoint_id,
                visible_session_ids=visible_sessions,
                graph_options=(
                    graph_options if isinstance(graph_options, dict) else {}
                ),
            )
            prefix_manifest = str(
                checkpoint_index_root / "prefix_manifest.json"
            )
        adapter = create_adapter(
            "HiveMem",
            config_overrides={
                "index_root": str(checkpoint_index_root),
                "top_k": top_k,
                "visual_categories": VISUAL_CATEGORIES,
                "graph_options": graph_options,
            },
        )
        try:
            adapter.reset(sample_id, Path())
            visible_session_set = set(visible_sessions)
            for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
                manifest_question_id = wma_manifest_question_id(
                    sample_id, checkpoint_id, qa_index
                )
                if (
                    selected_question_ids is not None
                    and manifest_question_id not in selected_question_ids
                ):
                    continue
                category = str(qa.get("question_type_abbrev", ""))
                if selected_question_ids is None and is_excluded_category(
                    category, excluded_categories
                ):
                    continue
                question = str(qa.get("question", ""))
                query_id = make_query_id(
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    qa_index=qa_index,
                    category=category,
                    question=question,
                )
                vector = query_cache.get_by_id(query_id)
                if vector is None:
                    raise KeyError(f"Missing cached query embedding: {query_id}")
                retrieval = adapter.retrieve(
                    RetrievalRequest(
                        query_id=query_id,
                        text=question,
                        category=category,
                        top_k=top_k,
                        visible_session_ids=tuple(visible_sessions),
                        query_vector=vector,
                    )
                )
                memory_items = result_context_items(retrieval)
                trace = result_trace_rows(retrieval)
                evidence = qa.get("evidence", []) or []
                evidence_ids = [
                    str(row.get("memory_id") or row.get("image_id") or "")
                    for row in evidence
                    if isinstance(row, dict)
                    and (row.get("memory_id") or row.get("image_id"))
                ]
                future_evidence_ids = [
                    value
                    for value in evidence_ids
                    if value in gold_points
                    and gold_points[value]["session_id"] not in visible_session_set
                ]
                unmapped_evidence_ids = [
                    value for value in evidence_ids if value not in gold_points
                ]
                jobs.append(
                    {
                        "query_id": query_id,
                        "manifest_question_id": manifest_question_id,
                        "sample_id": sample_id,
                        "dataset": sample_id,
                        "checkpoint_id": checkpoint_id,
                        "covered_sessions": covered_sessions,
                        "visible_sessions": visible_sessions,
                        "graph_prefix_manifest": prefix_manifest,
                        "qa_index": qa_index,
                        "question": question,
                        "category": category,
                        "question_type": qa.get("question_type", ""),
                        "difficulty": qa.get("difficulty", ""),
                        "original_answer": qa.get("answer", ""),
                        "evidence": evidence,
                        "gold_evidence_memory_ids": evidence_ids,
                        "gold_future_evidence_ids": future_evidence_ids,
                        "gold_unmapped_evidence_ids": unmapped_evidence_ids,
                        "gold_evidence_contents": [
                            gold_points[value]["content"]
                            for value in evidence_ids
                            if value in gold_points
                        ],
                        "gold_sessions": list(
                            dict.fromkeys(
                                gold_points[value]["session_id"]
                                for value in evidence_ids
                                if value in gold_points
                            )
                        ),
                        "gold_visible_sessions": list(
                            dict.fromkeys(
                                gold_points[value]["session_id"]
                                for value in evidence_ids
                                if value in gold_points
                                and gold_points[value]["session_id"] in visible_session_set
                            )
                        ),
                        "memory_items": memory_items,
                        "retrieval_top_k": trace,
                    }
                )
                if remaining_question_ids is not None:
                    remaining_question_ids.discard(manifest_question_id)
            # A strict manifest does not authorize ingesting sessions after
            # its last selected checkpoint question. Besides wasting work,
            # doing so would make the final memory snapshot contain future
            # information that was unavailable to every requested QA.
            if remaining_question_ids is not None and not remaining_question_ids:
                break
        finally:
            adapter.close()
    return _order_wma_jobs(jobs, ordered_question_ids, sample_id=sample_id)


def prepare_native_sample_jobs(
    sample_path: Path,
    query_cache: QueryEmbeddingCache | None,
    *,
    baseline: str,
    state_root: Path,
    top_k: int,
    config_overrides: dict[str, Any],
    memory_snapshots: list[dict[str, Any]] | None = None,
    excluded_categories: frozenset[str] = frozenset(),
    ordered_question_ids: tuple[str, ...] | None = None,
    call_recorder: CallRecorder | None = None,
    checkpoint_answer_client: VLMAnswerClient | None = None,
    checkpoint_answer_executor: ThreadPoolExecutor | None = None,
    checkpoint_results: list[dict[str, Any]] | None = None,
    checkpoint_traces: list[dict[str, Any]] | None = None,
    completed_jobs: dict[str, dict[str, Any]] | None = None,
    on_qa_completed: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], None
    ] | None = None,
    allow_native_qa_errors: bool = False,
) -> list[dict[str, Any]]:
    """Stream one WMA sample through a native baseline without future leakage."""
    baseline = canonical_name(baseline)
    if checkpoint_answer_client is not None and baseline not in {"M2A", "MIRIX", "MMA"}:
        raise ValueError(
            "checkpoint-inline answering is restricted to M2A, MIRIX, and MMA"
        )
    if baseline == "M2A" and checkpoint_answer_client is not None and (
        checkpoint_results is None or checkpoint_traces is None
    ):
        raise ValueError(
            "checkpoint_results and checkpoint_traces are required for inline answers"
        )
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_id = str(payload["sample_id"])
    ordered_sessions = session_ids(payload)
    session_order = {session_id: index for index, session_id in enumerate(ordered_sessions)}
    if baseline == "M2A":
        chunks = batch_wma_rounds_for_m2a(
            build_wma_chunks_from_data(
                payload,
                sample_path.parent,
                sample_path=sample_path,
            ),
            rounds_per_batch=int(
                config_overrides.get("m2a_wma_rounds_per_ingest") or 1
            ),
        )
    elif baseline in {"OmniSimpleMem", "MMA"}:
        chunks = build_omni_wma_chunks_from_data(
            payload,
            sample_path.parent,
            sample_path=sample_path,
        )
    elif baseline == "M3-Agent-caption":
        chunks = build_m3_wma_chunks_from_data(
            payload,
            sample_path.parent,
            sample_path=sample_path,
        )
    else:
        chunks = wma_chunks(config_overrides, sample_id)
    gold_points = build_gold_evidence_map(payload)
    checkpoints = []
    for position, checkpoint in enumerate(payload.get("qa_checkpoints", []) or []):
        covered = [str(value) for value in checkpoint.get("covered_sessions", [])]
        visible = visible_sessions_for_checkpoint(ordered_sessions, covered)
        last_index = max((session_order[value] for value in visible), default=-1)
        checkpoints.append((last_index, position, checkpoint, covered, visible))
    checkpoints.sort(key=lambda row: (row[0], row[1]))

    adapter_config = dict(config_overrides)
    reuse_root = str(adapter_config.get("m3_reuse_state_root") or "").strip()
    if baseline == "M3-Agent-caption" and reuse_root:
        adapter_config["m3_reuse_sample_state"] = str(Path(reuse_root) / sample_id)
    adapter = create_adapter(baseline, config_overrides=adapter_config)
    build_fault_policy = ConsecutiveBuildFaultPolicy(
        baseline=baseline,
        benchmark="WorldMemArena",
        enabled=(
            baseline == "MMA"
            or (
                baseline == "M2A"
                and bool(
                    config_overrides.get("m2a_skip_failed_build_points", False)
                )
            )
            or (
                baseline == "MIRIX"
                and bool(
                    config_overrides.get("mirix_skip_failed_build_points", False)
                )
            )
        ),
        maximum=int(
            config_overrides.get(
                (
                    "mirix_max_consecutive_failed_build_points"
                    if baseline == "MIRIX"
                    else "m2a_max_consecutive_failed_build_points"
                ),
                10,
            )
            or 10
        ),
        recorder=call_recorder,
    )
    completed_jobs = dict(completed_jobs or {})
    jobs: list[dict[str, Any]] = []
    selected_question_ids = (
        set(ordered_question_ids) if ordered_question_ids is not None else None
    )
    remaining_question_ids = (
        set(selected_question_ids) if selected_question_ids is not None else None
    )
    ingested_through = -1
    try:
        adapter.reset(sample_id, state_root / sample_id)
        if baseline in {"MIRIX", "MMA"}:
            completed_sessions = adapter.completed_session_ids()
            chunks = adapter.filter_completed_session_chunks(chunks)
            ingested_through = max(
                (session_order[value] for value in completed_sessions),
                default=-1,
            )
        chunks_by_session: dict[str, list[Any]] = {
            session_id: [] for session_id in ordered_sessions
        }
        for chunk in chunks:
            chunks_by_session.setdefault(
                str(chunk.metadata.get("session_id") or ""), []
            ).append(chunk)
        for last_index, _, checkpoint, covered_sessions, visible_sessions in checkpoints:
            checkpoint_jobs: list[dict[str, Any]] = []
            checkpoint_last_session = ""
            for session_index in range(ingested_through + 1, last_index + 1):
                session_id = ordered_sessions[session_index]
                checkpoint_last_session = session_id
                with (
                    call_recorder.phase("memory_build")
                    if call_recorder is not None
                    else nullcontext()
                ):
                    for chunk in chunks_by_session.get(session_id, []):
                        try:
                            adapter.ingest(chunk)
                        except Exception as exc:
                            build_fault_policy.handle(
                                exc,
                                chunk=chunk,
                                point_kind="ingest",
                                session_id=session_id,
                            )
                            continue
                        build_fault_policy.success()
                    if baseline != "MIRIX":
                        try:
                            adapter.end_session(session_id)
                        except Exception as exc:
                            build_fault_policy.handle(
                                exc,
                                chunk=None,
                                point_kind="end_session",
                                session_id=session_id,
                            )
                        else:
                            build_fault_policy.success()
            if baseline == "MIRIX" and checkpoint_last_session:
                with (
                    call_recorder.phase("memory_build")
                    if call_recorder is not None
                    else nullcontext()
                ):
                    # A checkpoint is the only early flush boundary: all
                    # residual messages must be visible before its QA begins.
                    try:
                        adapter.end_session(checkpoint_last_session)
                    except Exception as exc:
                        build_fault_policy.handle(
                            exc,
                            chunk=None,
                            point_kind="end_session",
                            session_id=checkpoint_last_session,
                        )
                    else:
                        build_fault_policy.success()
            ingested_through = max(ingested_through, last_index)
            checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
            visible_session_set = set(visible_sessions)
            for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
                manifest_question_id = wma_manifest_question_id(
                    sample_id, checkpoint_id, qa_index
                )
                if (
                    selected_question_ids is not None
                    and manifest_question_id not in selected_question_ids
                ):
                    continue
                category = str(qa.get("question_type_abbrev", ""))
                if selected_question_ids is None and is_excluded_category(
                    category, excluded_categories
                ):
                    continue
                if manifest_question_id in completed_jobs:
                    jobs.append(completed_jobs[manifest_question_id])
                    if remaining_question_ids is not None:
                        remaining_question_ids.discard(manifest_question_id)
                    continue
                question = str(qa.get("question", ""))
                query_id = make_query_id(
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    qa_index=qa_index,
                    category=category,
                    question=question,
                )
                vector = query_cache.get_by_id(query_id) if query_cache is not None else None
                retrieval_error = ""
                try:
                    with (
                        call_recorder.phase("retrieval")
                        if call_recorder is not None
                        else nullcontext()
                    ):
                        retrieval = adapter.retrieve(
                            RetrievalRequest(
                                query_id=query_id,
                                text=question,
                                category=category,
                                top_k=top_k,
                                visible_session_ids=tuple(visible_sessions),
                                query_vector=vector,
                            )
                        )
                except Exception as exc:
                    if not (allow_native_qa_errors and baseline == "MIRIX"):
                        raise
                    retrieval_error = str(exc)
                    retrieval = RetrievalResult(
                        trace={
                            "failed": True,
                            "stage": "retrieval",
                            "error": retrieval_error,
                        }
                    )
                trace = result_trace_rows(retrieval)
                memory_items = result_context_items(retrieval)
                native_answer = None
                terminal_native_error: Exception | None = None
                if baseline == "MIRIX" and retrieval_error:
                    native_answer = {
                        "text": "",
                        "error": retrieval_error,
                        "usage": None,
                        "attempts": int(config_overrides.get("retries") or 0) + 1,
                        "failed_attempts": int(config_overrides.get("retries") or 0) + 1,
                        "image_count": 0,
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
                        allow_empty_evidence=True,
                    )
                    with (
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
                                    top_k=top_k,
                                )
                            )
                        except Exception as exc:
                            if not allow_native_qa_errors:
                                raise
                            if baseline == "MMA" and is_mma_consecutive_bad_point_error(exc):
                                terminal_native_error = exc
                            error = str(exc)
                            native_answer = {
                                "text": "",
                                "error": error,
                                "usage": None,
                                "attempts": int(config_overrides.get("retries") or 0) + 1,
                                "failed_attempts": int(config_overrides.get("retries") or 0) + 1,
                                "image_count": 0,
                                "trace": {"failed": True, "error": error},
                            }
                        else:
                            if native_result.retrieval is not None:
                                retrieval = native_result.retrieval
                                memory_items = result_context_items(retrieval)
                                trace = result_trace_rows(retrieval)
                            native_answer = native_result.to_dict()
                evidence = qa.get("evidence", []) or []
                evidence_ids = [
                    str(row.get("memory_id") or row.get("image_id") or "")
                    for row in evidence
                    if isinstance(row, dict) and (row.get("memory_id") or row.get("image_id"))
                ]
                job = {
                        "query_id": query_id,
                        "manifest_question_id": manifest_question_id,
                        "sample_id": sample_id,
                        "dataset": sample_id,
                        "checkpoint_id": checkpoint_id,
                        "covered_sessions": covered_sessions,
                        "visible_sessions": visible_sessions,
                        "qa_index": qa_index,
                        "question": question,
                        "category": category,
                        "question_type": qa.get("question_type", ""),
                        "difficulty": qa.get("difficulty", ""),
                        "original_answer": qa.get("answer", ""),
                        "evidence": evidence,
                        "gold_evidence_memory_ids": evidence_ids,
                        "gold_future_evidence_ids": [
                            value for value in evidence_ids
                            if value in gold_points
                            and gold_points[value]["session_id"] not in visible_session_set
                        ],
                        "gold_unmapped_evidence_ids": [
                            value for value in evidence_ids if value not in gold_points
                        ],
                        "gold_evidence_contents": [
                            gold_points[value]["content"]
                            for value in evidence_ids if value in gold_points
                        ],
                        "gold_sessions": list(dict.fromkeys(
                            gold_points[value]["session_id"]
                            for value in evidence_ids if value in gold_points
                        )),
                        "gold_visible_sessions": list(dict.fromkeys(
                            gold_points[value]["session_id"]
                            for value in evidence_ids
                            if value in gold_points
                            and gold_points[value]["session_id"] in visible_session_set
                        )),
                        "memory_items": memory_items,
                        "retrieval_top_k": trace,
                        "retrieval_method_trace": dict(retrieval.trace),
                        "native_answer": native_answer,
                    }
                jobs.append(job)
                checkpoint_jobs.append(job)
                if on_qa_completed is not None:
                    if native_answer is None:
                        raise RuntimeError(
                            "durable native QA checkpoint requires a native answer"
                        )
                    result, answer_trace = answer_job(
                        checkpoint_answer_client, job
                    )
                    if result.get("error") and not allow_native_qa_errors:
                        raise RuntimeError(
                            f"native QA checkpoint failed: {result['error']}"
                        )
                    on_qa_completed(job, result, answer_trace)
                if terminal_native_error is not None:
                    raise terminal_native_error
                if remaining_question_ids is not None:
                    remaining_question_ids.discard(manifest_question_id)
            if (
                baseline == "M2A"
                and checkpoint_answer_client is not None
                and checkpoint_jobs
            ):
                if checkpoint_answer_executor is None:
                    answered = [
                        answer_job(checkpoint_answer_client, job)
                        for job in checkpoint_jobs
                    ]
                else:
                    answer_futures = [
                        checkpoint_answer_executor.submit(
                            answer_job, checkpoint_answer_client, job
                        )
                        for job in checkpoint_jobs
                    ]
                    # This is the checkpoint barrier: do not advance to the
                    # next session until every answer at the current
                    # checkpoint has completed.
                    answered = [future.result() for future in answer_futures]
                barrier_completed_at = time.time()
                answer_errors = [
                    str(result.get("error") or "")
                    for result, _ in answered
                    if result.get("error")
                ]
                if answer_errors:
                    raise RuntimeError(
                        f"M2A checkpoint {checkpoint_id} answer failed before future "
                        f"ingest: {answer_errors[0]}"
                    )
                for result, trace in answered:
                    protocol = {
                        "mode": "answer_before_future_ingest",
                        "checkpoint_id": checkpoint_id,
                        "barrier_completed_at": barrier_completed_at,
                    }
                    result["checkpoint_protocol"] = dict(protocol)
                    trace["checkpoint_protocol"] = dict(protocol)
                    checkpoint_results.append(result)
                    checkpoint_traces.append(trace)
            if remaining_question_ids is not None and not remaining_question_ids:
                break
        if memory_snapshots is not None:
            memory_snapshots.extend(row.to_dict() for row in adapter.snapshot())
    finally:
        adapter.close()
    return _order_wma_jobs(
        jobs, ordered_question_ids, sample_id=sample_id
    )


def answer_job(client: VLMAnswerClient, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    job = _with_manifest_question_id(job)
    started = time.time()
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], job["category"]
    )
    evidence, _ = build_retrieved_memory_evidence(
        job["memory_items"], job["category"]
    )
    messages = build_answer_messages(
        question=job["question"],
        question_type=job["category"],
        memory_evidence=evidence,
        allow_empty_evidence=True,
    )
    raw_answer = ""
    answer_response = None
    native = job.get("native_answer")
    answer_token_usage = None
    answer_attempts = 0
    answer_failed_attempts = 0
    answer_image_count = 0
    try:
        if native:
            answer_token_usage = native.get("usage")
            answer_attempts = int(native.get("attempts") or 1)
            answer_failed_attempts = int(native.get("failed_attempts") or 0)
            answer_image_count = int(native.get("image_count") or 0)
            if native.get("error"):
                raise RuntimeError(str(native["error"]))
            raw_answer = str(native.get("text") or "")
        else:
            answer_response = client.answer_messages_with_usage(
                messages=messages,
                memory_items=job["memory_items"],
                category=job["category"],
            )
            raw_answer = answer_response.text
            answer_token_usage = answer_response.usage
            answer_attempts = answer_response.attempts
            answer_failed_attempts = answer_response.failed_attempts
            answer_image_count = answer_response.image_count
        answer = parse_answer_response(raw_answer)
        error = ""
    except Exception as exc:
        answer, error = "", str(exc)
        if answer_response is not None:
            answer_token_usage = answer_response.usage
            answer_attempts = answer_response.attempts
            answer_failed_attempts = min(
                answer_attempts, answer_response.failed_attempts + 1
            )
            answer_image_count = answer_response.image_count
        elif not native:
            answer_token_usage = None
            answer_attempts = client.retries + 1
            answer_failed_attempts = answer_attempts
            answer_image_count = client.count_answer_images(
                job["memory_items"], category=job["category"]
            )
    top_k = job["retrieval_top_k"]
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "memory_items", "retrieval_top_k", "retrieval_method_trace",
            "native_answer",
        }
    }
    result.update(
        {
            "system_answer": answer,
            "answer_raw_response": raw_answer,
            "retrieved_ids": [row["memory_id"] for row in top_k],
            "retrieved_source_groups": [row["source_dialogue_ids"] for row in top_k],
            "retrieved_sessions": [row["session_id"] for row in top_k],
            "empty_retrieval": not bool(top_k),
            "error": error,
            "answer_token_usage": answer_token_usage,
            "answer_attempts": answer_attempts,
            "answer_failed_attempts": answer_failed_attempts,
            "answer_image_count": answer_image_count,
            "native_answer_trace": dict(
                (job.get("native_answer") or {}).get("trace") or {}
            ),
            "answer_seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    trace = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "sample_id": job["sample_id"],
        "checkpoint_id": job["checkpoint_id"],
        "question": job["question"],
        "category": job["category"],
        "covered_sessions": job["covered_sessions"],
        "visible_sessions": job["visible_sessions"],
        "top_k": top_k,
        "empty_retrieval": not bool(top_k),
        "memory_context": memory_context,
        "answer_prompt_messages": messages,
        "retrieval_method_trace": dict(job.get("retrieval_method_trace") or {}),
        "m2a_trace": dict(
            (job.get("retrieval_method_trace") or {}).get("m2a_trace") or {}
        ),
    }
    return result, trace


def to_pipeline_qa_record(result: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    items = trace.get("top_k", [])
    return {
        "sample_id": result["sample_id"],
        "sample_uuid": result["sample_id"],
        "manifest_question_id": result["manifest_question_id"],
        "checkpoint_id": result["checkpoint_id"],
        "question": result["question"],
        "gold_answer": result["original_answer"],
        "gold_evidence_memory_ids": result.get("gold_evidence_memory_ids", []),
        "gold_evidence_contents": result.get("gold_evidence_contents", []),
        "question_type": result.get("question_type", ""),
        "question_type_abbrev": result.get("category", ""),
        "difficulty": result.get("difficulty", ""),
        "retrieval": {
            "query": result["question"],
            "top_k": len(items),
            "items": [
                {
                    "rank": row["rank"],
                    "memory_id": row["memory_id"],
                    "text": row["content"],
                    "score": row["score"],
                    "raw_backend_id": row["memory_id"],
                    "image_path": (row.get("image_paths") or [None])[0],
                }
                for row in items
            ],
            "raw_trace": {
                "retrieval_source_sessions_by_rank": {
                    str(row["rank"]): [row.get("session_id", "")]
                    for row in items
                }
            },
        },
        "generated_answer": result.get("system_answer", ""),
        "cited_memories": [],
        "retrieval_seconds": 0.0,
        "answer_seconds": result.get("answer_seconds", 0.0),
        "retrieval_token_usage": {},
        "answer_token_usage": result.get("answer_token_usage"),
    }


def _run_signature(
    args: argparse.Namespace,
    sample_paths: list[Path],
) -> dict[str, Any]:
    ignored = {
        "answer_api_key",
        "answer_concurrency",
        "sample_concurrency",
        "allow_answer_errors",
        "checkpoint_every",
        "result_dir",
        "resume",
        # Retry-queue bookkeeping changes execution only; it must not make an
        # already-built MMA database incompatible with the same experiment.
        "sample_attempts",
        "skip_model_check",
        # Execution-only MIRIX fault policy, excluded so the resilient
        # watchdog can reuse checkpoints written before these CLI switches
        # were exposed.
        "mirix_skip_failed_build_points",
        "mirix_max_consecutive_failed_build_points",
        # These switches are consumed only by M2A and cannot affect MMA state.
        "m2a_skip_failed_build_points",
        "m2a_max_consecutive_failed_build_points",
    }
    input_paths: list[Path] = list(sample_paths)
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
        for sample_path in sample_paths:
            bank = index_root / "datasets" / sample_path.stem
            layout = DatasetLayout(bank)
            input_paths.extend(
                [
                    bank / "memories.jsonl",
                    layout.existing_vector_path("text.npy", "vectors.npy"),
                    layout.existing_vector_path("image.npy", "image_vectors.npy"),
                    layout.existing_vector_path("image_mask.npy", "image_mask.npy"),
                ]
            )
    signature = {
        "checkpoint_answer_protocol": (
            "m2a_answer_before_future_ingest_v1"
            if args.baseline == "M2A"
            else "legacy_deferred_answer"
        ),
        "prefix_graph_schema_version": PREFIX_GRAPH_SCHEMA_VERSION,
        "arguments": {
            key: value for key, value in vars(args).items() if key not in ignored
        },
        "inputs": file_manifest(input_paths),
        "prompt_version": PROMPT_VERSION,
        "prompt_source": PROMPT_SOURCE,
        "prompt_sha256": prompt_sha256(),
        "call_trace_version": TRACE_VERSION,
    }
    if args.baseline == "M2A":
        signature["m2a_input"] = m2a_wma_input_manifest(
            sample_paths,
            rounds_per_batch=args.m2a_wma_rounds_per_ingest,
        )
    return signature


def _mma_resume_signature_digests(
    args: argparse.Namespace, signature: dict[str, Any]
) -> tuple[str, ...]:
    """Accept the pre-normalization digest for execution-only WMA switches."""
    primary = signature_digest(signature)
    if args.baseline != "MMA":
        return (primary,)
    legacy_arguments = dict(signature["arguments"])
    for key in (
        "sample_attempts",
        "m2a_skip_failed_build_points",
        "m2a_max_consecutive_failed_build_points",
    ):
        legacy_arguments[key] = getattr(args, key)
    legacy = signature_digest({**signature, "arguments": legacy_arguments})
    return tuple(dict.fromkeys((primary, legacy)))


def m2a_wma_input_manifest(
    sample_paths: list[Path], *, rounds_per_batch: int = 1
) -> dict[str, Any]:
    """Describe the benchmark-native WMA files consumed by the M2A builder."""
    input_paths: list[Path] = []
    for sample_path in sample_paths:
        resolved_sample = sample_path.resolve()
        input_paths.append(resolved_sample)
        payload = json.loads(resolved_sample.read_text(encoding="utf-8"))
        for chunk in build_wma_chunks_from_data(
            payload,
            resolved_sample.parent,
            sample_path=resolved_sample,
        ):
            input_paths.extend(Path(value).resolve() for value in chunk.images)
    source_path = Path(inspect.getsourcefile(build_wma_chunks_from_data) or "").resolve()
    inputs = file_manifest(input_paths)
    return {
        "source": "wma_lifelong",
        "format": "M2A benchmark-native dialogue turns",
        "builder": "embedding.chunk_builder.build_wma_chunks_from_data",
        "builder_module_path": str(source_path),
        "builder_module_sha256": sha256_file(source_path),
        "ingest_batching": {
            "mode": "m2a_batched_complete_rounds",
            "max_rounds_per_call": int(rounds_per_batch),
            "same_session_only": True,
            "max_image_bearing_source_turns_per_call": 1,
        },
        "input_files": inputs,
        "input_file_count": len(inputs),
        "shared_fixed_chunks": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory baseline on WorldMemArena.")
    parser.add_argument("--baseline", default="HiveMem")
    parser.add_argument("--data-dir", default=str(DEFAULT_WMA_DATA_DIR))
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--index-root", default="")
    parser.add_argument("--query-embedding-dir", default="")
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument(
        "--m3-reuse-state-root",
        default="",
        help="Read-only M3 memory/datasets root from a completed prior run.",
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument(
        "--sample-attempts",
        type=int,
        default=3,
        help=(
            "Maximum attempts per isolated WMA sample. Failed samples are "
            "appended to a retry queue after the other samples run."
        ),
    )
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--exclude-categories",
        default="",
        help="Comma-separated QA categories to skip before embedding, retrieval, and answering.",
    )
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--answer-concurrency", type=int, default=16)
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
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--executor-temperature", type=float, default=0.0)
    parser.add_argument("--executor-max-tokens", type=int, default=512)
    parser.add_argument("--m2a-wma-rounds-per-ingest", type=int, default=1)
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
    parser.add_argument("--mma-native-batch-size", type=int, default=20)
    parser.add_argument("--executor-visual-input", choices=("image", "caption"), default="image")
    parser.add_argument(
        "--allow-answer-errors",
        action="store_true",
        help="Write metrics and pipeline records even when VLM answer requests failed.",
    )
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--graph-retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed-k", type=int, default=0)
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--graph-mode", choices=("rerank", "append"), default="append")
    parser.add_argument("--append-k", type=int, default=2)
    from hive_mem.build_memories import apply_config_defaults
    apply_config_defaults(
        parser,
        allowed_keys={
            "answer_base_url",
            "answer_concurrency",
            "sample_concurrency",
            "answer_model",
            "answer_api_key",
            "answer_temperature",
            "num_predict",
            "cost_mb_input_price",
            "cost_mb_output_price",
            "cost_qa_input_price",
            "cost_qa_output_price",
            "request_timeout",
            "retries",
            "think",
            "reasoning_effort",
            "top_k",
            "embedding_dim",
            "embedding_model",
            "embedding_base_url",
            "executor_model",
            "executor_base_url",
            "executor_temperature",
            "executor_max_tokens",
            "m2a_wma_rounds_per_ingest",
            "mirix_skip_failed_build_points",
            "mirix_max_consecutive_failed_build_points",
            "m2a_skip_failed_build_points",
            "m2a_max_consecutive_failed_build_points",
            "executor_visual_input",
            "graph_retrieval",
            "graph_mode",
            "append_k",
            "seed_k",
            "expansion_bonus",
            "efficiency_config",
            "m3_reuse_state_root",
        },
    )
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be provided together")
    manifest_index = (
        SplitManifestIndex(args.split_manifest) if args.split_manifest else None
    )
    manifest_split = normalize_split_name(args.split) if args.split else ""
    if manifest_index is not None and (args.sample_id or args.max_qa):
        parser.error(
            "--sample-id/--max-qa cannot be combined with strict manifest selection"
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
    if args.baseline == "HiveMem" and not args.query_embedding_dir:
        parser.error("--query-embedding-dir is required when --baseline=HiveMem")
    if (
        args.answer_concurrency < 1
        or args.sample_concurrency < 1
        or args.top_k < 1
        or args.checkpoint_every < 1
        or args.m2a_wma_rounds_per_ingest < 1
        or args.mirix_max_consecutive_failed_build_points < 1
        or args.m2a_max_consecutive_failed_build_points < 1
        or args.sample_attempts < 1
    ):
        parser.error("Sample/answer concurrency, top-k, and checkpoint interval must be positive")
    if args.max_qa < 0 or args.retries < 0 or args.request_timeout <= 0:
        parser.error("Invalid QA limit, retry count, or request timeout")

    data_dir = Path(args.data_dir)
    available_paths = iter_wma_sample_files(data_dir)
    ordered_ids_by_sample: dict[str, tuple[str, ...]] = {}
    if manifest_index is not None:
        manifest_rows = manifest_index.conversations(
            manifest_split, data_source="worldmemarena_lifelong"
        )
        paths_by_stem = {path.stem: path for path in available_paths}
        missing_samples = [
            row.source_id for row in manifest_rows if row.source_id not in paths_by_stem
        ]
        if missing_samples:
            raise FileNotFoundError(
                f"Missing WMA manifest sample(s): {missing_samples}"
            )
        paths = [paths_by_stem[row.source_id] for row in manifest_rows]
        ordered_ids_by_sample = {
            row.source_id: row.question_ids for row in manifest_rows
        }
    else:
        selected = set(args.sample_id)
        paths = [
            path for path in available_paths
            if not selected or path.stem in selected
        ]
    if not paths:
        raise FileNotFoundError(f"No matching WorldMemArena samples under {data_dir}")
    source_questions = 0
    source_excluded_questions = 0
    for path in paths:
        source_payload = json.loads(path.read_text(encoding="utf-8"))
        for checkpoint in source_payload.get("qa_checkpoints", []) or []:
            source_qas = checkpoint.get("questions", []) or []
            source_questions += len(source_qas)
            source_excluded_questions += sum(
                is_excluded_category(
                    qa.get("question_type_abbrev", ""), excluded_categories
                )
                for qa in source_qas
            )
    client = VLMAnswerClient(
        model=args.answer_model, base_url=args.answer_base_url,
        api_key=args.answer_api_key, temperature=args.answer_temperature,
        num_predict=args.num_predict,
        timeout=args.request_timeout, retries=args.retries, think=args.think,
        reasoning_effort=args.reasoning_effort,
    )
    cache = (
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
    graph_options = (
        {
            "seed_k": args.seed_k,
            "expansion_bonus": args.expansion_bonus,
            "mode": args.graph_mode,
            "append_k": args.append_k,
        }
        if args.graph_retrieval else False
    )
    result_dir = Path(args.result_dir)
    output_layout = BaselineOutputLayout(result_dir)
    checkpoint_dir = output_layout.checkpoint_dir
    baseline_state_root = output_layout.state_root(args.baseline_state_dir)
    signature = _run_signature(args, paths)
    compatible_sample_signatures = _mma_resume_signature_digests(args, signature)
    sample_signature = compatible_sample_signatures[0]

    checkpoint_answer_pool = (
        ThreadPoolExecutor(max_workers=args.answer_concurrency)
        if args.baseline == "M2A"
        else None
    )

    def prepare(path: Path) -> dict[str, Any]:
        if args.resume:
            cached = next(
                (
                    artifact
                    for compatible_signature in compatible_sample_signatures
                    if (
                        artifact := load_sample_artifact(
                            output_layout.sample_checkpoint_dir,
                            path.stem,
                            signature=compatible_signature,
                        )
                    )
                    is not None
                ),
                None,
            )
            if cached is not None:
                print(f"[resume] skip prepared WMA sample: {path.stem}", flush=True)
                return cached
        sample_snapshots: list[dict[str, Any]] = []
        native_progress_path = (
            checkpoint_dir
            / "native_samples"
            / Path(trace_filename(path.stem)).with_suffix(".json")
        )
        native_progress: dict[str, Any] = {}
        if (
            args.baseline in {"MIRIX", "MMA"}
            and args.resume
            and native_progress_path.is_file()
        ):
            try:
                loaded_progress = json.loads(
                    native_progress_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                loaded_progress = {}
            if (
                loaded_progress.get("version") == 1
                and loaded_progress.get("sample_id") == path.stem
                and (
                    loaded_progress.get("signature")
                    in compatible_sample_signatures
                    or os.getenv("BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT", "0")
                    == "1"
                )
            ):
                native_progress = loaded_progress
        sample_jobs: list[dict[str, Any]] = list(native_progress.get("jobs") or [])
        sample_results: list[dict[str, Any]] = list(
            native_progress.get("results") or []
        )
        sample_traces: list[dict[str, Any]] = list(native_progress.get("traces") or [])

        def save_native_progress(status: str, error: str = "") -> None:
            if args.baseline not in {"MIRIX", "MMA"}:
                return
            if native_progress_path.is_file():
                try:
                    existing = json.loads(
                        native_progress_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing = {}
                existing_count = int(existing.get("completed_questions") or 0)
                if existing_count > len(sample_results):
                    raise RuntimeError(
                        "refusing to regress native QA checkpoint from "
                        f"{existing_count} to {len(sample_results)} questions"
                    )
            write_json_atomic(
                native_progress_path,
                {
                    "version": 1,
                    "sample_id": path.stem,
                    "signature": sample_signature,
                    "status": status,
                    "completed_questions": len(sample_results),
                    "jobs": sample_jobs,
                    "results": sample_results,
                    "traces": sample_traces,
                    "last_error": error[:4000],
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

        def checkpoint_native_qa(
            job: dict[str, Any],
            result: dict[str, Any],
            trace: dict[str, Any],
        ) -> None:
            query_id = str(job["query_id"])
            if any(str(row.get("query_id")) == query_id for row in sample_jobs):
                return
            sample_jobs.append(job)
            sample_results.append(result)
            sample_traces.append(trace)
            save_native_progress("running")
        if args.baseline == "HiveMem":
            if cache is None:
                raise ValueError("HiveMem query embedding cache is unavailable")
            sample_jobs = prepare_sample_jobs(
                path, Path(args.index_root), cache,
                top_k=args.top_k, graph_options=graph_options,
                prefix_graph_root=output_layout.memory_dir / "prefix_graphs",
                excluded_categories=excluded_categories,
                ordered_question_ids=ordered_ids_by_sample.get(path.stem),
            )
        else:
            config_overrides = {
                    "answer_model": args.answer_model,
                    "answer_base_url": args.answer_base_url,
                    "answer_temperature": args.answer_temperature,
                    "executor_model": args.executor_model,
                    "executor_base_url": args.executor_base_url,
                    "executor_temperature": args.executor_temperature,
                    "executor_max_tokens": args.executor_max_tokens,
                    "m2a_wma_rounds_per_ingest": args.m2a_wma_rounds_per_ingest,
                    "mirix_skip_failed_build_points": (
                        args.mirix_skip_failed_build_points
                    ),
                    "mirix_max_consecutive_failed_build_points": (
                        args.mirix_max_consecutive_failed_build_points
                    ),
                    "m2a_skip_failed_build_points": (
                        args.m2a_skip_failed_build_points
                    ),
                    "m2a_max_consecutive_failed_build_points": (
                        args.m2a_max_consecutive_failed_build_points
                    ),
                    "mma_native_batch_size": args.mma_native_batch_size,
                    "executor_visual_input": args.executor_visual_input,
                    "executor_native_tool_calls": args.baseline in {"MIRIX", "MMA"},
                    "embedding_model": args.embedding_model,
                    "embedding_base_url": args.embedding_base_url,
                    "embedding_dim": args.embedding_dim,
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
            recorder = CallRecorder(
                trace_path=call_trace_path,
                baseline=args.baseline,
                benchmark="WorldMemArena",
                sample_id=path.stem,
                reset=not (
                    args.baseline in {"MMA", "MIRIX"}
                    and
                    args.resume
                    and (
                        baseline_state_root
                        / path.stem
                        / (
                            ".offline_mma_resume.json"
                            if args.baseline == "MMA"
                            else ".offline_mirix_resume.json"
                        )
                    ).is_file()
                ),
            )
            with CountingProxy(
                args.executor_base_url,
                recorder,
                args.request_timeout,
                max_output_tokens=args.executor_max_tokens,
                temperature=args.executor_temperature,
                qa_max_output_tokens=args.num_predict,
                reasoning_effort=args.reasoning_effort,
            ) as proxy:
                config_overrides["executor_base_url"] = proxy.endpoint
                try:
                    prepared_jobs = prepare_native_sample_jobs(
                        path,
                        cache,
                        baseline=args.baseline,
                        state_root=baseline_state_root,
                        top_k=args.top_k,
                        config_overrides=config_overrides,
                        memory_snapshots=sample_snapshots,
                        excluded_categories=excluded_categories,
                        ordered_question_ids=ordered_ids_by_sample.get(path.stem),
                        call_recorder=recorder,
                        checkpoint_answer_client=(
                            client
                            if args.baseline in {"M2A", "MIRIX", "MMA"}
                            else None
                        ),
                        checkpoint_answer_executor=checkpoint_answer_pool,
                        checkpoint_results=(
                            sample_results if args.baseline == "M2A" else None
                        ),
                        checkpoint_traces=(
                            sample_traces if args.baseline == "M2A" else None
                        ),
                        completed_jobs={
                            str(row.get("manifest_question_id") or ""): row
                            for row in sample_jobs
                            if row.get("manifest_question_id")
                        },
                        on_qa_completed=(
                            checkpoint_native_qa
                            if args.baseline in {"MIRIX", "MMA"}
                            else None
                        ),
                        allow_native_qa_errors=args.allow_answer_errors,
                    )
                except Exception as exc:
                    save_native_progress("pending_retry", str(exc))
                    raise
                if args.baseline in {"MIRIX", "MMA"}:
                    # The callback has already persisted every completed QA.
                    # Use source-order jobs returned by the resumed native pass.
                    sample_jobs = prepared_jobs
                else:
                    sample_jobs = prepared_jobs
        artifact = {
            "sample_id": path.stem,
            "jobs": sample_jobs,
            "snapshots": sample_snapshots,
            "results": sample_results,
            "traces": sample_traces,
        }
        if args.baseline != "HiveMem":
            artifact["call_trace_path"] = str(call_trace_path)
        save_sample_artifact(
            output_layout.sample_checkpoint_dir,
            path.stem,
            signature=sample_signature,
            artifact=artifact,
        )
        save_native_progress("completed")
        print(f"[prepared] {path.stem}: {len(sample_jobs)} question(s)", flush=True)
        return artifact

    if args.baseline == "M2A" and not args.skip_model_check:
        client.assert_model_available()
    try:
        if args.baseline == "MIRIX":
            artifacts = run_sample_retry_queue(
                paths,
                prepare,
                max_attempts=args.sample_attempts,
                status_path=checkpoint_dir / "sample_status.json",
            )
        else:
            artifacts = parallel_map_ordered(
                paths,
                prepare,
                max_workers=args.sample_concurrency,
                item_key=lambda path: path.stem,
            )
    finally:
        if checkpoint_answer_pool is not None:
            checkpoint_answer_pool.shutdown(wait=True, cancel_futures=True)
    jobs = [
        _with_manifest_question_id(job)
        for artifact in artifacts
        for job in artifact["jobs"]
    ]
    expected_manifest_question_ids = (
        manifest_index.ordered_question_ids(
            manifest_split, data_source="worldmemarena_lifelong"
        )
        if manifest_index is not None
        else None
    )
    if expected_manifest_question_ids is not None:
        actual = tuple(str(job.get("manifest_question_id") or "") for job in jobs)
        if actual != expected_manifest_question_ids:
            raise RuntimeError(
                "WMA prepared jobs do not exactly match manifest question order"
            )
    memory_snapshots = [
        row for artifact in artifacts for row in artifact.get("snapshots", [])
    ]
    if args.max_qa:
        jobs = jobs[: args.max_qa]

    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(checkpoint_dir / "prepared_qa.jsonl", jobs)
    checkpoint_manifest = checkpoint_dir / "manifest.json"
    checkpoint_results = checkpoint_dir / "results.json"
    checkpoint_traces = checkpoint_dir / "retrieval_trace.jsonl"
    job_ids = [str(job["query_id"]) for job in jobs]
    job_id_set = set(job_ids)
    if len(job_ids) != len(job_id_set):
        raise RuntimeError("WorldMemArena jobs contain duplicate query_id values")
    results_by_id: dict[str, dict[str, Any]] = {}
    trace_by_id: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        for row in artifact.get("results", []):
            query_id = str(row.get("query_id") or "")
            if query_id not in job_id_set or query_id in results_by_id:
                raise RuntimeError(
                    f"Invalid or duplicate inline WMA answer: {query_id!r}"
                )
            results_by_id[query_id] = row
        for row in artifact.get("traces", []):
            query_id = str(row.get("query_id") or "")
            if query_id not in job_id_set or query_id in trace_by_id:
                raise RuntimeError(
                    f"Invalid or duplicate inline WMA trace: {query_id!r}"
                )
            trace_by_id[query_id] = row
    if args.baseline == "M2A" and set(results_by_id) != set(trace_by_id):
        raise RuntimeError("M2A inline checkpoint answers and traces do not match")
    if args.resume and checkpoint_manifest.exists():
        saved_manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if saved_manifest.get("signature") != signature:
            raise RuntimeError(
                f"Checkpoint settings or input files changed: {checkpoint_manifest}; "
                "rerun with --no-resume"
            )
        if not checkpoint_results.is_file() or not checkpoint_traces.is_file():
            raise RuntimeError(
                f"Incomplete checkpoint under {checkpoint_dir}; rerun with --no-resume"
            )
        for row in json.loads(checkpoint_results.read_text(encoding="utf-8")):
            query_id = str(row.get("query_id") or "")
            if query_id in job_id_set:
                results_by_id[query_id] = row
        for line in checkpoint_traces.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                query_id = str(row.get("query_id") or "")
                if query_id in job_id_set:
                    trace_by_id[query_id] = row
        print(
            f"[resume] loaded {len(results_by_id)} checkpointed answer(s)",
            flush=True,
        )

    def save_checkpoint() -> None:
        ordered_ids = [query_id for query_id in job_ids if query_id in results_by_id]
        write_json_atomic(
            checkpoint_results,
            [results_by_id[query_id] for query_id in ordered_ids],
        )
        write_jsonl_atomic(
            checkpoint_traces,
            [trace_by_id[query_id] for query_id in ordered_ids if query_id in trace_by_id],
        )
        write_json_atomic(
            checkpoint_manifest,
            {
                "signature": signature,
                "completed": len(ordered_ids),
                "expected": len(job_ids),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    completed = {
        query_id
        for query_id, row in results_by_id.items()
        if query_id in trace_by_id
        and (
            args.baseline == "MMA"
            or args.allow_answer_errors
            or not row.get("error")
        )
    }
    pending = [job for job in jobs if job["query_id"] not in completed]
    if pending and not args.skip_model_check:
        client.assert_model_available()
    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
        futures = {pool.submit(answer_job, client, job): job["query_id"] for job in pending}
        for future in as_completed(futures):
            result, trace = future.result()
            query_id = futures[future]
            results_by_id[query_id] = result
            trace_by_id[query_id] = trace
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                since_checkpoint = 0
            print(
                f"[{len(results_by_id)}/{len(jobs)}] {query_id} "
                f"error={str(result.get('error') or '')[:100]!r}",
                flush=True,
            )
    save_checkpoint()
    missing_results = [query_id for query_id in job_ids if query_id not in results_by_id]
    missing_traces = [query_id for query_id in job_ids if query_id not in trace_by_id]
    if missing_results or missing_traces:
        raise RuntimeError(
            f"Incomplete WMA output: {len(missing_results)} results and "
            f"{len(missing_traces)} traces missing"
        )
    results = [results_by_id[query_id] for query_id in job_ids]
    if expected_manifest_question_ids is not None:
        result_ids = tuple(
            str(row.get("manifest_question_id") or "") for row in results
        )
        trace_ids = tuple(
            str(trace_by_id[query_id].get("manifest_question_id") or "")
            for query_id in job_ids
        )
        if result_ids != expected_manifest_question_ids:
            raise RuntimeError("WMA results do not match manifest question order")
        if trace_ids != expected_manifest_question_ids:
            raise RuntimeError("WMA retrieval traces do not match manifest question order")
    write_json_atomic(result_dir / "results.json", results)
    if args.baseline == "HiveMem":
        memory_snapshots = load_hivemem_snapshot(
            args.index_root,
            (path.stem for path in paths),
        )
    write_jsonl_atomic(output_layout.snapshot, memory_snapshots)
    trace_path = result_dir / "retrieval_trace.jsonl"
    write_jsonl_atomic(trace_path, [trace_by_id[query_id] for query_id in job_ids])
    answer_errors = sum(bool(row.get("error")) for row in results)
    public_args = {
        key: value for key, value in vars(args).items() if key != "answer_api_key"
    }
    if args.baseline != "HiveMem":
        public_args["baseline_state_dir"] = str(baseline_state_root)
    public_args["memory_snapshot"] = str(output_layout.snapshot)
    manifest = public_args | {
        "samples": len(paths),
        "questions": len(jobs),
        "excluded_categories": sorted(excluded_categories),
        "source_questions": source_questions,
        "source_excluded_questions": source_excluded_questions,
        "source_eligible_questions": source_questions - source_excluded_questions,
        "completed": len(results),
        "answer_errors": answer_errors,
        "baseline_runtime": baseline_metadata(args.baseline),
        "selection_mode": "strict_manifest" if manifest_index is not None else "legacy",
        "split_manifest_sha256": (
            manifest_index.file_sha256 if manifest_index is not None else ""
        ),
        "ordered_question_ids": list(expected_manifest_question_ids or ()),
        "prompt_version": PROMPT_VERSION,
        "prompt_source": PROMPT_SOURCE,
        "prompt_sha256": prompt_sha256(),
        "chunk_input": (
            m2a_wma_input_manifest(
                paths,
                rounds_per_batch=args.m2a_wma_rounds_per_ingest,
            )
            if args.baseline == "M2A"
            else omni_input_manifest("wma_lifelong")
            if args.baseline in {"OmniSimpleMem", "MMA"}
            else m3_input_manifest("wma_lifelong")
            if args.baseline == "M3-Agent-caption"
            else chunk_source_manifest({}, "wma_lifelong")
        ),
    }
    sample_status_path = checkpoint_dir / "sample_status.json"
    if sample_status_path.is_file():
        sample_status = json.loads(sample_status_path.read_text(encoding="utf-8"))
        manifest["resilient_sample_execution"] = {
            "status_file": str(sample_status_path),
            "counts": dict(sample_status.get("counts") or {}),
            "max_attempts_per_sample": int(
                sample_status.get("max_attempts_per_sample") or args.sample_attempts
            ),
            "per_qa_native_checkpoint": args.baseline == "MIRIX",
            "mirix_sqlite_checkpoint": args.baseline == "MIRIX",
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
        manifest["omni_conformance"] = omni_conformance_manifest("worldmemarena")
    if args.baseline == "M3-Agent-caption":
        manifest["m3_conformance"] = m3_conformance_manifest(
            "worldmemarena", answer_prompt_sha256=prompt_sha256()
        )
    write_json_atomic(result_dir / "run_manifest.json", manifest | {"run_signature": signature})
    pipeline_path = result_dir / "pipeline_qa.jsonl"
    if answer_errors and not args.allow_answer_errors:
        (result_dir / "metrics.json").unlink(missing_ok=True)
        pipeline_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{answer_errors}/{len(results)} answer requests failed; "
            f"partial results were saved under {result_dir}, but metrics were not written"
        )
    effective_top_k = (
        args.top_k + args.append_k
        if args.graph_retrieval and args.graph_mode == "append"
        else args.top_k
    )
    summary = summarize_results(results, k=effective_top_k)
    evaluated_sample_ids = sorted(
        {str(row.get("sample_id") or "").strip() for row in results}
        - {""}
    )
    if args.baseline == "HiveMem":
        summary["calls"] = combine_call_metrics(
            calculate_calls_mb(Path(args.index_root), evaluated_sample_ids),
            calculate_calls_qa(results, sample_id_field="sample_id"),
        )
    else:
        summary["calls"] = write_runtime_call_metrics(
            [
                artifact["call_trace_path"]
                for artifact in artifacts
                if artifact.get("call_trace_path")
            ],
            result_dir,
            results,
            sample_id_field="sample_id",
            sample_ids=evaluated_sample_ids,
        )
    try:
        if args.baseline == "HiveMem":
            memory_metrics = write_memory_metrics(
                Path(args.index_root),
                result_dir,
                tokenizer_name=args.executor_model,
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
        results,
        sample_id_field="sample_id",
        sample_ids=evaluated_sample_ids,
        model=args.answer_model,
        config_path=args.efficiency_config,
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
    write_jsonl_atomic(
        pipeline_path,
        [
            to_pipeline_qa_record(result, trace_by_id[result["query_id"]])
            for result in results
        ],
    )
    print(json.dumps({"result_dir": str(result_dir), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
