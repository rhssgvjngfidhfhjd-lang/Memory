from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

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
from benchmarks.wma_harness.runner.metrics import summarize_results
from benchmarks.wma_harness.runner.prompts import SYSTEM_PROMPT, format_question_prompt
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
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.question_filter import is_excluded_category, parse_excluded_categories
from embedding.chunk_builder import build_wma_chunks_from_data, iter_wma_sample_files
from evidence_policy.split_manifest import SplitManifestIndex, normalize_split_name


VISUAL_CATEGORIES = {"VFR", "VS", "VU", "CMR", ""}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WMA_DATA_DIR = Path(
    os.getenv(
        "WMA_DATA_DIR",
        PROJECT_ROOT / "WorldMemArena" / "WorldMemArena" / "lifelong",
    )
)


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
    graph_options: dict[str, Any] | None,
    excluded_categories: frozenset[str] = frozenset(),
    ordered_question_ids: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if graph_options is not None:
        raise ValueError(
            "Graph retrieval is disabled for WMA checkpoints: the current graph "
            "statistics are built from the full memory bank and are not prefix-safe."
        )
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_id = str(payload["sample_id"])
    adapter = create_adapter(
        "HiveMem",
        config_overrides={
            "index_root": str(index_root),
            "top_k": top_k,
            "visual_categories": VISUAL_CATEGORIES,
        },
    )
    ordered_sessions = session_ids(payload)
    gold_points = build_gold_evidence_map(payload)
    jobs: list[dict[str, Any]] = []
    selected_question_ids = (
        set(ordered_question_ids) if ordered_question_ids is not None else None
    )
    try:
        adapter.reset(sample_id, Path())
        for checkpoint in payload.get("qa_checkpoints", []) or []:
            checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
            covered_sessions = [str(value) for value in checkpoint.get("covered_sessions", [])]
            visible_sessions = visible_sessions_for_checkpoint(
                ordered_sessions, covered_sessions
            )
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
    finally:
        adapter.close()
    return _order_wma_jobs(
        jobs, ordered_question_ids, sample_id=sample_id
    )


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
) -> list[dict[str, Any]]:
    """Stream one WMA sample through a native baseline without future leakage."""
    baseline = canonical_name(baseline)
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_id = str(payload["sample_id"])
    ordered_sessions = session_ids(payload)
    session_order = {session_id: index for index, session_id in enumerate(ordered_sessions)}
    chunks = build_wma_chunks_from_data(
        payload,
        sample_path.parent,
        sample_path=sample_path,
    )
    chunks_by_session: dict[str, list[Any]] = {session_id: [] for session_id in ordered_sessions}
    for chunk in chunks:
        chunks_by_session.setdefault(str(chunk.metadata.get("session_id") or ""), []).append(chunk)
    gold_points = build_gold_evidence_map(payload)
    checkpoints = []
    for position, checkpoint in enumerate(payload.get("qa_checkpoints", []) or []):
        covered = [str(value) for value in checkpoint.get("covered_sessions", [])]
        visible = visible_sessions_for_checkpoint(ordered_sessions, covered)
        last_index = max((session_order[value] for value in visible), default=-1)
        checkpoints.append((last_index, position, checkpoint, covered, visible))
    checkpoints.sort(key=lambda row: (row[0], row[1]))

    adapter = create_adapter(baseline, config_overrides=config_overrides)
    jobs: list[dict[str, Any]] = []
    selected_question_ids = (
        set(ordered_question_ids) if ordered_question_ids is not None else None
    )
    ingested_through = -1
    try:
        adapter.reset(sample_id, state_root / sample_id)
        for last_index, _, checkpoint, covered_sessions, visible_sessions in checkpoints:
            for session_index in range(ingested_through + 1, last_index + 1):
                session_id = ordered_sessions[session_index]
                with (
                    call_recorder.phase("memory_build")
                    if call_recorder is not None
                    else nullcontext()
                ):
                    for chunk in chunks_by_session.get(session_id, []):
                        adapter.ingest(chunk)
                    adapter.end_session(session_id)
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
                question = str(qa.get("question", ""))
                query_id = make_query_id(
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    qa_index=qa_index,
                    category=category,
                    question=question,
                )
                vector = query_cache.get_by_id(query_id) if query_cache is not None else None
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
                trace = result_trace_rows(retrieval)
                evidence = qa.get("evidence", []) or []
                evidence_ids = [
                    str(row.get("memory_id") or row.get("image_id") or "")
                    for row in evidence
                    if isinstance(row, dict) and (row.get("memory_id") or row.get("image_id"))
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
                        "memory_items": result_context_items(retrieval),
                        "retrieval_top_k": trace,
                    }
                )
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
    try:
        answer_response = client.answer_with_usage(
            system_prompt=SYSTEM_PROMPT,
            memory_items=job["memory_items"],
            question_prompt=format_question_prompt(job["question"], job["category"]),
            category=job["category"],
        )
        answer = answer_response.text
        answer_token_usage = answer_response.usage
        answer_attempts = answer_response.attempts
        answer_failed_attempts = answer_response.failed_attempts
        answer_image_count = answer_response.image_count
        error = ""
    except Exception as exc:
        answer, error = "", str(exc)
        answer_token_usage = None
        answer_attempts = client.retries + 1
        answer_failed_attempts = client.retries + 1
        answer_image_count = client.count_answer_images(
            job["memory_items"], category=job["category"]
        )
    memory_context, _ = build_retrieved_memory_context(job["memory_items"], job["category"])
    top_k = job["retrieval_top_k"]
    result = {key: value for key, value in job.items() if key not in {"memory_items", "retrieval_top_k"}}
    result.update(
        {
            "system_answer": answer,
            "retrieved_ids": [row["memory_id"] for row in top_k],
            "retrieved_source_groups": [row["source_dialogue_ids"] for row in top_k],
            "retrieved_sessions": [row["session_id"] for row in top_k],
            "error": error,
            "answer_token_usage": answer_token_usage,
            "answer_attempts": answer_attempts,
            "answer_failed_attempts": answer_failed_attempts,
            "answer_image_count": answer_image_count,
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
        "memory_context": memory_context,
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
        "skip_model_check",
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
            input_paths.extend(
                [
                    bank / "memories.jsonl",
                    bank / "text_vectors.npy",
                    bank / "image_vectors.npy",
                    bank / "image_mask.npy",
                ]
            )
    prompt_source = SYSTEM_PROMPT + "\n" + inspect.getsource(format_question_prompt)
    return {
        "arguments": {
            key: value for key, value in vars(args).items() if key not in ignored
        },
        "inputs": file_manifest(input_paths),
        "prompt_sha256": hashlib.sha256(prompt_source.encode("utf-8")).hexdigest(),
        "call_trace_version": TRACE_VERSION,
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
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--exclude-categories",
        default="MB",
        help="Comma-separated QA categories to skip before embedding, retrieval, and answering.",
    )
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default="EMPTY")
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
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--executor-temperature", type=float, default=0.0)
    parser.add_argument("--executor-visual-input", choices=("image", "caption"), default="image")
    parser.add_argument(
        "--allow-answer-errors",
        action="store_true",
        help="Write metrics and pipeline records even when VLM answer requests failed.",
    )
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--graph-retrieval", action="store_true")
    parser.add_argument("--seed-k", type=int, default=0)
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--graph-mode", choices=("rerank", "append"), default="rerank")
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
            "top_k",
            "embedding_dim",
            "embedding_model",
            "embedding_base_url",
            "executor_model",
            "executor_base_url",
            "executor_temperature",
            "executor_visual_input",
            "efficiency_config",
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
        if args.graph_retrieval else None
    )
    result_dir = Path(args.result_dir)
    output_layout = BaselineOutputLayout(result_dir)
    baseline_state_root = output_layout.state_root(args.baseline_state_dir)
    signature = _run_signature(args, paths)
    sample_signature = signature_digest(signature)

    def prepare(path: Path) -> dict[str, Any]:
        if args.resume:
            cached = load_sample_artifact(
                output_layout.sample_checkpoint_dir,
                path.stem,
                signature=sample_signature,
            )
            if cached is not None:
                print(f"[resume] skip prepared WMA sample: {path.stem}", flush=True)
                return cached
        sample_snapshots: list[dict[str, Any]] = []
        if args.baseline == "HiveMem":
            if cache is None:
                raise ValueError("HiveMem query embedding cache is unavailable")
            sample_jobs = prepare_sample_jobs(
                path, Path(args.index_root), cache,
                top_k=args.top_k, graph_options=graph_options,
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
                    "executor_visual_input": args.executor_visual_input,
                    "embedding_model": args.embedding_model,
                    "embedding_base_url": args.embedding_base_url,
                    "embedding_dim": args.embedding_dim,
                    "top_k": args.top_k,
                    "request_timeout": args.request_timeout,
                    "retries": args.retries,
                }
            call_trace_path = result_dir / "call_traces" / trace_filename(path.stem)
            recorder = CallRecorder(
                trace_path=call_trace_path,
                baseline=args.baseline,
                benchmark="WorldMemArena",
                sample_id=path.stem,
                reset=True,
            )
            with CountingProxy(
                args.executor_base_url,
                recorder,
                args.request_timeout,
            ) as proxy:
                config_overrides["executor_base_url"] = proxy.endpoint
                sample_jobs = prepare_native_sample_jobs(
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
                )
        artifact = {
            "sample_id": path.stem,
            "jobs": sample_jobs,
            "snapshots": sample_snapshots,
        }
        if args.baseline != "HiveMem":
            artifact["call_trace_path"] = str(call_trace_path)
        save_sample_artifact(
            output_layout.sample_checkpoint_dir,
            path.stem,
            signature=sample_signature,
            artifact=artifact,
        )
        print(f"[prepared] {path.stem}: {len(sample_jobs)} question(s)", flush=True)
        return artifact

    artifacts = parallel_map_ordered(
        paths,
        prepare,
        max_workers=args.sample_concurrency,
        item_key=lambda path: path.stem,
    )
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
    checkpoint_dir = output_layout.checkpoint_dir
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
        if not row.get("error") and query_id in trace_by_id
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
    }
    write_json_atomic(result_dir / "run_manifest.json", manifest | {"run_signature": signature})
    pipeline_path = result_dir / "pipeline_qa.jsonl"
    if answer_errors and not args.allow_answer_errors:
        (result_dir / "metrics.json").unlink(missing_ok=True)
        pipeline_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{answer_errors}/{len(results)} answer requests failed; "
            f"partial results were saved under {result_dir}, but metrics were not written"
        )
    summary = summarize_results(results, k=args.top_k)
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
