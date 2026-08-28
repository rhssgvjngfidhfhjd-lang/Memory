from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
)
from benchmarks.memgallery_harness.runner.prompts import (
    SYSTEM_PROMPT,
    format_question_prompt,
    prompt_manifest,
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
    summarize_results,
    write_memory_metrics,
    write_snapshot_memory_metrics,
    write_retrieval_memory_token,
)
from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.baseline_runtime.protocol import (
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.question_filter import is_excluded_category, parse_excluded_categories
from embedding.chunk_builder import build_chunks_from_data
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMGALLERY_DATA_DIR = WORKSPACE_ROOT / "Mem-Gallery" / "benchmark" / "data"


def run_dataset(
    dataset_path: Path,
    data_dir: Path,
    index_root: Path,
    client: VLMAnswerClient,
    query_cache: QueryEmbeddingCache | None,
    *,
    top_k: int = 5,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
    graph_options: dict | None = None,
    system_prompt: str | None = None,
    baseline: str = "HiveMem",
    state_root: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    memory_snapshots: list[dict[str, Any]] | None = None,
    allow_answer_errors: bool = True,
    excluded_categories: frozenset[str] = frozenset(),
    qa_stats: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
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
    native_adapter = create_adapter(baseline, config_overrides=overrides)
    sample_state = (state_root or Path("outputs") / "memory" / "datasets") / dataset_name
    native_adapter.reset(dataset_name, sample_state)
    if baseline != "HiveMem":
        chunks = build_chunks_from_data(dataset, data_dir, dataset_name)
        current_session = ""
        for chunk in chunks:
            session_id = str(chunk.metadata.get("session_id") or "")
            if current_session and session_id != current_session:
                native_adapter.end_session(current_session)
            native_adapter.ingest(chunk)
            current_session = session_id
        if current_session:
            native_adapter.end_session(current_session)
    qa_pairs = dataset.get("human-annotated QAs", [])
    qa_end = qa_end or len(qa_pairs)
    results = []
    retrieval_traces = []
    processed = 0
    excluded = 0
    try:
        for qa_index, qa in enumerate(qa_pairs, start=1):
            if qa_index < qa_start or qa_index > qa_end:
                continue
            category = str(qa.get("point", ""))
            if is_excluded_category(category, excluded_categories):
                excluded += 1
                continue
            if max_qa and processed >= max_qa:
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
            query_vector = query_cache.get_by_id(query_id) if query_cache is not None else None
            if baseline == "HiveMem" and query_vector is None:
                raise KeyError(f"Missing cached query embedding: {query_id}")
            retrieval = native_adapter.retrieve(
                RetrievalRequest(
                    query_id=query_id,
                    text=f"[{category}] {question}",
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
            memory_items = result_context_items(retrieval)
            trace_rows = result_trace_rows(retrieval)
            retrieved_groups = [row["source_dialogue_ids"] for row in trace_rows]
            retrieved_ids = list(
                dict.fromkeys(source for group in retrieved_groups for source in group)
            )
            memory_context, _ = build_retrieved_memory_context(memory_items, category)
            prompt = format_question_prompt(question, category, speaker_a, "assistant")
            try:
                answer_response = client.answer_with_usage(
                    system_prompt=system_prompt or SYSTEM_PROMPT,
                    memory_items=memory_items,
                    question_prompt=prompt,
                    query_image=query_image,
                    category=category,
                )
                answer = answer_response.text
                answer_token_usage = answer_response.usage
                answer_attempts = answer_response.attempts
                answer_failed_attempts = answer_response.failed_attempts
                error = ""
            except Exception as exc:
                if not allow_answer_errors:
                    raise RuntimeError(
                        f"Answer request failed for {dataset_name} QA {qa_index}: {exc}"
                    ) from exc
                answer = ""
                answer_token_usage = None
                answer_attempts = client.retries + 1
                answer_failed_attempts = client.retries + 1
                error = str(exc)
            clue = qa.get("clue", []) if isinstance(qa.get("clue", []), list) else []
            result = {
            "sample_id": profile.get("name", dataset_name),
            "dataset": dataset_name,
            "session_id": qa.get("session_id", ""),
            "question": question,
            "category": category,
            "system_answer": answer,
            "original_answer": qa.get("answer", ""),
            "retrieved_ids": retrieved_ids,
            "retrieved_source_groups": retrieved_groups,
            "clue": clue,
            "error": error,
            "answer_token_usage": answer_token_usage,
            "answer_attempts": answer_attempts,
            "answer_failed_attempts": answer_failed_attempts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
            results.append(result)
            retrieval_traces.append(
                {
                "dataset": dataset_name,
                "qa_index": qa_index,
                "question": question,
                "category": category,
                "clue": clue,
                "top_k": trace_rows,
                "memory_context": memory_context,
                }
            )
            print(
                f"[{dataset_name} {qa_index}/{len(qa_pairs)}] {category} "
                f"answer={answer[:80]!r} error={error[:80]!r}",
                flush=True,
            )
        if baseline != "HiveMem" and memory_snapshots is not None:
            memory_snapshots.extend(row.to_dict() for row in native_adapter.snapshot())
    finally:
        native_adapter.close()
    if qa_stats is not None:
        qa_stats.update(
            {
                "eligible_questions": processed,
                "excluded_questions": excluded,
            }
        )
    return results, retrieval_traces


def _checkpoint_signature(
    args: argparse.Namespace,
    dataset_paths: list[Path],
) -> dict[str, Any]:
    ignored = {
        "resume",
        "allow_answer_errors",
        "memory_tokenizer",
        "retrieval_memory_tokenizer",
        "result_dir",
        "answer_api_key",
    }
    input_paths: list[Path] = list(dataset_paths)
    if args.profiles_file:
        input_paths.append(Path(args.profiles_file))
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
    parser.add_argument(
        "--index-root",
        default="",
        help="HiveMem run directory containing datasets/.",
    )
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument("--query-embedding-dir", default="data/qwen3_vl_embedding_2b/query_embeddings")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--qa-start", type=int, default=1)
    parser.add_argument("--qa-end", type=int, default=0)
    parser.add_argument(
        "--exclude-categories",
        default="AR",
        help="Comma-separated QA categories to skip before embedding, retrieval, and answering.",
    )
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default="EMPTY")
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--cost-mb-input-price", type=float, default=None)
    parser.add_argument("--cost-mb-output-price", type=float, default=None)
    parser.add_argument("--cost-qa-input-price", type=float, default=None)
    parser.add_argument("--cost-qa-output-price", type=float, default=None)
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
        help="Write metrics even when one or more VLM answer requests failed.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from the dataset-level checkpoint under RESULT_DIR/.checkpoint.",
    )
    parser.add_argument(
        "--profiles-file",
        default="",
        help="JSON file mapping dataset name -> profile_summary text; appended to the "
        "answer system prompt per dataset (use with profile-free memory banks).",
    )
    parser.add_argument("--graph-retrieval", action="store_true",
                        help="Plan-A graph-expanded retrieval: vector seeds + one-hop expansion + rerank")
    parser.add_argument("--seed-k", type=int, default=0, help="Seed count for expansion (0 = top_k)")
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--expand-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expand-related", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--related-types", default="",
                        help="Comma-separated edge types to expand via links.related (empty = all)")
    parser.add_argument("--graph-mode", default="rerank", choices=["rerank", "append"],
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
    excluded_categories = parse_excluded_categories(args.exclude_categories)
    try:
        args.baseline = canonical_name(args.baseline)
    except KeyError as exc:
        parser.error(str(exc))
    if args.baseline == "HiveMem" and not args.index_root:
        parser.error("--index-root is required when --baseline=HiveMem")
    if args.resume and args.baseline != "HiveMem":
        parser.error("--resume currently supports only --baseline=HiveMem")
    if args.top_k < 1 or args.max_qa < 0 or args.qa_start < 1 or args.qa_end < 0:
        parser.error("--top-k and --qa-start must be positive; QA limits cannot be negative")
    if args.qa_end and args.qa_end < args.qa_start:
        parser.error("--qa-end must be 0 or at least --qa-start")
    if args.retries < 0 or args.request_timeout <= 0 or args.num_predict < 1:
        parser.error("Invalid answer retry, timeout, or token limit")

    graph_options = None
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
    paths = sorted((data_dir / "dialog").glob("*.json")) if args.all_datasets else [data_dir / "dialog" / f"{args.data_name}.json"]
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
    profiles: dict[str, str] = {}
    if args.profiles_file:
        profiles = json.loads(Path(args.profiles_file).read_text(encoding="utf-8"))
        if not isinstance(profiles, dict):
            raise ValueError("--profiles-file must contain a JSON object")
    result_dir = Path(args.result_dir)
    output_layout = BaselineOutputLayout(result_dir)
    baseline_state_root = output_layout.state_root(args.baseline_state_dir)
    checkpoint_dir = result_dir / ".checkpoint"
    checkpoint_results = checkpoint_dir / "results.json"
    checkpoint_traces = checkpoint_dir / "retrieval_trace.jsonl"
    checkpoint_manifest = checkpoint_dir / "manifest.json"
    signature = _checkpoint_signature(args, paths)
    all_results: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    excluded_questions = 0
    completed_datasets: list[str] = []
    if args.resume and checkpoint_manifest.exists():
        saved_manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
        if not _checkpoint_signatures_match(
            saved_manifest.get("signature") or {}, signature
        ):
            raise RuntimeError(
                f"Checkpoint settings do not match this run: {checkpoint_manifest}"
            )
        if not checkpoint_results.is_file() or not checkpoint_traces.is_file():
            raise RuntimeError(
                f"Incomplete checkpoint under {checkpoint_dir}; rerun with --no-resume"
            )
        all_results = json.loads(checkpoint_results.read_text(encoding="utf-8"))
        all_traces = [
            json.loads(line)
            for line in checkpoint_traces.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed_datasets = [
            str(value) for value in saved_manifest.get("completed_datasets", [])
        ]
        excluded_questions = int(saved_manifest.get("excluded_questions", 0))
        if len(all_results) != len(all_traces):
            raise RuntimeError(
                f"Checkpoint has {len(all_results)} results but {len(all_traces)} traces"
            )
        if int(saved_manifest.get("questions", -1)) != len(all_results):
            raise RuntimeError("Checkpoint manifest question count does not match results")
        completed_set = set(completed_datasets)
        if any(str(row.get("dataset") or "") not in completed_set for row in all_results):
            raise RuntimeError("Checkpoint contains results outside completed_datasets")
        print(
            f"[resume] loaded {len(all_results)} answers from "
            f"{len(completed_datasets)} completed dataset(s)",
            flush=True,
        )
    remaining_paths = [path for path in paths if path.stem not in completed_datasets]
    if remaining_paths:
        client.assert_model_available()
    memory_snapshots: list[dict[str, Any]] = []
    for path in paths:
        if path.stem in completed_datasets:
            print(f"[resume] skip completed dataset: {path.stem}", flush=True)
            continue
        dataset_profile = profiles.get(path.stem, "")
        dataset_system_prompt = (
            SYSTEM_PROMPT + "\n\nUser profile (background about the person the "
            "memories are about):\n" + dataset_profile
        ) if dataset_profile else None
        dataset_qa_stats: dict[str, int] = {}
        results, traces = run_dataset(
            path,
            data_dir,
            Path(args.index_root) if args.index_root else Path(),
            client,
            query_cache,
            top_k=args.top_k,
            max_qa=args.max_qa,
            qa_start=args.qa_start,
            qa_end=args.qa_end,
            graph_options=graph_options,
            system_prompt=dataset_system_prompt,
            baseline=args.baseline,
            state_root=baseline_state_root,
            config_overrides={
                "index_root": args.index_root,
                "graph_options": graph_options,
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
            },
            memory_snapshots=memory_snapshots,
            allow_answer_errors=args.allow_answer_errors,
            excluded_categories=excluded_categories,
            qa_stats=dataset_qa_stats,
        )
        all_results.extend(results)
        all_traces.extend(traces)
        excluded_questions += dataset_qa_stats["excluded_questions"]
        completed_datasets.append(path.stem)
        write_json_atomic(checkpoint_results, all_results)
        write_jsonl_atomic(checkpoint_traces, all_traces)
        write_json_atomic(
            checkpoint_manifest,
            {
                "signature": signature,
                "completed_datasets": completed_datasets,
                "questions": len(all_results),
                "excluded_questions": excluded_questions,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print(
            f"[checkpoint] {len(completed_datasets)}/{len(paths)} datasets, "
            f"{len(all_results)} answers",
            flush=True,
        )

    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", all_results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", all_traces)
    if args.baseline != "HiveMem":
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
    }
    write_json_atomic(result_dir / "run_manifest.json", manifest)
    if answer_errors and not args.allow_answer_errors:
        (result_dir / "metrics.json").unlink(missing_ok=True)
        raise RuntimeError(
            f"{answer_errors}/{len(all_results)} answer requests failed; "
            f"partial results were saved under {result_dir}, but metrics were not written"
        )
    summary = summarize_results(all_results, k=args.top_k)
    summary["cost_qa"] = calculate_cost_qa(
        all_results,
        sample_id_field="dataset",
        input_price=args.cost_qa_input_price,
        output_price=args.cost_qa_output_price,
    )
    evaluated_sample_ids = sorted(
        {str(row.get("dataset") or "").strip() for row in all_results}
        - {""}
    )
    summary["calls"] = combine_call_metrics(
        calculate_calls_mb(
            Path(args.index_root) if args.baseline == "HiveMem" else None,
            evaluated_sample_ids,
            unavailable_reason=(
                f"The {args.baseline} baseline does not expose exact "
                "Memory-Bank call counts."
            ),
        ),
        calculate_calls_qa(all_results, sample_id_field="dataset"),
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
    write_json_atomic(result_dir / "metrics.json", summary)


if __name__ == "__main__":
    main()
