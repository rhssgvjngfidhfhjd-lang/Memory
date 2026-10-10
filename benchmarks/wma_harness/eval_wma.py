from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from src.utils import dataset_root, wma_framework_root, write_jsonl_atomic
from benchmarks.runtime import create_adapter
from benchmarks.runtime.adapter import RetrievalRequest, result_context_items, result_trace_rows
from benchmarks.runtime.evaluation import (
    add_common_arguments, parse_arguments, answer_client, graph_options,
    run_signature, prepare_samples, evaluate_jobs, finish_evaluation,
)
from benchmarks.common.utils import is_excluded_category, parse_excluded_categories
from embedding.chunks import iter_wma_sample_files
from src.graph import materialize_prefix_graph
from benchmarks.common.query_cache import QueryEmbeddingCache
from benchmarks.wma_harness.questions import (
    build_gold_evidence_map, make_query_id, session_ids, visible_sessions_for_checkpoint,
)
from benchmarks.common.answer_client import VLMAnswerClient as _BaseClient
from benchmarks.common.prompts import (
    WMA_PROMPT_SOURCE as PROMPT_SOURCE, WMA_PROMPT_VERSION as PROMPT_VERSION,
    build_wma_answer_messages as build_answer_messages,
    wma_prompt_sha256 as prompt_sha256, parse_answer_response,
    evidence_with_zero_hit_marker,
)
from benchmarks.wma_harness.metrics import summarize_results

VISUAL_CATEGORIES = {'VFR', 'VS', 'VU', 'CMR', ''}
DEFAULT_WMA_DATA_DIR = dataset_root('wma')
DEFAULT_WORLDMEMARENA_ROOT = wma_framework_root()


# Agent-domain rows labelled question_type="unknown" use an empty abbreviation,
# including some questions whose gold evidence is an image.
MEMORY_IMAGE_CATEGORIES = frozenset({"VFR", "VS", "VU", "CMR", ""})


def build_retrieved_memory_context(
    memory_items: list[dict[str, Any]], category: str = ""
) -> tuple[str, list[str]]:
    evidence, image_paths = build_retrieved_memory_evidence(memory_items, category)
    return "\n\n".join(["The retrieved memory contents are as follows:", *evidence]), image_paths


def build_retrieved_memory_evidence(
    memory_items: list[dict[str, Any]], category: str = ""
) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    image_paths: list[str] = []
    seen_image_paths: set[str] = set()
    include_images = category.upper() in MEMORY_IMAGE_CATEGORIES
    for rank, item in enumerate(memory_items, start=1):
        metadata = item.get("metadata", {}) or {}
        raw_images = item.get("images")
        if not isinstance(raw_images, list):
            legacy = item.get("image")
            raw_images = [legacy] if isinstance(legacy, dict) else []
        attached_images: list[dict[str, Any]] = []
        if include_images:
            for image in raw_images:
                if not isinstance(image, dict) or not image.get("path"):
                    continue
                image_path = str(image["path"])
                if image_path in seen_image_paths:
                    continue
                seen_image_paths.add(image_path)
                attached_images.append(image)
        attached_original = any(
            str(image.get("kind", "image")) == "image" for image in attached_images
        )
        header = (
            f"[{rank}] SESSION:{metadata.get('session_id', '')} "
            f"ROUND:{metadata.get('dialogue_id', '')}"
        )
        if attached_original and metadata.get("image_id"):
            header += f" IMG:{metadata['image_id']}"
        text = str(item.get("text", ""))
        if not attached_original:
            for image_id in metadata.get("image_ids", []) or []:
                if image_id:
                    text = text.replace(str(image_id), "[IMAGE_ID_REDACTED]")
        block = [header, text]
        for image in attached_images:
            image_paths.append(str(image["path"]))
            raw_kind = str(image.get("kind", "image")).lower()
            kind = "image" if raw_kind == "image" else raw_kind.upper()
            block.append(
                f"Attached memory {kind} {len(image_paths)}: {image.get('img_id', '')}"
            )
        evidence.append("\n\n".join(block))
    return evidence, image_paths


class VLMAnswerClient(_BaseClient):
    def _build_text_and_image_paths(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
        *,
        prepend_memory_context: bool = True,
    ) -> tuple[str, list[str]]:
        memory_text, image_paths = build_retrieved_memory_context(memory_items, category)
        lines = (
            [memory_text, "", question_prompt]
            if prepend_memory_context
            else [question_prompt]
        )
        if query_image and query_image.get("path"):
            lines.append(f"Attached question image {len(image_paths) + 1}.")
            image_paths.append(str(query_image["path"]))
        return "\n\n".join(lines), image_paths


def wma_manifest_question_id(sample_id: str, checkpoint_id: str, qa_index: int) -> str:
    """Canonical WMA question ID used by the split manifest."""
    return f'{sample_id}:{checkpoint_id}:Q{qa_index:03d}'

def _with_manifest_question_id(job: dict[str, Any]) -> dict[str, Any]:
    """Backfill canonical IDs in sample checkpoints written before manifests."""
    if job.get('manifest_question_id'):
        return job
    qa_index = job.get('qa_index')
    if qa_index is None:
        raise KeyError(f"WMA job lacks both manifest_question_id and qa_index: {job.get('query_id', '<unknown>')}")
    normalized = dict(job)
    normalized['manifest_question_id'] = wma_manifest_question_id(str(job['sample_id']), str(job['checkpoint_id']), int(qa_index))
    return normalized

def _order_wma_jobs(jobs: list[dict[str, Any]], ordered_question_ids: tuple[str, ...] | None, *, sample_id: str) -> list[dict[str, Any]]:
    if ordered_question_ids is None:
        return jobs
    by_manifest_id = {str(row['manifest_question_id']): row for row in jobs}
    missing = [value for value in ordered_question_ids if value not in by_manifest_id]
    if missing:
        raise KeyError(f'WMA manifest references {len(missing)} missing question(s) for {sample_id}: {missing[:5]}')
    return [by_manifest_id[value] for value in ordered_question_ids]

def prepare_sample_jobs(
    sample_path: Path,
    index_root: Path,
    query_cache: QueryEmbeddingCache,
    *,
    top_k: int,
    graph_options: dict[str, Any],
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
                        "retrieval_method_trace": dict(retrieval.trace),
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



def answer_job(client: VLMAnswerClient, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    job = _with_manifest_question_id(job)
    started = time.time()
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], job["category"]
    )
    evidence, _ = build_retrieved_memory_evidence(
        job["memory_items"], job["category"]
    )
    prompt_evidence, zero_hit_prompt_marker_used = evidence_with_zero_hit_marker(
        evidence
    )
    messages = build_answer_messages(
        question=job["question"],
        question_type=job["category"],
        memory_evidence=prompt_evidence,
        allow_empty_evidence=True,
    )
    raw_answer = ""
    answer_response = None
    answer_token_usage = None
    answer_attempts = 0
    answer_failed_attempts = 0
    answer_image_count = 0
    answer_total_image_count = 0
    try:
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
        answer_total_image_count = getattr(answer_response, 'total_image_count', None)
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
            answer_total_image_count = getattr(answer_response, 'total_image_count', None)
        else:
            answer_token_usage = getattr(exc, 'usage', None)
            answer_attempts = getattr(exc, 'attempts', client.retries + 1)
            answer_failed_attempts = getattr(exc, 'failed_attempts', answer_attempts)
            answer_image_count = getattr(exc, 'image_count', None)
            if answer_image_count is None:
                answer_image_count = client.count_answer_images(
                    job["memory_items"], category=job["category"]
                )
            answer_total_image_count = getattr(exc, 'total_image_count', None)
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
            "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
            "error": error,
            "answer_token_usage": answer_token_usage,
            "answer_attempts": answer_attempts,
            "answer_failed_attempts": answer_failed_attempts,
            "answer_image_count": answer_image_count,
            "answer_total_image_count": answer_total_image_count if answer_total_image_count is not None else answer_image_count * answer_attempts,
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
        "zero_hit_prompt_marker_used": zero_hit_prompt_marker_used,
        "memory_context": memory_context,
        "answer_prompt_messages": messages,
        "retrieval_method_trace": dict(job.get("retrieval_method_trace") or {}),
    }
    return result, trace


def to_pipeline_qa_record(result: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    items = trace.get('top_k', [])
    return {'sample_id': result['sample_id'], 'sample_uuid': result['sample_id'], 'manifest_question_id': result['manifest_question_id'], 'checkpoint_id': result['checkpoint_id'], 'question': result['question'], 'gold_answer': result['original_answer'], 'gold_evidence_memory_ids': result.get('gold_evidence_memory_ids', []), 'gold_evidence_contents': result.get('gold_evidence_contents', []), 'question_type': result.get('question_type', ''), 'question_type_abbrev': result.get('category', ''), 'difficulty': result.get('difficulty', ''), 'retrieval': {'query': result['question'], 'top_k': len(items), 'items': [{'rank': row['rank'], 'memory_id': row['memory_id'], 'text': row['content'], 'score': row['score'], 'raw_backend_id': row['memory_id'], 'image_path': (row.get('image_paths') or [None])[0]} for row in items], 'raw_trace': {'retrieval_source_sessions_by_rank': {str(row['rank']): [row.get('session_id', '')] for row in items}}}, 'generated_answer': result.get('system_answer', ''), 'cited_memories': [], 'retrieval_seconds': 0.0, 'answer_seconds': result.get('answer_seconds', 0.0), 'retrieval_token_usage': {}, 'answer_token_usage': result.get('answer_token_usage')}


def evaluate_results_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate WorldMemArena result JSON.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--official-mode", choices=("none", "metrics", "judge"), default="none"
    )
    parser.add_argument(
        "--worldmemarena-root",
        default=str(DEFAULT_WORLDMEMARENA_ROOT),
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    metrics = summarize_results(rows, k=args.top_k)
    if args.official_mode != "none":
        framework_root = Path(args.worldmemarena_root)
        if not (framework_root / "eval_framework").is_dir():
            raise FileNotFoundError(
                f"Official WMA evaluator not found in {framework_root}. "
                "Set HIVE_WMA_FRAMEWORK_ROOT or --worldmemarena-root."
            )
        sys.path.insert(0, str(framework_root))
        from eval_framework.cli import _qa_record_from_dict
        from eval_framework.evaluators.qa import (
            evaluate_checkpoint_qa,
            evaluate_checkpoint_qa_metrics_only,
        )

        pipeline_path = Path(args.results).with_name("pipeline_qa.jsonl")
        pipeline_rows = [
            json.loads(line)
            for line in pipeline_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluator = (
            evaluate_checkpoint_qa_metrics_only
            if args.official_mode == "metrics"
            else evaluate_checkpoint_qa
        )
        official: list[dict[str, Any]] = [
            evaluator(_qa_record_from_dict(row)) for row in pipeline_rows
        ]
        count = len(official)
        metrics["official"] = {
            "mode": args.official_mode,
            "count": count,
            "answer_f1": (
                sum(float(row.get("answer_f1") or 0.0) for row in official) / count
                if count else 0.0
            ),
            "answer_bleu1": (
                sum(float(row.get("answer_bleu1") or 0.0) for row in official) / count
                if count else 0.0
            ),
            "correct": sum(row.get("answer_label") == "Correct" for row in official),
        }
        official_path = Path(args.output).with_name("official_qa_eval.jsonl")
        with official_path.open("w", encoding="utf-8") as handle:
            for row in official:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))



def main() -> None:
    argv = sys.argv[1:]
    if '--summarize-only' in argv:
        evaluate_results_main([value for value in argv if value != '--summarize-only'])
        return
    parser = argparse.ArgumentParser(description='Evaluate HiVe_mem on WorldMemArena.')
    parser.add_argument(
        '--summarize-only', action='store_true', default=argparse.SUPPRESS,
        help='Score existing result JSON; use --summarize-only --help for options.',
    )
    add_common_arguments(parser)
    parser.add_argument('--data-dir', default=str(DEFAULT_WMA_DATA_DIR))
    parser.add_argument('--query-embedding-dir', required=True)
    parser.add_argument('--sample-id', action='append', default=[])
    parser.add_argument('--exclude-categories', default='')
    args, manifest_index, split = parse_arguments(parser)
    available = {path.stem: path for path in iter_wma_sample_files(Path(args.data_dir))}
    ordered = {}
    if manifest_index is not None:
        if args.sample_id:
            parser.error('--sample-id cannot be combined with strict manifest selection')
        rows = manifest_index.conversations(split, data_source='worldmemarena_lifelong')
        missing = [row.source_id for row in rows if row.source_id not in available]
        if missing:
            raise FileNotFoundError(f'WorldMemArena manifest sample files are missing: {missing[:5]}')
        paths = [available[row.source_id] for row in rows]
        ordered = {row.source_id: row.question_ids for row in rows}
    else:
        selected = set(args.sample_id)
        paths = [path for name, path in available.items() if not selected or name in selected]
    if not paths:
        raise FileNotFoundError(f'No WorldMemArena samples selected under {args.data_dir}')
    excluded = frozenset() if manifest_index else parse_excluded_categories(args.exclude_categories)
    cache = QueryEmbeddingCache(args.query_embedding_dir, expected_dim=args.embedding_dim,
                                expected_model=args.embedding_model, expected_revision=args.embedding_revision)
    query_root = Path(args.query_embedding_dir)
    prompt = {'prompt_version': PROMPT_VERSION, 'prompt_source': PROMPT_SOURCE, 'prompt_sha256': prompt_sha256()}
    signature = run_signature(args, [*paths, *(query_root / name for name in ('vectors.npy', 'metadata.jsonl', 'manifest.json'))],
                              (path.stem for path in paths), prompt)
    def prepare(path: Path) -> dict[str, Any]:
        jobs = prepare_sample_jobs(path, Path(args.index_root), cache, top_k=args.top_k,
                                   graph_options=graph_options(args),
                                   prefix_graph_root=Path(args.result_dir) / 'memory' / 'prefix_graphs',
                                   excluded_categories=excluded, ordered_question_ids=ordered.get(path.stem))
        return {'sample_id': path.stem, 'jobs': jobs}
    artifacts = prepare_samples(paths, prepare, args=args, signature=signature, item_key=lambda path: path.stem)
    jobs = [job for artifact in artifacts for job in artifact['jobs']]
    if args.max_qa:
        jobs = jobs[:args.max_qa]
    expected = manifest_index.ordered_question_ids(split, data_source='worldmemarena_lifelong') if manifest_index else None
    client = answer_client(VLMAnswerClient, args)
    results, traces = evaluate_jobs(jobs, client, answer_job, args=args, signature=signature, expected_question_ids=expected)
    write_jsonl_atomic(Path(args.result_dir) / 'pipeline_qa.jsonl',
                       [to_pipeline_qa_record(result, trace) for result, trace in zip(results, traces)])
    finish_evaluation(results, traces, args=args, signature=signature,
                      manifest={'benchmark': 'WorldMemArena', **prompt,
                                'selection_mode': 'strict_manifest' if manifest_index else 'selected_samples',
                                'split_manifest_sha256': manifest_index.file_sha256 if manifest_index else '',
                                'ordered_question_ids': list(expected or ())},
                      summarize=summarize_results, sample_id_field='sample_id', bank_id=lambda row: row['sample_id'])


if __name__ == '__main__':
    main()
