from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from src.utils import dataset_root, write_jsonl_atomic
from benchmarks.runtime import create_adapter
from benchmarks.runtime.adapter import RetrievalRequest, result_context_items, result_trace_rows
from benchmarks.runtime.evaluation import (
    add_common_arguments, parse_arguments, answer_client, graph_options,
    run_signature, prepare_samples, evaluate_jobs, finish_evaluation,
)
from benchmarks.common.utils import is_excluded_category, parse_excluded_categories
from benchmarks.common.query_cache import QueryEmbeddingCache, make_query_id
from benchmarks.common.answer_client import (
    VLMAnswerClient, build_retrieved_memory_context, build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.common.prompts import (
    build_memgallery_answer_messages as build_answer_messages,
    memgallery_prompt_manifest as prompt_manifest,
    parse_answer_response, resolve_question_image, evidence_with_zero_hit_marker,
)
from benchmarks.common.metrics import summarize_results

DEFAULT_MEMGALLERY_DATA_DIR = dataset_root('memgallery')

def memgallery_manifest_question_id(dataset_name: str, qa_index: int) -> str:
    """Canonical ID used by multimodal_split_manifest.json."""
    return f'{dataset_name}_q{qa_index - 1:04d}'

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
    usage = None
    attempts = 0
    failed_attempts = 0
    image_count = 0
    total_image_count = 0
    try:
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
        total_image_count = getattr(response, 'total_image_count', None)
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
            total_image_count = getattr(response, 'total_image_count', None)
        else:
            usage = getattr(exc, 'usage', None)
            attempts = getattr(exc, 'attempts', client.retries + 1)
            failed_attempts = getattr(exc, 'failed_attempts', attempts)
            image_count = getattr(exc, 'image_count', None)
            if image_count is None:
                image_count = client.count_answer_images(
                    job["memory_items"],
                    query_image=job.get("query_image"),
                    category=job["category"],
                )
            total_image_count = getattr(exc, 'total_image_count', None)
    memory_context, _ = build_retrieved_memory_context(
        job["memory_items"], job["category"]
    )
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "qa_index", "question_prompt", "system_prompt",
            "query_image", "memory_items", "retrieval_top_k",
            "retrieval_method_trace",
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
            "answer_total_image_count": total_image_count if total_image_count is not None else image_count * attempts,
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
    }
    return result, trace



def prepare_dataset_jobs(dataset_path: Path, data_dir: Path, index_root: Path, query_cache: QueryEmbeddingCache,
                         *, top_k: int = 5, max_qa: int = 0, qa_start: int = 1, qa_end: int = 0,
                         graph_options: dict | None = None,
                         excluded_categories: frozenset[str] = frozenset(),
                         ordered_question_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    name = dataset_path.stem
    profile = dataset.get('character_profile') or {}
    indexed = [(memgallery_manifest_question_id(name, index), index, qa)
               for index, qa in enumerate(dataset.get('human-annotated QAs', []) or [], start=1)]
    if ordered_question_ids is not None:
        by_id = {row[0]: row for row in indexed}
        missing = [key for key in ordered_question_ids if key not in by_id]
        if missing:
            raise KeyError(f'Mem-Gallery manifest references missing questions for {name}: {missing[:5]}')
        selected = [by_id[key] for key in ordered_question_ids]
    else:
        selected = [row for row in indexed if row[1] >= qa_start and (not qa_end or row[1] <= qa_end)
                    and not is_excluded_category(row[2].get('point', ''), excluded_categories)]
        if max_qa:
            selected = selected[:max_qa]
    adapter = create_adapter(config_overrides={'index_root': str(index_root), 'graph_options': graph_options})
    jobs = []
    try:
        adapter.reset(name, Path())
        for manifest_id, qa_index, qa in selected:
            category, question = str(qa.get('point', '')), str(qa.get('question', ''))
            image = resolve_question_image(data_dir, qa)
            query_id = make_query_id(dataset_name=name, qa_index=qa_index, category=category, question=question, query_image=image)
            vector = query_cache.get(dataset_name=name, qa_index=qa_index, category=category, question=question, query_image=image)
            if vector is None:
                raise KeyError(f'Missing cached query embedding: {query_id}')
            retrieval = adapter.retrieve(RetrievalRequest(query_id=query_id, text=f'[{category}] {question}',
                                         category=category, top_k=top_k,
                                         query_image=str(image.get('path') or '') if isinstance(image, dict) else None,
                                         query_vector=vector))
            trace = result_trace_rows(retrieval)
            groups = [row['source_dialogue_ids'] for row in trace]
            jobs.append({'query_id': query_id, 'manifest_question_id': manifest_id,
                         'sample_id': profile.get('name', name), 'dataset': name,
                         'session_id': qa.get('session_id', ''), 'qa_index': qa_index,
                         'question': question, 'category': category,
                         'speaker_a': f"user ({profile['name']})" if profile.get('name') else 'user',
                         'query_image': image, 'original_answer': qa.get('answer', ''),
                         'retrieved_ids': list(dict.fromkeys(source for group in groups for source in group)),
                         'retrieved_source_groups': groups,
                         'clue': qa.get('clue', []) if isinstance(qa.get('clue'), list) else [],
                         'memory_items': result_context_items(retrieval), 'retrieval_top_k': trace,
                         'retrieval_method_trace': dict(retrieval.trace)})
    finally:
        adapter.close()
    return {'sample_id': name, 'jobs': jobs, 'eligible_questions': len(jobs), 'excluded_questions': len(indexed) - len(selected)}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate HiVe_mem on Mem-Gallery.')
    add_common_arguments(parser)
    parser.add_argument('--data-dir', default=str(DEFAULT_MEMGALLERY_DATA_DIR))
    parser.add_argument('--data-name', default='AI_Robotics_Automation_Future_Tech')
    parser.add_argument('--all-datasets', action='store_true')
    parser.add_argument('--query-embedding-dir', required=True)
    parser.add_argument('--qa-start', type=int, default=1)
    parser.add_argument('--qa-end', type=int, default=0)
    parser.add_argument('--exclude-categories', default='AR')
    args, manifest_index, split = parse_arguments(parser)
    if args.qa_start < 1 or args.qa_end < 0 or (args.qa_end and args.qa_end < args.qa_start):
        parser.error('Invalid QA range')
    if manifest_index is not None and (args.qa_start != 1 or args.qa_end):
        parser.error('QA ranges cannot be combined with strict manifest selection')
    data_dir = Path(args.data_dir)
    ordered = {}
    if manifest_index is not None:
        rows = manifest_index.conversations(split, data_source='mem_gallery')
        paths = [data_dir / 'dialog' / f'{row.source_id}.json' for row in rows]
        ordered = {row.source_id: row.question_ids for row in rows}
    else:
        paths = sorted((data_dir / 'dialog').glob('*.json')) if args.all_datasets else [data_dir / 'dialog' / f'{args.data_name}.json']
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError(f'Mem-Gallery dataset selection is missing under {data_dir / "dialog"}')
    excluded = frozenset() if manifest_index is not None else parse_excluded_categories(args.exclude_categories)
    cache = QueryEmbeddingCache(args.query_embedding_dir, expected_dim=args.embedding_dim,
                                expected_model=args.embedding_model, expected_revision=args.embedding_revision)
    query_root = Path(args.query_embedding_dir)
    signature = run_signature(args, [*paths, *(query_root / name for name in ('vectors.npy', 'metadata.jsonl', 'manifest.json'))],
                              (path.stem for path in paths), prompt_manifest())
    def prepare(path: Path) -> dict[str, Any]:
        return prepare_dataset_jobs(path, data_dir, Path(args.index_root), cache,
                                    top_k=args.top_k, max_qa=args.max_qa, qa_start=args.qa_start, qa_end=args.qa_end,
                                    graph_options=graph_options(args), excluded_categories=excluded,
                                    ordered_question_ids=ordered.get(path.stem))
    artifacts = prepare_samples(paths, prepare, args=args, signature=signature, item_key=lambda path: path.stem)
    jobs = [job for artifact in artifacts for job in artifact['jobs']]
    expected = manifest_index.ordered_question_ids(split, data_source='mem_gallery') if manifest_index else None
    client = answer_client(VLMAnswerClient, args)
    results, traces = evaluate_jobs(jobs, client, lambda client, job: answer_dataset_job(client, job, allow_answer_errors=True),
                                   args=args, signature=signature, expected_question_ids=expected)
    write_jsonl_atomic(Path(args.result_dir) / 'pipeline_qa.jsonl', jobs)
    finish_evaluation(results, traces, args=args, signature=signature,
                      manifest={'benchmark': 'Mem-Gallery', **prompt_manifest(),
                                'selection_mode': 'strict_manifest' if manifest_index else 'selected_datasets',
                                'split_manifest_sha256': manifest_index.file_sha256 if manifest_index else '',
                                'ordered_question_ids': list(expected or ()),
                                'excluded_questions': sum(row['excluded_questions'] for row in artifacts)},
                      summarize=summarize_results, sample_id_field='dataset', bank_id=lambda row: row['dataset'])


if __name__ == '__main__':
    main()
