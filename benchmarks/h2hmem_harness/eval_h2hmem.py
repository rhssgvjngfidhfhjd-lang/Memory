from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any
import urllib.request

from src.utils import api_key_for, dataset_root, write_json_atomic, write_jsonl_atomic
from benchmarks.runtime import create_adapter
from benchmarks.runtime.adapter import RetrievalRequest, result_context_items, result_trace_rows
from benchmarks.runtime.evaluation import (
    add_common_arguments, parse_arguments, answer_client, graph_options,
    run_signature, prepare_samples, evaluate_jobs, finish_evaluation,
)
from embedding.chunks import iter_h2h_session_files
from benchmarks.common.answer_client import (
    VLMAnswerClient, build_retrieved_memory_context, build_retrieved_memory_evidence, query_image_prompt_metadata,
)
from benchmarks.common.prompts import (
    H2HMEM_PROMPT_SOURCE as PROMPT_SOURCE,
    H2HMEM_PROMPT_VERSION as PROMPT_VERSION,
    build_h2hmem_answer_messages as build_answer_messages,
    evidence_with_zero_hit_marker, parse_answer_response,
    h2hmem_prompt_sha256 as prompt_sha256,
)
from benchmarks.common.metrics import summarize_results as _summarize_answers
from benchmarks.common.query_cache import QueryEmbeddingCache, h2hmem_question_id

DEFAULT_H2HMEM_DATA_DIR = dataset_root('h2hmem')


def embed_texts(texts: list[str], config: dict[str, Any]) -> list[list[float]]:
    """Legacy text embedding helper; graph evaluation uses prepared query vectors."""
    if not texts:
        return []
    base_url = str(config.get("embedding_base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("embedding_base_url is required for HiVe_mem")
    endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
    body = json.dumps(
        {"model": str(config["embedding_model"]), "input": texts},
        # Real benchmark files can contain lone UTF-16 surrogates. Escaping
        # non-ASCII here keeps the JSON valid without losing those code units.
        ensure_ascii=True,
    ).encode("utf-8")
    key_env = str(config.get("embedding_api_key_env") or "EMBEDDING_API_KEY")
    headers = {"Content-Type": "application/json"}
    api_key = api_key_for("embedding", config.get("embedding_api_key"), env_name=key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    timeout = float(config.get("request_timeout") or 180)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = sorted(payload.get("data") or [], key=lambda row: int(row.get("index", 0)))
    vectors = [[float(value) for value in row["embedding"]] for row in rows]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
        )
    expected = int(config.get("embedding_dim") or 0)
    if expected and any(len(vector) != expected for vector in vectors):
        dimensions = sorted({len(vector) for vector in vectors})
        raise ValueError(f"embedding dimension mismatch: expected {expected}, got {dimensions}")
    return vectors

def summarize_results(results: list[dict], k: int = 5) -> dict:
    """Score H2HMEM retrieval against the annotated answer sessions."""
    normalized = [{**row, "clue": [str(value) for value in row.get("answer_session", [])],
                   "retrieved_source_groups": [[str(session)] for session in row.get("retrieved_sessions", [])]}
                  for row in results]
    return _summarize_answers(normalized, k=k)


def _natural_key(path: Path) -> tuple[Any, ...]:
    return tuple((int(value) if value.isdigit() else value for part in path.parts for value in re.split('(\\d+)', part.casefold())))

def _question_files(conversation_dir: Path) -> list[Path]:
    return sorted(conversation_dir.glob('scenes/session*/questions.json'), key=_natural_key)

def _question_image(question_file: Path, raw: Any) -> dict[str, str] | None:
    value = str(raw or '').strip()
    if not value:
        return None
    scenes_dir = question_file.parents[2] / 'scenes'
    if '/' in value or '\\' in value:
        session_name, filename = re.split('[/\\\\]', value, maxsplit=1)
        path = scenes_dir / session_name / 'image' / filename
    else:
        path = question_file.parent / 'image' / value
    if not path.is_file():
        raise FileNotFoundError(f'H2HMem question image not found: {path}')
    return {'path': str(path.resolve()), 'img_id': value}

def _question_rows(conversation_dir: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for path in _question_files(conversation_dir):
        payload = json.loads(path.read_text(encoding='utf-8'))
        rows.extend(((path, index, question) for index, question in enumerate(payload.get('questions') or [], start=1) if question.get('validated', True)))
    return rows

def h2hmem_manifest_question_id(variant: str, conversation_id: str, session_id: str, qa_index: int, qa: dict[str, Any], *, expected_question_ids=None) -> str:
    """Canonical H2HMem question ID used by the split manifest."""
    return h2hmem_question_id(variant, conversation_id, session_id, qa_index, qa,
                              expected_question_ids=expected_question_ids)

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
    total_image_count = 0
    try:
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
        total_image_count = getattr(response, 'total_image_count', None)
        answer = parse_answer_response(raw_answer)
        error = ""
    except Exception as exc:
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
                    query_image=job.get("query_image_payload"),
                    category="VR",
                )
            total_image_count = getattr(exc, 'total_image_count', None)
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "question_prompt", "query_image_payload", "memory_items", "retrieval_top_k",
            "retrieval_method_trace",
        }
    }
    result.update(
        {
            "system_answer": answer,
            "answer_raw_response": raw_answer,
            "retrieved_ids": [row["memory_id"] for row in job["retrieval_top_k"]],
            "retrieved_sessions": [row["session_id"] for row in job["retrieval_top_k"]],
            "retrieved_source_groups": [
                row["source_dialogue_ids"] for row in job["retrieval_top_k"]
            ],
            "error": error,
            "answer_seconds": time.time() - started,
            "answer_token_usage": usage,
            "answer_attempts": attempts,
            "answer_failed_attempts": failed_attempts,
            "answer_image_count": image_count,
            "answer_total_image_count": total_image_count if total_image_count is not None else image_count * attempts,
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
    }
    return result, trace



def prepare_conversation_jobs(*, data_dir: Path, variant: str, conversation_id: str,
                              config: dict[str, Any], query_cache: QueryEmbeddingCache, max_qa: int = 0,
                              ordered_question_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    conversation_dir = data_dir / ('multi-party' if variant == 'multiparty' else variant) / conversation_id
    allowed = set(ordered_question_ids) if ordered_question_ids is not None else None
    indexed = [(h2hmem_manifest_question_id(variant, conversation_id, path.parent.name, index, qa,
                                          expected_question_ids=allowed), path, index, qa)
               for path, index, qa in _question_rows(conversation_dir)]
    if ordered_question_ids is not None:
        by_id = {row[0]: row for row in indexed}
        missing = [key for key in ordered_question_ids if key not in by_id]
        if missing:
            raise KeyError(f'H2HMEM manifest references missing questions for {variant}/{conversation_id}: {missing[:5]}')
        selected = [by_id[key] for key in ordered_question_ids]
    else:
        selected = indexed[:max_qa] if max_qa else indexed
    sample_id = f'{variant}_{conversation_id}'
    adapter = create_adapter(config_overrides=config)
    jobs = []
    try:
        adapter.reset(sample_id, Path())
        for manifest_id, question_file, qa_index, qa in selected:
            question_data = qa.get('question') or {}
            question = str(question_data.get('text') or '')
            question_type = qa.get('question_type') or {}
            category = str(question_type.get('sub_type') or question_type.get('main_type') or '')
            session = question_file.parent.name
            question_id = str(qa.get('question_id') or qa.get('original_question_id') or f'{conversation_id}:{session}:{qa_index}')
            query_id = manifest_id
            image = _question_image(question_file, question_data.get('image'))
            vector = query_cache.get_by_id(manifest_id)
            if vector is None:
                raise KeyError(f'H2HMEM query vector missing for {manifest_id}; prepare the matching query cache first')
            retrieval = adapter.retrieve(RetrievalRequest(query_id=query_id, text=question, category=category,
                                         top_k=int(config['top_k']), query_image=str(image['path']) if image else None,
                                         query_vector=vector))
            jobs.append({'uid': query_id, 'query_id': query_id, 'manifest_question_id': manifest_id,
                         'question_id': question_id, 'sample_id': conversation_id, 'conversation_id': conversation_id,
                         'dialogue_name': conversation_id, 'variant': variant, 'session_id': session,
                         'question': question, 'question_text': question,
                         'question_image': str(question_data.get('image') or ''),
                         'question_type': question_type, 'category': category, 'difficulty': qa.get('difficulty', ''),
                         'original_answer': qa.get('original_answer', ''), 'answer_session': qa.get('answer_session') or [],
                         'query_image_payload': image, 'memory_items': result_context_items(retrieval),
                         'retrieval_top_k': result_trace_rows(retrieval), 'retrieval_method_trace': dict(retrieval.trace)})
    finally:
        adapter.close()
    return {'sample_id': sample_id, 'variant': variant, 'conversation_id': conversation_id, 'jobs': jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate HiVe_mem on H2HMEM.')
    add_common_arguments(parser)
    parser.add_argument('--data-dir', default=str(DEFAULT_H2HMEM_DATA_DIR))
    parser.add_argument('--variant', choices=('dyadic', 'multiparty', 'all'), default='all')
    parser.add_argument('--conversation-id', action='append', default=[])
    parser.add_argument('--query-embedding-dir', required=True)
    args, manifest_index, split = parse_arguments(parser)
    cache = QueryEmbeddingCache(args.query_embedding_dir, expected_dim=args.embedding_dim,
                                expected_model=args.embedding_model, expected_revision=args.embedding_revision)
    cache.load()
    data_dir = Path(args.data_dir)
    if (data_dir / 'dataset').is_dir():
        data_dir = data_dir / 'dataset'
    if manifest_index is not None:
        if args.variant != 'all' or args.conversation_id:
            parser.error('--variant/--conversation-id cannot be combined with strict manifest selection')
        rows = [row for source in manifest_index.data_sources if source in {'h2hmem_dyadic', 'h2hmem_multiparty'}
                for row in manifest_index.conversations(split, data_source=source)]
        specs = [(row.variant, row.source_id, row.question_ids) for row in rows]
    else:
        variants = ('dyadic', 'multiparty') if args.variant == 'all' else (args.variant,)
        selected = set(args.conversation_id)
        specs = [(variant, conversation, None) for variant in variants
                 for conversation in dict.fromkeys(path.parents[2].name for path in iter_h2h_session_files(data_dir, variant=variant)
                                                   if not selected or path.parents[2].name in selected)]
    if not specs:
        raise FileNotFoundError(f'No H2HMEM conversations selected under {data_dir}')
    banks = [f'{variant}_{conversation}' for variant, conversation, _ in specs]
    paths = [path for variant, conversation, _ in specs
             for path in sorted((data_dir / ('multi-party' if variant == 'multiparty' else variant) / conversation).rglob('*.json'))]
    paths.extend((cache.vectors_path, cache.metadata_path, cache.manifest_path))
    prompt = {'prompt_version': PROMPT_VERSION, 'prompt_source': PROMPT_SOURCE, 'prompt_sha256': prompt_sha256()}
    signature = run_signature(args, paths, banks, prompt)
    config = {'top_k': args.top_k, 'index_root': args.index_root, 'graph_options': graph_options(args),
              'embedding_dim': args.embedding_dim, 'embedding_model': args.embedding_model}
    def prepare(spec: tuple) -> dict[str, Any]:
        variant, conversation, ordered_ids = spec
        return prepare_conversation_jobs(data_dir=data_dir, variant=variant, conversation_id=conversation,
                                         config=config, query_cache=cache, ordered_question_ids=ordered_ids)
    artifacts = prepare_samples(specs, prepare, args=args, signature=signature, item_key=lambda spec: f'{spec[0]}/{spec[1]}')
    jobs = [job for artifact in artifacts for job in artifact['jobs']]
    if args.max_qa:
        jobs = jobs[:args.max_qa]
    expected = tuple(question for _, _, questions in specs for question in (questions or ())) if manifest_index else None
    client = answer_client(VLMAnswerClient, args)
    results, traces = evaluate_jobs(jobs, client, answer_conversation_job, args=args, signature=signature, expected_question_ids=expected)
    for variant in dict.fromkeys(spec[0] for spec in specs):
        name = 'prediction_multi_party.json' if variant == 'multiparty' else 'prediction_dyadic.json'
        write_json_atomic(Path(args.result_dir) / name, {'predictions': [row for row in results if row['variant'] == variant]})
    write_jsonl_atomic(Path(args.result_dir) / 'pipeline_qa.jsonl', jobs)
    finish_evaluation(results, traces, args=args, signature=signature,
                      manifest={'benchmark': 'H2HMEM', **prompt,
                                'selection_mode': 'strict_manifest' if manifest_index else 'selected_conversations',
                                'split_manifest_sha256': manifest_index.file_sha256 if manifest_index else '',
                                'ordered_question_ids': list(expected or ())},
                      summarize=summarize_results, sample_id_field='sample_id',
                      bank_id=lambda row: f"{row['variant']}_{row['conversation_id']}")


if __name__ == '__main__':
    main()
