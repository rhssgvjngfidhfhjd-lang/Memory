"""Benchmark preparation, concurrency, checkpoints, output layout, and metrics."""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Iterable, TypeVar

from src.utils import (
    DatasetLayout, file_manifest, resolve_reference, validate_embedding_settings,
    write_json_atomic, write_jsonl_atomic,
)
from benchmarks.runtime.call_trace import TRACE_VERSION
from benchmarks.common.utils import require_service_url
from benchmarks.common.metrics import (
    add_memory_metrics, add_retrieval_memory_tokens, calculate_calls_mb,
    calculate_calls_qa, calculate_cost_mb, combine_call_metrics,
    merge_existing_llm_judge_metrics, write_efficiency_metrics,
    write_memory_metrics, write_retrieval_memory_token,
)
from evidence_policy.evidence import SplitManifestIndex, normalize_split_name


Item = TypeVar("Item")
Value = TypeVar("Value")


@dataclass(frozen=True)
class OutputLayout:
    root: Path

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def datasets_dir(self) -> Path:
        return self.memory_dir / "datasets"

    @property
    def snapshot(self) -> Path:
        return self.memory_dir / "memory_snapshot.jsonl"

    @property
    def pipeline_qa(self) -> Path:
        return self.root / "pipeline_qa.jsonl"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / ".checkpoint"

    @property
    def sample_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir / "samples"

    def state_root(self, override: str | Path = "") -> Path:
        return Path(override) if override else self.datasets_dir


def load_hivemem_snapshot(
    index_root: str | Path,
    sample_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Normalize HiVe_mem's native JSONL banks into the shared snapshot schema."""
    root = Path(index_root) / "datasets"
    if not root.is_dir():
        return []
    selected = {str(value) for value in sample_ids} if sample_ids is not None else None
    rows: list[dict] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected is not None and dataset_dir.name not in selected:
            continue
        memories_path = dataset_dir / "memories.jsonl"
        if not memories_path.is_file():
            continue
        with memories_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                memory = json.loads(line)
                metadata = dict(memory.get("metadata") or {})
                rows.append(
                    {
                        "memory_id": str(memory.get("memory_id") or memory.get("id") or ""),
                        "text": str(memory.get("content") or memory.get("summary") or ""),
                        "session_id": str(metadata.get("session_id") or ""),
                        "source_dialogue_ids": [
                            str(value) for value in metadata.get("source_dialogue_ids") or []
                        ],
                        "image_ids": [str(value) for value in metadata.get("image_ids") or []],
                        "image_paths": [
                            str(value) for value in metadata.get("image_paths") or []
                        ],
                        "backend_type": "hivemem",
                        "metadata": {**metadata, "dataset": dataset_dir.name},
                    }
                )
    return rows


def sample_artifact_path(root: Path, sample_id: str) -> Path:
    """Return a collision-safe, portable checkpoint filename for one sample."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:12]
    return root / f"{slug[:96]}-{digest}.json"


def load_sample_artifact(
    root: Path,
    sample_id: str,
    *,
    signature: str,
) -> dict[str, Any] | None:
    path = sample_artifact_path(root, sample_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("sample_id") != sample_id:
        return None
    signature_matches = payload.get("signature") == signature
    if not signature_matches:
        return None
    artifact = payload.get("artifact")
    return artifact if isinstance(artifact, dict) else None


def save_sample_artifact(
    root: Path,
    sample_id: str,
    *,
    signature: str,
    artifact: dict[str, Any],
) -> Path:
    path = sample_artifact_path(root, sample_id)
    write_json_atomic(
        path,
        {
            "version": 1,
            "sample_id": sample_id,
            "signature": signature,
            "artifact": artifact,
        },
    )
    return path


def signature_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parallel_map_ordered(
    items: Iterable[Item],
    worker: Callable[[Item], Value],
    *,
    max_workers: int,
    item_key: Callable[[Item], str] = str,
    on_complete: Callable[[str, Value], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[Value]:
    """Run all independent samples and return input-ordered values.

    Failures are collected until every submitted sample has had a chance to
    finish.  This matters for benchmark runners because each successful sample
    writes its own checkpoint; aborting iteration on the first exception hides
    later failures and makes recovery appear less complete than it is.
    """
    ordered = list(items)
    if not ordered:
        return []
    failures: list[tuple[str, Exception]] = []
    if max_workers <= 1:
        values = []
        for item in ordered:
            key = item_key(item)
            try:
                value = worker(item)
            except Exception as exc:
                failures.append((key, exc))
                _report_sample_failure(key, exc)
                if on_error is not None:
                    on_error(key, exc)
                continue
            if on_complete is not None:
                on_complete(key, value)
            values.append(value)
        if failures:
            _raise_parallel_failures(failures)
        return values

    completed: dict[str, Value] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future[Value], str] = {
            pool.submit(worker, item): item_key(item) for item in ordered
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                failures.append((key, exc))
                _report_sample_failure(key, exc)
                if on_error is not None:
                    on_error(key, exc)
                continue
            completed[key] = value
            if on_complete is not None:
                on_complete(key, value)
    if failures:
        _raise_parallel_failures(failures)
    return [completed[item_key(item)] for item in ordered]


def _report_sample_failure(key: str, exc: Exception) -> None:
    """Expose deferred sample failures immediately while peers checkpoint."""
    print(
        "[sample-error] "
        + json.dumps(
            {
                "sample_id": str(key),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "action": "continue_other_samples_then_fail_job",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _raise_parallel_failures(failures: list[tuple[str, Exception]]) -> None:
    details = "; ".join(
        f"{key}: {type(exc).__name__}: {exc}" for key, exc in failures
    )
    raise RuntimeError(
        f"{len(failures)} parallel sample(s) failed after other samples were allowed "
        f"to finish: {details}"
    ) from failures[0][1]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--split-manifest', default='')
    parser.add_argument('--split', default='')
    parser.add_argument('--index-root', required=True, help='HiVe_mem build directory containing datasets/.')
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--top-k', type=int, default=5, help='Number of vector hits before graph append.')
    parser.add_argument('--seed-k', type=int, default=0)
    parser.add_argument('--append-k', type=int, default=2)
    parser.add_argument('--degree-cap', type=int, default=4)
    parser.add_argument('--embedding-dim', type=int, default=None)
    parser.add_argument('--embedding-model', default='')
    parser.add_argument('--embedding-revision', default='')
    parser.add_argument('--embedding-base-url', default='')
    parser.add_argument('--answer-base-url', default='')
    parser.add_argument('--answer-model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--answer-api-key', default='EMPTY')
    parser.add_argument('--answer-temperature', type=float, default=0)
    parser.add_argument('--num-predict', type=int, default=512)
    parser.add_argument('--request-timeout', type=int, default=180)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--think', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reasoning-effort', default='')
    parser.add_argument('--sample-concurrency', type=int, default=4)
    parser.add_argument('--answer-concurrency', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--skip-model-check', action='store_true')
    parser.add_argument('--allow-answer-errors', action='store_true')
    parser.add_argument('--max-qa', type=int, default=0)
    parser.add_argument('--memory-tokenizer', default='')
    parser.add_argument('--retrieval-memory-tokenizer', default='')
    parser.add_argument('--cost-mb-input-price', type=float, default=None)
    parser.add_argument('--cost-mb-output-price', type=float, default=None)
    parser.add_argument('--cost-qa-input-price', type=float, default=None)
    parser.add_argument('--cost-qa-output-price', type=float, default=None)
    parser.add_argument('--efficiency-config', default='config://defaults.json')


def parse_arguments(parser: argparse.ArgumentParser, *, require_embedding_url: bool = False) -> tuple[argparse.Namespace, SplitManifestIndex | None, str]:
    from src.build import apply_config_defaults
    apply_config_defaults(parser)
    args = parser.parse_args()
    args.answer_base_url = require_service_url(parser, args.answer_base_url,
                                               flag='--answer-base-url', env_name='HIVE_ANSWER_BASE_URL')
    if require_embedding_url:
        args.embedding_base_url = require_service_url(parser, args.embedding_base_url,
                                                      flag='--embedding-base-url', env_name='HIVE_EMBEDDING_BASE_URL')
    args.embedding_model, args.embedding_dim = validate_embedding_settings(
        parser, args.embedding_model, args.embedding_dim,
    )
    if bool(args.split_manifest) != bool(args.split):
        parser.error('--split-manifest and --split must be provided together')
    if min(args.top_k, args.sample_concurrency, args.answer_concurrency, args.checkpoint_every, args.request_timeout, args.num_predict) < 1:
        parser.error('Retrieval count, concurrency, checkpoint interval, timeout, and token limit must be positive')
    if min(args.max_qa, args.append_k, args.seed_k, args.retries) < 0 or args.degree_cap < 1:
        parser.error('QA limit, graph counts, and retries cannot be negative; degree cap must be positive')
    index = SplitManifestIndex(resolve_reference(args.split_manifest)) if args.split_manifest else None
    split = normalize_split_name(args.split) if args.split else ''
    if index is not None and args.max_qa:
        parser.error('--max-qa cannot be combined with strict manifest selection')
    return args, index, split


def graph_options(args: argparse.Namespace) -> dict[str, Any]:
    return {'seed_k': args.seed_k, 'mode': 'append', 'append_k': args.append_k, 'degree_cap': args.degree_cap}


def answer_client(client_class: type, args: argparse.Namespace):
    return client_class(model=args.answer_model, base_url=args.answer_base_url,
                        api_key=args.answer_api_key, temperature=args.answer_temperature,
                        num_predict=args.num_predict, timeout=args.request_timeout,
                        retries=args.retries, think=args.think, reasoning_effort=args.reasoning_effort)


def run_signature(args: argparse.Namespace, inputs: Iterable[Path], bank_ids: Iterable[str], prompt: dict[str, Any]) -> dict[str, Any]:
    ignored = {'resume', 'sample_concurrency', 'answer_concurrency', 'checkpoint_every',
               'skip_model_check', 'allow_answer_errors', 'result_dir', 'answer_api_key',
               'memory_tokenizer', 'retrieval_memory_tokenizer'}
    paths = list(inputs)
    if args.split_manifest:
        paths.append(Path(resolve_reference(args.split_manifest)))
    root = Path(args.index_root)
    paths.append(root / 'build_manifest.json')
    for bank_id in bank_ids:
        bank = root / 'datasets' / bank_id
        layout = DatasetLayout(bank)
        paths.extend((bank / 'memories.jsonl',
                      layout.existing_vector_path('text.npy', 'vectors.npy'),
                      layout.existing_vector_path('image.npy', 'image_vectors.npy'),
                      layout.existing_vector_path('image_mask.npy', 'image_mask.npy'),
                      layout.attributes, layout.attribute_vectors, layout.edges_manifest,
                      layout.build_stats, layout.build_trace))
    return {'arguments': {key: value for key, value in vars(args).items() if key not in ignored},
            'inputs': file_manifest(paths), 'call_trace_version': TRACE_VERSION, **prompt}


def prepare_samples(specs: list[Any], worker: Callable[[Any], dict[str, Any]], *, args: argparse.Namespace,
                    signature: dict[str, Any], item_key: Callable[[Any], str]) -> list[dict[str, Any]]:
    layout = OutputLayout(Path(args.result_dir))
    digest = signature_digest(signature)
    status_path = layout.checkpoint_dir / 'preparation_status.json'
    statuses: dict[str, dict[str, Any]] = {item_key(spec): {'status': 'pending'} for spec in specs}
    lock = threading.Lock()

    def status(key: str, state: str, **details: Any) -> None:
        with lock:
            statuses[key] = {'status': state, **details}
            write_json_atomic(status_path, {'signature': digest, 'expected_samples': len(specs),
                                           'completed_samples': sum(row['status'] == 'completed' for row in statuses.values()),
                                           'samples': statuses, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})

    def prepare(spec: Any) -> dict[str, Any]:
        key = item_key(spec)
        if args.resume:
            artifact = load_sample_artifact(layout.sample_checkpoint_dir, key, signature=digest)
            if artifact is not None:
                status(key, 'completed', questions=len(artifact['jobs']), resumed=True)
                print(f'[resume] prepared sample {key}', flush=True)
                return artifact
        status(key, 'running')
        try:
            artifact = worker(spec)
            save_sample_artifact(layout.sample_checkpoint_dir, key, signature=digest, artifact=artifact)
        except Exception as exc:
            status(key, 'failed', error=f'{type(exc).__name__}: {exc}')
            raise
        status(key, 'completed', questions=len(artifact['jobs']))
        print(f"[prepared] {key}: {len(artifact['jobs'])} question(s)", flush=True)
        return artifact

    return parallel_map_ordered(specs, prepare, max_workers=args.sample_concurrency, item_key=item_key)


def evaluate_jobs(jobs: list[dict[str, Any]], client: Any, answer: Callable[[Any, dict[str, Any]], tuple[dict, dict]],
                  *, args: argparse.Namespace, signature: dict[str, Any], expected_question_ids: tuple[str, ...] | None) -> tuple[list[dict], list[dict]]:
    layout = OutputLayout(Path(args.result_dir))
    if expected_question_ids is not None:
        actual = tuple(str(job.get('manifest_question_id') or '') for job in jobs)
        if actual != expected_question_ids:
            raise RuntimeError('Prepared jobs do not exactly match split manifest question order')
    job_ids = [str(job['query_id']) for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError('Prepared jobs contain duplicate query IDs')
    write_jsonl_atomic(layout.checkpoint_dir / 'prepared_qa.jsonl', jobs)
    results_path = layout.checkpoint_dir / 'results.json'
    traces_path = layout.checkpoint_dir / 'retrieval_trace.jsonl'
    manifest_path = layout.checkpoint_dir / 'manifest.json'
    results_by_id: dict[str, dict] = {}
    traces_by_id: dict[str, dict] = {}
    if args.resume and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if saved.get('signature') != signature:
            raise RuntimeError(f'Checkpoint settings or input files changed: {manifest_path}; rerun with --no-resume')
        if not results_path.is_file() or not traces_path.is_file():
            raise RuntimeError(f'Incomplete answer checkpoint under {layout.checkpoint_dir}')
        results_by_id = {str(row['query_id']): row for row in json.loads(results_path.read_text(encoding='utf-8'))}
        traces_by_id = {str(row['query_id']): row for row in (json.loads(line) for line in traces_path.read_text(encoding='utf-8').splitlines() if line.strip())}
        print(f'[resume] loaded {len(results_by_id)} answer(s)', flush=True)

    def checkpoint() -> None:
        completed = [key for key in job_ids if key in results_by_id and key in traces_by_id]
        write_json_atomic(results_path, [results_by_id[key] for key in completed])
        write_jsonl_atomic(traces_path, [traces_by_id[key] for key in completed])
        write_json_atomic(manifest_path, {'signature': signature, 'completed': len(completed),
                                          'expected': len(job_ids), 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')})

    completed = {key for key in job_ids if key in results_by_id and key in traces_by_id
                 and not results_by_id[key].get('error')}
    pending = [job for job in jobs if job['query_id'] not in completed]
    if pending and not args.skip_model_check:
        client.assert_model_available()
    since_checkpoint = 0
    try:
        with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
            futures = {pool.submit(answer, client, job): str(job['query_id']) for job in pending}
            for future in as_completed(futures):
                result, trace = future.result()
                key = futures[future]
                results_by_id[key], traces_by_id[key] = result, trace
                since_checkpoint += 1
                if since_checkpoint >= args.checkpoint_every:
                    checkpoint()
                    since_checkpoint = 0
                print(f"[{len(results_by_id)}/{len(jobs)}] {key} error={str(result.get('error') or '')[:100]!r}", flush=True)
    finally:
        checkpoint()
    results = [results_by_id[key] for key in job_ids]
    traces = [traces_by_id[key] for key in job_ids]
    if expected_question_ids is not None:
        for rows in (results, traces):
            if tuple(str(row.get('manifest_question_id') or '') for row in rows) != expected_question_ids:
                raise RuntimeError('Evaluation output does not exactly match split manifest question order')
    write_json_atomic(layout.root / 'results.json', results)
    write_jsonl_atomic(layout.root / 'retrieval_trace.jsonl', traces)
    return results, traces


def finish_evaluation(results: list[dict], traces: list[dict], *, args: argparse.Namespace,
                      signature: dict[str, Any], manifest: dict[str, Any], summarize: Callable,
                      sample_id_field: str, bank_id: Callable[[dict], str]) -> dict[str, Any]:
    from benchmarks.runtime import method_metadata

    layout = OutputLayout(Path(args.result_dir))
    normalized = [{**row, '_metric_sample_id': bank_id(row)} for row in results]
    samples = sorted({row['_metric_sample_id'] for row in normalized})
    write_jsonl_atomic(layout.snapshot, load_hivemem_snapshot(args.index_root, samples))
    public_args = {key: value for key, value in vars(args).items() if key != 'answer_api_key'}
    errors = sum(bool(row.get('error')) for row in results)
    write_json_atomic(layout.root / 'run_manifest.json', {**public_args, **manifest,
                      'questions': len(results), 'answer_errors': errors, 'method_runtime': method_metadata(),
                      'graph_retrieval': True, 'graph_mode': 'append',
                      'memory_snapshot': str(layout.snapshot), 'run_signature': signature})
    if errors and not args.allow_answer_errors:
        (layout.root / 'metrics.json').unlink(missing_ok=True)
        raise RuntimeError(f'{errors}/{len(results)} answer requests failed; partial results and checkpoints were saved under {layout.root}')
    summary = summarize(results, k=args.top_k + args.append_k)
    summary['calls'] = combine_call_metrics(calculate_calls_mb(Path(args.index_root), samples),
                                            calculate_calls_qa(normalized, sample_id_field='_metric_sample_id'))
    write_json_atomic(layout.root / 'call_metrics.json', summary['calls'])
    try:
        memory = write_memory_metrics(Path(args.index_root), layout.root, tokenizer_name=args.memory_tokenizer,
                                      sample_ids=samples, cost_mb_input_price=args.cost_mb_input_price,
                                      cost_mb_output_price=args.cost_mb_output_price)
        summary = add_memory_metrics(summary, memory)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f'memory metrics unavailable: {exc}', flush=True)
        summary['cost_mb'] = calculate_cost_mb(Path(args.index_root), samples,
                                              input_price=args.cost_mb_input_price, output_price=args.cost_mb_output_price)
    try:
        retrieval_tokens = write_retrieval_memory_token(layout.root, tokenizer_name=args.retrieval_memory_tokenizer or args.answer_model)
        summary = add_retrieval_memory_tokens(summary, retrieval_tokens)
    except (OSError, KeyError, ValueError) as exc:
        print(f'retrieval memory token metrics unavailable: {exc}', flush=True)
    efficiency = write_efficiency_metrics(layout.root, normalized, sample_id_field='_metric_sample_id',
                                         sample_ids=samples, model=args.answer_model, config_path=args.efficiency_config,
                                         hivemem_index_root=Path(args.index_root))
    summary.update({key: efficiency[key] for key in ('cost_mb', 'cost_qa', 'cost_total', 'latency_mb', 'latency_qa', 'latency_total')})
    summary = merge_existing_llm_judge_metrics(summary, layout.root)
    write_json_atomic(layout.root / 'metrics.json', summary)
    print(json.dumps({'result_dir': str(layout.root), **summary}, ensure_ascii=False), flush=True)
    return summary
