from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
)
from benchmarks.baseline_runtime.openai_compat import embed_texts
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.protocol import (
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import file_manifest, write_json_atomic, write_jsonl_atomic
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from embedding.chunk_builder import (
    build_h2h_chunks_from_directory,
    iter_h2h_session_files,
)
from hive_mem.build_memories import apply_config_defaults


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_H2HMEM_DATA_DIR = WORKSPACE_ROOT / "H2HMEM-main" / "dataset"
SYSTEM_PROMPT = """You answer questions using only the supplied retrieved memories.
Be concise and return only the answer. H2HMem records human-to-human conversations;
track speakers, dates, updates, and visual evidence carefully. If the evidence does
not support an answer, say that it is unknown instead of guessing."""


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


def _question_prompt(text: str, category: str) -> str:
    return (
        "Answer the following H2HMem question from the retrieved conversation "
        "memories. Return only the answer, without an 'Answer:' prefix.\n\n"
        f"Question type: {category}\nQuestion: {text}"
    )


def prepare_conversation_jobs(
    *,
    data_dir: Path,
    variant: str,
    conversation_id: str,
    baseline: str,
    state_root: Path,
    config: dict[str, Any],
    max_qa: int = 0,
) -> dict[str, Any]:
    variant_dir = "multi-party" if variant == "multiparty" else variant
    conversation_dir = data_dir / variant_dir / conversation_id
    adapter = create_adapter(baseline, config_overrides=config)
    sample_id = f"{variant}_{conversation_id}"
    try:
        adapter.reset(sample_id, state_root / variant / conversation_id)
        if baseline != "HiveMem":
            chunks = build_h2h_chunks_from_directory(
                data_dir,
                variant=variant,
                conversation_ids={conversation_id},
            )
            current_session = ""
            for chunk in chunks:
                session_id = str(chunk.metadata.get("session_id") or "")
                if current_session and session_id != current_session:
                    adapter.end_session(current_session)
                adapter.ingest(chunk)
                current_session = session_id
            if current_session:
                adapter.end_session(current_session)

        jobs: list[dict[str, Any]] = []
        for question_file, qa_index, qa in _question_rows(conversation_dir):
            if max_qa and len(jobs) >= max_qa:
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
            query_vector = (
                embed_texts([question], config)[0] if baseline == "HiveMem" else None
            )
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
            memory_items = result_context_items(retrieval)
            trace_rows = result_trace_rows(retrieval)
            query_image = _question_image(question_file, question_data.get("image"))
            jobs.append({
                "uid": query_id,
                "query_id": query_id,
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
                "question_prompt": _question_prompt(question, category),
                "query_image_payload": query_image,
                "memory_items": memory_items,
                "retrieval_top_k": trace_rows,
            })
        snapshots = [row.to_dict() for row in adapter.snapshot()]
        return {
            "sample_id": sample_id,
            "variant": variant,
            "conversation_id": conversation_id,
            "jobs": jobs,
            "snapshots": snapshots,
        }
    finally:
        adapter.close()


def answer_conversation_job(
    client: VLMAnswerClient,
    job: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    try:
        response = client.answer_with_usage(
            system_prompt=SYSTEM_PROMPT,
            memory_items=job["memory_items"],
            question_prompt=job["question_prompt"],
            query_image=job.get("query_image_payload"),
            category="VR",
        )
        answer, error = response.text, ""
        usage, attempts = response.usage, response.attempts
    except Exception as exc:
        answer, error, usage = "", str(exc), None
        attempts = client.retries + 1
    result = {
        key: value
        for key, value in job.items()
        if key not in {
            "question_prompt", "query_image_payload", "memory_items", "retrieval_top_k"
        }
    }
    result.update(
        {
            "system_answer": answer,
            "retrieved_ids": [row["memory_id"] for row in job["retrieval_top_k"]],
            "retrieved_source_groups": [
                row["source_dialogue_ids"] for row in job["retrieval_top_k"]
            ],
            "error": error,
            "answer_seconds": time.time() - started,
            "answer_token_usage": usage,
            "answer_attempts": attempts,
        }
    )
    trace = {
        "query_id": job["query_id"],
        "conversation_id": job["conversation_id"],
        "session_id": job["session_id"],
        "question": job["question"],
        "category": job["category"],
        "top_k": job["retrieval_top_k"],
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
    )
    pairs = [answer_conversation_job(client, job) for job in artifact["jobs"]]
    return [x[0] for x in pairs], [x[1] for x in pairs], artifact["snapshots"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory baseline on H2HMem.")
    parser.add_argument("--baseline", default="HiveMem")
    parser.add_argument("--data-dir", default=str(DEFAULT_H2HMEM_DATA_DIR))
    parser.add_argument("--variant", choices=("dyadic", "multiparty", "all"), default="all")
    parser.add_argument("--conversation-id", action="append", default=[])
    parser.add_argument("--index-root", default="")
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-concurrency", type=int, default=4)
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:18000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default="EMPTY")
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--executor-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--executor-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--executor-temperature", type=float, default=0.0)
    parser.add_argument("--executor-visual-input", choices=("image", "caption"), default="image")
    parser.add_argument("--skip-model-check", action="store_true")
    apply_config_defaults(
        parser,
        allowed_keys={
            "answer_base_url", "answer_model", "answer_api_key",
            "answer_temperature", "num_predict", "request_timeout", "retries",
            "think", "top_k", "embedding_dim", "embedding_model",
            "embedding_base_url", "executor_model", "executor_base_url",
            "executor_temperature", "executor_visual_input",
            "sample_concurrency", "answer_concurrency", "checkpoint_every",
        },
    )
    args = parser.parse_args()
    args.baseline = canonical_name(args.baseline)
    if args.baseline == "HiveMem" and not args.index_root:
        parser.error("--index-root is required when --baseline=HiveMem")
    if args.sample_concurrency < 1 or args.answer_concurrency < 1 or args.checkpoint_every < 1:
        parser.error("Sample/answer concurrency and checkpoint interval must be positive")

    data_dir = Path(args.data_dir)
    if (data_dir / "dataset").is_dir():
        data_dir = data_dir / "dataset"
    variants = ("dyadic", "multiparty") if args.variant == "all" else (args.variant,)
    selected = set(args.conversation_id)
    conversations: dict[str, list[str]] = {}
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
    )
    if not args.skip_model_check:
        client.assert_model_available()

    result_dir = Path(args.result_dir)
    layout = BaselineOutputLayout(result_dir)
    state_root = layout.state_root(args.baseline_state_dir)
    config = {
        "top_k": args.top_k,
        "index_root": args.index_root,
        "embedding_dim": args.embedding_dim,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "answer_temperature": args.answer_temperature,
        "num_predict": args.num_predict,
        "think": args.think,
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_temperature": args.executor_temperature,
        "executor_visual_input": args.executor_visual_input,
        "request_timeout": args.request_timeout,
        "retries": args.retries,
    }
    sample_specs: list[tuple[str, str, int]] = []
    remaining_limit = args.max_qa
    for variant in variants:
        for conversation_id in conversations[variant]:
            if args.max_qa and remaining_limit <= 0:
                break
            conversation_dir = data_dir / (
                "multi-party" if variant == "multiparty" else variant
            ) / conversation_id
            question_count = len(_question_rows(conversation_dir))
            quota = min(remaining_limit, question_count) if args.max_qa else 0
            sample_specs.append((variant, conversation_id, quota))
            if args.max_qa:
                remaining_limit -= quota

    signature = {
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {
                "answer_api_key", "sample_concurrency", "answer_concurrency",
                "checkpoint_every", "resume", "skip_model_check", "result_dir",
            }
        },
        "inputs": file_manifest(
            path
            for variant, conversation_id, _ in sample_specs
            for path in sorted(
                (
                    data_dir
                    / ("multi-party" if variant == "multiparty" else variant)
                    / conversation_id
                ).rglob("*.json")
            )
        ),
        "system_prompt": SYSTEM_PROMPT,
    }
    sample_signature = signature_digest(signature)

    def prepare(spec: tuple[str, str, int]) -> dict[str, Any]:
        variant, conversation_id, quota = spec
        sample_id = f"{variant}/{conversation_id}"
        if args.resume:
            cached = load_sample_artifact(
                layout.sample_checkpoint_dir,
                sample_id,
                signature=sample_signature,
            )
            if cached is not None:
                print(f"[resume] skip prepared conversation: {sample_id}", flush=True)
                return cached
        artifact = prepare_conversation_jobs(
                data_dir=data_dir,
                variant=variant,
                conversation_id=conversation_id,
                baseline=args.baseline,
                state_root=state_root,
                config=config,
                max_qa=quota,
            )
        save_sample_artifact(
            layout.sample_checkpoint_dir,
            sample_id,
            signature=sample_signature,
            artifact=artifact,
        )
        print(f"[prepared] {sample_id}: {len(artifact['jobs'])} question(s)", flush=True)
        return artifact

    artifacts = parallel_map_ordered(
        sample_specs,
        prepare,
        max_workers=args.sample_concurrency,
        item_key=lambda spec: f"{spec[0]}/{spec[1]}",
    )
    jobs = [job for artifact in artifacts for job in artifact["jobs"]]
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
        if not row.get("error") and query_id in traces_by_id
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

    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", traces)
    write_jsonl_atomic(layout.snapshot, snapshots)
    for variant in variants:
        filename = "prediction_multi_party.json" if variant == "multiparty" else "prediction_dyadic.json"
        write_json_atomic(
            result_dir / filename,
            {"predictions": [row for row in results if row["variant"] == variant]},
        )
    write_json_atomic(
        result_dir / "run_manifest.json",
        {
            "benchmark": "H2HMEM",
            "baseline": baseline_metadata(args.baseline),
            "variants": list(variants),
            "conversations": conversations,
            "questions": len(results),
            "configuration": {
                **config,
                "sample_concurrency": args.sample_concurrency,
                "answer_concurrency": args.answer_concurrency,
                "checkpoint_every": args.checkpoint_every,
            },
            "memory_snapshot": str(layout.snapshot),
        },
    )


if __name__ == "__main__":
    main()
