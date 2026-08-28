from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from benchmarks.baseline_runtime import baseline_metadata, canonical_name, create_adapter
from benchmarks.baseline_runtime.openai_compat import embed_texts
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.protocol import (
    RetrievalRequest,
    result_context_items,
    result_trace_rows,
)
from benchmarks.io_utils import write_json_atomic, write_jsonl_atomic
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
    variant_dir = "multi-party" if variant == "multiparty" else variant
    conversation_dir = data_dir / variant_dir / conversation_id
    adapter = create_adapter(baseline, config_overrides=config)
    sample_id = f"{variant}_{conversation_id}"
    adapter.reset(sample_id, state_root / variant / conversation_id)
    try:
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

        results: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for question_file, qa_index, qa in _question_rows(conversation_dir):
            if max_qa and len(results) >= max_qa:
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
            started = time.time()
            try:
                response = client.answer_with_usage(
                    system_prompt=SYSTEM_PROMPT,
                    memory_items=memory_items,
                    question_prompt=_question_prompt(question, category),
                    query_image=query_image,
                    # H2HMem is multimodal; attach any retrieved memory images.
                    category="VR",
                )
                answer, error = response.text, ""
                usage = response.usage
                attempts = response.attempts
            except Exception as exc:
                answer, error, usage = "", str(exc), None
                attempts = client.retries + 1
            result = {
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
                "system_answer": answer,
                "original_answer": qa.get("original_answer", ""),
                "answer_session": qa.get("answer_session") or [],
                "retrieved_ids": [row["memory_id"] for row in trace_rows],
                "retrieved_source_groups": [
                    row["source_dialogue_ids"] for row in trace_rows
                ],
                "error": error,
                "answer_seconds": time.time() - started,
                "answer_token_usage": usage,
                "answer_attempts": attempts,
            }
            results.append(result)
            traces.append(
                {
                    "query_id": query_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "question": question,
                    "category": category,
                    "top_k": trace_rows,
                }
            )
            print(
                f"[{variant}/{conversation_id}/{session_id}/{qa_index}] "
                f"answer={answer[:80]!r} error={error[:80]!r}",
                flush=True,
            )
        snapshots = [row.to_dict() for row in adapter.snapshot()]
        return results, traces, snapshots
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a memory baseline on H2HMem.")
    parser.add_argument("--baseline", default="HiveMem")
    parser.add_argument("--data-dir", default=str(DEFAULT_H2HMEM_DATA_DIR))
    parser.add_argument("--variant", choices=("dyadic", "multiparty", "all"), default="all")
    parser.add_argument("--conversation-id", action="append", default=[])
    parser.add_argument("--index-root", default="")
    parser.add_argument("--baseline-state-dir", default="")
    parser.add_argument("--result-dir", required=True)
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
        },
    )
    args = parser.parse_args()
    args.baseline = canonical_name(args.baseline)
    if args.baseline == "HiveMem" and not args.index_root:
        parser.error("--index-root is required when --baseline=HiveMem")

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
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_temperature": args.executor_temperature,
        "executor_visual_input": args.executor_visual_input,
        "request_timeout": args.request_timeout,
        "retries": args.retries,
    }
    results: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for variant in variants:
        for conversation_id in conversations[variant]:
            remaining = max(args.max_qa - len(results), 0) if args.max_qa else 0
            if args.max_qa and remaining == 0:
                break
            new_results, new_traces, new_snapshots = run_conversation(
                data_dir=data_dir,
                variant=variant,
                conversation_id=conversation_id,
                baseline=args.baseline,
                state_root=state_root,
                client=client,
                config=config,
                max_qa=remaining,
            )
            results.extend(new_results)
            traces.extend(new_traces)
            snapshots.extend(new_snapshots)

    result_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_dir / "results.json", results)
    write_jsonl_atomic(result_dir / "retrieval_trace.jsonl", traces)
    if snapshots:
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
            "configuration": config,
            "memory_snapshot": str(layout.snapshot) if snapshots else "",
        },
    )


if __name__ == "__main__":
    main()
