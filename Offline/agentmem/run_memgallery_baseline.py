from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from memgallery_harness.runner.ollama_client import OllamaVLMClient
from memgallery_harness.runner.run_memgallery import (
    SYSTEM_PROMPT,
    format_question_prompt,
    resolve_question_image,
)
from memgallery_harness.retrieval.query_embedding_cache import QueryEmbeddingCache, make_query_id

from .memgallery.adapter import SimpleMemoryMemGalleryAdapter
from .memgallery.metrics import summarize_results


def run_dataset(
    dataset_path: Path,
    data_dir: Path,
    index_root: Path,
    client: OllamaVLMClient,
    query_cache: QueryEmbeddingCache,
    *,
    top_k: int = 5,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
) -> tuple[list[dict], list[dict]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset_path.stem
    profile = dataset.get("character_profile") or {}
    speaker_a = f"user ({profile.get('name')})" if profile.get("name") else "user"
    adapter = SimpleMemoryMemGalleryAdapter(index_root, dataset_name, top_k=top_k)
    qa_pairs = dataset.get("human-annotated QAs", [])
    qa_end = qa_end or len(qa_pairs)
    results = []
    retrieval_traces = []
    processed = 0
    for qa_index, qa in enumerate(qa_pairs, start=1):
        if qa_index < qa_start or qa_index > qa_end:
            continue
        if max_qa and processed >= max_qa:
            break
        processed += 1
        question = str(qa.get("question", ""))
        category = str(qa.get("point", ""))
        query_image = resolve_question_image(data_dir, qa)
        query_id = make_query_id(
            dataset_name=dataset_name,
            qa_index=qa_index,
            category=category,
            question=question,
            query_image=query_image,
        )
        query_vector = query_cache.get_by_id(query_id)
        if query_vector is None:
            raise KeyError(f"Missing cached query embedding: {query_id}")
        memory_items = adapter.recall(
            {"text": f"[{category}] {question}", "category": category, "vector": query_vector}
        )
        prompt = format_question_prompt(question, category, speaker_a, "assistant")
        try:
            answer = client.answer(
                system_prompt=SYSTEM_PROMPT,
                memory_items=memory_items,
                question_prompt=prompt,
                query_image=query_image,
                category=category,
            )
            error = ""
        except Exception as exc:
            answer = ""
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
            "retrieved_ids": adapter.last_retrieved_ids,
            "retrieved_source_groups": adapter.last_retrieved_source_groups,
            "clue": clue,
            "error": error,
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
                "top_k": adapter.last_trace,
            }
        )
        print(f"[{dataset_name} {qa_index}/{len(qa_pairs)}] {category} answer={answer[:80]!r} error={error[:80]!r}")
    return results, retrieval_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mem-Gallery with SimpleMem memories.")
    parser.add_argument("--data-name", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--data-dir", default="/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--query-embedding-dir", default="artifacts/query_embeddings")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--qa-start", type=int, default=1)
    parser.add_argument("--qa-end", type=int, default=0)
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default="EMPTY")
    parser.add_argument("--num-predict", type=int, default=8000)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    client = OllamaVLMClient(
        model=args.answer_model,
        base_url=args.answer_base_url,
        api_key=args.answer_api_key,
        num_predict=args.num_predict,
        timeout=args.request_timeout,
        retries=args.retries,
        think=args.think,
        backend="openai",
    )
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(args.query_embedding_dir, expected_dim=args.embedding_dim)
    data_dir = Path(args.data_dir)
    paths = sorted((data_dir / "dialog").glob("*.json")) if args.all_datasets else [data_dir / "dialog" / f"{args.data_name}.json"]
    all_results = []
    all_traces = []
    for path in paths:
        results, traces = run_dataset(
            path,
            data_dir,
            Path(args.index_root),
            client,
            query_cache,
            top_k=args.top_k,
            max_qa=args.max_qa,
            qa_start=args.qa_start,
            qa_end=args.qa_end,
        )
        all_results.extend(results)
        all_traces.extend(traces)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(
        json.dumps(summarize_results(all_results, k=args.top_k), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (result_dir / "retrieval_trace.jsonl").open("w", encoding="utf-8") as handle:
        for trace in all_traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    (result_dir / "run_manifest.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
