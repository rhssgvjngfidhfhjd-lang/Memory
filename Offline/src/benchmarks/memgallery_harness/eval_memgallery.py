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
    resolve_question_image,
)
from benchmarks.memgallery_harness.retrieval.query_embedding_cache import QueryEmbeddingCache, make_query_id

from benchmarks.memgallery_harness.runner.metrics import (
    summarize_results,
    write_memory_metrics,
    write_retrieval_memory_token,
)
from typing import Any, Optional
from hive_mem.retriever import SimpleMemoryIndex


class SimpleMemoryMemGalleryAdapter:
    def __init__(
        self,
        index_root: str | Path,
        dataset_name: str,
        top_k: int = 5,
        graph_options: Optional[dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.top_k = int(top_k)
        directory = Path(index_root) / "datasets" / dataset_name
        self.graph_categories: Optional[set[str]] = None
        if graph_options is not None:
            from hive_mem.retriever import GraphExpandedIndex

            options = dict(graph_options)
            categories = options.pop("categories", None)
            if categories:
                self.graph_categories = {str(c).strip().upper() for c in categories}
            self.index: SimpleMemoryIndex = GraphExpandedIndex(directory, **options)
        else:
            self.index = SimpleMemoryIndex(directory)
        self.last_retrieved_ids: list[str] = []
        self.last_retrieved_source_groups: list[list[str]] = []
        self.last_trace: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.last_retrieved_ids = []
        self.last_retrieved_source_groups = []
        self.last_trace = []

    def recall(self, query: str | dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(query, dict) or query.get("vector") is None:
            raise ValueError("SimpleMemory adapter requires a cached query vector")
        category = str(query.get("category", ""))
        gated_off = (
            self.graph_categories is not None
            and category.upper() not in self.graph_categories
        )
        if gated_off:
            # Category gating: fall back to the plain vector search path.
            hits = SimpleMemoryIndex.search(
                self.index, query["vector"], self.top_k, category=category
            )
        else:
            hits = self.index.search(query["vector"], self.top_k, category=category)
        groups = [list(hit.item.metadata.get("source_dialogue_ids", [])) for hit in hits]
        self.last_retrieved_source_groups = groups
        self.last_retrieved_ids = list(dict.fromkeys(source for group in groups for source in group))
        self.last_trace = [
            {
                "rank": hit.rank,
                "memory_id": hit.item.memory_id,
                "score": hit.score,
                "via": hit.via,
                "content": hit.item.content,
                "source_dialogue_ids": groups[index],
                "image_ids": hit.item.metadata.get("image_ids", []),
                "image_paths": hit.item.metadata.get("image_paths", []),
            }
            for index, hit in enumerate(hits)
        ]
        append_mode = getattr(self.index, "mode", "rerank") == "append"
        items = []
        for hit in hits:
            item = hit.to_context_item()
            if append_mode and hit.via == "graph":
                item["text"] = f"(related background memory) {item['text']}"
            items.append(item)
        return items

    def store(self, observation: Any) -> None:
        return None


def run_dataset(
    dataset_path: Path,
    data_dir: Path,
    index_root: Path,
    client: VLMAnswerClient,
    query_cache: QueryEmbeddingCache,
    *,
    top_k: int = 5,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
    graph_options: dict | None = None,
    system_prompt: str | None = None,
) -> tuple[list[dict], list[dict]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset_path.stem
    profile = dataset.get("character_profile") or {}
    speaker_a = f"user ({profile.get('name')})" if profile.get("name") else "user"
    adapter = SimpleMemoryMemGalleryAdapter(
        index_root, dataset_name, top_k=top_k, graph_options=graph_options
    )
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
        memory_context, _ = build_retrieved_memory_context(memory_items, category)
        prompt = format_question_prompt(question, category, speaker_a, "assistant")
        try:
            answer = client.answer(
                system_prompt=system_prompt or SYSTEM_PROMPT,
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
                "memory_context": memory_context,
            }
        )
        print(f"[{dataset_name} {qa_index}/{len(qa_pairs)}] {category} answer={answer[:80]!r} error={error[:80]!r}")
    return results, retrieval_traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mem-Gallery with SimpleMem memories.")
    parser.add_argument("--data-name", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--data-dir", default="/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data")
    parser.add_argument(
        "--index-root",
        required=True,
        help="HiveMem run directory containing datasets/.",
    )
    parser.add_argument("--query-embedding-dir", default="data/qwen3_vl_embedding_2b/query_embeddings")
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
    profiles: dict[str, str] = {}
    if args.profiles_file:
        profiles = json.loads(Path(args.profiles_file).read_text(encoding="utf-8"))
    all_results = []
    all_traces = []
    for path in paths:
        dataset_profile = profiles.get(path.stem, "")
        dataset_system_prompt = (
            SYSTEM_PROMPT + "\n\nUser profile (background about the person the "
            "memories are about):\n" + dataset_profile
        ) if dataset_profile else None
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
            graph_options=graph_options,
            system_prompt=dataset_system_prompt,
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
    try:
        write_retrieval_memory_token(
            result_dir,
            tokenizer_name=args.retrieval_memory_tokenizer or args.answer_model,
        )
    except (OSError, KeyError, ValueError) as exc:
        print(f"retrieval memory token metrics unavailable: {exc}", flush=True)
    try:
        write_memory_metrics(
            Path(args.index_root),
            result_dir,
            tokenizer_name=args.memory_tokenizer,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # Legacy baseline traces (including A-Mem) do not always contain the
        # tokenizer metadata needed for an honest memory-token estimate.  QA
        # results are still complete and should not prevent the judge stage.
        print(f"memory metrics unavailable: {exc}", flush=True)


if __name__ == "__main__":
    main()
