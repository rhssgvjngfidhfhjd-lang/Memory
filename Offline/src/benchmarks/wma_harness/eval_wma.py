from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from benchmarks.wma_harness.runner.metrics import summarize_results
from benchmarks.wma_harness.runner.prompts import SYSTEM_PROMPT, format_question_prompt
from embedding.chunk_builder import iter_wma_sample_files
from hive_mem.retriever import SimpleMemoryIndex


VISUAL_CATEGORIES = {"VFR", "VS", "VU", "CMR", ""}
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WMA_DATA_DIR = Path(
    os.getenv(
        "WMA_DATA_DIR",
        PROJECT_ROOT / "WorldMemArena" / "WorldMemArena",
    )
)


class WMAIndexAdapter:
    def __init__(
        self,
        index_root: str | Path,
        sample_id: str,
        *,
        top_k: int = 5,
        graph_options: dict[str, Any] | None = None,
    ):
        self.top_k = int(top_k)
        if graph_options is not None:
            raise ValueError(
                "Graph retrieval is disabled for WMA checkpoints: the current graph "
                "statistics are built from the full memory bank and are not prefix-safe."
            )
        directory = Path(index_root) / "datasets" / sample_id
        self.index = SimpleMemoryIndex(
            directory, visual_categories=VISUAL_CATEGORIES
        )

    def recall(
        self,
        query_vector: list[float],
        *,
        category: str,
        visible_sessions: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        hits = self.index.search(
            query_vector,
            self.top_k,
            category=category,
            allowed_session_ids=set(visible_sessions),
        )
        items = [hit.to_context_item() for hit in hits]
        trace = [
            {
                "rank": hit.rank,
                "memory_id": hit.item.id,
                "score": hit.score,
                "via": hit.via,
                "content": hit.item.content,
                "session_id": hit.item.metadata.get("session_id", ""),
                "source_dialogue_ids": hit.item.metadata.get("source_dialogue_ids", []),
                "image_ids": hit.item.metadata.get("image_ids", []),
                "image_paths": hit.item.metadata.get("image_paths", []),
            }
            for hit in hits
        ]
        return items, trace


def prepare_sample_jobs(
    sample_path: Path,
    index_root: Path,
    query_cache: QueryEmbeddingCache,
    *,
    top_k: int,
    graph_options: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    sample_id = str(payload["sample_id"])
    adapter = WMAIndexAdapter(
        index_root, sample_id, top_k=top_k, graph_options=graph_options
    )
    ordered_sessions = session_ids(payload)
    gold_points = build_gold_evidence_map(payload)
    jobs: list[dict[str, Any]] = []
    for checkpoint in payload.get("qa_checkpoints", []) or []:
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        covered_sessions = [str(value) for value in checkpoint.get("covered_sessions", [])]
        visible_sessions = visible_sessions_for_checkpoint(
            ordered_sessions, covered_sessions
        )
        visible_session_set = set(visible_sessions)
        for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
            category = str(qa.get("question_type_abbrev", ""))
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
            memory_items, trace = adapter.recall(
                vector, category=category, visible_sessions=visible_sessions
            )
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
    return jobs


def answer_job(client: VLMAnswerClient, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    try:
        answer = client.answer(
            system_prompt=SYSTEM_PROMPT,
            memory_items=job["memory_items"],
            question_prompt=format_question_prompt(job["question"], job["category"]),
            category=job["category"],
        )
        error = ""
    except Exception as exc:
        answer, error = "", str(exc)
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
            "answer_seconds": time.time() - started,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    trace = {
        "query_id": job["query_id"],
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
        "answer_token_usage": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HiveMem on WorldMemArena.")
    parser.add_argument("--data-dir", default=str(DEFAULT_WMA_DATA_DIR))
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--query-embedding-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--answer-concurrency", type=int, default=16)
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--answer-api-key", default="EMPTY")
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-retrieval", action="store_true")
    parser.add_argument("--seed-k", type=int, default=0)
    parser.add_argument("--expansion-bonus", type=float, default=0.2)
    parser.add_argument("--graph-mode", choices=("rerank", "append"), default="rerank")
    parser.add_argument("--append-k", type=int, default=2)
    args = parser.parse_args()
    if args.answer_concurrency < 1:
        raise ValueError("--answer-concurrency must be at least 1")

    data_dir = Path(args.data_dir)
    selected = set(args.sample_id)
    if args.small:
        selected.update(json.loads((data_dir / "small_ids.json").read_text(encoding="utf-8")))
    paths = [
        path for path in iter_wma_sample_files(data_dir)
        if not selected or path.stem in selected
    ]
    client = VLMAnswerClient(
        model=args.answer_model, base_url=args.answer_base_url,
        api_key=args.answer_api_key, num_predict=args.num_predict,
        timeout=args.request_timeout, retries=args.retries, think=args.think,
    )
    if not args.skip_model_check:
        client.assert_model_available()
    cache = QueryEmbeddingCache(args.query_embedding_dir, expected_dim=args.embedding_dim)
    graph_options = (
        {
            "seed_k": args.seed_k,
            "expansion_bonus": args.expansion_bonus,
            "mode": args.graph_mode,
            "append_k": args.append_k,
        }
        if args.graph_retrieval else None
    )
    jobs: list[dict[str, Any]] = []
    for path in paths:
        jobs.extend(
            prepare_sample_jobs(
                path, Path(args.index_root), cache,
                top_k=args.top_k, graph_options=graph_options,
            )
        )
        if args.max_qa and len(jobs) >= args.max_qa:
            jobs = jobs[: args.max_qa]
            break

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "results.json"
    existing = json.loads(result_path.read_text(encoding="utf-8")) if args.resume and result_path.exists() else []
    completed = {str(row.get("query_id", "")) for row in existing}
    pending = [job for job in jobs if job["query_id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.answer_concurrency) as pool:
        answered = list(pool.map(lambda job: answer_job(client, job), pending))
    results = existing + [row[0] for row in answered]
    traces = [row[1] for row in answered]
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (result_dir / "metrics.json").write_text(
        json.dumps(summarize_results(results, k=args.top_k), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace_mode = "a" if args.resume else "w"
    with (result_dir / "retrieval_trace.jsonl").open(trace_mode, encoding="utf-8") as handle:
        for row in traces:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    trace_by_id = {row["query_id"]: row for row in traces}
    pipeline_path = result_dir / "pipeline_qa.jsonl"
    pipeline_mode = "a" if args.resume and pipeline_path.exists() else "w"
    with pipeline_path.open(pipeline_mode, encoding="utf-8") as handle:
        for result in (row[0] for row in answered):
            handle.write(
                json.dumps(
                    to_pipeline_qa_record(result, trace_by_id[result["query_id"]]),
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest = vars(args) | {"samples": len(paths), "questions": len(jobs), "completed": len(results)}
    (result_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"result_dir": str(result_dir), **summarize_results(results, args.top_k)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
