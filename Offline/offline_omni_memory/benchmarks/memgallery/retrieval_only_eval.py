from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline_omni_memory.benchmarks.memgallery.adapter import MemGalleryOfflineAdapter
from offline_omni_memory.benchmarks.memgallery.run_memgallery import resolve_question_image
from offline_omni_memory.core.config import OfflineOmniConfig
from offline_omni_memory.retrieval.query_embedding_cache import QueryEmbeddingCache, make_query_id


def retrieval_hit(retrieved_ids: list[str], clue_ids: list[str], k: int = 5) -> float:
    if not clue_ids:
        return 0.0
    return float(bool(set(retrieved_ids[:k]) & set(clue_ids)))


def summarize_retrieval(rows: list[dict], k: int = 5) -> dict:
    from collections import defaultdict

    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row.get("category", "")].append(row)

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "count": len(rows),
        f"retrieval_hitrate@{k}": avg([r["_hit"] for r in rows]),
        "by_category": {},
    }
    for cat, cat_rows in sorted(by_cat.items()):
        metrics["by_category"][cat] = {
            "count": len(cat_rows),
            f"retrieval_hitrate@{k}": avg([r["_hit"] for r in cat_rows]),
        }
    return metrics


def run_dataset(
    dataset_path: Path,
    cfg: OfflineOmniConfig,
    max_qa: int,
    qa_start: int,
    qa_end: int,
    query_cache: QueryEmbeddingCache | None,
    allow_query_embed_fallback: bool,
    k: int,
) -> list[dict]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset_path.stem
    adapter = MemGalleryOfflineAdapter(cfg, dataset_name=dataset_name)
    results = []
    qa_pairs = dataset.get("human-annotated QAs", [])
    total_qa = len(qa_pairs)
    if qa_end <= 0:
        qa_end = total_qa
    processed = 0
    for idx, qa in enumerate(qa_pairs, start=1):
        if idx < qa_start or idx > qa_end:
            continue
        if max_qa and processed >= max_qa:
            break
        processed += 1
        question = qa.get("question", "")
        category = str(qa.get("point", ""))
        query_image = resolve_question_image(Path(cfg.data_dir), qa)
        query_vector = None
        if query_cache is not None:
            query_id = make_query_id(
                dataset_name=dataset_name,
                qa_index=idx,
                category=category,
                question=question,
                query_image=query_image,
            )
            query_vector = query_cache.get_by_id(query_id)
            if query_vector is None and not allow_query_embed_fallback:
                raise KeyError(f"Missing cached query embedding: {query_id}")
        query = {"text": f"[{category}] {question}", "image": query_image, "vector": query_vector}
        adapter.recall(query)
        clue = qa.get("clue", []) if isinstance(qa.get("clue", []), list) else []
        retrieved_ids = list(adapter.last_retrieved_ids)
        hit = retrieval_hit(retrieved_ids, clue, k=k)
        results.append(
            {
                "dataset": dataset_name,
                "question": question,
                "category": category,
                "retrieved_ids": retrieved_ids,
                "clue": clue,
                "_hit": hit,
            }
        )
        print(f"[{dataset_name} {idx}/{total_qa}] {category} hit={hit}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only Mem-Gallery evaluation (no answer model calls).")
    parser.add_argument("--data-name", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--data-dir", default="/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--query-embedding-dir", default="")
    parser.add_argument("--result-dir", default="artifacts/results/retrieval_only")
    parser.add_argument("--max-qa", type=int, default=0)
    parser.add_argument("--qa-start", type=int, default=1)
    parser.add_argument("--qa-end", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-query-embed-fallback", action="store_true")
    args = parser.parse_args()

    cfg = OfflineOmniConfig.from_env(data_dir=args.data_dir, index_dir=args.index_dir)

    data_dir = Path(cfg.data_dir)
    query_cache = (
        QueryEmbeddingCache(args.query_embedding_dir, expected_dim=cfg.embedding_dim)
        if args.query_embedding_dir
        else None
    )
    if args.all_datasets:
        paths = sorted((data_dir / "dialog").glob("*.json"))
    else:
        paths = [data_dir / "dialog" / f"{args.data_name}.json"]

    all_results = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        all_results.extend(
            run_dataset(
                path,
                cfg,
                max_qa=args.max_qa,
                qa_start=args.qa_start,
                qa_end=args.qa_end,
                query_cache=query_cache,
                allow_query_embed_fallback=args.allow_query_embed_fallback,
                k=args.k,
            )
        )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results_path = result_dir / "results.json"
    metrics_path = result_dir / "metrics.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = summarize_retrieval(all_results, k=args.k)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved results: {results_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
