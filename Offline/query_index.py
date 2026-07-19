#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from offline_memgallery_qwen.faiss_store import FaissChunkStore
from offline_memgallery_qwen.qwen3vl_embedding import Qwen3VLEmbeddingService


def _category_from_query(query: str) -> tuple[str, str]:
    m = re.match(r"^\[([A-Z]+)\]\s*", query)
    if not m:
        return "", query
    return m.group(1), query[m.end():]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a Mem-Gallery Qwen3-VL FAISS index.")
    parser.add_argument("query")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    category, clean_query = _category_from_query(args.query)
    embedder = Qwen3VLEmbeddingService(
        model_name=args.model_name,
        device=args.device,
        expected_dim=args.dim,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    vector = embedder.embed_query(clean_query)
    store = FaissChunkStore(args.index_dir, dim=args.dim)
    hits = store.search(vector, top_k=args.top_k, category=category, query_text=clean_query)

    for hit in hits:
        md = hit.chunk.metadata
        header = (
            f"[{hit.rank}] score={hit.score:.4f} "
            f"chunk={hit.chunk.chunk_id} session={md.get('session_id')} "
            f"round={md.get('dialogue_id')} image={md.get('image_id', '')}"
        )
        print(header)
        print(hit.chunk.text[:1200])
        print()


if __name__ == "__main__":
    main()
