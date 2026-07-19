#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offline_memgallery_qwen.faiss_store import FaissChunkStore
from offline_memgallery_qwen.io_utils import read_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single FAISS index from embedded chunks.")
    parser.add_argument("--chunks", default="artifacts/chunks.jsonl")
    parser.add_argument("--embedding-dir", default="artifacts/embeddings")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--dim", type=int, default=2048)
    args = parser.parse_args()

    chunks = {c.chunk_id: c for c in read_chunks_jsonl(args.chunks)}
    manifest = json.loads((Path(args.embedding_dir) / "manifest.json").read_text(encoding="utf-8"))
    ordered_chunks = []
    vectors = []
    for item in sorted(manifest, key=lambda x: x["worker_id"]):
        ids = json.loads(Path(item["ids"]).read_text(encoding="utf-8"))
        arr = np.load(item["vectors"])
        if len(ids) != len(arr):
            raise ValueError(f"Embedding shard mismatch: {item}")
        for chunk_id, vec in zip(ids, arr):
            ordered_chunks.append(chunks[chunk_id])
            vectors.append(vec)

    store = FaissChunkStore(args.index_dir, dim=args.dim)
    store.build(ordered_chunks, np.asarray(vectors, dtype=np.float32))
    print({"chunks": len(ordered_chunks), "index_dir": str(Path(args.index_dir).resolve())})


if __name__ == "__main__":
    main()
