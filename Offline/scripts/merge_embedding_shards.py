#!/usr/bin/env python3
"""Merge two embedding manifests by chunk_id: keep base vectors for chunk_ids
not present in the override dir, and take override vectors for the rest.

Used for Stage 2: reuse scheme c's original (uncapped) embeddings for the
~2959 non-image chunks (their embedding is unaffected by the vision-token cap
since they carry no image), and substitute the freshly re-embedded,
pixel-capped vectors for the ~1003 image-bearing chunks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_manifest_as_dict(embedding_dir: str) -> dict[str, np.ndarray]:
    manifest = json.loads((Path(embedding_dir) / "manifest.json").read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    for item in sorted(manifest, key=lambda x: x["worker_id"]):
        ids = json.loads(Path(item["ids"]).read_text(encoding="utf-8"))
        arr = np.load(item["vectors"])
        if len(ids) != len(arr):
            raise ValueError(f"Embedding shard mismatch: {item}")
        for chunk_id, vec in zip(ids, arr):
            out[chunk_id] = vec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge base + override embedding shards by chunk_id.")
    parser.add_argument("--base-embedding-dir", required=True)
    parser.add_argument("--override-embedding-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = load_manifest_as_dict(args.base_embedding_dir)
    override = load_manifest_as_dict(args.override_embedding_dir)

    original_base_count = len(base)
    overridden = sum(1 for chunk_id in override if chunk_id in base)
    base.update(override)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(base.keys())
    vectors = np.asarray([base[i] for i in ids], dtype=np.float32)
    np.save(out_dir / "embeddings_worker0.npy", vectors)
    (out_dir / "ids_worker0.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = [
        {
            "worker_id": 0,
            "ids": str(out_dir / "ids_worker0.json"),
            "vectors": str(out_dir / "embeddings_worker0.npy"),
            "count": len(ids),
        }
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        {
            "original_base_count": original_base_count,
            "overridden_count": overridden,
            "total": len(ids),
            "out_dir": str(out_dir.resolve()),
        }
    )


if __name__ == "__main__":
    main()
