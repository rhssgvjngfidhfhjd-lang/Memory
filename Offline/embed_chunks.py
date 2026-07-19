#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from offline_memgallery_qwen.io_utils import read_chunks_jsonl
from offline_memgallery_qwen.qwen3vl_embedding import Qwen3VLEmbeddingService


def _worker(payload):
    worker_id, device, chunk_dicts, args_dict = payload
    if device.lower().isdigit():
        device_name = f"cuda:{device}"
    else:
        device_name = device
    embedder = Qwen3VLEmbeddingService(
        model_name=args_dict["model_name"],
        device=device_name,
        expected_dim=args_dict["dim"],
        dtype=args_dict["dtype"],
        local_files_only=args_dict["local_files_only"],
        max_pixels=args_dict.get("max_pixels") or None,
        min_pixels=args_dict.get("min_pixels") or None,
    )
    vectors = []
    ids = []
    for item in chunk_dicts:
        images = [] if args_dict["no_images"] else item.get("images") or []
        vector = embedder.embed_chunk(item["text"], images)
        vectors.append(vector)
        ids.append(item["chunk_id"])
    return worker_id, ids, np.asarray(vectors, dtype=np.float32)


def _split_evenly(items, n):
    return [items[i::n] for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Mem-Gallery chunks with Qwen3-VL.")
    parser.add_argument("--chunks", default="artifacts/chunks.jsonl")
    parser.add_argument("--out-dir", default="artifacts/embeddings")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA ids or device names.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=0, help="Cap total pixels per image before embedding (0 = no cap).")
    parser.add_argument("--min-pixels", type=int, default=0, help="Floor total pixels per image before embedding (0 = model default).")
    args = parser.parse_args()

    chunks = read_chunks_jsonl(args.chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = ["cpu"]
    if args.workers:
        devices = devices[: args.workers]

    chunk_dicts = [c.to_dict() for c in chunks]
    shards = _split_evenly(chunk_dicts, len(devices))
    args_dict = {
        "model_name": args.model_name,
        "dim": args.dim,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "no_images": args.no_images,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
    }
    payloads = [(i, devices[i], shards[i], args_dict) for i in range(len(devices)) if shards[i]]

    if len(payloads) == 1:
        results = [_worker(payloads[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            results = pool.map(_worker, payloads)

    manifest = []
    for worker_id, ids, vectors in sorted(results, key=lambda x: x[0]):
        npy_path = out_dir / f"embeddings_worker{worker_id}.npy"
        ids_path = out_dir / f"ids_worker{worker_id}.json"
        np.save(npy_path, vectors)
        ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"worker_id": worker_id, "ids": str(ids_path), "vectors": str(npy_path), "count": len(ids)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"chunks": len(chunks), "workers": len(payloads), "out_dir": str(out_dir.resolve())})


if __name__ == "__main__":
    main()
