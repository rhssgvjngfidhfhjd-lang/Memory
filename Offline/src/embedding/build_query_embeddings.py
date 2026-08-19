#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np

from .qwen3_text_embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    create_embedding_service,
    is_qwen3_text_embedding_model,
)
from benchmarks.memgallery_harness.retrieval.query_embedding_cache import make_query_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMGALLERY_DATA_DIR = PROJECT_ROOT.parent / "Mem-Gallery" / "benchmark" / "data"


def resolve_question_image(data_dir: Path, qa: dict[str, Any]) -> dict[str, str] | None:
    raw = qa.get("question_image")
    if not raw:
        return None
    raw = str(raw)
    if os.path.isabs(raw):
        path = raw
    elif raw.startswith("../image/"):
        path = str((data_dir / "image" / raw.replace("../image/", "")).resolve())
    else:
        path = str((data_dir / "image" / raw).resolve())
    out = {"path": path}
    if qa.get("image_caption"):
        out["caption"] = str(qa["image_caption"])
    return out


def iter_qa_items(data_dir: Path, data_name: str = "", all_datasets: bool = True) -> list[dict[str, Any]]:
    if data_name:
        paths = [data_dir / "dialog" / f"{data_name}.json"]
    elif all_datasets:
        paths = sorted((data_dir / "dialog").glob("*.json"))
    else:
        raise ValueError("Either --data-name or --all-datasets must be set")

    items: list[dict[str, Any]] = []
    for path in paths:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        dataset_name = path.stem
        qa_pairs = dataset.get("human-annotated QAs", []) or []
        for qa_index, qa in enumerate(qa_pairs, start=1):
            category = str(qa.get("point", ""))
            question = str(qa.get("question", ""))
            query_image = resolve_question_image(data_dir, qa)
            query_id = make_query_id(
                dataset_name=dataset_name,
                qa_index=qa_index,
                category=category,
                question=question,
                query_image=query_image,
            )
            items.append(
                {
                    "query_id": query_id,
                    "dataset": dataset_name,
                    "qa_index": qa_index,
                    "category": category,
                    "question": question,
                    "query_image": query_image,
                    "answer": qa.get("answer", ""),
                    "clue": qa.get("clue", []) if isinstance(qa.get("clue", []), list) else [],
                }
            )
    return items


def _split_evenly(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    return [items[i::n] for i in range(n)]


def _prepare_text_query(item: dict[str, Any]) -> str:
    question = str(item["question"])
    image = item.get("query_image")
    caption = str(image.get("caption", "")).strip() if isinstance(image, dict) else ""
    if caption:
        return f"{question}\nImage caption: {caption}"
    return question


def _worker(payload):
    worker_id, device, items, args_dict = payload
    if str(device).lower().isdigit():
        device_name = f"cuda:{device}"
    else:
        device_name = str(device)
    embedder = create_embedding_service(
        model_name=args_dict["model_name"],
        device=device_name,
        expected_dim=args_dict["dim"],
        dtype=args_dict["dtype"],
        local_files_only=args_dict["local_files_only"],
        batch_size=args_dict["batch_size"],
    )
    if embedder.supports_images:
        vectors = []
        for item in items:
            image = item.get("query_image")
            images = [image["path"]] if isinstance(image, dict) and image.get("path") else []
            vectors.append(embedder.embed_query(str(item["question"]), images))
        matrix = np.asarray(vectors, dtype=np.float32)
    else:
        matrix = embedder.embed_queries([_prepare_text_query(item) for item in items])
    return worker_id, items, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute benchmark QA query embeddings.")
    parser.add_argument("--benchmark", choices=("memgallery", "wma"), default="memgallery")
    parser.add_argument("--data-dir", default=str(DEFAULT_MEMGALLERY_DATA_DIR))
    parser.add_argument("--data-name", default="")
    parser.add_argument("--all-datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", default="data/qwen3_vl_embedding_2b/query_embeddings")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--devices", default="0,1,2,3", help="Comma-separated CUDA ids or device names.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.benchmark == "wma":
        from benchmarks.wma_harness.retrieval.query_embedding_cache import iter_qa_items as iter_wma_qa_items

        sample_ids = {args.data_name} if args.data_name else None
        items = iter_wma_qa_items(data_dir, sample_ids=sample_ids)
    else:
        items = iter_qa_items(data_dir, data_name=args.data_name, all_datasets=args.all_datasets)
    if args.limit:
        items = items[: args.limit]

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if not devices:
        devices = ["cpu"]
    if args.workers:
        devices = devices[: args.workers]

    shards = _split_evenly(items, len(devices))
    args_dict = {
        "model_name": args.model_name,
        "dim": args.dim,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "batch_size": args.batch_size,
    }
    payloads = [(i, devices[i], shards[i], args_dict) for i in range(len(devices)) if shards[i]]

    if len(payloads) == 1:
        results = [_worker(payloads[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            results = pool.map(_worker, payloads)

    ordered_rows: list[dict[str, Any]] = []
    vectors = []
    for _worker_id, rows, arr in sorted(results, key=lambda x: x[0]):
        ordered_rows.extend(rows)
        vectors.append(arr)
    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, args.dim), dtype=np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", matrix)
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "count": len(ordered_rows),
        "dim": args.dim,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "data_dir": str(data_dir.resolve()),
        "benchmark": args.benchmark,
        "vectors": str((out_dir / "vectors.npy").resolve()),
        "metadata": str((out_dir / "metadata.jsonl").resolve()),
    }
    if is_qwen3_text_embedding_model(args.model_name):
        manifest.update(
            {
                "modality": "text",
                "query_instruction": DEFAULT_QUERY_INSTRUCTION,
                "query_image_policy": "append image_caption when available; raw image is not encoded",
            }
        )
    else:
        manifest.update({"modality": "vision-language", "query_image_policy": "encode raw image"})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
