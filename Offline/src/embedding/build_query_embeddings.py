#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

from .qwen3_text_embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    create_embedding_service,
    is_qwen3_text_embedding_model,
)
from benchmarks.memgallery_harness.retrieval.query_embedding_cache import make_query_id
from benchmarks.memgallery_harness.runner.prompts import resolve_question_image
from benchmarks.question_filter import is_excluded_category, parse_excluded_categories
from evidence_policy.episode_sources import iter_source_questions
from evidence_policy.split_manifest import SplitManifestIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMGALLERY_DATA_DIR = PROJECT_ROOT.parent / "Mem-Gallery" / "benchmark" / "data"


def iter_qa_items(
    data_dir: Path,
    data_name: str = "",
    all_datasets: bool = True,
    excluded_categories: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
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
            if is_excluded_category(category, excluded_categories):
                continue
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


def iter_h2hmem_qa_items(
    manifest_path: str | Path,
    *,
    variant: str = "all",
) -> list[dict[str, Any]]:
    """Return manifest-selected H2HMem questions with collision-free IDs."""
    from benchmarks.h2hmem_harness.eval_h2hmem import _question_image

    if variant not in {"all", "dyadic", "multiparty"}:
        raise ValueError(f"Unknown H2HMem variant: {variant!r}")
    sources = (
        ("h2hmem_dyadic", "h2hmem_multiparty")
        if variant == "all"
        else (f"h2hmem_{variant}",)
    )
    manifest = SplitManifestIndex(manifest_path)
    rows = iter_source_questions(
        manifest,
        PROJECT_ROOT.parent,
        data_sources=sources,
    )
    items: list[dict[str, Any]] = []
    for row in rows:
        raw_image = str(row.metadata.get("question_image", ""))
        query_image = (
            _question_image(Path(row.source_path), raw_image) if raw_image else None
        )
        items.append(
            {
                "query_id": row.question_id,
                "dataset": f"{row.metadata['variant']}_{row.source_id}",
                "qa_index": row.question_index + 1,
                "category": row.category,
                "question": row.question,
                "query_image": query_image,
                "answer": row.answer,
                "clue": list(row.metadata.get("answer_session", [])),
                "manifest_question_id": row.question_id,
                "split": row.split,
                "variant": row.metadata["variant"],
            }
        )
    return items


def _split_evenly(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    return [items[i::n] for i in range(n)]


def _resolve_devices(
    specification: str,
    *,
    workers: int = 0,
    cuda_device_count: int | None = None,
) -> list[str]:
    """Resolve ``auto`` or an explicit device list and reject unavailable GPUs."""
    if cuda_device_count is None:
        try:
            import torch

            cuda_device_count = torch.cuda.device_count()
        except (ImportError, RuntimeError):
            cuda_device_count = 0

    if specification.strip().lower() == "auto":
        devices = [f"cuda:{index}" for index in range(cuda_device_count)] or ["cpu"]
    else:
        devices = [value.strip() for value in specification.split(",") if value.strip()]
        devices = [f"cuda:{value}" if value.isdigit() else value for value in devices]
        if not devices:
            devices = ["cpu"]

    for device in devices:
        if device.startswith("cuda:"):
            try:
                index = int(device.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device: {device}") from exc
            if index < 0 or index >= cuda_device_count:
                raise ValueError(
                    f"Requested {device}, but only {cuda_device_count} CUDA device(s) are visible"
                )
        elif device == "cuda" and cuda_device_count < 1:
            raise ValueError("Requested CUDA, but no CUDA devices are visible")

    return devices[:workers] if workers else devices


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
    parser.add_argument(
        "--benchmark", choices=("memgallery", "wma", "h2hmem"), default="memgallery"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_MEMGALLERY_DATA_DIR))
    parser.add_argument("--data-name", default="")
    parser.add_argument("--all-datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Required for H2HMem so query IDs exactly match the PPO split manifest.",
    )
    parser.add_argument(
        "--h2h-variant",
        choices=("all", "dyadic", "multiparty"),
        default="all",
    )
    parser.add_argument("--out-dir", default="data/qwen3_vl_embedding_2b/query_embeddings")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument(
        "--devices",
        default="auto",
        help="'auto' uses every visible CUDA GPU (or CPU); alternatively pass comma-separated devices.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--exclude-categories",
        default=None,
        help="Comma-separated QA categories to omit (defaults: AR for Mem-Gallery, MB for WMA).",
    )
    args = parser.parse_args()

    default_excluded = "MB" if args.benchmark == "wma" else (
        "AR" if args.benchmark == "memgallery" else ""
    )
    excluded_categories = parse_excluded_categories(
        default_excluded if args.exclude_categories is None else args.exclude_categories
    )

    data_dir = Path(args.data_dir)
    if args.benchmark == "wma":
        from benchmarks.wma_harness.retrieval.query_embedding_cache import iter_qa_items as iter_wma_qa_items

        sample_ids = {args.data_name} if args.data_name else None
        items = iter_wma_qa_items(
            data_dir,
            sample_ids=sample_ids,
            excluded_categories=excluded_categories,
        )
    elif args.benchmark == "h2hmem":
        if not args.split_manifest:
            parser.error("--split-manifest is required for --benchmark h2hmem")
        items = iter_h2hmem_qa_items(
            args.split_manifest,
            variant=args.h2h_variant,
        )
    else:
        items = iter_qa_items(
            data_dir,
            data_name=args.data_name,
            all_datasets=args.all_datasets,
            excluded_categories=excluded_categories,
        )
    if args.limit:
        items = items[: args.limit]

    devices = _resolve_devices(args.devices, workers=args.workers)

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
        "excluded_categories": sorted(excluded_categories),
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
