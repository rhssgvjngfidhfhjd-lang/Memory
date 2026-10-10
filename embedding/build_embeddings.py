"""Generate query or dialogue-chunk vector caches."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import (
    api_key_for, benchmark_path, dataset_root, image_data_uri, load_runtime_config, portable_value,
    query_embedding_path, validate_embedding_settings,
)
from src.utils import sha256_file as file_sha256, write_json_atomic
from .backends import (
    DEFAULT_QUERY_INSTRUCTION, create_embedding_service, is_qwen3_text_embedding_model,
)
from benchmarks.common.query_cache import (
    QUERY_ID_SCHEME, legacy_query_id, make_query_id,
)
from benchmarks.common.prompts import resolve_question_image
from benchmarks.common.utils import is_excluded_category, parse_excluded_categories
from evidence_policy.evidence import SplitManifestIndex, iter_source_questions


DEFAULT_MEMGALLERY_DATA_DIR = dataset_root("memgallery")


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
                    "legacy_query_id": legacy_query_id(
                        dataset_name=dataset_name, qa_index=qa_index,
                        category=category, question=question, query_image=query_image,
                    ),
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
        None,
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
        revision=args_dict.get("model_revision") or None,
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


def request_embeddings(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int,
) -> list[list[float]]:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key_for('embedding')}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return [row["embedding"] for row in result.get("data") or []]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"embedding request failed: {exc}") from exc
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def scan_input(input_path: Path, metadata_path: Path) -> int:
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    count = 0
    with input_path.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            output.write(
                json.dumps(
                    {
                        "index": count,
                        "chunk_id": row.get("chunk_id"),
                        "dataset": metadata.get("dataset"),
                        "session_id": metadata.get("session_id"),
                        "source_dialogue_ids": metadata.get("source_dialogue_ids")
                        or [metadata.get("dialogue_id") or row.get("chunk_id")],
                        "image_count": len(row.get("images") or []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    temporary.replace(metadata_path)
    return count


def embedding_payload(
    rows: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    if len(rows) == 1 and rows[0].get("images"):
        content = [
            {"type": "image_url", "image_url": {"url": image_data_uri(path)}}
            for path in rows[0].get("images") or []
        ]
        content.append({"type": "text", "text": str(rows[0].get("text") or " ")})
        return {
            "model": model,
            "mode": "context",
            "messages": [{"role": "user", "content": content}],
        }
    return {
        "model": model,
        "mode": "context",
        "input": [str(row.get("text") or " ") for row in rows],
    }


def _add_query_arguments(parser: argparse.ArgumentParser, runtime: dict[str, Any]) -> None:
    parser.add_argument(
        "--benchmark", choices=("memgallery", "wma", "h2hmem"), default="memgallery"
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--data-name", default="")
    parser.add_argument("--all-datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--split-manifest",
        default=str(benchmark_path("multimodal_split_manifest.json")),
        help="Split manifest that defines the H2HMem query IDs and allowlist.",
    )
    parser.add_argument(
        "--h2h-variant",
        choices=("all", "dyadic", "multiparty"),
        default="all",
    )
    parser.add_argument("--out-dir", default="", help="Defaults to data/<benchmark>/query_embeddings/<model>/.")
    parser.add_argument("--model-name", default=runtime.get("embedding_model", ""))
    parser.add_argument("--model-revision", default=runtime.get("embedding_revision", ""),
                        help="Hugging Face model commit/tag; record and reuse the same revision for memory embeddings.")
    parser.add_argument("--dim", type=int, default=runtime.get("embedding_dim"))
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
        help="Comma-separated QA categories to omit (defaults: AR for Mem-Gallery; none otherwise).",
    )


def _add_chunk_arguments(parser: argparse.ArgumentParser, runtime: dict[str, Any]) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=runtime.get("embedding_base_url", ""))
    parser.add_argument("--model", default=runtime.get("embedding_model", ""))
    parser.add_argument("--model-revision", default=runtime.get("embedding_revision", ""),
                        help="Revision used by the embedding service; changed pins require a new cache.")
    parser.add_argument("--dim", type=int, default=runtime.get("embedding_dim"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)


def build_query_embeddings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.model_name, args.dim = validate_embedding_settings(
        parser, args.model_name, args.dim, model_flag="--model-name", dimension_flag="--dim",
    )

    if not args.data_dir:
        args.data_dir = str({
            "memgallery": DEFAULT_MEMGALLERY_DATA_DIR,
            "wma": dataset_root("wma"),
            "h2hmem": dataset_root("h2hmem"),
        }[args.benchmark])
    if not args.out_dir:
        try:
            args.out_dir = str(query_embedding_path(args.benchmark, args.model_name))
        except ValueError as error:
            parser.error(str(error))

    default_excluded = "AR" if args.benchmark == "memgallery" else ""
    excluded_categories = parse_excluded_categories(
        default_excluded if args.exclude_categories is None else args.exclude_categories
    )

    data_dir = Path(args.data_dir)
    if args.benchmark == "wma":
        from benchmarks.wma_harness.questions import iter_qa_items as iter_wma_qa_items

        sample_ids = {args.data_name} if args.data_name else None
        items = iter_wma_qa_items(
            data_dir,
            sample_ids=sample_ids,
            excluded_categories=excluded_categories,
        )
    elif args.benchmark == "h2hmem":
        # Source readers use the shared dataset-root resolver; explicit CLI input wins.
        os.environ["HIVE_H2HMEM_ROOT"] = str(data_dir.expanduser().resolve())
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
    if args.limit < 0:
        parser.error("--limit must be non-negative.")
    if args.limit:
        items = items[: args.limit]

    if not items:
        parser.error(f"No query questions found for {args.benchmark} in {data_dir}; check the raw-data layout and category filters.")
    if args.workers < 0 or args.batch_size <= 0:
        parser.error("--workers must be non-negative and --batch-size must be positive.")

    devices = _resolve_devices(args.devices, workers=args.workers)

    shards = _split_evenly(items, len(devices))
    args_dict = {
        "model_name": args.model_name,
        "dim": args.dim,
        "dtype": args.dtype,
        "local_files_only": args.local_files_only,
        "batch_size": args.batch_size,
        "model_revision": getattr(args, "model_revision", ""),
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
            f.write(json.dumps(portable_value(row), ensure_ascii=False) + "\n")
    manifest = {
        "count": len(ordered_rows),
        "dim": args.dim,
        "model_name": args.model_name,
        "model_revision": getattr(args, "model_revision", ""),
        "dtype": args.dtype,
        "data_dir": str(data_dir.resolve()),
        "benchmark": args.benchmark,
        "excluded_categories": sorted(excluded_categories),
        "vectors": str((out_dir / "vectors.npy").resolve()),
        "metadata": str((out_dir / "metadata.jsonl").resolve()),
    }
    if args.benchmark == "memgallery":
        manifest["query_id_scheme"] = QUERY_ID_SCHEME
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
    (out_dir / "manifest.json").write_text(json.dumps(portable_value(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest)


def build_chunk_embeddings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.model, args.dim = validate_embedding_settings(
        parser, args.model, args.dim, model_flag="--model", dimension_flag="--dim",
    )
    args.base_url = args.base_url.strip()
    if not args.base_url:
        parser.error("Set HIVE_EMBEDDING_BASE_URL or pass --base-url.")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    vectors_path = output_dir / "vectors.npy"
    input_hash = file_sha256(input_path)
    with input_path.open(encoding="utf-8") as source:
        total = sum(bool(line.strip()) for line in source)
    if not total:
        parser.error(f"No dialogue chunks found in {input_path}.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    signature = {
        "input": str(input_path),
        "input_sha256": input_hash,
        "model": args.model,
        "model_revision": str(getattr(args, "model_revision", "") or "").strip(),
        "dim": args.dim,
        "include_images": True,
        "total": total,
    }

    completed = 0
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [
            key for key, value in signature.items()
            if (previous.get(key, "") if key == "model_revision" else previous.get(key)) != value
        ]
        if mismatched:
            raise RuntimeError(f"existing cache signature mismatch: {', '.join(mismatched)}")
        completed = int(previous.get("completed") or 0)
    if not 0 <= completed <= total:
        raise RuntimeError(f"invalid cache progress: {completed}/{total}")
    if completed and not vectors_path.is_file():
        raise RuntimeError(f"cached vectors are missing: {vectors_path}; use a new cache directory to rebuild")

    if vectors_path.is_file():
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        if vectors.shape != (total, args.dim):
            raise RuntimeError(f"unexpected vector shape: {vectors.shape}")
        if vectors.dtype != np.float32:
            raise RuntimeError(f"unexpected vector dtype: {vectors.dtype}")
        for start in range(0, completed, 4096):
            if not np.isfinite(vectors[start:min(start + 4096, completed)]).all():
                raise RuntimeError("committed cached vectors contain NaN or Inf; rebuild in a new cache directory")
    else:
        vectors = np.lib.format.open_memmap(
            vectors_path, mode="w+", dtype=np.float32, shape=(total, args.dim)
        )

    # Rejected inputs and invalid existing vectors must leave cached metadata intact.
    scan_input(input_path, metadata_path)
    if completed == total:
        print(f"already complete: {total}/{total} {output_dir}", flush=True)
        return

    manifest = {**signature, "completed": completed, "status": "running"}
    write_json_atomic(manifest_path, manifest)
    endpoint = args.base_url.rstrip("/") + "/embeddings"
    started = time.time()
    pending: list[tuple[int, dict[str, Any]]] = []

    def commit(batch: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal completed
        if not batch:
            return
        rows = [row for _, row in batch]
        payload = embedding_payload(
            rows, model=args.model
        )
        values = request_embeddings(
            endpoint, payload, timeout=args.timeout, retries=args.retries
        )
        if len(values) != len(batch):
            raise RuntimeError(f"expected {len(batch)} vectors, received {len(values)}")
        validated = []
        for (index, _), value in zip(batch, values):
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (args.dim,):
                raise RuntimeError(f"vector {index} has shape {vector.shape}")
            if not np.isfinite(vector).all():
                raise RuntimeError(f"vector {index} contains NaN or Inf")
            validated.append((index, vector))
        for index, vector in validated:
            vectors[index] = vector
            completed = index + 1
        vectors.flush()
        manifest.update(
            {
                "completed": completed,
                "status": "running",
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        write_json_atomic(manifest_path, manifest)
        if completed % 100 < len(batch) or completed == total:
            print(f"progress {completed}/{total}", flush=True)

    with input_path.open(encoding="utf-8") as source:
        index = 0
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if index < completed:
                index += 1
                continue
            has_images = bool(row.get("images"))
            if has_images:
                commit(pending)
                pending = []
                commit([(index, row)])
            else:
                pending.append((index, row))
                if len(pending) >= args.batch_size:
                    commit(pending)
                    pending = []
            index += 1
        commit(pending)

    manifest.update(
        {
            "completed": completed,
            "status": "complete",
            "elapsed_seconds": round(time.time() - started, 2),
            "vectors": str(vectors_path),
            "metadata": str(metadata_path),
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(f"complete {completed}/{total}: {output_dir}", flush=True)


def main(argv: list[str] | None = None) -> None:
    runtime = load_runtime_config()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    queries = commands.add_parser("queries", help="Precompute benchmark query embeddings")
    _add_query_arguments(queries, runtime)
    queries.set_defaults(handler=build_query_embeddings, command_parser=queries)
    chunks = commands.add_parser("chunks", help="Build resumable embeddings from chunk JSONL")
    _add_chunk_arguments(chunks, runtime)
    chunks.set_defaults(handler=build_chunk_embeddings, command_parser=chunks)
    args = parser.parse_args(argv)
    args.handler(args, args.command_parser)


if __name__ == "__main__":
    main()
