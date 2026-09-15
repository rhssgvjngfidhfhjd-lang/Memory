#!/usr/bin/env python3
"""Count Qwen3-VL visual tokens for train/val/test splits across benchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from count_benchmark_image_tokens import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ImageInput,
    ImageResult,
    _percentile,
    count_image,
    h2hmem_images,
    memgallery_images,
    wma_images,
)

DEFAULT_MANIFEST = WORKSPACE_ROOT / "Offline/configs/multimodal_split_manifest.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "Offline/outputs/image_token_splits_qwen3vl4b"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _benchmark_name(data_source: str) -> str:
    if data_source.startswith("h2hmem_"):
        return "H2HMEM"
    if data_source == "mem_gallery":
        return "Mem-Gallery"
    if data_source == "worldmemarena_lifelong":
        return "WorldMemArena"
    raise ValueError(f"Unsupported data_source: {data_source}")


def _dataset_path(data_source: str, conversation: dict[str, Any]) -> Path:
    source_id = str(conversation["source_id"])
    variant = str(conversation.get("variant") or "")
    if data_source == "mem_gallery":
        return WORKSPACE_ROOT / f"Mem-Gallery/benchmark/data/dialog/{source_id}.json"
    if data_source.startswith("h2hmem_"):
        disk_variant = "multi-party" if variant == "multiparty" else variant
        return WORKSPACE_ROOT / f"H2HMEM-main/dataset/{disk_variant}/{source_id}"
    if data_source == "worldmemarena_lifelong":
        personal_path = WORKSPACE_ROOT / f"WorldMemArena/WorldMemArena/lifelong/personal/{source_id}.json"
        if personal_path.is_file():
            return personal_path
        domain = source_id.split("_", 1)[0]
        return (
            WORKSPACE_ROOT
            / f"WorldMemArena/WorldMemArena/lifelong/project/{domain}/{source_id}.json"
        )
    raise ValueError(f"Unsupported data_source: {data_source}")


def _dataset_images(data_source: str, dataset_path: Path) -> list[ImageInput]:
    if data_source == "mem_gallery":
        return memgallery_images(dataset_path)
    if data_source.startswith("h2hmem_"):
        return h2hmem_images(dataset_path)
    if data_source == "worldmemarena_lifelong":
        return wma_images(dataset_path)
    raise ValueError(f"Unsupported data_source: {data_source}")


def _with_split(item: ImageInput, split: str) -> ImageInput:
    return ImageInput(
        benchmark=f"{item.benchmark}:{split}",
        dataset=item.dataset,
        path=item.path,
        sources=item.sources,
    )


def _values(results: list[ImageResult]) -> list[int]:
    return [
        int(row.image_pad_tokens)
        for row in results
        if row.status == "ok" and row.image_pad_tokens is not None
    ]


def _summary_row(
    benchmark: str,
    split: str,
    dataset_count: int,
    results: list[ImageResult],
) -> dict[str, Any]:
    values = _values(results)
    row: dict[str, Any] = {
        "benchmark": benchmark,
        "split": split,
        "datasets": dataset_count,
        "discovered_images": len(results),
        "successful_images": len(values),
        "failed_images": len(results) - len(values),
    }
    if not values:
        return row
    row.update(
        {
            "total_image_pad_tokens": sum(values),
            "mean_image_pad_tokens": statistics.mean(values),
            "median_image_pad_tokens": statistics.median(values),
            "stddev_image_pad_tokens": statistics.pstdev(values),
            "min_image_pad_tokens": min(values),
            "max_image_pad_tokens": max(values),
            "p95_image_pad_tokens": _percentile(values, 0.95),
            "mean_vision_sequence_tokens": statistics.mean(value + 2 for value in values),
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| Benchmark | Split | Datasets | Images | Failed | Total tokens | Mean | Median | Min | Max | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None:
                return "N/A"
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)

        lines.append(
            "| {benchmark} | {split} | {datasets} | {images} | {failed} | {total} | "
            "{mean} | {median} | {minimum} | {maximum} | {p95} |".format(
                benchmark=row["benchmark"],
                split=row["split"],
                datasets=row["datasets"],
                images=row["successful_images"],
                failed=row["failed_images"],
                total=fmt("total_image_pad_tokens"),
                mean=fmt("mean_image_pad_tokens"),
                median=fmt("median_image_pad_tokens"),
                minimum=fmt("min_image_pad_tokens"),
                maximum=fmt("max_image_pad_tokens"),
                p95=fmt("p95_image_pad_tokens"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-images-per-dataset",
        type=int,
        default=0,
        help="Deterministic smoke-test limit; 0 processes every discovered image.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_details: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    split_groups: dict[tuple[str, str], list[ImageResult]] = {}
    split_dataset_counts: dict[tuple[str, str], int] = {}

    for source in manifest["datasets"]:
        data_source = str(source["data_source"])
        benchmark = _benchmark_name(data_source)
        for split in ("train", "val", "test"):
            conversations = source["splits"][split]["conversations"]
            for conversation in conversations:
                dataset_path = _dataset_path(data_source, conversation)
                images = _dataset_images(data_source, dataset_path)
                if args.max_images_per_dataset:
                    images = images[: args.max_images_per_dataset]
                if not images:
                    raise RuntimeError(f"No images discovered for {dataset_path}")

                print(
                    f"Processing {benchmark}/{split}/{images[0].dataset}: "
                    f"{len(images)} images",
                    flush=True,
                )
                results = [count_image(processor, _with_split(item, split)) for item in images]
                for result in results:
                    detail = asdict(result)
                    detail["benchmark"] = benchmark
                    detail["split"] = split
                    detail["data_source"] = data_source
                    detail["source_id"] = conversation["source_id"]
                    detail["dataset_path"] = str(dataset_path)
                    all_details.append(detail)

                dataset_summary = _summary_row(benchmark, split, 1, results)
                dataset_summary["data_source"] = data_source
                dataset_summary["dataset"] = images[0].dataset
                dataset_summary["source_id"] = conversation["source_id"]
                dataset_summary["dataset_path"] = str(dataset_path)
                dataset_rows.append(dataset_summary)

                key = (benchmark, split)
                split_groups.setdefault(key, []).extend(results)
                split_dataset_counts[key] = split_dataset_counts.get(key, 0) + 1

    split_rows = [
        _summary_row(
            benchmark,
            split,
            split_dataset_counts[(benchmark, split)],
            split_groups[(benchmark, split)],
        )
        for benchmark in ("H2HMEM", "Mem-Gallery", "WorldMemArena")
        for split in ("train", "val", "test")
        if (benchmark, split) in split_groups
    ]

    aggregate_rows = [
        _summary_row(
            benchmark,
            "all",
            sum(
                count
                for (row_benchmark, _split), count in split_dataset_counts.items()
                if row_benchmark == benchmark
            ),
            [
                result
                for (row_benchmark, _split), results in split_groups.items()
                if row_benchmark == benchmark
                for result in results
            ],
        )
        for benchmark in ("H2HMEM", "Mem-Gallery", "WorldMemArena")
    ]

    _write_csv(args.output_dir / "split_image_token_details.csv", all_details)
    _write_csv(args.output_dir / "split_image_token_by_dataset.csv", dataset_rows)
    _write_csv(args.output_dir / "split_image_token_summary.csv", split_rows)
    _write_csv(args.output_dir / "split_image_token_benchmark_totals.csv", aggregate_rows)
    _write_markdown(args.output_dir / "split_image_token_summary.md", split_rows)
    (args.output_dir / "split_image_token_summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(args.manifest.resolve()),
                    "model_path": str(args.model_path.resolve()),
                    "processor_class": type(processor).__name__,
                    "image_processor_class": type(processor.image_processor).__name__,
                    "transformers_version": transformers.__version__,
                    "counting_unit": "unique physical image within each dataset",
                    "primary_metric": "count of <|image_pad|> in processor input_ids",
                    "vision_boundary_tokens_per_image": 2,
                    "image_processor_size": processor.image_processor.size,
                    "patch_size": processor.image_processor.patch_size,
                    "temporal_patch_size": processor.image_processor.temporal_patch_size,
                    "merge_size": processor.image_processor.merge_size,
                },
                "summary": split_rows,
                "benchmark_totals": aggregate_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSplit summary:")
    for row in split_rows:
        print(
            "{benchmark}/{split}: datasets={datasets}, images={successful_images}, "
            "failed={failed_images}, mean={mean_image_pad_tokens:.2f}, "
            "median={median_image_pad_tokens:.2f}, total={total_image_pad_tokens}".format(
                **row
            )
        )
    print(f"Results written to {args.output_dir.resolve()}")
    return 1 if any(row["status"] != "ok" for row in all_details) else 0


if __name__ == "__main__":
    sys.exit(main())
