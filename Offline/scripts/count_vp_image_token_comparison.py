#!/usr/bin/env python3
"""Compare original-image tokens against corresponding VP crop tokens."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import transformers
from transformers import AutoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
OFFLINE_SRC = WORKSPACE_ROOT / "Offline/src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(OFFLINE_SRC) not in sys.path:
    sys.path.insert(0, str(OFFLINE_SRC))

from count_benchmark_image_tokens import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ImageInput,
    count_image,
)
from evidence_policy.vp_store import VPArtifactIndex  # noqa: E402

DEFAULT_ORIGINAL_DETAILS = (
    WORKSPACE_ROOT
    / "Offline/outputs/image_token_benchmark_qwen3vl4b/image_token_details.csv"
)
DEFAULT_VP_RUN_DIR = WORKSPACE_ROOT / "vp_extractor/outputs/qwen3vl4b_all_v1"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "Offline/outputs/image_token_vp_comparison_qwen3vl4b"


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--original-details", type=Path, default=DEFAULT_ORIGINAL_DETAILS)
    parser.add_argument("--vp-run-dir", type=Path, default=DEFAULT_VP_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Deterministic smoke-test limit; 0 processes all original images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.original_details.open(encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    if args.max_images:
        original_rows = original_rows[: args.max_images]

    index = VPArtifactIndex(args.vp_run_dir)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict[str, Any]] = []
    vp_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(original_rows, start=1):
        benchmark = row["benchmark"]
        dataset = row["dataset"]
        original_path = Path(row["image_path"])
        original_tokens = int(row["image_pad_tokens"])
        record = index.record_for(original_path, dataset=benchmark)
        primitives = record.primitives if record is not None else ()

        crop_tokens: list[int] = []
        missing_crops = 0
        for primitive in primitives:
            crop_input = ImageInput(
                benchmark=benchmark,
                dataset=dataset,
                path=primitive.crop_path,
                sources=(f"vp:{primitive.vp_id}:{primitive.label}",),
            )
            result = count_image(processor, crop_input)
            vp_row = asdict(result)
            vp_row.update(
                {
                    "original_image_path": str(original_path),
                    "original_image_tokens": original_tokens,
                    "vp_id": primitive.vp_id,
                    "vp_label": primitive.label,
                    "bbox_norm": json.dumps(primitive.bbox_norm),
                }
            )
            vp_rows.append(vp_row)
            if result.status == "ok" and result.image_pad_tokens is not None:
                crop_tokens.append(int(result.image_pad_tokens))
            else:
                missing_crops += 1

        vp_total_tokens = sum(crop_tokens)
        token_delta = vp_total_tokens - original_tokens
        ratio = vp_total_tokens / original_tokens if original_tokens else 0.0
        image_rows.append(
            {
                "benchmark": benchmark,
                "dataset": dataset,
                "image_path": str(original_path),
                "matched_vp_record": record is not None,
                "vp_status": record.status if record is not None else "",
                "vp_count": len(primitives),
                "vp_successful_count": len(crop_tokens),
                "vp_failed_count": missing_crops,
                "original_image_tokens": original_tokens,
                "vp_total_image_tokens": vp_total_tokens,
                "vp_mean_image_tokens": _mean(crop_tokens),
                "vp_median_image_tokens": _median(crop_tokens),
                "vp_min_image_tokens": min(crop_tokens) if crop_tokens else 0,
                "vp_max_image_tokens": max(crop_tokens) if crop_tokens else 0,
                "vp_total_minus_original_tokens": token_delta,
                "vp_total_to_original_ratio": ratio,
                "vp_total_token_saving_percent": (1.0 - ratio) * 100.0,
            }
        )

        if row_index % 25 == 0 or row_index == len(original_rows):
            print(f"Processed {row_index}/{len(original_rows)} original images", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted({(row["benchmark"], row["dataset"]) for row in image_rows}):
        benchmark, dataset = key
        rows = [row for row in image_rows if (row["benchmark"], row["dataset"]) == key]
        original_tokens = [int(row["original_image_tokens"]) for row in rows]
        vp_total_tokens = [int(row["vp_total_image_tokens"]) for row in rows]
        vp_counts = [int(row["vp_count"]) for row in rows]
        successful_vp_counts = [int(row["vp_successful_count"]) for row in rows]
        summary_rows.append(
            {
                "benchmark": benchmark,
                "dataset": dataset,
                "images": len(rows),
                "matched_vp_records": sum(bool(row["matched_vp_record"]) for row in rows),
                "images_with_vp": sum(int(row["vp_count"]) > 0 for row in rows),
                "vp_crops": sum(vp_counts),
                "successful_vp_crops": sum(successful_vp_counts),
                "mean_vps_per_image": _mean(vp_counts),
                "original_total_tokens": sum(original_tokens),
                "original_mean_tokens": _mean(original_tokens),
                "vp_total_tokens": sum(vp_total_tokens),
                "vp_total_mean_per_image": _mean(vp_total_tokens),
                "vp_crop_mean_tokens": _mean(
                    [
                        int(row["image_pad_tokens"])
                        for row in vp_rows
                        if row["benchmark"] == benchmark
                        and row["dataset"] == dataset
                        and row["status"] == "ok"
                    ]
                ),
                "vp_total_to_original_ratio": (
                    sum(vp_total_tokens) / sum(original_tokens) if sum(original_tokens) else 0.0
                ),
                "vp_total_token_saving_percent": (
                    1.0 - (sum(vp_total_tokens) / sum(original_tokens))
                )
                * 100.0
                if sum(original_tokens)
                else 0.0,
            }
        )

    all_original = [int(row["original_image_tokens"]) for row in image_rows]
    all_vp_total = [int(row["vp_total_image_tokens"]) for row in image_rows]
    all_vp_crop_tokens = [
        int(row["image_pad_tokens"]) for row in vp_rows if row["status"] == "ok"
    ]
    aggregate = {
        "images": len(image_rows),
        "matched_vp_records": sum(bool(row["matched_vp_record"]) for row in image_rows),
        "images_with_vp": sum(int(row["vp_count"]) > 0 for row in image_rows),
        "vp_crops": len(vp_rows),
        "successful_vp_crops": len(all_vp_crop_tokens),
        "original_total_tokens": sum(all_original),
        "original_mean_tokens": _mean(all_original),
        "vp_total_tokens": sum(all_vp_total),
        "vp_total_mean_per_image": _mean(all_vp_total),
        "vp_crop_mean_tokens": _mean(all_vp_crop_tokens),
        "vp_total_to_original_ratio": (
            sum(all_vp_total) / sum(all_original) if sum(all_original) else 0.0
        ),
        "vp_total_token_saving_percent": (
            1.0 - (sum(all_vp_total) / sum(all_original))
        )
        * 100.0
        if sum(all_original)
        else 0.0,
    }

    _write_csv(args.output_dir / "vp_image_token_comparison_by_image.csv", image_rows)
    _write_csv(args.output_dir / "vp_image_token_comparison_by_crop.csv", vp_rows)
    _write_csv(args.output_dir / "vp_image_token_comparison_summary.csv", summary_rows)
    (args.output_dir / "vp_image_token_comparison_summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model_path": str(args.model_path.resolve()),
                    "processor_class": type(processor).__name__,
                    "image_processor_class": type(processor.image_processor).__name__,
                    "transformers_version": transformers.__version__,
                    "original_details": str(args.original_details.resolve()),
                    "vp_run_dir": str(args.vp_run_dir.resolve()),
                    "counting_unit": "original image and all matched VP crops",
                    "primary_metric": "count of <|image_pad|> in processor input_ids",
                },
                "aggregate": aggregate,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nBenchmark summary:")
    for row in summary_rows:
        print(
            "{benchmark}/{dataset}: images={images}, vp_crops={vp_crops}, "
            "original_mean={original_mean_tokens:.2f}, "
            "vp_total_mean={vp_total_mean_per_image:.2f}, "
            "ratio={vp_total_to_original_ratio:.3f}, "
            "saving={vp_total_token_saving_percent:.2f}%".format(**row)
        )
    print(
        "\nAll: images={images}, vp_crops={vp_crops}, original_mean={original_mean_tokens:.2f}, "
        "vp_total_mean={vp_total_mean_per_image:.2f}, ratio={vp_total_to_original_ratio:.3f}, "
        "saving={vp_total_token_saving_percent:.2f}%".format(**aggregate)
    )
    print(f"Results written to {args.output_dir.resolve()}")
    return 1 if len(all_vp_crop_tokens) != len(vp_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
