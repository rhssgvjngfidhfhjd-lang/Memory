#!/usr/bin/env python3
"""Count Qwen3-VL visual tokens for one dataset from each benchmark.

The script counts each unique physical image once.  Its primary metric is the
number of ``<|image_pad|>`` tokens produced by the model's own processor.  It
also verifies that count against ``image_grid_thw`` and reports the two vision
boundary tokens separately.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

from PIL import Image
import transformers
from transformers import AutoProcessor


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path("/data/shared_models/Qwen3-VL-4B-Instruct")
DEFAULT_MEMGALLERY_JSON = (
    WORKSPACE_ROOT
    / "Mem-Gallery/benchmark/data/dialog/Dog_Behavior_Research_Academic_Life.json"
)
DEFAULT_H2HMEM_DIR = WORKSPACE_ROOT / "H2HMEM-main/dataset/dyadic/dialogue10"
DEFAULT_WMA_JSON = (
    WORKSPACE_ROOT
    / "WorldMemArena/WorldMemArena/lifelong/project/academic/academic_03.json"
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "Offline/outputs/image_token_benchmark_qwen3vl4b"
FIXED_PROMPT = "Describe the image."


@dataclass(frozen=True)
class ImageInput:
    benchmark: str
    dataset: str
    path: Path
    sources: tuple[str, ...]


@dataclass
class ImageResult:
    benchmark: str
    dataset: str
    image_path: str
    sources: str
    width: int | None = None
    height: int | None = None
    image_format: str = ""
    image_pad_tokens: int | None = None
    vision_boundary_tokens: int | None = None
    vision_sequence_tokens: int | None = None
    grid_t: int | None = None
    grid_h: int | None = None
    grid_w: int | None = None
    grid_formula_tokens: int | None = None
    status: str = "ok"
    error: str = ""


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _deduplicate(
    benchmark: str, dataset: str, rows: Iterable[tuple[Path, str]]
) -> list[ImageInput]:
    by_path: dict[Path, set[str]] = {}
    for path, source in rows:
        resolved = path.resolve()
        by_path.setdefault(resolved, set()).add(source)
    return [
        ImageInput(
            benchmark=benchmark,
            dataset=dataset,
            path=path,
            sources=tuple(sorted(sources)),
        )
        for path, sources in sorted(by_path.items(), key=lambda item: str(item[0]))
    ]


def memgallery_images(dataset_json: Path) -> list[ImageInput]:
    payload = _load_json(dataset_json)
    data_dir = dataset_json.parents[1]
    rows: list[tuple[Path, str]] = []

    def resolve(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        if raw.startswith("../image/"):
            return data_dir / "image" / raw.removeprefix("../image/")
        return data_dir / raw

    for session in payload.get("multi_session_dialogues", []) or []:
        session_id = str(session.get("session_id") or "")
        for turn in session.get("dialogues", []) or []:
            round_id = str(turn.get("round") or "")
            for raw in _values(turn.get("input_image")):
                rows.append((resolve(raw), f"dialogue:{session_id}:{round_id}"))

    for index, qa in enumerate(payload.get("human-annotated QAs", []) or [], start=1):
        for raw in _values(qa.get("question_image")):
            rows.append((resolve(raw), f"question:{index}"))

    return _deduplicate("Mem-Gallery", dataset_json.stem, rows)


def h2hmem_images(conversation_dir: Path) -> list[ImageInput]:
    scenes_dir = conversation_dir / "scenes"
    rows: list[tuple[Path, str]] = []
    for session_dir in sorted(path for path in scenes_dir.glob("session*") if path.is_dir()):
        session_path = session_dir / "session.json"
        if session_path.is_file():
            payload = _load_json(session_path)
            for turn_index, turn in enumerate(payload.get("dialogue", []) or [], start=1):
                content = turn.get("content") or {}
                for raw in _values(content.get("image")):
                    rows.append(
                        (session_dir / "image" / raw, f"dialogue:{session_dir.name}:{turn_index}")
                    )

        questions_path = session_dir / "questions.json"
        if questions_path.is_file():
            payload = _load_json(questions_path)
            for qa_index, qa in enumerate(payload.get("questions", []) or [], start=1):
                question = qa.get("question") or {}
                for raw in _values(question.get("image")):
                    parts = raw.replace("\\", "/").split("/", maxsplit=1)
                    path = (
                        scenes_dir / parts[0] / "image" / parts[1]
                        if len(parts) == 2
                        else session_dir / "image" / raw
                    )
                    rows.append((path, f"question:{session_dir.name}:{qa_index}"))

    variant = conversation_dir.parent.name
    dataset = f"{variant}/{conversation_dir.name}"
    return _deduplicate("H2HMEM", dataset, rows)


def wma_images(dataset_json: Path) -> list[ImageInput]:
    payload = _load_json(dataset_json)
    rows: list[tuple[Path, str]] = []

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            attachments = value.get("attachments")
            if isinstance(attachments, list):
                for index, attachment in enumerate(attachments, start=1):
                    if not isinstance(attachment, dict):
                        continue
                    for raw in _values(attachment.get("file_path")):
                        path = Path(raw)
                        if not path.is_absolute():
                            path = dataset_json.parent / path
                        rows.append((path, f"{location}:attachment:{index}"))
            for key, nested in value.items():
                if key != "attachments":
                    walk(nested, f"{location}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{location}[{index}]")

    walk(payload, "root")
    dataset = str(payload.get("sample_id") or dataset_json.stem)
    return _deduplicate("WorldMemArena", dataset, rows)


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def count_image(processor: Any, item: ImageInput) -> ImageResult:
    result = ImageResult(
        benchmark=item.benchmark,
        dataset=item.dataset,
        image_path=str(item.path),
        sources=";".join(item.sources),
    )
    try:
        if not item.path.is_file():
            raise FileNotFoundError(f"image does not exist: {item.path}")
        with Image.open(item.path) as opened:
            result.width, result.height = opened.size
            result.image_format = str(opened.format or "")
            image = opened.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": FIXED_PROMPT},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        image_pad_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
        grid_t, grid_h, grid_w = [int(value) for value in inputs["image_grid_thw"][0]]
        merge_size = int(processor.image_processor.merge_size)
        grid_formula_tokens = grid_t * grid_h * grid_w // (merge_size**2)
        if image_pad_tokens != grid_formula_tokens:
            raise RuntimeError(
                "processor token count and grid formula disagree: "
                f"{image_pad_tokens} != {grid_formula_tokens}"
            )

        result.image_pad_tokens = image_pad_tokens
        result.vision_boundary_tokens = 2
        result.vision_sequence_tokens = image_pad_tokens + 2
        result.grid_t = grid_t
        result.grid_h = grid_h
        result.grid_w = grid_w
        result.grid_formula_tokens = grid_formula_tokens
    except Exception as exc:  # Continue so failures remain visible in the report.
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def summarize(results: list[ImageResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[ImageResult]] = {}
    for row in results:
        groups.setdefault((row.benchmark, row.dataset), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (benchmark, dataset), rows in groups.items():
        values = [
            int(row.image_pad_tokens)
            for row in rows
            if row.status == "ok" and row.image_pad_tokens is not None
        ]
        if not values:
            summaries.append(
                {
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "discovered_images": len(rows),
                    "successful_images": 0,
                    "failed_images": len(rows),
                }
            )
            continue
        summaries.append(
            {
                "benchmark": benchmark,
                "dataset": dataset,
                "discovered_images": len(rows),
                "successful_images": len(values),
                "failed_images": len(rows) - len(values),
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
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Benchmark | Dataset | Images | Failed | Mean image tokens | Median | Min | Max | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        def number(key: str) -> str:
            value = row.get(key)
            if value is None:
                return "N/A"
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        lines.append(
            "| {benchmark} | {dataset} | {successful_images} | {failed_images} | "
            "{mean} | {median} | {minimum} | {maximum} | {p95} |".format(
                benchmark=row["benchmark"],
                dataset=row["dataset"],
                successful_images=row["successful_images"],
                failed_images=row["failed_images"],
                mean=number("mean_image_pad_tokens"),
                median=number("median_image_pad_tokens"),
                minimum=number("min_image_pad_tokens"),
                maximum=number("max_image_pad_tokens"),
                p95=number("p95_image_pad_tokens"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--memgallery-json", type=Path, default=DEFAULT_MEMGALLERY_JSON)
    parser.add_argument("--h2hmem-dir", type=Path, default=DEFAULT_H2HMEM_DIR)
    parser.add_argument("--wma-json", type=Path, default=DEFAULT_WMA_JSON)
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
    inputs_by_dataset = [
        memgallery_images(args.memgallery_json),
        h2hmem_images(args.h2hmem_dir),
        wma_images(args.wma_json),
    ]
    if args.max_images_per_dataset:
        inputs_by_dataset = [
            rows[: args.max_images_per_dataset] for rows in inputs_by_dataset
        ]
    if any(not rows for rows in inputs_by_dataset):
        empty = [index for index, rows in enumerate(inputs_by_dataset) if not rows]
        raise RuntimeError(f"no images discovered for dataset group(s): {empty}")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ImageResult] = []
    for rows in inputs_by_dataset:
        print(f"Processing {rows[0].benchmark}/{rows[0].dataset}: {len(rows)} images")
        for index, item in enumerate(rows, start=1):
            results.append(count_image(processor, item))
            if index % 25 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)}", flush=True)

    summaries = summarize(results)
    details = [asdict(row) for row in results]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model_path.resolve()),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "transformers_version": transformers.__version__,
        "fixed_prompt": FIXED_PROMPT,
        "counting_unit": "unique physical image",
        "primary_metric": "count of <|image_pad|> in processor input_ids",
        "validation": "image_pad_tokens == product(image_grid_thw) / merge_size^2",
        "vision_boundary_tokens_per_image": 2,
        "image_processor_size": processor.image_processor.size,
        "patch_size": processor.image_processor.patch_size,
        "temporal_patch_size": processor.image_processor.temporal_patch_size,
        "merge_size": processor.image_processor.merge_size,
        "datasets": [
            {
                "benchmark": rows[0].benchmark,
                "dataset": rows[0].dataset,
                "source": str(
                    [args.memgallery_json, args.h2hmem_dir, args.wma_json][index].resolve()
                ),
                "selection_rule": "first dataset listed in test split",
            }
            for index, rows in enumerate(inputs_by_dataset)
        ],
    }

    _write_csv(args.output_dir / "image_token_details.csv", details)
    _write_csv(args.output_dir / "image_token_summary.csv", summaries)
    (args.output_dir / "image_token_summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    markdown = _markdown_table(summaries)
    (args.output_dir / "image_token_summary.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Results written to {args.output_dir.resolve()}")
    return 1 if any(row.status != "ok" for row in results) else 0


if __name__ == "__main__":
    sys.exit(main())
