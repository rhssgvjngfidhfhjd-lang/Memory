from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .extractor import VPExtractor
from .io import ArtifactStore, load_caption_map, scan_datasets, scan_path
from .models import Settings
from .vlm import ObjectDiscoverer, OpenAICompatibleVLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _read_json(config_path)
    settings = _apply_overrides(Settings.from_mapping(config), args)
    output_root = _resolve_project_path(settings.output_root)
    store = ArtifactStore(
        output_root,
        settings.run_id,
        jpeg_quality=settings.jpeg_quality,
        create_preview=settings.create_preview,
    )

    if args.command == "export":
        images, primitives = store.export()
        print(f"Wrote {images}")
        print(f"Wrote {primitives}")
        return 0

    client = OpenAICompatibleVLM(
        settings, api_key=os.getenv(settings.api_key_env, "EMPTY")
    )
    if args.command == "check-model":
        client.assert_available()
        print(f"Available: {settings.model} at {settings.base_url}")
        return 0

    prompts = config.get("prompts", {})
    discovery_prompt = _read_text(
        _resolve_project_path(str(prompts.get("discovery", "prompts/discovery_v1.txt")))
    )
    relocalize_prompt = _read_text(
        _resolve_project_path(str(prompts.get("relocalize", "prompts/relocalize_v1.txt")))
    )
    caption_guided_prompt = _read_text(
        _resolve_project_path(
            str(prompts.get("caption_guided", "prompts/caption_guided_v1.txt"))
        )
    )
    signature = {
        "settings": settings.public_dict(),
        "prompts": {
            "discovery_sha256": _text_sha256(discovery_prompt),
            "relocalize_sha256": _text_sha256(relocalize_prompt),
            "caption_guided_sha256": _text_sha256(caption_guided_prompt),
        },
    }
    store.initialize(signature)
    if not args.skip_model_check:
        client.assert_available()

    sources = _resolve_sources(args)
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    sources = sources[args.shard_index :: args.num_shards]
    if args.limit:
        sources = sources[: args.limit]
    discoverer = ObjectDiscoverer(
        client,
        discovery_prompt,
        relocalize_prompt,
        settings.max_primitives,
        caption_guided_prompt,
        settings.retries,
    )
    extractor = VPExtractor(discoverer, settings)

    processed = skipped = failed = 0
    per_dataset: dict[str, dict[str, int]] = {}
    for index, source in enumerate(sources, start=1):
        stats = per_dataset.setdefault(
            source.dataset, {"processed": 0, "skipped": 0, "failed": 0}
        )
        if store.is_complete(source.image_id) and not args.force:
            skipped += 1
            stats["skipped"] += 1
            continue
        try:
            store.save(extractor.extract_image(source))
            processed += 1
            stats["processed"] += 1
        except Exception as exc:
            failed += 1
            stats["failed"] += 1
            store.save_failure(source, exc)
            print(f"FAILED {source.relative_path}: {exc}")
        if index % 25 == 0 or index == len(sources):
            print(f"Progress {index}/{len(sources)}")

    store.export()
    for dataset, stats in sorted(per_dataset.items()):
        print(f"{dataset}: " + " ".join(f"{key}={value}" for key, value in stats.items()))
    print(f"Done: processed={processed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract object-centric visual primitives")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "configs" / "default.json")
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract one image, a directory, or a dataset")
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--dataset")
    extract.add_argument("--dataset-name", default="custom")
    extract.add_argument(
        "--caption-file",
        type=Path,
        help="Optional Mem-Gallery dialog JSON used for caption-guided extraction",
    )
    extract.add_argument(
        "--datasets-config",
        default=str(PROJECT_ROOT / "configs" / "datasets.json"),
    )
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--skip-model-check", action="store_true")

    subparsers.add_parser("check-model", help="Verify the configured model endpoint")
    subparsers.add_parser("export", help="Rebuild JSONL exports from record.json files")
    return parser


def _resolve_sources(args: argparse.Namespace):
    if args.input:
        captions = load_caption_map(args.caption_file.resolve()) if args.caption_file else None
        return scan_path(args.input, dataset=args.dataset_name, captions=captions)
    datasets = _read_json(Path(args.datasets_config).resolve())
    names = sorted(datasets) if str(args.dataset).lower() == "all" else [args.dataset]
    return scan_datasets(names, datasets, PROJECT_ROOT)


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    values = {}
    for argument, field_name in (
        (args.model, "model"),
        (args.base_url, "base_url"),
        (args.output_root, "output_root"),
        (args.run_id, "run_id"),
    ):
        if argument:
            values[field_name] = argument
    return replace(settings, **values)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
