#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from embedding.chunk_builder import (
    build_chunks_from_directory,
    build_h2h_chunks_from_directory,
    build_wma_chunks_from_directory,
)
from embedding.chunk_builder import write_chunks_jsonl, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMGALLERY_DATA_DIR = PROJECT_ROOT.parent / "Mem-Gallery" / "benchmark" / "data"
DEFAULT_WMA_DATA_DIR = PROJECT_ROOT.parent / "WorldMemArena" / "WorldMemArena" / "lifelong"
DEFAULT_H2HMEM_DATA_DIR = PROJECT_ROOT.parent / "H2HMEM-main" / "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark dialogue-round chunks.")
    parser.add_argument(
        "--benchmark", choices=("memgallery", "wma", "h2hmem"), default="memgallery"
    )
    parser.add_argument("--data-name", default="", help="Optional sample/conversation id.")
    parser.add_argument("--variant", choices=("dyadic", "multiparty"), default="dyadic")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output", default="data/qwen3_vl_embedding_2b/chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--no-previous-summary", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    common = {
        "max_tokens": args.max_tokens,
        "include_previous_summary": not args.no_previous_summary,
        "include_captions": (
            False if args.benchmark == "h2hmem" else not args.no_captions
        ),
        "include_images": not args.no_images,
    }
    default_data_dirs = {
        "memgallery": DEFAULT_MEMGALLERY_DATA_DIR,
        "wma": DEFAULT_WMA_DATA_DIR,
        "h2hmem": DEFAULT_H2HMEM_DATA_DIR,
    }
    data_dir = args.data_dir or str(default_data_dirs[args.benchmark])
    if args.benchmark == "wma":
        chunks = build_wma_chunks_from_directory(
            data_dir,
            sample_ids={args.data_name} if args.data_name else None,
            **common,
        )
    elif args.benchmark == "h2hmem":
        chunks = build_h2h_chunks_from_directory(
            data_dir,
            variant=args.variant,
            conversation_ids={args.data_name} if args.data_name else None,
            max_tokens=args.max_tokens,
            include_previous_summary=not args.no_previous_summary,
            include_images=not args.no_images,
        )
    else:
        chunks = build_chunks_from_directory(data_dir, **common)
    count = write_chunks_jsonl(chunks, args.output)
    stats = {
        "chunks": count,
        "benchmark": args.benchmark,
        "variant": args.variant if args.benchmark == "h2hmem" else None,
        "with_images": sum(1 for c in chunks if c.images),
        "data_dir": str(Path(data_dir).resolve()),
        "output": str(Path(args.output).resolve()),
        "max_tokens": args.max_tokens,
        "include_captions": common["include_captions"],
        "include_images": not args.no_images,
    }
    write_json(Path(args.output).with_suffix(".stats.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
