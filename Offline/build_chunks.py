#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from offline_memgallery_qwen.chunk_builder import build_chunks_from_directory
from offline_memgallery_qwen.io_utils import write_chunks_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Mem-Gallery dialogue-round chunks.")
    parser.add_argument("--data-dir", default="/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data")
    parser.add_argument("--output", default="artifacts/chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--no-previous-summary", action="store_true")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    chunks = build_chunks_from_directory(
        args.data_dir,
        max_tokens=args.max_tokens,
        include_previous_summary=not args.no_previous_summary,
        include_captions=not args.no_captions,
        include_images=not args.no_images,
    )
    count = write_chunks_jsonl(chunks, args.output)
    stats = {
        "chunks": count,
        "with_images": sum(1 for c in chunks if c.images),
        "data_dir": str(Path(args.data_dir).resolve()),
        "output": str(Path(args.output).resolve()),
        "max_tokens": args.max_tokens,
        "include_captions": not args.no_captions,
        "include_images": not args.no_images,
    }
    write_json(Path(args.output).with_suffix(".stats.json"), stats)
    print(stats)


if __name__ == "__main__":
    main()
