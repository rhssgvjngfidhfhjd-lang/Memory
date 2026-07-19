#!/usr/bin/env python3
"""Stage 4: build a "separate vectors" chunk set that mirrors the
WorldMemArena qwen_embed_adapter.py reference design -- text and image are
never fused into one embedding. For each image-bearing chunk, emit:

  1. the same chunk with images=[] (identical to its scheme-b twin)
  2. one synthetic "image companion" row per image: text = caption only,
     images=[path], so the image's own vector is never diluted by the
     surrounding round's dialogue text, and the text vector for the round
     is never diluted by vision tokens either.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from offline_memgallery_qwen.io_utils import read_chunks_jsonl, write_chunks_jsonl
from offline_memgallery_qwen.schema import Chunk


def build_separate_vectors_chunks(chunks: list[Chunk]) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        if not chunk.images:
            out.append(chunk)
            continue

        text_only_metadata = dict(chunk.metadata)
        out.append(replace(chunk, images=[], metadata=text_only_metadata))

        image_ids = chunk.metadata.get("image_ids") or []
        image_captions = chunk.metadata.get("image_captions") or []
        for i, image_path in enumerate(chunk.images):
            image_id = image_ids[i] if i < len(image_ids) else chunk.metadata.get("image_id", "")
            caption = image_captions[i] if i < len(image_captions) else chunk.metadata.get("image_caption", "")
            companion_metadata = dict(chunk.metadata)
            companion_metadata.update(
                {
                    "image_id": image_id,
                    "image_ids": [image_id] if image_id else [],
                    "image_caption": caption,
                    "image_captions": [caption] if caption else [],
                    "has_image": True,
                    "row_kind": "image_companion",
                    "parent_chunk_id": chunk.chunk_id,
                }
            )
            companion_text = caption or f"image {image_id}".strip()
            out.append(
                Chunk(
                    chunk_id=f"{chunk.chunk_id}::img{i}",
                    text=companion_text,
                    images=[image_path],
                    metadata=companion_metadata,
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate-vectors chunk set (Stage 4 ablation).")
    parser.add_argument("--chunks", default="artifacts/chunks.jsonl")
    parser.add_argument("--output", default="artifacts/chunks_separate_vectors.jsonl")
    args = parser.parse_args()

    chunks = read_chunks_jsonl(args.chunks)
    out_chunks = build_separate_vectors_chunks(chunks)
    count = write_chunks_jsonl(out_chunks, args.output)
    with_images = sum(1 for c in out_chunks if c.images)
    companions = sum(1 for c in out_chunks if c.metadata.get("row_kind") == "image_companion")
    print(
        {
            "input_chunks": len(chunks),
            "output_rows": count,
            "with_images": with_images,
            "image_companion_rows": companions,
            "output": args.output,
        }
    )


if __name__ == "__main__":
    main()
