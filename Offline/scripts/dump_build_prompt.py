#!/usr/bin/env python3
"""脚本1：输出真实的建库 prompt（与建库流程逐字节一致），写入 outputs/temp/。

Usage:
  python scripts/dump_build_prompt.py                  # 第 0 条 chunk
  python scripts/dump_build_prompt.py --chunk-index 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hive_mem.executor import MemoryExecutor
from hive_mem.executor import EXECUTOR_VISUAL_INPUTS

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        default=str(ROOT / "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl"),
    )
    parser.add_argument("--profiles", default=str(ROOT / "configs/profiles.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/temp"))
    parser.add_argument(
        "--executor-visual-input",
        choices=EXECUTOR_VISUAL_INPUTS,
        default="",
        help="Override configs/defaults.json when rendering the prompt.",
    )
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/defaults.json").read_text(encoding="utf-8"))
    visual_input = args.executor_visual_input or config.get(
        "executor_visual_input", "image"
    )
    chunks = [json.loads(line) for line in open(args.chunks, encoding="utf-8")]
    chunk = chunks[args.chunk_index]
    dataset = chunk["metadata"]["dataset"]
    profile = json.loads(Path(args.profiles).read_text(encoding="utf-8")).get(dataset, "")

    executor = MemoryExecutor(llm_client=None, embedder=None)
    image_paths = [str(path) for path in (chunk.get("images") or []) if str(path)]
    prepared_chunk = executor.prepare_chunk_text(chunk["text"], visual_input)
    prompt = executor._build_prompt(
        prepared_chunk,
        profile=profile,
        has_images=visual_input == "image" and bool(image_paths),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt_{chunk['chunk_id'].replace(':', '_')}.txt"
    out_path.write_text(prompt, encoding="utf-8")

    print(prompt)
    print(
        f"=== executor_visual_input={visual_input} | "
        f"attached_images={len(image_paths) if visual_input == 'image' else 0} ==="
    )
    if visual_input == "image":
        for image_path in image_paths:
            print(f"=== image={image_path} ===")
    print(f"=== chunk={chunk['chunk_id']} | 总长 {len(prompt)} 字符 ≈ {len(prompt) // 4} tokens ===")
    print(f"=== 已写入 {out_path} ===")


if __name__ == "__main__":
    main()
