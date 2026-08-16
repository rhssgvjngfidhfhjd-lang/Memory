#!/usr/bin/env python3
"""脚本2：对一条 chunk 走一遍真实建库流程（vLLM 生成 + 真实 embedder 编码 +
入库），把产生的 MAU 输出到 outputs/temp/。

与全量建库唯一的差别是只处理一条 chunk、不落正式库目录。
模型与服务地址读取 configs/defaults.json（executor_model / executor_base_url）。

Usage:
  python scripts/test_build_one_mau.py                   # 第 0 条 chunk
  python scripts/test_build_one_mau.py --chunk-index 40
  python scripts/test_build_one_mau.py --chunk-index 40 --no-embed   # 跳过真实编码（快）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hive_mem.llm_client import LLMClient
from hive_mem.builder import load_events
from hive_mem.executor import MemoryExecutor
from hive_mem.executor import EXECUTOR_VISUAL_INPUTS, visual_input_uses_images
from hive_mem.mau import MAUBank

ROOT = Path(__file__).resolve().parents[1]


class _ZeroEmbedder:
    """--no-embed 用：跳过 GPU，向量置零（MAU 的 json 本来不含向量）。"""

    def embed_texts(self, texts, mode="context"):
        import numpy as np
        n = len(texts) if isinstance(texts, list) else 1
        return np.zeros((n, 2048), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        default=str(ROOT / "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl"),
    )
    parser.add_argument("--profiles", default=str(ROOT / "configs/profiles.json"))
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/temp"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--executor-visual-input",
        choices=EXECUTOR_VISUAL_INPUTS,
        default="",
        help="Override configs/defaults.json for this one-item build.",
    )
    parser.add_argument("--no-embed", action="store_true", help="跳过真实 embedding（不加载 GPU 模型）")
    args = parser.parse_args()

    config = json.loads((ROOT / "configs/defaults.json").read_text(encoding="utf-8"))
    visual_input = args.executor_visual_input or config.get(
        "executor_visual_input", "image"
    )
    events = load_events(args.chunks)
    event = events[args.chunk_index]
    profile = json.loads(Path(args.profiles).read_text(encoding="utf-8")).get(event.dataset, "")

    llm_client = LLMClient(
        model=config["executor_model"],
        api_base=config["executor_base_url"],
        api_key="EMPTY",
        temperature=0.0,
        max_new_tokens=1024,
    )
    if args.no_embed:
        embedder = _ZeroEmbedder()
    else:
        from embedding.qwen3_text_embedding import create_memory_embedder
        embedder = create_memory_embedder(
            model_name=config["embedding_model"],
            device=args.device,
            expected_dim=config["embedding_dim"],
        )

    executor = MemoryExecutor(llm_client, embedder)
    raw_response, actions = executor.execute(
        chunk_text=event.text,
        profile=profile,
        image_paths=event.image_paths,
        visual_input=visual_input,
    )

    bank = MAUBank()
    executor.apply_to_memory_bank(actions, bank, event_metadata=event.metadata)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = event.source_chunk_id.replace(":", "_")
    maus = [item.to_dict() for item in bank.memories]
    (out_dir / f"mau_{tag}.json").write_text(
        json.dumps(maus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / f"raw_response_{tag}.txt").write_text(raw_response, encoding="utf-8")

    print(f"=== chunk={event.source_chunk_id} dataset={event.dataset} ===")
    print(
        f"executor_visual_input={visual_input} "
        f"attached_images={len(event.image_paths) if visual_input_uses_images(visual_input) else 0}"
    )
    print(f"LLM 返回 {len(actions)} 个块，入库 {len(bank)} 条 MAU")
    for mau in maus:
        print(json.dumps(mau, ensure_ascii=False, indent=2))
    if not args.no_embed and len(bank):
        import numpy as np
        v = bank.memories[0].embedding
        print(f"embedding 校验: dim={v.shape[0]} L2norm={float(np.linalg.norm(v)):.4f}")
    print(f"=== 已写入 {out_dir}/mau_{tag}.json 和 raw_response_{tag}.txt ===")


if __name__ == "__main__":
    main()
