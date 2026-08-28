"""Stage 2: embed baseline memory banks with Qwen3-VL-Embedding-2B.

Reads ``<memory-root>/datasets/<dataset>/memories.jsonl`` (written by
``WorldMemArena/eval_framework/scripts/build_memgallery_baselines.py``) and
writes ``vectors/text.npy`` so that
``benchmarks.memgallery_harness.eval_memgallery`` can load the directory as a
MAUBank index. Memories whose metadata carries ``image_paths`` also get
``vectors/image.npy`` + ``vectors/image_mask.npy`` (VS/VR max-fusion, same as mode c).

Run from anywhere (package installed with pip -e)::

    python scripts/embed_baseline_memories.py \
        --memory-root outputs/baseline_memories/amem --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embedding.qwen3_text_embedding import create_memory_embedder
from hive_mem.output_layout import DatasetLayout


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embed_dataset(
    dataset_dir: Path,
    embedder,
    *,
    image_vectors: bool,
    force: bool,
) -> dict:
    rows = read_jsonl(dataset_dir / "memories.jsonl")
    layout = DatasetLayout(dataset_dir)
    layout.vectors_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = layout.text_vectors
    if vectors_path.exists() and not force:
        existing = np.load(vectors_path)
        if len(existing) == len(rows):
            return {"dataset": dataset_dir.name, "memories": len(rows), "skipped": True}

    started = time.time()
    vectors = np.zeros((0, embedder.expected_dim), dtype=np.float32)
    if rows:
        contents = [str(row.get("content", "")) for row in rows]
        vectors = embedder.embed_texts(contents, mode="context")
    np.save(vectors_path, vectors.astype(np.float32))

    image_rows = 0
    if image_vectors and embedder.supports_images:
        paths = [
            (row.get("metadata") or {}).get("image_paths") or [] for row in rows
        ]
        mask = np.array([bool(p) for p in paths], dtype=bool)
        if mask.any():
            image_matrix = np.zeros((len(rows), embedder.expected_dim), dtype=np.float32)
            for index, row_paths in enumerate(paths):
                if row_paths:
                    image_matrix[index] = embedder.embed_images([row_paths[0]])[0]
            np.save(layout.image_vectors, image_matrix)
            np.save(layout.image_mask, mask)
            image_rows = int(mask.sum())

    return {
        "dataset": dataset_dir.name,
        "memories": len(rows),
        "image_rows": image_rows,
        "elapsed_sec": round(time.time() - started, 1),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed baseline memory banks.")
    parser.add_argument("--memory-root", required=True, help="e.g. outputs/baseline_memories/amem")
    parser.add_argument("--datasets", default="", help="Comma-separated subset; empty = all")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-dim", type=int, default=2048)
    parser.add_argument("--image-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Re-embed even if vectors exist")
    args = parser.parse_args()

    root = Path(args.memory_root) / "datasets"
    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    dataset_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "memories.jsonl").exists() and (not wanted or d.name in wanted)
    )
    if not dataset_dirs:
        raise SystemExit(f"No dataset directories with memories.jsonl under {root}")

    embedder = create_memory_embedder(
        model_name=args.model, device=args.device, expected_dim=args.expected_dim
    )
    summary = []
    for pos, dataset_dir in enumerate(dataset_dirs, start=1):
        stats = embed_dataset(
            dataset_dir, embedder, image_vectors=args.image_vectors, force=args.force
        )
        summary.append(stats)
        print(f"[{pos}/{len(dataset_dirs)}] {json.dumps(stats, ensure_ascii=False)}", flush=True)

    manifest = {
        "model": args.model,
        "expected_dim": args.expected_dim,
        "datasets": summary,
        "total_memories": sum(s["memories"] for s in summary),
    }
    (Path(args.memory_root) / "embed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[embed] wrote {Path(args.memory_root) / 'embed_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
