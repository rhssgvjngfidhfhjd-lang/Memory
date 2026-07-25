from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import OpenAICompatibleBackend
from .memgallery.builder import MemGalleryMemoryBuilder
from .memgallery.input_loader import load_events
from .memgallery.qwen_embedder import QwenMemoryEmbedder
from .operations import get_operations


MODE_CHUNKS = {
    "a": "artifacts/chunks_text_only.jsonl",
    "b": "artifacts/chunks_text_caption.jsonl",
    "c": "artifacts/chunks.jsonl",
}
MODE_NAMES = {"a": "a_text_only", "b": "b_text_caption", "c": "c_text_caption_image"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgentMem memories for Mem-Gallery.")
    parser.add_argument("--mode", default="c", choices=["a", "b", "c"])
    parser.add_argument("--chunks", default="")
    parser.add_argument("--dataset", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--output-root", default="artifacts/agentmem")
    parser.add_argument("--executor-model", required=True)
    parser.add_argument("--executor-base-url", required=True)
    parser.add_argument("--executor-api-key", default="EMPTY")
    parser.add_argument("--executor-max-tokens", type=int, default=1024)
    parser.add_argument("--executor-timeout", type=int, default=180)
    parser.add_argument("--executor-retries", type=int, default=2)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--operations",
        default="insert",
        help="Comma-separated operations to enable. Default 'insert' (insert-only); "
        "full set: 'insert,update,delete,noop'.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    chunks_path = args.chunks or MODE_CHUNKS[args.mode]
    all_events = load_events(chunks_path)
    datasets = sorted({event.dataset for event in all_events}) if args.all_datasets else [args.dataset]
    backend = OpenAICompatibleBackend(
        model=args.executor_model,
        api_base=args.executor_base_url,
        api_key=args.executor_api_key,
        temperature=0.0,
        max_new_tokens=args.executor_max_tokens,
        max_retries=args.executor_retries + 1,
        timeout=args.executor_timeout,
    )
    embedder = QwenMemoryEmbedder(
        model_name=args.embedding_model,
        device=args.device,
        expected_dim=args.embedding_dim,
        dtype=args.dtype,
    )
    operations = get_operations(args.operations.split(","))
    builder = MemGalleryMemoryBuilder(backend, embedder, top_k=args.top_k, operations=operations)
    summaries = {}
    for dataset in datasets:
        events = [event for event in all_events if event.dataset == dataset]
        output_dir = Path(args.output_root) / MODE_NAMES[args.mode] / "datasets" / dataset
        summaries[dataset] = builder.build(
            events,
            output_dir,
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
            max_events=args.max_events,
            build_image_vectors=args.mode == "c",
        )
        print(json.dumps({dataset: summaries[dataset]}, ensure_ascii=False))
    manifest_path = Path(args.output_root) / MODE_NAMES[args.mode] / "build_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
