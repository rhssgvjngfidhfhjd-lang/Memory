from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .llm_client import LLMClient
from .builder import MAUBuilder
from .builder import load_events
from .executor import EXECUTOR_VISUAL_INPUTS
from .output_layout import RunLayout
from .output_layout import DatasetLayout
from embedding.qwen3_text_embedding import create_memory_embedder



def apply_config_defaults(parser: argparse.ArgumentParser) -> None:
    """Overlay configs/defaults.json (project root) onto argparse defaults.
    CLI arguments still take precedence; unknown keys are ignored."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "defaults.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    known = {action.dest for action in parser._actions}
    parser.set_defaults(
        **{k: v for k, v in config.items() if not k.startswith("_") and k in known}
    )


def completed_dataset_stats(
    dataset_layout: DatasetLayout,
    checkpoint_dir: Path,
    *,
    expected_events: int,
    expected_dim: int,
    expected_executor_visual_input: str = "image",
) -> dict | None:
    """Return stats only for a complete build made with compatible settings."""
    if (checkpoint_dir / "builder_state.json").exists():
        return None
    memories_path = dataset_layout.root / "memories.jsonl"
    if not dataset_layout.build_stats.exists() or not memories_path.exists():
        return None
    try:
        stats = json.loads(dataset_layout.build_stats.read_text(encoding="utf-8"))
        vectors = np.load(dataset_layout.text_vectors, mmap_mode="r")
        with memories_path.open(encoding="utf-8") as handle:
            memory_count = sum(1 for line in handle if line.strip())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if int(stats.get("input_events", -1)) != int(expected_events):
        return None
    if vectors.shape != (memory_count, int(expected_dim)):
        return None
    if int(stats.get("final_memories", -1)) != memory_count:
        return None
    # Builds made before this field existed used captions and no raw images.
    stored_visual_input = str(stats.get("executor_visual_input") or "caption")
    if stored_visual_input != expected_executor_visual_input:
        return None
    return {**stats, "skipped_complete": True}

MODE_CHUNKS = {
    # a/b ablation chunks are not currently generated; regenerate with
    # scripts/build_chunks.py if needed. Mode c uses the profile-free chunks
    # (persona is injected via --profiles-file instead).
    "a": "data/qwen3_vl_embedding_2b/chunks_text_only.jsonl",
    "b": "data/qwen3_vl_embedding_2b/chunks_text_caption.jsonl",
    "c": "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl",
}
def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgentMem memories for Mem-Gallery.")
    parser.add_argument("--mode", default="c", choices=["a", "b", "c"])
    parser.add_argument("--chunks", default="")
    parser.add_argument("--dataset", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument(
        "--output-root",
        default="outputs/hive_mem",
        help="Run directory; datasets are written directly under <run>/datasets.",
    )
    parser.add_argument("--executor-model", default="")
    parser.add_argument("--executor-base-url", default="")
    parser.add_argument("--executor-api-key", default="EMPTY")
    parser.add_argument("--executor-max-tokens", type=int, default=1024)
    parser.add_argument("--executor-timeout", type=int, default=180)
    parser.add_argument("--executor-retries", type=int, default=2)
    parser.add_argument(
        "--executor-visual-input",
        choices=EXECUTOR_VISUAL_INPUTS,
        default="image",
        help=(
            "Build evidence: raw image only, caption only, or both raw image and "
            "caption."
        ),
    )
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--profiles-file",
        default="",
        help="JSON file mapping dataset name -> profile_summary text. When set, the "
        "profile is injected into the executor prompt per dataset (use with "
        "profile-free chunks).",
    )
    apply_config_defaults(parser)
    args = parser.parse_args()
    if not args.executor_model or not args.executor_base_url:
        parser.error("--executor-model/--executor-base-url required (set them or configs/defaults.json)")

    profiles: dict[str, str] = {}
    if args.profiles_file:
        profiles = json.loads(Path(args.profiles_file).read_text(encoding="utf-8"))

    chunks_path = args.chunks or MODE_CHUNKS[args.mode]
    layout = RunLayout.from_path(args.output_root)
    all_events = load_events(chunks_path)
    datasets = sorted({event.dataset for event in all_events}) if args.all_datasets else [args.dataset]
    llm_client = LLMClient(
        model=args.executor_model,
        api_base=args.executor_base_url,
        api_key=args.executor_api_key,
        temperature=0.0,
        max_new_tokens=args.executor_max_tokens,
        max_retries=args.executor_retries + 1,
        timeout=args.executor_timeout,
    )
    embedder = create_memory_embedder(
        model_name=args.embedding_model,
        device=args.device,
        expected_dim=args.embedding_dim,
        dtype=args.dtype,
    )
    builder = MAUBuilder(llm_client, embedder)
    summaries = {}
    for dataset in datasets:
        events = [event for event in all_events if event.dataset == dataset]
        dataset_layout = layout.dataset(dataset)
        output_dir = dataset_layout.root
        if not args.no_resume and not args.max_events:
            existing = completed_dataset_stats(
                dataset_layout,
                layout.checkpoint(dataset),
                expected_events=len(events),
                expected_dim=args.embedding_dim,
                expected_executor_visual_input=args.executor_visual_input,
            )
            if existing is not None:
                summaries[dataset] = existing
                print(json.dumps({dataset: existing}, ensure_ascii=False))
                continue
        summaries[dataset] = builder.build(
            events,
            output_dir,
            checkpoint_dir=layout.checkpoint(dataset),
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
            max_events=args.max_events,
            build_image_vectors=(args.mode == "c" and embedder.supports_images),
            profile=profiles.get(dataset, ""),
            executor_visual_input=args.executor_visual_input,
        )
        print(json.dumps({dataset: summaries[dataset]}, ensure_ascii=False))
    manifest_path = layout.build_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
