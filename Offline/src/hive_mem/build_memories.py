from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from .llm_client import LLMClient
from .builder import MAUBuilder
from .builder import load_events
from .executor import EXECUTOR_VISUAL_INPUTS
from .output_layout import RunLayout
from .output_layout import DatasetLayout
from embedding.qwen3_text_embedding import create_memory_embedder
from embedding.openai_memory_embedder import OpenAIMemoryEmbedder
from benchmarks.io_utils import write_json_atomic



def apply_config_defaults(
    parser: argparse.ArgumentParser,
    *,
    allowed_keys: set[str] | None = None,
) -> None:
    """Overlay configs/defaults.json (project root) onto argparse defaults.
    CLI arguments still take precedence; unknown keys are ignored."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "defaults.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    known = {action.dest for action in parser._actions}
    if allowed_keys is not None:
        known &= allowed_keys
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
    expected_signature: dict | None = None,
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
    if expected_signature is not None and stats.get("build_signature") != expected_signature:
        return None
    return {**stats, "skipped_complete": True}


def build_signature(args: argparse.Namespace, dataset: str, events, profile: str) -> dict:
    event_payload = json.dumps(
        [event.to_dict() for event in events],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "dataset": dataset,
        "events_sha256": hashlib.sha256(event_payload).hexdigest(),
        "profile_sha256": hashlib.sha256(profile.encode("utf-8")).hexdigest(),
        "mode": args.mode,
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_max_tokens": args.executor_max_tokens,
        "executor_visual_input": args.executor_visual_input,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "embedding_dim": args.embedding_dim,
        "dtype": args.dtype,
        "build_image_vectors": args.mode == "c",
    }

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
    parser.add_argument("--executor-max-tokens", type=int, default=512)
    parser.add_argument("--executor-timeout", type=int, default=180)
    parser.add_argument("--executor-retries", type=int, default=2)
    parser.add_argument(
        "--executor-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent executor LLM requests (results commit in event order).",
    )
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
    parser.add_argument("--embedding-base-url", default="")
    parser.add_argument("--embedding-api-key", default="EMPTY")
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
    if args.embedding_base_url:
        embedder = OpenAIMemoryEmbedder(
            base_url=args.embedding_base_url,
            model_name=args.embedding_model,
            expected_dim=args.embedding_dim,
            api_key=args.embedding_api_key,
            timeout=args.executor_timeout,
        )
    else:
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
        if not events:
            raise ValueError(f"No input events found for dataset {dataset!r}")
        profile = profiles.get(dataset, "")
        signature = build_signature(args, dataset, events, profile)
        dataset_layout = layout.dataset(dataset)
        output_dir = dataset_layout.root
        if not args.no_resume and not args.max_events:
            existing = completed_dataset_stats(
                dataset_layout,
                layout.checkpoint(dataset),
                expected_events=len(events),
                expected_dim=args.embedding_dim,
                expected_executor_visual_input=args.executor_visual_input,
                expected_signature=signature,
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
            profile=profile,
            executor_visual_input=args.executor_visual_input,
            executor_concurrency=args.executor_concurrency,
            checkpoint_signature=signature,
        )
        print(json.dumps({dataset: summaries[dataset]}, ensure_ascii=False))
    manifest_path = layout.build_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    public_manifest = {
        key: value
        for key, value in vars(args).items()
        if key not in {"executor_api_key", "embedding_api_key"}
    }
    write_json_atomic(manifest_path, public_manifest)
    # H2HMEM builds dyadic and multiparty banks into the same root. Preserve
    # both build configurations instead of letting the second overwrite the
    # only audit record.
    write_json_atomic(
        layout.root / f"build_manifest.{Path(chunks_path).stem}.json",
        public_manifest,
    )


if __name__ == "__main__":
    main()
