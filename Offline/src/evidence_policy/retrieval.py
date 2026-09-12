from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from hive_mem.prefix_graph import materialize_prefix_graph
from hive_mem.retriever import (
    DEFAULT_HIVEMEM_GRAPH_OPTIONS,
    GraphExpandedIndex,
    MemoryHit,
)


GRAPH_OPTION_KEYS = {
    "seed_k",
    "expansion_bonus",
    "mode",
    "append_k",
    "expand_temporal",
    "expand_related",
    "expand_entity",
    "expand_attribute",
    "related_types",
    "df_max",
    "df_stop",
    "min_shared",
    "degree_cap",
}


def resolve_graph_options(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the shared HiveMem graph defaults plus explicit PPO overrides.

    Graph retrieval is enabled when ``graph_options`` is absent.  Setting it to
    ``false`` remains available for controlled vector-only ablations.
    """

    raw = config.get("graph_options")
    if raw is False:
        return None
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("graph_options must be an object or false")
    options = {**DEFAULT_HIVEMEM_GRAPH_OPTIONS, **dict(raw or {})}
    unknown = sorted(set(options) - GRAPH_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown graph_options: {', '.join(unknown)}")
    if options.get("mode") != "append":
        raise ValueError("Evidence-policy graph retrieval requires mode='append'")
    if int(options.get("append_k", 0)) != 2:
        raise ValueError("Evidence-policy graph retrieval requires append_k=2")
    return options


def validate_graph_config(config: dict[str, Any]) -> None:
    options = resolve_graph_options(config)
    if options is not None and int(config.get("top_k", 0)) != 5:
        raise ValueError("Evidence-policy graph retrieval requires vector top_k=5")


def build_graph_index(
    dataset_dir: str | Path,
    options: dict[str, Any],
    *,
    visual_categories: set[str] | None = None,
) -> GraphExpandedIndex:
    kwargs = dict(options)
    if visual_categories:
        kwargs["visual_categories"] = visual_categories
    return GraphExpandedIndex(dataset_dir, **kwargs)


def build_wma_prefix_graph_index(
    source_dataset_dir: str | Path,
    cache_root: str | Path,
    *,
    sample_id: str,
    checkpoint_id: str,
    visible_session_ids: Iterable[str],
    options: dict[str, Any],
    visual_categories: set[str] | None = None,
) -> tuple[GraphExpandedIndex, str]:
    checkpoint_root = Path(cache_root) / sample_id / checkpoint_id
    prefix_root = materialize_prefix_graph(
        source_dataset_dir,
        checkpoint_root,
        sample_id=sample_id,
        checkpoint_id=checkpoint_id,
        visible_session_ids=tuple(visible_session_ids),
        graph_options=options,
    )
    manifest = json.loads(
        (prefix_root / "prefix_manifest.json").read_text(encoding="utf-8")
    )
    index = build_graph_index(
        prefix_root / "datasets" / sample_id,
        options,
        visual_categories=visual_categories,
    )
    return index, str(manifest["signature"])


def retrieval_signature(
    dataset_dir: str | Path,
    options: dict[str, Any] | None,
    *,
    prefix_signature: str = "",
) -> str:
    payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "graph_options": options,
        "prefix_signature": prefix_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def retrieval_trace(hits: Iterable[MemoryHit]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": str(hit.item.id),
            "rank": int(hit.rank),
            "score": float(hit.score),
            "via": str(hit.via),
            "session_id": str(hit.item.metadata.get("session_id", "")),
            "source_dialogue_ids": [
                str(value)
                for value in hit.item.metadata.get("source_dialogue_ids", [])
            ],
        }
        for hit in hits
    ]
