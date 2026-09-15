from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hive_mem.prefix_graph import materialize_prefix_graph
from hive_mem.retriever import (
    DEFAULT_HIVEMEM_GRAPH_OPTIONS,
    GraphExpandedIndex,
    MemoryHit,
    SimpleMemoryIndex,
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

RETRIEVAL_MODES = frozenset({"vector", "random_append", "graph_append"})


def resolve_retrieval_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a backwards-compatible, auditable retrieval configuration."""

    explicit_mode = str(config.get("retrieval_mode") or "").strip().lower()
    if explicit_mode and explicit_mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"retrieval_mode must be one of {sorted(RETRIEVAL_MODES)}, "
            f"got {explicit_mode!r}"
        )
    if not explicit_mode:
        explicit_mode = (
            "vector" if config.get("graph_options") is False else "graph_append"
        )

    vector_k = int(config.get("top_k", 0))
    if vector_k < 1:
        raise ValueError("top_k must be at least 1")
    graph_options = resolve_graph_options(config) if explicit_mode == "graph_append" else None
    append_k = (
        int(graph_options["append_k"])
        if graph_options is not None
        else int(config.get("random_append_k", 2))
        if explicit_mode == "random_append"
        else 0
    )
    retrieval_seed = int(config.get("retrieval_seed", config.get("seed", 0)))

    if explicit_mode == "graph_append" and vector_k != 5:
        raise ValueError("graph_append retrieval requires top_k=5")
    if explicit_mode == "random_append" and (vector_k != 5 or append_k != 2):
        raise ValueError("random_append retrieval requires top_k=5 and random_append_k=2")
    return {
        "mode": explicit_mode,
        "vector_k": vector_k,
        "append_k": append_k,
        "retrieval_seed": retrieval_seed,
        "graph_options": graph_options,
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
    resolve_retrieval_settings(config)


def question_retrieval_seed(
    global_seed: int,
    benchmark: str,
    manifest_question_id: str,
) -> int:
    payload = f"{int(global_seed)}\n{benchmark}\n{manifest_question_id}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def retrieve_hits(
    index: SimpleMemoryIndex,
    query_vector: list[float] | np.ndarray,
    settings: dict[str, Any],
    *,
    benchmark: str,
    manifest_question_id: str,
    category: str = "",
    allowed_session_ids: set[str] | None = None,
) -> tuple[list[MemoryHit], dict[str, Any]]:
    """Retrieve one question and return both hits and audit metadata."""

    vector_k = int(settings["vector_k"])
    hits = list(
        index.search(
            query_vector,
            top_k=vector_k,
            category=category,
            allowed_session_ids=allowed_session_ids,
        )
    )
    mode = str(settings["mode"])
    requested = int(settings["append_k"])
    per_question_seed: int | None = None
    if mode == "random_append":
        if not isinstance(index, SimpleMemoryIndex) or isinstance(index, GraphExpandedIndex):
            raise TypeError("random_append requires a SimpleMemoryIndex")
        per_question_seed = question_retrieval_seed(
            int(settings["retrieval_seed"]), benchmark, manifest_question_id
        )
        scores = index._scores(query_vector, category, allowed_session_ids)
        selected_ids = {str(hit.item.id) for hit in hits}
        eligible = [
            position
            for position, score in enumerate(scores)
            if np.isfinite(score)
            and str(index.bank.memories[position].id) not in selected_ids
        ]
        if len(eligible) < requested:
            raise ValueError(
                f"random_append requires {requested} eligible memories for "
                f"{manifest_question_id}, found {len(eligible)}"
            )
        selected = random.Random(per_question_seed).sample(eligible, requested)
        base_rank = len(hits)
        hits.extend(
            MemoryHit(
                item=index.bank.memories[position],
                score=float(scores[position]),
                rank=base_rank + offset,
                via="random",
            )
            for offset, position in enumerate(selected, start=1)
        )

    vector_ids = [str(hit.item.id) for hit in hits if hit.via == "vector"]
    appended = [hit for hit in hits if hit.via != "vector"]
    actual = len(appended)
    return hits, {
        "retrieval_mode": mode,
        "vector_k": vector_k,
        "append_k_requested": requested,
        "append_k_actual": actual,
        "retrieval_seed": per_question_seed,
        "retrieval_global_seed": (
            int(settings["retrieval_seed"]) if mode == "random_append" else None
        ),
        "retrieval_vector_ids": vector_ids,
        "retrieval_append_ids": [str(hit.item.id) for hit in appended],
        "retrieval_final_ids": [str(hit.item.id) for hit in hits],
        "append_shortfall_reason": (
            "no_eligible_graph_candidates"
            if mode == "graph_append" and actual < requested
            else ""
        ),
    }


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
    retrieval_mode: str = "",
    vector_k: int | None = None,
    append_k: int = 0,
    retrieval_seed: int | None = None,
) -> str:
    payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "retrieval_mode": retrieval_mode or ("graph_append" if options else "vector"),
        "vector_k": vector_k,
        "append_k": int(append_k),
        "retrieval_seed": retrieval_seed,
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
