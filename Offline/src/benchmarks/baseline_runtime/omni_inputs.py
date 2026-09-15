"""Benchmark-native observations for the OmniSimpleMem protocol bridge."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Iterable

from embedding.chunk_builder import (
    Chunk,
    build_h2h_chunks_from_directory,
    build_wma_chunks_from_data,
)


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _memgallery_image_path(data_dir: Path, raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if raw.startswith("../image/"):
        return str((data_dir / "image" / raw.removeprefix("../image/")).resolve())
    return str((data_dir / raw).resolve())


def build_omni_memgallery_chunks(
    dataset: dict[str, Any], data_dir: str | Path, dataset_name: str
) -> list[Chunk]:
    """Map each Mem-Gallery round to one source-grounded Omni observation."""
    root = Path(data_dir)
    profile = dataset.get("character_profile") or {}
    speaker = f"user ({profile.get('name')})" if profile.get("name") else "user"
    chunks: list[Chunk] = []
    for session in dataset.get("multi_session_dialogues", []) or []:
        session_id = str(session.get("session_id") or "")
        timestamp = str(session.get("date") or "")
        for offset, dialogue in enumerate(session.get("dialogues", []) or [], start=1):
            dialogue_id = str(dialogue.get("round") or f"{session_id}:{offset}")
            image_ids = [str(value) for value in _values(dialogue.get("image_id")) if value]
            captions = [
                str(value) for value in _values(dialogue.get("image_caption")) if value
            ]
            images = [
                _memgallery_image_path(root, str(value))
                for value in _values(dialogue.get("input_image"))
                if value
            ]
            lines = [
                f"{speaker}: {str(dialogue.get('user') or '')}",
                f"assistant: {str(dialogue.get('assistant') or '')}",
            ]
            lines.extend(f"image_caption: {caption}" for caption in captions)
            chunks.append(
                Chunk(
                    chunk_id=f"{dataset_name}:{dialogue_id}",
                    text="\n".join(lines),
                    images=images,
                    metadata={
                        "benchmark": "memgallery",
                        "dataset": dataset_name,
                        "session_id": session_id,
                        "dialogue_id": dialogue_id,
                        "timestamp": timestamp,
                        "image_ids": image_ids,
                        "image_captions": captions,
                        "omni_input_mode": "multimodal_source_round",
                    },
                )
            )
    return chunks


def _source_turn_chunks(
    parent_chunks: Iterable[Chunk], *, benchmark: str
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for parent in parent_chunks:
        for source_turn in parent.metadata.get("m2a_turns") or []:
            turn_id = str(source_turn["turn_id"])
            speaker = str(source_turn.get("speaker") or source_turn.get("role") or "")
            text = str(source_turn.get("text") or "")
            images = [str(value) for value in source_turn.get("images") or [] if value]
            image_ids = [Path(value).name for value in images]
            chunks.append(
                Chunk(
                    chunk_id=turn_id,
                    text=f"{speaker}: {text}" if speaker else text,
                    images=images,
                    metadata={
                        "benchmark": benchmark,
                        "dataset": parent.metadata.get("dataset", ""),
                        "conversation_id": parent.metadata.get("conversation_id", ""),
                        "session_id": parent.metadata.get("session_id", ""),
                        "dialogue_id": turn_id,
                        "source_dialogue_id": source_turn.get("source_dialogue_id", ""),
                        "timestamp": source_turn.get("timestamp", ""),
                        "speaker": speaker,
                        "role": source_turn.get("role", ""),
                        "image_ids": image_ids,
                        "omni_input_mode": "multimodal_source_turn",
                    },
                )
            )
    return chunks


def build_omni_h2h_chunks_from_directory(
    data_dir: str | Path, *, variant: str, conversation_id: str
) -> list[Chunk]:
    parents = build_h2h_chunks_from_directory(
        data_dir,
        variant=variant,
        conversation_ids={conversation_id},
        include_previous_summary=False,
    )
    return _source_turn_chunks(parents, benchmark="h2hmem")


def build_omni_wma_chunks_from_data(
    sample: dict[str, Any], data_dir: str | Path, *, sample_path: str | Path
) -> list[Chunk]:
    parents = build_wma_chunks_from_data(
        sample,
        data_dir,
        sample_path=sample_path,
        include_previous_summary=False,
    )
    return _source_turn_chunks(parents, benchmark="worldmemarena")


def omni_input_manifest(source: str) -> dict[str, Any]:
    path = Path(__file__).resolve()
    normalized = source.strip().casefold()
    functions = (
        (_values, _memgallery_image_path, build_omni_memgallery_chunks)
        if normalized == "memgallery"
        else (_source_turn_chunks, build_omni_h2h_chunks_from_directory)
        if normalized.startswith("h2hmem_")
        else (_source_turn_chunks, build_omni_wma_chunks_from_data)
    )
    mapping_source = "\n".join(inspect.getsource(function) for function in functions)
    return {
        "source": source,
        "path": str(path),
        "mapping_sha256": hashlib.sha256(mapping_source.encode("utf-8")).hexdigest(),
        "format": "benchmark-native source observations",
        "shared_fixed_chunks": False,
    }


def omni_conformance_manifest(benchmark: str) -> dict[str, Any]:
    normalized = benchmark.strip().casefold()
    return {
        "official_core_used": True,
        "official_benchmark_entry_available": normalized == "memgallery",
        "official_benchmark_entry_used": False,
        "protocol_bridge": "Offline source observation to official Omni core",
        "ingest_path": [
            "OmniMemoryOrchestrator.add_text",
            "OmniMemoryOrchestrator.add_image",
        ],
        "retrieval_path": "OmniMemoryOrchestrator.query",
        "answer_path": "benchmark QA prompt",
        "orchestrator_answer_used": False,
        "fixed_top_k": 7,
        "modality_hard_filter_disabled": True,
        "modality_filter_policy": (
            "Record upstream lexical modality hints but do not apply them as "
            "post-retrieval hard filters over the unified Qwen3-VL embedding space."
        ),
        "visual_hard_filter_disabled": True,
        "visual_filter_policy": (
            "Backward-compatible alias of modality_hard_filter_disabled."
        ),
        "force_ingest": False,
        "adapter_silent_fallback": False,
        "graph_entity_flow_enabled": True,
        "shared_fixed_chunks": False,
        "upstream_add_multimodal_used": False,
        "upstream_add_multimodal_reason": (
            "Pinned upstream MAULinks.from_dict preserves related=null, causing "
            "add_multimodal to fail in add_related; upstream source is not patched."
        ),
        "upstream_entity_error_policy": (
            "Preserved: EntityExtractor logs invalid JSON and returns an empty extraction."
        ),
    }
