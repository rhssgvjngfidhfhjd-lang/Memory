"""Benchmark-native dialogue observations for the M3-Agent caption bridge.

M3-Agent's official input unit is a temporally ordered video clip.  These
builders preserve the benchmark's native dialogue round as that unit without
passing through the shared, expanded retrieval chunks.  The adapter consumes
``metadata.m3_observation`` and treats ``Chunk`` only as the worker protocol
envelope.
"""

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


def _resolve_memgallery_image(data_dir: Path, raw: str) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    if raw.startswith("../image/"):
        return str((data_dir / "image" / raw.removeprefix("../image/")).resolve())
    return str((data_dir / raw).resolve())


def _observation_text(observation: dict[str, Any]) -> str:
    lines = [
        f"Timestamp: {observation.get('timestamp', '')}",
        f"Session: {observation.get('session_id', '')}",
        f"Dialogue round: {observation.get('dialogue_id', '')}",
    ]
    for turn in observation.get("turns") or []:
        speaker = str(turn.get("speaker") or turn.get("role") or "speaker")
        lines.append(f"{speaker}: {str(turn.get('text') or '')}")
    for image in observation.get("images") or []:
        lines.append(f"Image ID: {str(image.get('image_id') or '')}")
        caption = str(image.get("caption") or "")
        if caption:
            lines.append(f"Image annotation: {caption}")
    return "\n".join(lines)


def _make_chunk(
    *,
    benchmark: str,
    dataset: str,
    session_id: str,
    dialogue_id: str,
    timestamp: str,
    turns: list[dict[str, Any]],
    image_paths: list[str],
    image_ids: list[str],
    captions: list[str],
    extra: dict[str, Any] | None = None,
) -> Chunk:
    width = max(len(image_paths), len(image_ids), len(captions), 0)
    paths = image_paths + [""] * (width - len(image_paths))
    ids = image_ids + [""] * (width - len(image_ids))
    texts = captions + [""] * (width - len(captions))
    images = [
        {"path": path, "image_id": image_id, "caption": caption}
        for path, image_id, caption in zip(paths, ids, texts)
        if path or image_id or caption
    ]
    observation = {
        "schema_version": 1,
        "input_mode": "dialogue_round_as_clip",
        "benchmark": benchmark,
        "dataset": dataset,
        "session_id": session_id,
        "dialogue_id": dialogue_id,
        "timestamp": timestamp,
        "turns": turns,
        "images": images,
        **dict(extra or {}),
    }
    return Chunk(
        chunk_id=f"m3:{dataset}:{dialogue_id}",
        text=_observation_text(observation),
        images=[str(row["path"]) for row in images if row.get("path")],
        metadata={
            "benchmark": benchmark,
            "dataset": dataset,
            "session_id": session_id,
            "dialogue_id": dialogue_id,
            "timestamp": timestamp,
            "image_ids": [str(row["image_id"]) for row in images if row.get("image_id")],
            "image_captions": [str(row["caption"]) for row in images if row.get("caption")],
            "m3_observation": observation,
            "m3_input_mode": "benchmark_native_dialogue_round",
        },
    )


def build_m3_memgallery_chunks(
    dataset: dict[str, Any], data_dir: str | Path, dataset_name: str
) -> list[Chunk]:
    root = Path(data_dir)
    profile = dataset.get("character_profile") or {}
    user_name = str(profile.get("name") or "user")
    chunks: list[Chunk] = []
    for session in dataset.get("multi_session_dialogues", []) or []:
        session_id = str(session.get("session_id") or "")
        timestamp = str(session.get("date") or "")
        for offset, dialogue in enumerate(session.get("dialogues", []) or [], start=1):
            dialogue_id = str(dialogue.get("round") or f"{session_id}:R{offset:04d}")
            raw_paths = [str(value) for value in _values(dialogue.get("input_image")) if value]
            chunks.append(
                _make_chunk(
                    benchmark="memgallery",
                    dataset=dataset_name,
                    session_id=session_id,
                    dialogue_id=dialogue_id,
                    timestamp=timestamp,
                    turns=[
                        {"role": "user", "speaker": user_name, "text": str(dialogue.get("user") or "")},
                        {"role": "assistant", "speaker": "assistant", "text": str(dialogue.get("assistant") or "")},
                    ],
                    image_paths=[_resolve_memgallery_image(root, value) for value in raw_paths],
                    image_ids=[str(value) for value in _values(dialogue.get("image_id")) if value],
                    captions=[str(value) for value in _values(dialogue.get("image_caption")) if value],
                    extra={"profile_name": user_name},
                )
            )
    return chunks


def build_m3_chunks_from_round_chunks(
    parent_chunks: Iterable[Chunk], *, benchmark: str
) -> list[Chunk]:
    """Adapt round-level ``Chunk`` objects to M3's clip-observation protocol.

    This is the shared entry point for datasets whose loaders already preserve
    the original dialogue turns in ``metadata.m2a_turns``.  It deliberately
    rebuilds ``Chunk.text`` from those turns so baseline-specific expanded text
    (for example a previous-round summary) cannot leak into M3's observation.
    """
    chunks: list[Chunk] = []
    for parent in parent_chunks:
        turns = [
            {
                "turn_id": str(turn.get("turn_id") or ""),
                "role": str(turn.get("role") or ""),
                "speaker": str(turn.get("speaker") or turn.get("role") or ""),
                "text": str(turn.get("text") or ""),
                "timestamp": str(turn.get("timestamp") or ""),
            }
            for turn in parent.metadata.get("m2a_turns") or []
            if isinstance(turn, dict)
        ]
        source_ids = [
            str(value)
            for value in parent.metadata.get("source_dialogue_ids") or []
            if value
        ]
        dialogue_id = str(parent.metadata.get("dialogue_id") or parent.chunk_id)
        if not source_ids:
            source_ids = [dialogue_id]
        chunks.append(
            _make_chunk(
                benchmark=benchmark,
                dataset=str(parent.metadata.get("dataset") or ""),
                session_id=str(parent.metadata.get("session_id") or ""),
                dialogue_id=dialogue_id,
                timestamp=str(parent.metadata.get("timestamp") or parent.metadata.get("date") or ""),
                turns=turns,
                image_paths=[str(value) for value in parent.images if value],
                image_ids=[str(value) for value in parent.metadata.get("image_ids") or [] if value],
                captions=[str(value) for value in parent.metadata.get("image_captions") or [] if value],
                extra={
                    "conversation_id": str(parent.metadata.get("conversation_id") or ""),
                    "source_dialogue_ids": source_ids,
                },
            )
        )
    return chunks


def build_m3_h2h_chunks_from_directory(
    data_dir: str | Path, *, variant: str, conversation_id: str
) -> list[Chunk]:
    parents = build_h2h_chunks_from_directory(
        data_dir,
        variant=variant,
        conversation_ids={conversation_id},
        include_previous_summary=False,
    )
    return build_m3_chunks_from_round_chunks(parents, benchmark="h2hmem")


def build_m3_wma_chunks_from_data(
    sample: dict[str, Any], data_dir: str | Path, *, sample_path: str | Path
) -> list[Chunk]:
    parents = build_wma_chunks_from_data(
        sample,
        data_dir,
        sample_path=sample_path,
        include_previous_summary=False,
    )
    return build_m3_chunks_from_round_chunks(parents, benchmark="worldmemarena")


def m3_input_manifest(source: str) -> dict[str, Any]:
    normalized = source.strip().casefold()
    functions = (
        (_values, _resolve_memgallery_image, _observation_text, _make_chunk, build_m3_memgallery_chunks)
        if normalized == "memgallery"
        else (
            _observation_text,
            _make_chunk,
            build_m3_chunks_from_round_chunks,
            build_m3_h2h_chunks_from_directory,
        )
        if normalized.startswith("h2hmem_")
        else (_observation_text, _make_chunk, build_m3_chunks_from_round_chunks)
        if normalized in {"memeye", "memlens"}
        else (
            _observation_text,
            _make_chunk,
            build_m3_chunks_from_round_chunks,
            build_m3_wma_chunks_from_data,
        )
    )
    mapping_source = "\n".join(inspect.getsource(function) for function in functions)
    return {
        "source": source,
        "path": str(Path(__file__).resolve()),
        "mapping_sha256": hashlib.sha256(mapping_source.encode("utf-8")).hexdigest(),
        "format": "M3 benchmark-native dialogue observations",
        "input_unit": "one original dialogue round per M3 clip",
        "shared_fixed_chunks": False,
    }
