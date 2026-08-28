from __future__ import annotations

from typing import Any

from embedding.chunk_builder import Chunk


class ProvenanceIndex:
    """Small adapter-side map from backend ids to benchmark source ids."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        self._rows.clear()

    def register(self, memory_id: str, chunk: Chunk) -> None:
        metadata = chunk.metadata
        dialogue_id = str(metadata.get("dialogue_id") or chunk.chunk_id)
        session_id = str(metadata.get("session_id") or "")
        row = self._rows.setdefault(
            str(memory_id),
            {
                "session_id": session_id,
                "session_ids": [],
                "source_dialogue_ids": [],
                "image_ids": [],
                "image_paths": [],
            },
        )
        _append_unique(row["session_ids"], [session_id])
        _append_unique(row["source_dialogue_ids"], [dialogue_id])
        _append_unique(row["image_ids"], metadata.get("image_ids") or [])
        _append_unique(row["image_paths"], chunk.images)

    def merge(self, target_id: str, source_ids: list[str]) -> None:
        target = self._rows.setdefault(
            str(target_id),
            {
                "session_id": "",
                "session_ids": [],
                "source_dialogue_ids": [],
                "image_ids": [],
                "image_paths": [],
            },
        )
        for source_id in source_ids:
            source = self._rows.get(str(source_id)) or {}
            for key in ("source_dialogue_ids", "image_ids", "image_paths"):
                _append_unique(target[key], source.get(key) or [])
            _append_unique(
                target["session_ids"],
                source.get("session_ids") or [source.get("session_id", "")],
            )
            target["session_id"] = target["session_id"] or str(source.get("session_id") or "")

    def get(self, memory_id: str) -> dict[str, Any]:
        return dict(self._rows.get(str(memory_id)) or {})

    def visible(self, memory_id: str, session_ids: tuple[str, ...]) -> bool:
        if not session_ids:
            return True
        row = self.get(memory_id)
        memory_sessions = {
            str(value)
            for value in (
                row.get("session_ids") or [row.get("session_id", "")]
            )
            if value
        }
        return bool(memory_sessions.intersection(str(value) for value in session_ids))


def _append_unique(target: list[str], values: Any) -> None:
    known = set(target)
    for value in values:
        text = str(value)
        if text and text not in known:
            target.append(text)
            known.add(text)
