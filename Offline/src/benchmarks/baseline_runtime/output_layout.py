from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BaselineOutputLayout:
    root: Path

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"

    @property
    def datasets_dir(self) -> Path:
        return self.memory_dir / "datasets"

    @property
    def snapshot(self) -> Path:
        return self.memory_dir / "memory_snapshot.jsonl"

    @property
    def pipeline_qa(self) -> Path:
        return self.root / "pipeline_qa.jsonl"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / ".checkpoint"

    @property
    def sample_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir / "samples"

    def state_root(self, override: str | Path = "") -> Path:
        return Path(override) if override else self.datasets_dir


def load_hivemem_snapshot(
    index_root: str | Path,
    sample_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Normalize HiveMem's native JSONL banks into the shared snapshot schema."""
    root = Path(index_root) / "datasets"
    if not root.is_dir():
        return []
    selected = {str(value) for value in sample_ids} if sample_ids is not None else None
    rows: list[dict] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected is not None and dataset_dir.name not in selected:
            continue
        memories_path = dataset_dir / "memories.jsonl"
        if not memories_path.is_file():
            continue
        with memories_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                memory = json.loads(line)
                metadata = dict(memory.get("metadata") or {})
                rows.append(
                    {
                        "memory_id": str(memory.get("memory_id") or memory.get("id") or ""),
                        "text": str(memory.get("content") or memory.get("summary") or ""),
                        "session_id": str(metadata.get("session_id") or ""),
                        "source_dialogue_ids": [
                            str(value) for value in metadata.get("source_dialogue_ids") or []
                        ],
                        "image_ids": [str(value) for value in metadata.get("image_ids") or []],
                        "image_paths": [
                            str(value) for value in metadata.get("image_paths") or []
                        ],
                        "backend_type": "hivemem",
                        "metadata": {**metadata, "dataset": dataset_dir.name},
                    }
                )
    return rows
