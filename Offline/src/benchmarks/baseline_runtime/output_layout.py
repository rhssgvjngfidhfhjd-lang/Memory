from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    def state_root(self, override: str | Path = "") -> Path:
        return Path(override) if override else self.datasets_dir
