"""Canonical on-disk layout for one HiveMem run.

HiveMem build stages derive their paths from this module so experiments do not
grow ad-hoc mode/operation directory layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "RunLayout":
        return cls(Path(root))

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / ".checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def build_manifest(self) -> Path:
        return self.root / "build_manifest.json"

    def dataset(self, name: str) -> "DatasetLayout":
        return DatasetLayout(self.datasets_dir / name)

    def checkpoint(self, name: str) -> Path:
        return self.checkpoints_dir / name


@dataclass(frozen=True)
class DatasetLayout:
    root: Path

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def text_vectors(self) -> Path:
        return self.vectors_dir / "text.npy"

    @property
    def image_vectors(self) -> Path:
        return self.vectors_dir / "image.npy"

    @property
    def image_mask(self) -> Path:
        return self.vectors_dir / "image_mask.npy"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def build_stats(self) -> Path:
        return self.reports_dir / "build.json"

    @property
    def edges_manifest(self) -> Path:
        return self.reports_dir / "edges.json"

    @property
    def conflict_candidates(self) -> Path:
        return self.reports_dir / "conflicts.json"

    @property
    def build_trace(self) -> Path:
        return self.traces_dir / "build.jsonl"

    @property
    def edge_progress(self) -> Path:
        return self.traces_dir / "edges.jsonl"

    def existing_vector_path(self, filename: str, legacy_filename: str) -> Path:
        """Prefer the new vectors/ layout while accepting historical banks."""
        path = self.vectors_dir / filename
        return path if path.exists() else self.root / legacy_filename
