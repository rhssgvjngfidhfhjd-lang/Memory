from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VPPrimitive:
    vp_id: str
    label: str
    crop_path: Path
    bbox_norm: tuple[int, int, int, int]


@dataclass(frozen=True)
class VPImageRecord:
    image_id: str
    dataset: str
    relative_path: str
    source_sha256: str
    status: str
    primitives: tuple[VPPrimitive, ...]


class VPArtifactIndex:
    """Read-only lookup over one vp_extractor artifact run."""

    def __init__(self, run_dir: str | Path, *, max_vps_per_image: int = 0):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if max_vps_per_image < 0:
            raise ValueError("max_vps_per_image must be non-negative")
        self.max_vps_per_image = int(max_vps_per_image)
        self.run_path = self.run_dir / "run.json"
        self.images_path = self.run_dir / "exports" / "images.jsonl"
        if not self.run_path.is_file():
            raise FileNotFoundError(f"Missing VP run metadata: {self.run_path}")
        if not self.images_path.is_file():
            raise FileNotFoundError(f"Missing VP image index: {self.images_path}")
        self.run_metadata: dict[str, Any] = json.loads(
            self.run_path.read_text(encoding="utf-8")
        )
        self.run_id = str(self.run_metadata.get("run_id", ""))
        self.signature = self._signature()
        self._by_sha256: dict[str, VPImageRecord] = {}
        self._by_dataset_relative: dict[tuple[str, str], VPImageRecord] = {}
        self._by_basename: dict[str, list[VPImageRecord]] = {}
        self._path_cache: dict[tuple[str, str], VPImageRecord | None] = {}
        self._load()

    def _signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.run_path.read_bytes())
        digest.update(self.images_path.read_bytes())
        return digest.hexdigest()

    def _load(self) -> None:
        with self.images_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if str(raw.get("schema_version", "")) != "1.0":
                    raise ValueError(
                        f"Unsupported VP schema at {self.images_path}:{line_number}"
                    )
                source = raw.get("source") or {}
                dataset = str(source.get("dataset", "")).strip()
                relative = _normalize_path(source.get("relative_path", ""))
                source_sha256 = str(source.get("sha256", "")).lower()
                primitives: list[VPPrimitive] = []
                for item in raw.get("primitives", []) or []:
                    raw_crop_path = str(item.get("crop_path", "")).strip()
                    if not raw_crop_path:
                        raise ValueError(
                            f"Missing VP crop path at {self.images_path}:{line_number}"
                        )
                    crop_path = (self.run_dir / raw_crop_path).resolve()
                    bbox = tuple(int(value) for value in item.get("bbox_norm", []))
                    if len(bbox) != 4:
                        raise ValueError(
                            f"Invalid VP bbox at {self.images_path}:{line_number}"
                        )
                    primitives.append(
                        VPPrimitive(
                            vp_id=str(item.get("vp_id", "")),
                            label=str(item.get("label", "")),
                            crop_path=crop_path,
                            bbox_norm=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        )
                    )
                if self.max_vps_per_image:
                    primitives = primitives[: self.max_vps_per_image]
                record = VPImageRecord(
                    image_id=str(raw.get("image_id", "")),
                    dataset=dataset,
                    relative_path=relative,
                    source_sha256=source_sha256,
                    status=str(raw.get("status", "")),
                    primitives=tuple(primitives),
                )
                key = (dataset.lower(), relative)
                if not dataset or not relative or not record.image_id:
                    raise ValueError(
                        f"Incomplete VP record at {self.images_path}:{line_number}"
                    )
                if key in self._by_dataset_relative:
                    raise ValueError(f"Duplicate VP source record: {dataset}:{relative}")
                self._by_dataset_relative[key] = record
                if source_sha256:
                    self._by_sha256.setdefault(source_sha256, record)
                self._by_basename.setdefault(Path(relative).name.lower(), []).append(record)

    def record_for(
        self, image_path: str | Path, *, dataset: str | None = None
    ) -> VPImageRecord | None:
        raw_path = str(image_path)
        dataset_key = str(dataset or "").lower()
        cache_key = (dataset_key, raw_path)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        normalized = _normalize_path(raw_path)
        basename = Path(normalized).name.lower()
        blob_sha256 = (
            basename
            if len(basename) == 64
            and all(character in "0123456789abcdef" for character in basename)
            else ""
        )
        if blob_sha256:
            record = self._by_sha256.get(blob_sha256)
            if record is not None:
                self._path_cache[cache_key] = record
                return record
        candidates = self._by_basename.get(basename, ())
        suffix_matches = [
            row
            for row in candidates
            if (not dataset_key or row.dataset.lower() == dataset_key)
            and (normalized == row.relative_path or normalized.endswith("/" + row.relative_path))
        ]
        if len(suffix_matches) == 1:
            record = suffix_matches[0]
        else:
            path = Path(image_path)
            record = self._by_sha256.get(_file_sha256(path)) if path.is_file() else None
        self._path_cache[cache_key] = record
        return record

    def primitives_for(
        self, image_path: str | Path, *, dataset: str | None = None
    ) -> tuple[VPPrimitive, ...]:
        record = self.record_for(image_path, dataset=dataset)
        return record.primitives if record is not None else ()

    def has_primitives(self, image_path: str | Path, *, dataset: str | None = None) -> bool:
        return bool(self.primitives_for(image_path, dataset=dataset))

    def audit(self, image_paths: Iterable[str | Path]) -> dict[str, int]:
        total = matched = with_primitives = missing_crops = 0
        for image_path in dict.fromkeys(str(value) for value in image_paths):
            total += 1
            record = self.record_for(image_path)
            if record is None:
                continue
            matched += 1
            if record.primitives:
                with_primitives += 1
            missing_crops += sum(not row.crop_path.is_file() for row in record.primitives)
        return {
            "image_count": total,
            "matched_records": matched,
            "missing_records": total - matched,
            "with_primitives": with_primitives,
            "without_primitives": matched - with_primitives,
            "missing_crop_files": missing_crops,
        }


def _normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
