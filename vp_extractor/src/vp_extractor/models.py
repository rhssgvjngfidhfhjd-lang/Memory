from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


NormBox = tuple[int, int, int, int]
PixelBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 180
    retries: int = 1
    enable_thinking: bool = False
    max_primitives: int = 20
    dedup_iou: float = 0.6
    full_frame_area_threshold: float = 0.9
    enable_relocalization: bool = True
    min_box_side_norm: int = 8
    jpeg_quality: int = 95
    create_preview: bool = True
    output_root: str = "outputs"
    run_id: str = "qwen3vl4b_v1"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Settings":
        vlm = data.get("vlm", {})
        discovery = data.get("discovery", {})
        crop = data.get("crop", {})
        output = data.get("output", {})
        return cls(
            model=str(vlm.get("model", "Qwen/Qwen3-VL-4B-Instruct")),
            base_url=str(vlm.get("base_url", "http://127.0.0.1:18000/v1")),
            api_key_env=str(vlm.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(vlm.get("temperature", 0.0)),
            max_tokens=int(vlm.get("max_tokens", 1024)),
            timeout_seconds=int(vlm.get("timeout_seconds", 180)),
            retries=max(0, int(vlm.get("retries", 1))),
            enable_thinking=bool(vlm.get("enable_thinking", False)),
            max_primitives=max(1, int(discovery.get("max_primitives", 20))),
            dedup_iou=float(discovery.get("dedup_iou", 0.6)),
            full_frame_area_threshold=float(
                discovery.get("full_frame_area_threshold", 0.9)
            ),
            enable_relocalization=bool(discovery.get("enable_relocalization", True)),
            min_box_side_norm=max(1, int(discovery.get("min_box_side_norm", 8))),
            jpeg_quality=int(crop.get("jpeg_quality", 95)),
            create_preview=bool(crop.get("create_preview", True)),
            output_root=str(output.get("root", "outputs")),
            run_id=str(output.get("run_id", "qwen3vl4b_v1")),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "vlm": {
                "model": self.model,
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "timeout_seconds": self.timeout_seconds,
                "retries": self.retries,
                "enable_thinking": self.enable_thinking,
            },
            "discovery": {
                "max_primitives": self.max_primitives,
                "dedup_iou": self.dedup_iou,
                "full_frame_area_threshold": self.full_frame_area_threshold,
                "enable_relocalization": self.enable_relocalization,
                "min_box_side_norm": self.min_box_side_norm,
            },
            "crop": {
                "jpeg_quality": self.jpeg_quality,
                "create_preview": self.create_preview,
            },
            "output": {"root": self.output_root, "run_id": self.run_id},
        }


@dataclass(frozen=True)
class ImageInput:
    path: Path
    dataset: str
    relative_path: str
    image_id: str
    caption: str | None = None


@dataclass(frozen=True)
class PrimitiveCandidate:
    label: str
    bbox_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class PrimitiveRecord:
    vp_id: str
    label: str
    bbox_norm: NormBox
    bbox_px: PixelBox
    crop_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "vp_id": self.vp_id,
            "label": self.label,
            "bbox_norm": list(self.bbox_norm),
            "bbox_px": list(self.bbox_px),
            "crop_path": self.crop_path,
        }


@dataclass(frozen=True)
class ImageRecord:
    schema_version: str
    run_id: str
    image_id: str
    source: dict[str, Any]
    status: str
    primitives: tuple[PrimitiveRecord, ...] = field(default_factory=tuple)
    rejected_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "image_id": self.image_id,
            "source": self.source,
            "status": self.status,
            "primitives": [item.to_dict() for item in self.primitives],
        }
        if self.rejected_candidates:
            data["rejected_candidates"] = self.rejected_candidates
        return data


@dataclass
class ExtractionResult:
    record: ImageRecord
    source_image: Any
    crops: dict[str, Any]
