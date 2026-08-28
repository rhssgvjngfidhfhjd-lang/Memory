from __future__ import annotations

import math

from .io import crop_extension, file_sha256, load_canonical_image
from .models import (
    ExtractionResult,
    ImageInput,
    ImageRecord,
    NormBox,
    PixelBox,
    PrimitiveCandidate,
    PrimitiveRecord,
    Settings,
)
from .vlm import ObjectDiscoverer


class VPExtractor:
    """Reusable single-image visual primitive extractor."""

    def __init__(self, discoverer: ObjectDiscoverer, settings: Settings):
        self.discoverer = discoverer
        self.settings = settings

    def extract_image(self, source: ImageInput) -> ExtractionResult:
        image = load_canonical_image(source.path)
        raw_candidates = self.discoverer.discover(image, source.caption)
        candidates: list[PrimitiveCandidate] = []

        for candidate in raw_candidates:
            normalized = normalize_bbox(candidate.bbox_norm)
            uncertain = normalized is None or min(
                normalized[2] - normalized[0], normalized[3] - normalized[1]
            ) < self.settings.min_box_side_norm
            if uncertain and self.settings.enable_relocalization:
                replacement = self.discoverer.relocalize(image, candidate)
                if replacement is not None:
                    corrected = normalize_bbox(replacement.bbox_norm)
                    if corrected is not None:
                        candidate = PrimitiveCandidate(
                            label=replacement.label, bbox_norm=corrected
                        )
                        normalized = corrected
            if normalized is None:
                continue
            label = " ".join(candidate.label.split())
            if label:
                candidates.append(PrimitiveCandidate(label=label, bbox_norm=normalized))

        candidates = suppress_full_frame_parents(
            candidates, self.settings.full_frame_area_threshold
        )
        candidates = deduplicate(candidates, self.settings.dedup_iou)
        candidates.sort(key=lambda item: (item.bbox_norm[1], item.bbox_norm[0], item.label))
        candidates = candidates[: self.settings.max_primitives]

        primitives: list[PrimitiveRecord] = []
        crops = {}
        extension = crop_extension(source.path)
        for index, candidate in enumerate(candidates, start=1):
            bbox_norm = normalize_bbox(candidate.bbox_norm)
            if bbox_norm is None:
                continue
            bbox_px = norm_to_pixels(bbox_norm, image.width, image.height)
            vp_id = f"{source.image_id}_vp_{index:04d}"
            crop_path = f"items/{source.image_id}/vp_{index:04d}{extension}"
            primitives.append(
                PrimitiveRecord(
                    vp_id=vp_id,
                    label=candidate.label,
                    bbox_norm=bbox_norm,
                    bbox_px=bbox_px,
                    crop_path=crop_path,
                )
            )
            crops[vp_id] = image.crop(bbox_px)

        source_data = {
            "dataset": source.dataset,
            "relative_path": source.relative_path,
            "sha256": file_sha256(source.path),
            "width": image.width,
            "height": image.height,
            "extraction_mode": "caption_guided" if source.caption else "generic",
        }
        if source.caption:
            source_data["caption"] = source.caption

        record = ImageRecord(
            schema_version="1.0",
            run_id=self.settings.run_id,
            image_id=source.image_id,
            source=source_data,
            status="success" if primitives else "no_primitives",
            primitives=tuple(primitives),
            rejected_candidates=max(0, len(raw_candidates) - len(primitives)),
        )
        return ExtractionResult(record=record, source_image=image, crops=crops)


def normalize_bbox(values: tuple[float, float, float, float]) -> NormBox | None:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = (max(0, min(1000, round(value))) for value in values)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def norm_to_pixels(box: NormBox, width: int, height: int) -> PixelBox:
    x1, y1, x2, y2 = box
    px1 = max(0, min(width - 1, math.floor(x1 * width / 1000)))
    py1 = max(0, min(height - 1, math.floor(y1 * height / 1000)))
    px2 = max(px1 + 1, min(width, math.ceil(x2 * width / 1000)))
    py2 = max(py1 + 1, min(height, math.ceil(y2 * height / 1000)))
    return px1, py1, px2, py2


def bbox_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def bbox_area(box: tuple[float, ...]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment_ratio(parent: tuple[float, ...], child: tuple[float, ...]) -> float:
    left = max(parent[0], child[0])
    top = max(parent[1], child[1])
    right = min(parent[2], child[2])
    bottom = min(parent[3], child[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    child_area = bbox_area(child)
    return intersection / child_area if child_area else 0.0


def suppress_full_frame_parents(
    candidates: list[PrimitiveCandidate], area_threshold: float
) -> list[PrimitiveCandidate]:
    """Drop a near-full-image parent when a clearly smaller child exists."""
    full_area = 1000 * 1000
    kept: list[PrimitiveCandidate] = []
    for parent in candidates:
        parent_area = bbox_area(parent.bbox_norm)
        if parent_area / full_area < area_threshold:
            kept.append(parent)
            continue
        has_specific_child = any(
            child is not parent
            and bbox_area(child.bbox_norm) < parent_area * 0.8
            and containment_ratio(parent.bbox_norm, child.bbox_norm) >= 0.95
            for child in candidates
        )
        if not has_specific_child:
            kept.append(parent)
    return kept


def deduplicate(
    candidates: list[PrimitiveCandidate], threshold: float
) -> list[PrimitiveCandidate]:
    ordered = sorted(candidates, key=lambda item: bbox_area(item.bbox_norm), reverse=True)
    kept: list[PrimitiveCandidate] = []
    for candidate in ordered:
        if any(bbox_iou(candidate.bbox_norm, item.bbox_norm) >= threshold for item in kept):
            continue
        kept.append(candidate)
    return kept
