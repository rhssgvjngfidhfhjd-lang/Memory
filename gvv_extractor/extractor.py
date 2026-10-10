"""Grounded visual view records, model discovery, and single-image extraction."""
from __future__ import annotations

import base64
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from PIL import Image


NormBox = tuple[int, int, int, int]
PixelBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str
    api_key_env: str = "EXECUTOR_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: int = 180
    retries: int = 1
    enable_thinking: bool = False
    max_views_per_image: int = 20
    dedup_iou: float = 0.6
    full_frame_area_threshold: float = 0.9
    enable_relocalization: bool = True
    min_box_side_norm: int = 8
    jpeg_quality: int = 95
    create_preview: bool = True
    output_root: str = "outputs/gvv"
    run_id: str = "default"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Settings":
        vlm = data.get("vlm", {})
        discovery = data.get("discovery", {})
        crop = data.get("crop", {})
        output = data.get("output", {})
        return cls(
            model=str(vlm.get("model", "Qwen/Qwen3-VL-4B-Instruct")),
            base_url=str(vlm.get("base_url") or ""),
            api_key_env=str(vlm.get("api_key_env", "EXECUTOR_API_KEY")),
            temperature=float(vlm.get("temperature", 0.0)),
            max_tokens=int(vlm.get("max_tokens", 1024)),
            timeout_seconds=int(vlm.get("timeout_seconds", 180)),
            retries=max(0, int(vlm.get("retries", 1))),
            enable_thinking=bool(vlm.get("enable_thinking", False)),
            max_views_per_image=max(1, int(discovery.get("max_views_per_image", 20))),
            dedup_iou=float(discovery.get("dedup_iou", 0.6)),
            full_frame_area_threshold=float(
                discovery.get("full_frame_area_threshold", 0.9)
            ),
            enable_relocalization=bool(discovery.get("enable_relocalization", True)),
            min_box_side_norm=max(1, int(discovery.get("min_box_side_norm", 8))),
            jpeg_quality=int(crop.get("jpeg_quality", 95)),
            create_preview=bool(crop.get("create_preview", True)),
            output_root=str(output.get("root", "outputs/gvv")),
            run_id=str(output.get("run_id", "default")),
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
                "max_views_per_image": self.max_views_per_image,
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
class GVVCandidate:
    label: str
    bbox_norm: tuple[float, float, float, float]


@dataclass(frozen=True)
class GVVRecord:
    gvv_id: str
    label: str
    bbox_norm: NormBox
    bbox_px: PixelBox
    crop_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gvv_id": self.gvv_id,
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
    grounded_visual_views: tuple[GVVRecord, ...] = field(default_factory=tuple)
    rejected_candidates: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "image_id": self.image_id,
            "source": self.source,
            "status": self.status,
            "grounded_visual_views": [item.to_dict() for item in self.grounded_visual_views],
        }
        if self.rejected_candidates:
            data["rejected_candidates"] = self.rejected_candidates
        return data


@dataclass
class ExtractionResult:
    record: ImageRecord
    source_image: Any
    crops: dict[str, Any]


class VisionLanguageModel(Protocol):
    def generate(self, prompt: str, image: Image.Image) -> str: ...


class OpenAICompatibleVLM:
    """Small image-chat client for local OpenAI-compatible servers."""

    def __init__(self, settings: Settings, api_key: str = "EMPTY"):
        self.settings = settings
        self.api_key = api_key or "EMPTY"
        self.session = requests.Session()
        if (urlparse(settings.base_url).hostname or "").lower() in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            self.session.trust_env = False

    def assert_available(self) -> None:
        response = self.session.get(
            f"{self.settings.base_url.rstrip('/')}/models",
            headers=self._headers(),
            timeout=min(10, self.settings.timeout_seconds),
        )
        response.raise_for_status()
        models = {item.get("id") for item in response.json().get("data", [])}
        if self.settings.model not in models:
            raise RuntimeError(
                f"Model {self.settings.model!r} is unavailable; found {sorted(models)}"
            )

    def generate(self, prompt: str, image: Image.Image) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_url(image)},
                        },
                    ],
                }
            ],
            "chat_template_kwargs": {
                "enable_thinking": self.settings.enable_thinking
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                response = self.session.post(
                    f"{self.settings.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.timeout_seconds,
                )
                response.raise_for_status()
                choices = response.json().get("choices") or []
                if not choices:
                    raise RuntimeError("VLM returned no choices")
                message = choices[0].get("message") or {}
                text = (message.get("content") or message.get("reasoning_content") or "").strip()
                if not text:
                    raise RuntimeError("VLM returned an empty response")
                return text
            except Exception as exc:
                last_error = exc
                if attempt < self.settings.retries:
                    time.sleep(1 + attempt)
        raise RuntimeError(f"VLM request failed: {last_error}") from last_error

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class ObjectDiscoverer:
    def __init__(
        self,
        vlm: VisionLanguageModel,
        discovery_prompt: str,
        relocalize_prompt: str,
        max_views_per_image: int,
        caption_guided_prompt: str | None = None,
        response_retries: int = 1,
    ):
        self.vlm = vlm
        self.discovery_prompt = discovery_prompt.replace(
            "__MAX_VIEWS_PER_IMAGE", str(max_views_per_image)
        )
        self.relocalize_prompt = relocalize_prompt
        self.caption_guided_prompt = (
            caption_guided_prompt.replace("__MAX_VIEWS_PER_IMAGE", str(max_views_per_image))
            if caption_guided_prompt
            else None
        )
        self.max_views_per_image = max_views_per_image
        self.response_retries = max(0, int(response_retries))

    def discover(
        self, image: Image.Image, focus_context: str | None = None
    ) -> list[GVVCandidate]:
        last_error: ValueError | None = None
        for _ in range(self.response_retries + 1):
            raw = self.vlm.generate(self.build_prompt(focus_context), image)
            try:
                return parse_candidates(raw)[: self.max_views_per_image]
            except ValueError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def build_prompt(self, focus_context: str | None = None) -> str:
        """Build the exact discovery prompt sent to the VLM."""
        if focus_context and focus_context.strip() and self.caption_guided_prompt:
            return self.caption_guided_prompt.replace(
                "__CAPTION__", focus_context.strip()
            )
        context = (
            "Memory caption:\n" + focus_context.strip()
            if focus_context and focus_context.strip()
            else "No memory caption is available. Use generic image-only discovery."
        )
        return self.discovery_prompt.replace("__FOCUS_CONTEXT__", context)

    def relocalize(
        self, image: Image.Image, candidate: GVVCandidate
    ) -> GVVCandidate | None:
        prompt = self.relocalize_prompt.replace("__LABEL__", candidate.label).replace(
            "__BBOX__", str(list(candidate.bbox_norm))
        )
        last_error: ValueError | None = None
        for _ in range(self.response_retries + 1):
            try:
                candidates = parse_candidates(
                    self.vlm.generate(prompt, image), default_label=candidate.label
                )
                return candidates[0] if candidates else None
            except ValueError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error


def parse_candidates(
    text: str, *, default_label: str | None = None
) -> list[GVVCandidate]:
    """Parse the small JSON contract while tolerating markdown wrappers."""
    payload = _load_json_payload(text)
    if isinstance(payload, dict):
        if isinstance(payload.get("objects"), list):
            payload = payload["objects"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("VLM response must be a JSON array")

    candidates: list[GVVCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or default_label or "").strip()
        box = item.get("bbox_norm", item.get("bbox_2d"))
        if not label or not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            coords = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            continue
        candidates.append(GVVCandidate(label=label[:200], bbox_norm=coords))
    return candidates


def _load_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        starts = [pos for pos in (stripped.find("["), stripped.find("{")) if pos >= 0]
        if not starts:
            raise ValueError("VLM response contains no JSON")
        start = min(starts)
        end = max(stripped.rfind("]"), stripped.rfind("}"))
        if end < start:
            raise ValueError("VLM response contains incomplete JSON")
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            salvaged = _salvage_complete_objects(stripped[start:])
            if salvaged:
                return salvaged
            raise ValueError(f"Invalid VLM JSON: {exc}") from exc


def _salvage_complete_objects(text: str) -> list[dict[str, Any]]:
    """Recover complete object entries from a truncated JSON array/wrapper."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = text.find("{")
    while cursor >= 0:
        try:
            value, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            cursor = text.find("{", cursor + 1)
            continue
        if isinstance(value, dict):
            objects.append(value)
        cursor = text.find("{", end)
    return objects


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90,
        optimize=True,
        progressive=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class GVVExtractor:
    """Reusable single-image grounded visual view extractor."""

    def __init__(self, discoverer: ObjectDiscoverer, settings: Settings):
        self.discoverer = discoverer
        self.settings = settings

    def extract_image(self, source: ImageInput) -> ExtractionResult:
        # Import lazily because io uses the record types defined here.
        from .io import crop_extension, file_sha256, load_canonical_image

        image = load_canonical_image(source.path)
        raw_candidates = self.discoverer.discover(image, source.caption)
        candidates: list[GVVCandidate] = []

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
                        candidate = GVVCandidate(
                            label=replacement.label, bbox_norm=corrected
                        )
                        normalized = corrected
            if normalized is None:
                continue
            label = " ".join(candidate.label.split())
            if label:
                candidates.append(GVVCandidate(label=label, bbox_norm=normalized))

        candidates = suppress_full_frame_parents(
            candidates, self.settings.full_frame_area_threshold
        )
        candidates = deduplicate(candidates, self.settings.dedup_iou)
        candidates.sort(key=lambda item: (item.bbox_norm[1], item.bbox_norm[0], item.label))
        candidates = candidates[: self.settings.max_views_per_image]

        grounded_visual_views: list[GVVRecord] = []
        crops = {}
        extension = crop_extension(source.path)
        for index, candidate in enumerate(candidates, start=1):
            bbox_norm = normalize_bbox(candidate.bbox_norm)
            if bbox_norm is None:
                continue
            bbox_px = norm_to_pixels(bbox_norm, image.width, image.height)
            gvv_id = f"{source.image_id}_gvv_{index:04d}"
            crop_path = f"items/{source.image_id}/gvv_{index:04d}{extension}"
            grounded_visual_views.append(
                GVVRecord(
                    gvv_id=gvv_id,
                    label=candidate.label,
                    bbox_norm=bbox_norm,
                    bbox_px=bbox_px,
                    crop_path=crop_path,
                )
            )
            crops[gvv_id] = image.crop(bbox_px)

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
            status="success" if grounded_visual_views else "no_views",
            grounded_visual_views=tuple(grounded_visual_views),
            rejected_candidates=max(0, len(raw_candidates) - len(grounded_visual_views)),
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
    candidates: list[GVVCandidate], area_threshold: float
) -> list[GVVCandidate]:
    """Drop a near-full-image parent when a clearly smaller child exists."""
    full_area = 1000 * 1000
    kept: list[GVVCandidate] = []
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
    candidates: list[GVVCandidate], threshold: float
) -> list[GVVCandidate]:
    ordered = sorted(candidates, key=lambda item: bbox_area(item.bbox_norm), reverse=True)
    kept: list[GVVCandidate] = []
    for candidate in ordered:
        if any(bbox_iou(candidate.bbox_norm, item.bbox_norm) >= threshold for item in kept):
            continue
        kept.append(candidate)
    return kept
