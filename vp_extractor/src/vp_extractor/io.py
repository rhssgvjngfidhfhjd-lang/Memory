from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps

from .models import ExtractionResult, ImageInput, ImageRecord


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def load_canonical_image(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def scan_path(
    path: Path,
    dataset: str = "custom",
    captions: dict[str, str] | None = None,
) -> list[ImageInput]:
    path = path.resolve()
    if path.is_file():
        files = [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
        root = path.parent
        identities = {path: path.as_posix()}
    elif path.is_dir():
        root = path
        files = _walk_images(root)
        identities = {}
    else:
        raise FileNotFoundError(path)
    return [
        _image_input(item, dataset, root, identities.get(item), captions)
        for item in files
    ]


def scan_dataset(
    name: str, spec: dict[str, Any], project_root: Path
) -> list[ImageInput]:
    root = (project_root / str(spec["root"])).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    patterns = [str(pattern) for pattern in spec.get("include", ["**/*"])]
    files = [
        path
        for path in _walk_images(root)
        if any(path.relative_to(root).match(pattern) for pattern in patterns)
    ]
    caption_root = spec.get("caption_root")
    captions = (
        load_caption_map((project_root / str(caption_root)).resolve())
        if caption_root
        else None
    )
    return [_image_input(path, name, root, captions=captions) for path in files]


def load_caption_map(path: Path) -> dict[str, str]:
    """Load image captions from one Mem-Gallery dialog JSON or a directory."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    captions: dict[str, str] = {}
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        _collect_captions(data, captions)
    return captions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_extension(source: Path) -> str:
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return ".jpg"
    if suffix in {".png", ".webp"}:
        return suffix
    return ".png"


class ArtifactStore:
    def __init__(
        self,
        output_root: Path,
        run_id: str,
        *,
        jpeg_quality: int = 95,
        create_preview: bool = True,
    ):
        self.root = output_root.resolve() / run_id
        self.run_id = run_id
        self.jpeg_quality = jpeg_quality
        self.create_preview = create_preview
        self.items_dir = self.root / "items"
        self.exports_dir = self.root / "exports"
        self.failures_path = self.root / "failures.jsonl"

    def initialize(self, signature: dict[str, Any]) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        path = self.root / "run.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("signature") != signature:
                raise RuntimeError(
                    f"Run {self.run_id!r} already exists with different settings"
                )
            return
        _write_json(
            path,
            {
                "schema_version": "1.0",
                "run_id": self.run_id,
                "created_at": _utc_now(),
                "signature": signature,
            },
        )

    def is_complete(self, image_id: str) -> bool:
        return (self.items_dir / image_id / "record.json").is_file()

    def save(self, result: ExtractionResult) -> None:
        item_dir = self.items_dir / result.record.image_id
        item_dir.mkdir(parents=True, exist_ok=True)
        records = {primitive.vp_id: primitive for primitive in result.record.primitives}
        for vp_id, image in result.crops.items():
            primitive = records[vp_id]
            destination = self.root / primitive.crop_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save_image(image, destination, self.jpeg_quality)
        if self.create_preview and result.record.primitives:
            _save_preview(
                result.source_image,
                result.record,
                item_dir / "preview.jpg",
                self.jpeg_quality,
            )
        _write_json(item_dir / "record.json", result.record.to_dict())

    def save_failure(self, source: ImageInput, error: Exception) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": _utc_now(),
            "image_id": source.image_id,
            "dataset": source.dataset,
            "relative_path": source.relative_path,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        with self.failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def export(self) -> tuple[Path, Path]:
        image_rows: list[dict[str, Any]] = []
        primitive_rows: list[dict[str, Any]] = []
        for path in sorted(self.items_dir.glob("*/record.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            image_rows.append(record)
            for primitive in record.get("primitives", []):
                primitive_rows.append(
                    {
                        "run_id": record["run_id"],
                        "image_id": record["image_id"],
                        "source": record["source"],
                        **primitive,
                    }
                )
        images_path = self.exports_dir / "images.jsonl"
        primitives_path = self.exports_dir / "primitives.jsonl"
        _write_jsonl(images_path, image_rows)
        _write_jsonl(primitives_path, primitive_rows)
        return images_path, primitives_path


def _image_input(
    path: Path,
    dataset: str,
    root: Path,
    identity_path: str | None = None,
    captions: dict[str, str] | None = None,
) -> ImageInput:
    relative = path.relative_to(root).as_posix()
    identity = hashlib.sha256(
        f"{dataset}:{identity_path or relative}".encode("utf-8")
    ).hexdigest()[:12]
    return ImageInput(
        path=path,
        dataset=dataset,
        relative_path=relative,
        image_id=f"img_{identity}",
        caption=(captions or {}).get(relative) or (captions or {}).get(path.name),
    )


def _collect_captions(value: Any, captions: dict[str, str]) -> None:
    if isinstance(value, dict):
        _add_caption_pairs(value.get("input_image"), value.get("image_caption"), captions)
        _add_caption_pairs(value.get("question_image"), value.get("image_caption"), captions)
        for child in value.values():
            _collect_captions(child, captions)
    elif isinstance(value, list):
        for child in value:
            _collect_captions(child, captions)


def _add_caption_pairs(images: Any, texts: Any, captions: dict[str, str]) -> None:
    if isinstance(images, str):
        images = [images]
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(images, list) or not isinstance(texts, list):
        return
    for image, caption in zip(images, texts):
        if not isinstance(image, str) or not isinstance(caption, str):
            continue
        caption = " ".join(caption.split())
        if not caption:
            continue
        normalized = image.replace("\\", "/")
        marker = "/image/"
        if marker in normalized:
            captions[normalized.split(marker, 1)[1]] = caption
        captions[Path(normalized).name] = caption


def _walk_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                files.append(Path(directory) / filename)
    return files


def _save_image(image: Image.Image, path: Path, jpeg_quality: int) -> None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(path, format="JPEG", quality=jpeg_quality, subsampling=0)
    elif suffix == ".webp":
        image.save(path, format="WEBP", quality=jpeg_quality)
    else:
        image.save(path, format="PNG")


def _save_preview(
    image: Image.Image,
    record: ImageRecord,
    path: Path,
    jpeg_quality: int,
) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(2, round(max(preview.size) / 500))
    for primitive in record.primitives:
        x1, y1, x2, y2 = primitive.bbox_px
        draw.rectangle(
            (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
            outline="red",
            width=line_width,
        )
        label = "_".join(primitive.vp_id.rsplit("_", 2)[-2:])
        draw.text((x1 + line_width, y1 + line_width), label, fill="red")
    preview.save(path, format="JPEG", quality=jpeg_quality, subsampling=0)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
