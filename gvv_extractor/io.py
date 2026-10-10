from __future__ import annotations

import hashlib
import json
import os
import posixpath
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageOps
from src.utils import resolve_reference

from .extractor import ExtractionResult, ImageInput, ImageRecord


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
    basenames = Counter(item.name for item in files)
    return [
        _image_input(item, dataset, root, identities.get(item), captions,
                     allow_basename_caption=basenames[item.name] == 1)
        for item in files
    ]


def scan_dataset(
    name: str, spec: dict[str, Any], project_root: Path
) -> list[ImageInput]:
    root = _dataset_path(str(spec["root"]), project_root)
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
        load_caption_map(_dataset_path(str(caption_root), project_root))
        if caption_root
        else None
    )
    basenames = Counter(path.name for path in files)
    return [
        _image_input(path, name, root, captions=captions,
                     allow_basename_caption=basenames[path.name] == 1)
        for path in files
    ]


def _dataset_path(value: str, project_root: Path) -> Path:
    if value.startswith("dataset://"):
        return Path(resolve_reference(value))
    return (project_root / value).resolve()


def scan_datasets(
    names: Iterable[str], specs: dict[str, Any], project_root: Path
) -> list[ImageInput]:
    """Scan multiple configured datasets with one stable, collision-free result."""
    sources: list[ImageInput] = []
    seen_ids: dict[str, ImageInput] = {}
    for name in names:
        if name not in specs:
            raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(specs)}")
        for source in scan_dataset(name, specs[name], project_root):
            previous = seen_ids.get(source.image_id)
            if previous is not None:
                raise ValueError(
                    f"Duplicate image_id {source.image_id}: "
                    f"{previous.path} and {source.path}"
                )
            seen_ids[source.image_id] = source
            sources.append(source)
    return sources


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


def extraction_signatures_compatible(
    existing: dict[str, Any], requested: dict[str, Any]
) -> bool:
    """Allow only a loopback forwarding-port move in an otherwise equal run."""
    if existing == requested:
        return True
    old = deepcopy(existing)
    new = deepcopy(requested)
    try:
        old_vlm = old["settings"]["vlm"]
        new_vlm = new["settings"]["vlm"]
        old_url = str(old_vlm["base_url"])
        new_url = str(new_vlm["base_url"])
    except (KeyError, TypeError):
        return False
    if not _loopback_port_move(old_url, new_url):
        return False
    old_vlm["base_url"] = new_url
    return old == new


def _loopback_port_move(old_url: str, new_url: str) -> bool:
    old = urlsplit(old_url)
    new = urlsplit(new_url)
    loopback = {"127.0.0.1", "localhost", "::1"}
    return (
        old.hostname in loopback
        and new.hostname in loopback
        and old.scheme == new.scheme
        and old.path.rstrip("/") == new.path.rstrip("/")
        and old.query == new.query
        and old.fragment == new.fragment
        and old.username == new.username
        and old.password == new.password
        and old.port != new.port
    )


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
            if not extraction_signatures_compatible(
                existing.get("signature", {}), signature
            ):
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

    def is_complete(self, image_id: str, source: ImageInput | None = None) -> bool:
        """Require a valid record and every referenced crop before skipping work."""
        path = self.items_dir / image_id / "record.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("image_id") != image_id or record.get("run_id") != self.run_id:
                return False
            if record.get("schema_version") != "1.0" or record.get("status") not in {"success", "no_views"}:
                return False
            views = record.get("grounded_visual_views")
            if not isinstance(views, list):
                return False
            if (record["status"] == "success") != bool(views):
                return False
            if source is not None and record.get("source", {}).get("sha256") != file_sha256(source.path):
                return False
            if source is not None and record.get("source", {}).get("caption") != source.caption:
                return False
            seen_ids = set()
            for view in views:
                view_id = view["gvv_id"]
                if not view_id or view_id in seen_ids:
                    return False
                seen_ids.add(view_id)
                crop = (self.root / view["crop_path"]).resolve()
                if not crop.is_relative_to(self.root):
                    return False
                box = view["bbox_px"]
                if not isinstance(box, list) or len(box) != 4:
                    return False
                size = (box[2] - box[0], box[3] - box[1])
                with Image.open(crop) as image:
                    if image.size != size or min(size) <= 0:
                        return False
                    image.verify()
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False

    def save(self, result: ExtractionResult) -> None:
        item_dir = self.items_dir / result.record.image_id
        item_dir.mkdir(parents=True, exist_ok=True)
        records = {view.gvv_id: view for view in result.record.grounded_visual_views}
        for gvv_id, image in result.crops.items():
            view = records[gvv_id]
            destination = self.root / view.crop_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save_image(image, destination, self.jpeg_quality)
        if self.create_preview and result.record.grounded_visual_views:
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
        view_rows: list[dict[str, Any]] = []
        for path in sorted(self.items_dir.glob("*/record.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            image_rows.append(record)
            for view in record.get("grounded_visual_views", []):
                view_rows.append(
                    {
                        "run_id": record["run_id"],
                        "image_id": record["image_id"],
                        "source": record["source"],
                        **view,
                    }
                )
        images_path = self.exports_dir / "images.jsonl"
        views_path = self.exports_dir / "grounded_visual_views.jsonl"
        _write_jsonl(images_path, image_rows)
        _write_jsonl(views_path, view_rows)
        return images_path, views_path


def _image_input(
    path: Path,
    dataset: str,
    root: Path,
    identity_path: str | None = None,
    captions: dict[str, str] | None = None,
    *,
    allow_basename_caption: bool = True,
) -> ImageInput:
    relative = path.relative_to(root).as_posix()
    identity = hashlib.sha256(
        f"{dataset}:{identity_path or relative}".encode("utf-8")
    ).hexdigest()[:12]
    caption_map = captions or {}
    caption = caption_map.get(relative)
    if caption is None and allow_basename_caption:
        matching = [value for key, value in caption_map.items() if Path(key).name == path.name]
        if len(matching) == 1:
            caption = matching[0]
    return ImageInput(
        path=path,
        dataset=dataset,
        relative_path=relative,
        image_id=f"img_{identity}",
        caption=caption,
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
            normalized = normalized.split(marker, 1)[1]
        elif normalized.startswith("image/"):
            normalized = normalized[len("image/"):]
        captions[posixpath.normpath(normalized)] = caption


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
    for view in record.grounded_visual_views:
        x1, y1, x2, y2 = view.bbox_px
        draw.rectangle(
            (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
            outline="red",
            width=line_width,
        )
        label = "_".join(view.gvv_id.rsplit("_", 2)[-2:])
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
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
