import base64
from copy import deepcopy
import hashlib
import io
import os
import re
from functools import lru_cache
from typing import Any, Iterable


def is_truncated_completion(response: Any) -> bool:
    """Return whether an OpenAI-compatible response hit its output limit."""
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return False
    finish_reasons = {
        str(metadata.get(key) or "").lower()
        for key in ("finish_reason", "native_finish_reason")
    }
    return bool(finish_reasons & {"length", "max_tokens", "max_output_tokens"})


def raise_for_truncated_completion(response: Any) -> None:
    """Fail the current M2A operation instead of consuming truncated output."""
    if is_truncated_completion(response):
        raise RuntimeError(
            "M2A model response was truncated at the configured output-token limit"
        )

def extract_image_id(text: str) -> int:
    """
    Extract image id from string of format <image{id}>.
    
    Example: 
        extract_image_id("<image123>") -> 123
    """
    pattern = r'<image(\d+)>'
    match = re.search(pattern, text)
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Wrong format.")


@lru_cache(maxsize=256)
def _transport_image_bytes(
    image_path: str,
    source_size: int,
    source_mtime_ns: int,
    max_edge: int,
    quality: int,
) -> tuple[bytes, str]:
    """Return a size-controlled JPEG used only for remote VLM transport."""
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        return output.getvalue(), "image/jpeg"


def encode_image_to_base64(image_path: str, *, compress: bool = False) -> str:
    """Read an image and encode it as a data URL.

    ``compress`` is deliberately opt-in: M2A's embedding path keeps using the
    original bytes, while remote ChatAgent/MemoryManager requests use a
    size-controlled transport copy. The source image is never modified.
    """
    import mimetypes
    from pathlib import Path

    if image_path.startswith(("data:", "http://", "https://")):
        return image_path

    path = Path(image_path)
    original_data = path.read_bytes()
    if compress:
        max_edge = int(os.getenv("M2A_IMAGE_MAX_EDGE", "1536"))
        quality = int(os.getenv("M2A_IMAGE_JPEG_QUALITY", "85"))
        if max_edge <= 0:
            raise ValueError("M2A_IMAGE_MAX_EDGE must be positive")
        if not 1 <= quality <= 95:
            raise ValueError("M2A_IMAGE_JPEG_QUALITY must be between 1 and 95")
        stat = path.stat()
        try:
            compressed_data, compressed_mime = _transport_image_bytes(
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                max_edge,
                quality,
            )
        except OSError:
            compressed_data = original_data
            compressed_mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        if len(compressed_data) >= len(original_data):
            image_data = original_data
            mime_type = mimetypes.guess_type(str(path))[0]
        else:
            image_data, mime_type = compressed_data, compressed_mime
    else:
        image_data = original_data
        mime_type = mimetypes.guess_type(str(path))[0]

    if mime_type is None or mime_type.startswith('text/'):
        mime_type = 'image/jpeg'

    return f"data:{mime_type};base64,{base64.b64encode(image_data).decode('utf-8')}"


def _content_image_url(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "image" and isinstance(block.get("url"), str):
        return block["url"]
    if block.get("type") in {"image_url", "input_image"}:
        value = block.get("image_url")
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("url"), str):
            return value["url"]
    return None


def deduplicate_message_images(messages: Iterable[Any]) -> list[Any]:
    """Copy messages and attach each distinct image at most once per request.

    Text, tool calls, image tokens, and stored graph state are preserved. Only
    repeated binary image blocks in the outgoing copy are replaced by a short
    marker, so repeated memories can still be reasoned about without resending
    the same bytes.
    """
    request_messages = deepcopy(list(messages))
    seen: set[str] = set()
    for message in request_messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        normalized: list[Any] = []
        for block in content:
            url = _content_image_url(block)
            if url is None:
                normalized.append(block)
                continue
            identity = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if identity in seen:
                normalized.append(
                    {
                        "type": "text",
                        "text": "[Duplicate image bytes omitted; use the earlier identical image.]",
                    }
                )
                continue
            seen.add(identity)
            normalized.append(block)
        message.content = normalized
    return request_messages
