from __future__ import annotations

import base64
import io
import mimetypes
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


MAX_IMAGE_SIDE_FOR_ANSWER = 1344
IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b", re.IGNORECASE)


def build_retrieved_memory_context(
    memory_items: list[dict[str, Any]],
    category: str = "",
) -> tuple[str, list[str]]:
    """Render the exact retrieved-memory text and attached memory images.

    Keeping this formatter outside ``VLMAnswerClient`` lets offline metrics
    tokenize the same text that the answer model receives.
    """
    lines = ["The retrieved memory contents are as follows:"]
    image_paths: list[str] = []
    image_num = 0
    include_memory_images = category.upper() in {"VS", "VR"}
    for idx, item in enumerate(memory_items, start=1):
        md = item.get("metadata", {}) or {}
        image = item.get("image")
        has_attached_memory_image = (
            include_memory_images
            and isinstance(image, dict)
            and bool(image.get("path"))
        )
        header = f"[{idx}] SESSION:{md.get('session_id', '')} ROUND:{md.get('dialogue_id', '')}"
        # An image ID is a legitimate candidate label only when its image is
        # actually attached. Exposing an unattached ID leaks the answer to
        # text-only and caption-only ablations.
        if has_attached_memory_image and md.get("image_id"):
            header += f" IMG:{md.get('image_id')}"
        lines.append(header)
        memory_text = str(item.get("text", ""))
        if not has_attached_memory_image:
            memory_text = IMAGE_ID_PATTERN.sub("[IMAGE_ID_REDACTED]", memory_text)
        lines.append(memory_text)
        if has_attached_memory_image:
            image_num += 1
            lines.append(f"Attached memory image {image_num}: {image.get('img_id', '')}")
            image_paths.append(str(image["path"]))
    return "\n\n".join(lines), image_paths


class VLMAnswerClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        num_predict: int = 512,
        timeout: int = 180,
        retries: int = 0,
        think: bool | None = None,
        backend: str = "openai",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.num_predict = num_predict
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.think = think
        self.backend = backend

    def assert_model_available(self) -> None:
        if self.backend == "ollama":
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = {m.get("name") for m in resp.json().get("models", [])}
            models |= {m.split(":", 1)[0] for m in models if isinstance(m, str)}
        else:
            resp = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            models = {m.get("id") for m in resp.json().get("data", [])}
        resp.raise_for_status()
        if self.model not in models:
            raise RuntimeError(
                f"Answer model {self.model!r} is not available at {self.base_url}. "
                f"Available models: {sorted(models)}"
            )

    def answer(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
    ) -> str:
        if self.backend == "ollama":
            return self._answer_ollama_native(
                system_prompt=system_prompt,
                memory_items=memory_items,
                question_prompt=question_prompt,
                query_image=query_image,
                category=category,
            )
        return self._answer_openai_compatible(
            system_prompt=system_prompt,
            memory_items=memory_items,
            question_prompt=question_prompt,
            query_image=query_image,
            category=category,
        )

    def _answer_openai_compatible(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
    ) -> str:
        content = self._build_openai_content(memory_items, question_prompt, query_image, category)
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.num_predict,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        if self.think is not None:
            payload["extra_body"] = {"think": self.think}
            payload["think"] = self.think
        data = self._post_json(f"{self.base_url}/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or message.get("reasoning_content") or "").strip()

    def _answer_ollama_native(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
    ) -> str:
        content, images = self._build_user_content(memory_items, question_prompt, query_image, category)
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.num_predict},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content, "images": images},
            ],
        }
        if self.think is not None:
            payload["think"] = self.think
        data = self._post_json(f"{self.base_url}/api/chat", payload)
        message = (data.get("message", {}) or {})
        content = (message.get("content") or "").strip()
        if content:
            return content
        # Some local Qwen3-VL Ollama builds return only `thinking`.
        return (message.get("thinking") or "").strip()

    def _build_openai_content(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
    ) -> list[dict[str, Any]]:
        text, image_paths = self._build_text_and_image_paths(memory_items, question_prompt, query_image, category)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_url(path)},
                }
            )
        return content

    def _build_user_content(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
    ) -> tuple[str, list[str]]:
        text, image_paths = self._build_text_and_image_paths(memory_items, question_prompt, query_image, category)
        return text, [_encode_image(path) for path in image_paths]

    def _build_text_and_image_paths(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
    ) -> tuple[str, list[str]]:
        memory_text, image_paths = build_retrieved_memory_context(memory_items, category)
        lines = [memory_text, "", question_prompt]
        image_num = len(image_paths)
        if query_image and query_image.get("path"):
            image_num += 1
            lines.append(f"Attached question image {image_num}.")
            image_paths.append(str(query_image["path"]))
        return "\n\n".join(lines), image_paths

    @staticmethod
    def _include_memory_images(category: str) -> bool:
        return category.upper() in {"VS", "VR"}

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.retries:
                    raise
                time.sleep(1 + attempt)
        raise last_exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _encode_image(path: str) -> str:
    p = Path(path)
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_image_data_url(path: str) -> str:
    image_bytes, mime = _prepare_image_bytes(path)
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _prepare_image_bytes(path: str) -> tuple[bytes, str]:
    p = Path(path)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with Image.open(p) as image:
        width, height = image.size
        if max(width, height) <= MAX_IMAGE_SIDE_FOR_ANSWER:
            return p.read_bytes(), mime
        image.thumbnail((MAX_IMAGE_SIDE_FOR_ANSWER, MAX_IMAGE_SIDE_FOR_ANSWER))
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue(), "image/jpeg"
