from __future__ import annotations

import base64
import io
import mimetypes
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests
from PIL import Image

from benchmarks.answer_response import (
    ANSWER_BLOCK_REGEX,
    ANSWER_BLOCK_RETRY_REGEX,
    AnswerFormatError,
    parse_answer_block,
)


MAX_IMAGE_SIDE_FOR_ANSWER = 1344
CONTEXT_RECOVERY_IMAGE_SIDES = (896, 672, 448)
IMAGE_ID_PATTERN = re.compile(r"\bD\d+:IMG_\d+\b", re.IGNORECASE)
MEMORY_IMAGE_CATEGORIES = frozenset({"VS", "VR", "TTL"})


@dataclass(frozen=True)
class AnswerResponse:
    text: str
    usage: dict[str, int] | None
    attempts: int
    failed_attempts: int
    image_count: int = 0


class _AnswerAttemptError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, int] | None = None):
        super().__init__(message)
        self.usage = usage


def build_retrieved_memory_context(
    memory_items: list[dict[str, Any]],
    category: str = "",
) -> tuple[str, list[str]]:
    """Render the exact retrieved-memory text and attached memory images.

    Keeping this formatter outside ``VLMAnswerClient`` lets offline metrics
    tokenize the same text that the answer model receives.
    """
    evidence, image_paths = build_retrieved_memory_evidence(memory_items, category)
    return "\n\n".join(["The retrieved memory contents are as follows:", *evidence]), image_paths


def build_retrieved_memory_evidence(
    memory_items: list[dict[str, Any]],
    category: str = "",
) -> tuple[list[str], list[str]]:
    """Render one prompt evidence string per retrieval item plus image paths."""
    evidence: list[str] = []
    image_paths: list[str] = []
    image_num = 0
    include_memory_images = category.upper() in MEMORY_IMAGE_CATEGORIES
    for idx, item in enumerate(memory_items, start=1):
        md = item.get("metadata", {}) or {}
        raw_images = item.get("images")
        if not isinstance(raw_images, list):
            legacy = item.get("image")
            raw_images = [legacy] if isinstance(legacy, dict) else []
        attached_images = [
            image
            for image in raw_images
            if include_memory_images
            and isinstance(image, dict)
            and bool(image.get("path"))
        ]
        has_attached_original = any(
            str(image.get("kind", "image")) == "image" for image in attached_images
        )
        header = f"[{idx}] SESSION:{md.get('session_id', '')} ROUND:{md.get('dialogue_id', '')}"
        # An image ID is a legitimate candidate label only when its image is
        # actually attached. Exposing an unattached ID leaks the answer to
        # text-only and caption-only ablations.
        if has_attached_original and md.get("image_id"):
            header += f" IMG:{md.get('image_id')}"
        block = [header]
        memory_text = str(item.get("text", ""))
        if not has_attached_original:
            memory_text = IMAGE_ID_PATTERN.sub("[IMAGE_ID_REDACTED]", memory_text)
        block.append(memory_text)
        for image in attached_images:
            image_num += 1
            raw_kind = str(image.get("kind", "image")).lower()
            kind = "image" if raw_kind == "image" else raw_kind.upper()
            block.append(
                f"Attached memory {kind} {image_num}: {image.get('img_id', '')}"
            )
            image_paths.append(str(image["path"]))
        evidence.append("\n\n".join(block))
    return evidence, image_paths


def query_image_prompt_metadata(
    query_image: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if not query_image or not query_image.get("path"):
        return []
    image_id = str(
        query_image.get("img_id")
        or query_image.get("id")
        or Path(str(query_image["path"])).name
    )
    row = {"id": image_id}
    if query_image.get("caption"):
        row["caption"] = str(query_image["caption"])
    return [row]


def count_answer_images(
    memory_items: list[dict[str, Any]],
    query_image: dict[str, Any] | None = None,
    category: str = "",
) -> int:
    """Count images attached to each answer-model attempt."""
    _, memory_image_paths = build_retrieved_memory_context(memory_items, category)
    return len(memory_image_paths) + int(
        bool(query_image and query_image.get("path"))
    )


class VLMAnswerClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        base_url: str = "http://localhost:18000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout: int = 180,
        retries: int = 0,
        think: bool | None = None,
        backend: str = "openai",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.num_predict = num_predict
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.think = think
        self.backend = backend
        self._session = requests.Session()
        if (urlparse(self.base_url).hostname or "").lower() in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            # Local OpenAI-compatible servers must not inherit an unrelated
            # system HTTP proxy. This also removes the need for fragile
            # wildcard NO_PROXY settings.
            self._session.trust_env = False

    def assert_model_available(self) -> None:
        if self.backend == "ollama":
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = {m.get("name") for m in resp.json().get("models", [])}
            models |= {m.split(":", 1)[0] for m in models if isinstance(m, str)}
        else:
            resp = self._session.get(
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
        prepend_memory_context: bool = True,
    ) -> str:
        return self.answer_with_usage(
            system_prompt=system_prompt,
            memory_items=memory_items,
            question_prompt=question_prompt,
            query_image=query_image,
            category=category,
            prepend_memory_context=prepend_memory_context,
        ).text

    def answer_with_usage(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
        prepend_memory_context: bool = True,
    ) -> AnswerResponse:
        last_error: Exception | None = None
        cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_is_exact = True
        image_count = self.count_answer_images(
            memory_items, query_image=query_image, category=category
        )
        for attempt in range(self.retries + 1):
            try:
                if self.backend == "ollama":
                    text, usage = self._answer_ollama_native(
                        system_prompt=system_prompt,
                        memory_items=memory_items,
                        question_prompt=question_prompt,
                        query_image=query_image,
                        category=category,
                        prepend_memory_context=prepend_memory_context,
                    )
                else:
                    text, usage = self._answer_openai_compatible(
                        system_prompt=system_prompt,
                        memory_items=memory_items,
                        question_prompt=question_prompt,
                        query_image=query_image,
                        category=category,
                        prepend_memory_context=prepend_memory_context,
                    )
                if usage is None:
                    usage_is_exact = False
                else:
                    cumulative_usage = _sum_answer_usage(cumulative_usage, usage)
                return AnswerResponse(
                    text=text,
                    usage=cumulative_usage if usage_is_exact else None,
                    attempts=attempt + 1,
                    failed_attempts=attempt,
                    image_count=image_count,
                )
            except Exception as exc:
                attempt_usage = (
                    exc.usage if isinstance(exc, _AnswerAttemptError) else None
                )
                if attempt_usage is None:
                    usage_is_exact = False
                else:
                    cumulative_usage = _sum_answer_usage(
                        cumulative_usage, attempt_usage
                    )
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1 + attempt)
        assert last_error is not None
        raise last_error

    def answer_messages_with_usage(
        self,
        *,
        messages: list[dict[str, str]],
        memory_items: list[dict[str, Any]],
        query_image: dict[str, Any] | None = None,
        category: str = "",
    ) -> AnswerResponse:
        """Send prebuilt prompt messages verbatim while attaching actual images."""
        normalized = _validate_prebuilt_messages(messages)
        last_error: Exception | None = None
        cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_is_exact = True
        image_count = self.count_answer_images(
            memory_items, query_image=query_image, category=category
        )
        for attempt in range(self.retries + 1):
            try:
                if self.backend == "ollama":
                    text, usage = self._answer_prebuilt_ollama_native(
                        messages=normalized,
                        memory_items=memory_items,
                        query_image=query_image,
                        category=category,
                    )
                else:
                    text, usage = self._answer_prebuilt_openai_compatible(
                        messages=normalized,
                        memory_items=memory_items,
                        query_image=query_image,
                        category=category,
                        structured_regex=(
                            ANSWER_BLOCK_REGEX
                            if attempt == 0
                            else ANSWER_BLOCK_RETRY_REGEX
                        ),
                    )
                try:
                    parse_answer_block(text)
                except AnswerFormatError as exc:
                    raise _AnswerAttemptError(str(exc), usage) from exc
                if usage is None:
                    usage_is_exact = False
                else:
                    cumulative_usage = _sum_answer_usage(cumulative_usage, usage)
                return AnswerResponse(
                    text=text,
                    usage=cumulative_usage if usage_is_exact else None,
                    attempts=attempt + 1,
                    failed_attempts=attempt,
                    image_count=image_count,
                )
            except Exception as exc:
                attempt_usage = (
                    exc.usage if isinstance(exc, _AnswerAttemptError) else None
                )
                if attempt_usage is None:
                    usage_is_exact = False
                else:
                    cumulative_usage = _sum_answer_usage(
                        cumulative_usage, attempt_usage
                    )
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1 + attempt)
        assert last_error is not None
        raise last_error

    def count_answer_images(
        self,
        memory_items: list[dict[str, Any]],
        *,
        query_image: dict[str, Any] | None = None,
        category: str = "",
    ) -> int:
        """Count images attached by this client's benchmark-specific renderer."""
        _, image_paths = self._build_text_and_image_paths(
            memory_items, "", query_image, category
        )
        return len(image_paths)

    def _answer_openai_compatible(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
        prepend_memory_context: bool = True,
    ) -> tuple[str, dict[str, int] | None]:
        data: dict[str, Any] | None = None
        image_sides = (MAX_IMAGE_SIDE_FOR_ANSWER, *CONTEXT_RECOVERY_IMAGE_SIDES)
        for index, image_side in enumerate(image_sides):
            content = self._build_openai_content(
                memory_items,
                question_prompt,
                query_image,
                category,
                max_image_side=image_side,
                prepend_memory_context=prepend_memory_context,
            )
            messages: list[dict[str, Any]] = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})
            payload = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.num_predict,
                "messages": messages,
            }
            if self.think is not None:
                # vLLM applies Qwen's thinking switch while rendering the chat
                # template. Top-level ``think``/``extra_body`` fields are ignored
                # by its OpenAI-compatible request schema.
                payload["chat_template_kwargs"] = {"enable_thinking": self.think}
            try:
                data = self._post_json(f"{self.base_url}/chat/completions", payload)
                break
            except requests.HTTPError as exc:
                if index + 1 == len(image_sides) or not _is_context_length_error(exc):
                    raise
        assert data is not None
        usage = _normalize_answer_usage(data, backend="openai")
        choices = data.get("choices") or []
        if not choices:
            raise _AnswerAttemptError("Answer endpoint returned no choices", usage)
        message = choices[0].get("message") or {}
        answer = (
            message.get("content") or message.get("reasoning_content") or ""
        ).strip()
        if not answer:
            raise _AnswerAttemptError("Answer endpoint returned an empty response", usage)
        return answer, usage

    def _answer_prebuilt_openai_compatible(
        self,
        *,
        messages: list[dict[str, str]],
        memory_items: list[dict[str, Any]],
        query_image: dict[str, Any] | None,
        category: str,
        structured_regex: str,
    ) -> tuple[str, dict[str, int] | None]:
        _, image_paths = self._build_text_and_image_paths(
            memory_items,
            "",
            query_image,
            category,
            prepend_memory_context=False,
        )
        data: dict[str, Any] | None = None
        image_sides = (MAX_IMAGE_SIDE_FOR_ANSWER, *CONTEXT_RECOVERY_IMAGE_SIDES)
        for side_index, image_side in enumerate(image_sides):
            payload_messages: list[dict[str, Any]] = [dict(message) for message in messages]
            user_index = max(
                index
                for index, message in enumerate(payload_messages)
                if message["role"] == "user"
            )
            user_text = str(payload_messages[user_index]["content"])
            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for path in image_paths:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _encode_image_data_url(path, max_side=image_side)
                        },
                    }
                )
            payload_messages[user_index]["content"] = user_content
            payload: dict[str, Any] = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.num_predict,
                "messages": payload_messages,
                # Constrain only decoding; the harness-authored system/user
                # message text above remains byte-for-byte unchanged.
                "structured_outputs": {"regex": structured_regex},
            }
            if self.think is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.think}
            try:
                data = self._post_json(f"{self.base_url}/chat/completions", payload)
                break
            except requests.HTTPError as exc:
                if side_index + 1 == len(image_sides) or not _is_context_length_error(exc):
                    raise
        assert data is not None
        usage = _normalize_answer_usage(data, backend="openai")
        choices = data.get("choices") or []
        if not choices:
            raise _AnswerAttemptError("Answer endpoint returned no choices", usage)
        message = choices[0].get("message") or {}
        answer = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not answer:
            raise _AnswerAttemptError("Answer endpoint returned an empty response", usage)
        return answer, usage

    def _answer_ollama_native(
        self,
        *,
        system_prompt: str,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None = None,
        category: str = "",
        prepend_memory_context: bool = True,
    ) -> tuple[str, dict[str, int] | None]:
        content, images = self._build_user_content(
            memory_items,
            question_prompt,
            query_image,
            category,
            prepend_memory_context=prepend_memory_context,
        )
        messages: list[dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content, "images": images})
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            "messages": messages,
        }
        if self.think is not None:
            payload["think"] = self.think
        data = self._post_json(f"{self.base_url}/api/chat", payload)
        usage = _normalize_answer_usage(data, backend="ollama")
        message = (data.get("message", {}) or {})
        content = (message.get("content") or "").strip()
        if content:
            return content, usage
        # Some local Qwen3-VL Ollama builds return only `thinking`.
        thinking = (message.get("thinking") or "").strip()
        if not thinking:
            raise _AnswerAttemptError("Answer endpoint returned an empty response", usage)
        return thinking, usage

    def _answer_prebuilt_ollama_native(
        self,
        *,
        messages: list[dict[str, str]],
        memory_items: list[dict[str, Any]],
        query_image: dict[str, Any] | None,
        category: str,
    ) -> tuple[str, dict[str, int] | None]:
        _, image_paths = self._build_text_and_image_paths(
            memory_items,
            "",
            query_image,
            category,
            prepend_memory_context=False,
        )
        payload_messages: list[dict[str, Any]] = [dict(message) for message in messages]
        user_index = max(
            index for index, message in enumerate(payload_messages) if message["role"] == "user"
        )
        payload_messages[user_index]["images"] = [
            _encode_image(path) for path in image_paths
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
            "messages": payload_messages,
        }
        if self.think is not None:
            payload["think"] = self.think
        data = self._post_json(f"{self.base_url}/api/chat", payload)
        usage = _normalize_answer_usage(data, backend="ollama")
        message = data.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        if content:
            return content, usage
        thinking = (message.get("thinking") or "").strip()
        if not thinking:
            raise _AnswerAttemptError("Answer endpoint returned an empty response", usage)
        return thinking, usage

    def _build_openai_content(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
        *,
        max_image_side: int = MAX_IMAGE_SIDE_FOR_ANSWER,
        prepend_memory_context: bool = True,
    ) -> list[dict[str, Any]]:
        text, image_paths = self._build_text_and_image_paths(
            memory_items,
            question_prompt,
            query_image,
            category,
            prepend_memory_context=prepend_memory_context,
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _encode_image_data_url(path, max_side=max_image_side)
                    },
                }
            )
        return content

    def _build_user_content(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
        *,
        prepend_memory_context: bool = True,
    ) -> tuple[str, list[str]]:
        text, image_paths = self._build_text_and_image_paths(
            memory_items,
            question_prompt,
            query_image,
            category,
            prepend_memory_context=prepend_memory_context,
        )
        return text, [_encode_image(path) for path in image_paths]

    def _build_text_and_image_paths(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
        *,
        prepend_memory_context: bool = True,
    ) -> tuple[str, list[str]]:
        memory_text, image_paths = build_retrieved_memory_context(memory_items, category)
        lines = (
            [memory_text, "", question_prompt]
            if prepend_memory_context
            else [question_prompt]
        )
        image_num = len(image_paths)
        if query_image and query_image.get("path"):
            image_num += 1
            lines.append(f"Attached question image {image_num}.")
            image_paths.append(str(query_image["path"]))
        return "\n\n".join(lines), image_paths

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._session.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            # OpenAI-compatible servers normally return the actionable cause
            # (for example context length or image limits) in the response
            # body. Preserve it so endpoint recovery can distinguish a bad
            # payload from a transient tunnel failure.
            body = (resp.text or "").strip()
            if len(body) > 2000:
                body = body[:2000] + "..."
            detail = f"; response body: {body}" if body else ""
            raise requests.HTTPError(
                f"{exc}{detail}",
                response=resp,
                request=resp.request,
            ) from exc
        return resp.json()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _is_context_length_error(error: Exception) -> bool:
    message = str(error).lower()
    return "maximum model length" in message or "decoder prompt" in message


def _validate_prebuilt_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("Each prompt message must be a mapping")
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported prompt role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("Each prompt message must have non-empty string content")
        normalized.append({"role": role, "content": content})
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError("Prebuilt prompt messages require a user message")
    return normalized


def _normalize_answer_usage(
    payload: Mapping[str, Any],
    *,
    backend: str,
) -> dict[str, int] | None:
    if backend == "ollama":
        source: Mapping[str, Any] = payload
        prompt_keys = ("prompt_eval_count",)
        completion_keys = ("eval_count",)
        total_keys: tuple[str, ...] = ()
    else:
        raw = payload.get("usage")
        if not isinstance(raw, Mapping):
            return None
        source = raw
        prompt_keys = ("prompt_tokens", "input_tokens")
        completion_keys = ("completion_tokens", "output_tokens")
        total_keys = ("total_tokens",)
    if not any(key in source for key in prompt_keys) or not any(
        key in source for key in completion_keys
    ):
        return None
    prompt_tokens = _first_nonnegative_int(source, prompt_keys)
    completion_tokens = _first_nonnegative_int(source, completion_keys)
    total_tokens = _first_nonnegative_int(source, total_keys)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _first_nonnegative_int(source: Mapping[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key not in source:
            continue
        try:
            value = int(source.get(key) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid answer usage field {key}: {source.get(key)!r}") from exc
        if value < 0:
            raise ValueError(f"Invalid answer usage field {key}: {source.get(key)!r}")
        return value
    return 0


def _sum_answer_usage(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> dict[str, int]:
    return {
        key: int(left.get(key) or 0) + int(right.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _encode_image(path: str) -> str:
    p = Path(path)
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_image_data_url(
    path: str, *, max_side: int = MAX_IMAGE_SIDE_FOR_ANSWER
) -> str:
    image_bytes, mime = _prepare_image_bytes(path, max_side=max_side)
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _prepare_image_bytes(
    path: str, *, max_side: int = MAX_IMAGE_SIDE_FOR_ANSWER
) -> tuple[bytes, str]:
    p = Path(path)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with Image.open(p) as image:
        width, height = image.size
        if max(width, height) <= max_side:
            return p.read_bytes(), mime
        image.thumbnail((max_side, max_side))
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue(), "image/jpeg"
