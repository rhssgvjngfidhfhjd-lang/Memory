from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Protocol
from urllib.parse import urlparse

import requests
from PIL import Image

from .models import PrimitiveCandidate, Settings


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
        max_primitives: int,
        caption_guided_prompt: str | None = None,
    ):
        self.vlm = vlm
        self.discovery_prompt = discovery_prompt.replace(
            "__MAX_PRIMITIVES__", str(max_primitives)
        )
        self.relocalize_prompt = relocalize_prompt
        self.caption_guided_prompt = (
            caption_guided_prompt.replace("__MAX_PRIMITIVES__", str(max_primitives))
            if caption_guided_prompt
            else None
        )
        self.max_primitives = max_primitives

    def discover(
        self, image: Image.Image, focus_context: str | None = None
    ) -> list[PrimitiveCandidate]:
        raw = self.vlm.generate(self.build_prompt(focus_context), image)
        return parse_candidates(raw)[: self.max_primitives]

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
        self, image: Image.Image, candidate: PrimitiveCandidate
    ) -> PrimitiveCandidate | None:
        prompt = self.relocalize_prompt.replace("__LABEL__", candidate.label).replace(
            "__BBOX__", str(list(candidate.bbox_norm))
        )
        candidates = parse_candidates(
            self.vlm.generate(prompt, image), default_label=candidate.label
        )
        return candidates[0] if candidates else None


def parse_candidates(
    text: str, *, default_label: str | None = None
) -> list[PrimitiveCandidate]:
    """Parse the small JSON contract while tolerating markdown wrappers."""
    payload = _load_json_payload(text)
    if isinstance(payload, dict):
        if isinstance(payload.get("objects"), list):
            payload = payload["objects"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("VLM response must be a JSON array")

    candidates: list[PrimitiveCandidate] = []
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
        candidates.append(PrimitiveCandidate(label=label[:200], bbox_norm=coords))
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
            raise ValueError(f"Invalid VLM JSON: {exc}") from exc


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
