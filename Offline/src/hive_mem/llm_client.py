import threading
import base64
import mimetypes
from pathlib import Path
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Sequence


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    usage: dict[str, int]
    attempts: int = 1
    failed_attempts: int = 0

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> str:
        raise NotImplementedError


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        timeout: int = 60,
    ):
        keys = _normalize_api_keys(api_key)
        if not keys:
            raise ValueError("api=True requires at least one api_key.")

        self.model = model
        self.api_base = api_base
        self.api_keys = keys
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = retry_sleep
        self.timeout = timeout
        self._client_cache = {}
        self._lock = threading.Lock()
        self._key_index = 0

    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> str:
        return self.generate_with_usage(prompt, image_paths=image_paths).text

    def generate_with_usage(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> GenerationResponse:
        user_content = _build_user_content(prompt, image_paths)
        last_error = None
        cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage_is_exact = True
        for attempt in range(self.max_retries):
            client = self._next_client()
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                )
                message = completion.choices[0].message.content
                usage = _normalize_usage(completion.usage)
                if usage:
                    cumulative_usage = _sum_usage(cumulative_usage, usage)
                else:
                    usage_is_exact = False
                return GenerationResponse(
                    text=(message or "").strip(),
                    usage=cumulative_usage if usage_is_exact else {},
                    attempts=attempt + 1,
                    failed_attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                # Provider usage is unavailable for an exception, so an
                # eventual success cannot claim exact cumulative token cost.
                usage_is_exact = False
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"API generation failed after {self.max_retries} attempts: {last_error}")

    def _next_client(self):
        with self._lock:
            api_key = self.api_keys[self._key_index % len(self.api_keys)]
            self._key_index += 1
            client = self._client_cache.get(api_key)
            if client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "API mode requires the 'openai' package to be installed."
                    ) from exc
                client = OpenAI(
                    base_url=self.api_base,
                    api_key=api_key,
                    # Keep all retries in generate_with_usage so every actual
                    # API invocation is visible to Calls metrics.
                    max_retries=0,
                    timeout=self.timeout,
                )
                self._client_cache[api_key] = client
            return client


def _normalize_usage(usage: Any) -> dict[str, int]:
    """Return the three build-token counters used by result metrics."""
    if usage is None:
        return {}

    def value(name: str) -> int:
        raw = usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)
        return int(raw or 0)

    prompt_tokens = value("prompt_tokens")
    completion_tokens = value("completion_tokens")
    total_tokens = value("total_tokens") or prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _sum_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        key: int(left.get(key) or 0) + int(right.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _normalize_api_keys(api_key) -> List[str]:
    if api_key is None:
        return []
    if isinstance(api_key, str):
        key = api_key.strip()
        return [key] if key else []
    if isinstance(api_key, Sequence):
        return [str(key).strip() for key in api_key if str(key).strip()]
    raise ValueError("api_key must be a string or a list of strings.")


def _build_user_content(
    prompt: str,
    image_paths: Sequence[str] | None,
) -> str | list[dict[str, Any]]:
    """Build an OpenAI-compatible user message without altering text-only calls."""
    paths = [Path(path) for path in (image_paths or []) if str(path).strip()]
    if not paths:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(path)},
        }
        for path in paths
    )
    return content


def _encode_image_data_url(path: Path) -> str:
    """Encode the source image bytes directly; build-time input is not resized."""
    if not path.is_file():
        raise FileNotFoundError(f"Build image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
