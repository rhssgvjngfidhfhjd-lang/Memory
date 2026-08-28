"""OpenAI-compatible chat client that rotates across a ``KeyPool``.

Drop-in for ``openai.OpenAI`` at call sites that only use
``client.chat.completions.create(...)``: this exposes ``.create(**kwargs)``
with the same kwargs/return type, so callers can swap the client without
touching their retry or response-parsing logic.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from key_pool import KeyPool

logger = logging.getLogger(__name__)

# Errors worth rotating to a different key for (as opposed to a prompt-shaped
# error that would fail identically on any key).
_ROTATE_ON = (RateLimitError, APITimeoutError, APIConnectionError)


class MultiKeyOpenAI:
    def __init__(
        self,
        pool: KeyPool,
        base_url: str,
        max_retries_per_call: int | None = None,
    ) -> None:
        self._pool = pool
        self._base_url = base_url
        self._max_retries_per_call = max_retries_per_call or pool.size()
        self._clients: dict[str, OpenAI] = {}
        self._clients_lock = threading.Lock()

    def _client_for(self, key: str) -> OpenAI:
        client = self._clients.get(key)
        if client is None:
            with self._clients_lock:
                client = self._clients.get(key)
                if client is None:
                    client = OpenAI(api_key=key, base_url=self._base_url)
                    self._clients[key] = client
        return client

    def create(self, **kwargs: Any) -> Any:
        """Same signature/return as ``client.chat.completions.create(**kwargs)``."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries_per_call):
            key = self._pool.next_key()
            client = self._client_for(key)
            try:
                return client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                self._pool.mark_rate_limited(key)
                last_exc = exc
                logger.warning(
                    "key ...%s rate-limited (attempt %d/%d), rotating",
                    key[-6:], attempt + 1, self._max_retries_per_call,
                )
            except (APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "key ...%s transient error %r (attempt %d/%d), rotating",
                    key[-6:], exc, attempt + 1, self._max_retries_per_call,
                )
            except APIStatusError as exc:
                if exc.status_code in (401, 403):
                    # Key itself is broken (revoked/invalid) -- not a transient
                    # condition, so exclude it permanently and try another key.
                    self._pool.mark_invalid(key)
                    last_exc = exc
                    logger.warning(
                        "key ...%s invalid (HTTP %d), permanently excluding",
                        key[-6:], exc.status_code,
                    )
                    continue
                if exc.status_code in (429, 500, 502, 503, 504):
                    if exc.status_code == 429:
                        self._pool.mark_rate_limited(key)
                    last_exc = exc
                    logger.warning(
                        "key ...%s HTTP %d (attempt %d/%d), rotating",
                        key[-6:], exc.status_code, attempt + 1, self._max_retries_per_call,
                    )
                    continue
                raise
        assert last_exc is not None
        raise last_exc
