"""Thread-safe round-robin pool over multiple API keys, with per-key cooldown.

Generic, no dependency on ``openai`` or any specific project — any caller that
holds a list of API keys can use this to spread load and dodge per-key rate
limits (e.g. NVIDIA integrate API's 429s).
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

_KEY_RE = re.compile(r'"(nvapi-[^"]+)"')

_DEFAULT_KEY_FILE = Path(__file__).resolve().parent / "apikey"


def load_keys_from_file(path: str | Path) -> list[str]:
    """Parse a "bare list literal" file of quoted keys, one-per-line, with
    ``#`` comments and trailing commas allowed (i.e. the body of a Python
    list without the enclosing brackets). Dedupes while preserving order.
    """
    text = Path(path).read_text(encoding="utf-8")
    seen: set[str] = set()
    keys: list[str] = []
    for line in text.splitlines():
        code = line.split("#", 1)[0]
        for match in _KEY_RE.finditer(code):
            key = match.group(1)
            if key not in seen:
                seen.add(key)
                keys.append(key)
    if not keys:
        raise ValueError(f"no API keys found in {path}")
    return keys


class KeyPool:
    """Round-robin over ``keys``, skipping any key still in cooldown.

    Call ``mark_rate_limited(key)`` right after a 429 for that key; it will
    be skipped by ``next_key()`` for ``cooldown_s`` seconds, then rejoin the
    rotation automatically.
    """

    def __init__(self, keys: list[str], cooldown_s: float = 30.0) -> None:
        if not keys:
            raise ValueError("KeyPool requires at least one key")
        self._keys = list(keys)
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[str, float] = {}
        self._cursor = 0
        self._lock = threading.Lock()

    def size(self) -> int:
        return len(self._keys)

    def next_key(self) -> str:
        """Return the next non-cooling key. If every key is cooling down,
        return the one whose cooldown expires soonest (better to retry a
        near-ready key than to fail outright)."""
        now = time.monotonic()
        with self._lock:
            n = len(self._keys)
            for offset in range(n):
                idx = (self._cursor + offset) % n
                key = self._keys[idx]
                if self._cooldown_until.get(key, 0.0) <= now:
                    self._cursor = (idx + 1) % n
                    return key
            # all cooling down -- pick soonest-to-expire, don't advance cursor state oddly
            soonest_key = min(self._keys, key=lambda k: self._cooldown_until.get(k, 0.0))
            self._cursor = (self._keys.index(soonest_key) + 1) % n
            return soonest_key

    def mark_rate_limited(self, key: str) -> None:
        with self._lock:
            self._cooldown_until[key] = time.monotonic() + self._cooldown_s

    def mark_invalid(self, key: str) -> None:
        """Permanently exclude a key (e.g. revoked -> 401/403) from rotation."""
        with self._lock:
            self._cooldown_until[key] = float("inf")


_default_pool: KeyPool | None = None
_default_pool_lock = threading.Lock()


def default_pool(path: str | Path | None = None, cooldown_s: float = 30.0) -> KeyPool:
    """Process-wide singleton pool loaded from ``apikey`` (or ``path``)."""
    global _default_pool
    if _default_pool is not None and path is None:
        return _default_pool
    with _default_pool_lock:
        if _default_pool is not None and path is None:
            return _default_pool
        keys = load_keys_from_file(path or _DEFAULT_KEY_FILE)
        pool = KeyPool(keys, cooldown_s=cooldown_s)
        if path is None:
            _default_pool = pool
        return pool
