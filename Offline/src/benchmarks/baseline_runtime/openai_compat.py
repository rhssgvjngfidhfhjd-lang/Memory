from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def embed_texts(texts: list[str], config: dict[str, Any]) -> list[list[float]]:
    if not texts:
        return []
    base_url = str(config.get("embedding_base_url") or "").rstrip("/")
    if not base_url:
        raise ValueError("embedding_base_url is required for this baseline")
    endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
    body = json.dumps(
        {"model": str(config["embedding_model"]), "input": texts},
        # Real benchmark files can contain lone UTF-16 surrogates. Escaping
        # non-ASCII here keeps the JSON valid without losing those code units.
        ensure_ascii=True,
    ).encode("utf-8")
    key_env = str(config.get("embedding_api_key_env") or "EMBEDDING_API_KEY")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(key_env) or str(config.get("embedding_api_key") or "EMPTY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    timeout = float(config.get("request_timeout") or 180)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = sorted(payload.get("data") or [], key=lambda row: int(row.get("index", 0)))
    vectors = [[float(value) for value in row["embedding"]] for row in rows]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
        )
    expected = int(config.get("embedding_dim") or 0)
    if expected and any(len(vector) != expected for vector in vectors):
        dimensions = sorted({len(vector) for vector in vectors})
        raise ValueError(f"embedding dimension mismatch: expected {expected}, got {dimensions}")
    return vectors
