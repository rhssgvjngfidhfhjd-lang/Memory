#!/usr/bin/env python3
"""Serve Qwen3-Embedding-0.6B with an isolated 2048-D compatibility output.

The model natively emits 1024 dimensions.  This endpoint appends 1024 zeros
to every normalized vector so consumers configured for 2048 dimensions can
use it without changing cosine similarities or nearest-neighbour rankings.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from typing import Any, Protocol

from serve_embeddings import EmbeddingApplication, Handler


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
NATIVE_DIM = 1024
OUTPUT_DIM = 2048


class EmbeddingDelegate(Protocol):
    model_name: str

    def embed(self, payload: dict[str, Any]) -> list[list[float]]: ...


def right_zero_pad(vector: list[float], output_dim: int = OUTPUT_DIM) -> list[float]:
    """Return a copied vector padded on the right to ``output_dim``."""
    if len(vector) > output_dim:
        raise ValueError(
            f"cannot zero-pad embedding of dimension {len(vector)} to {output_dim}"
        )
    return list(vector) + [0.0] * (output_dim - len(vector))


class PaddedEmbeddingApplication:
    """OpenAI-compatible application wrapper with auditable zero padding."""

    def __init__(
        self,
        delegate: EmbeddingDelegate,
        *,
        native_dim: int = NATIVE_DIM,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        if output_dim < native_dim:
            raise ValueError("output_dim must be greater than or equal to native_dim")
        self.delegate = delegate
        self.model_name = delegate.model_name
        self.native_dim = int(native_dim)
        self.dim = int(output_dim)

    def embed(self, payload: dict[str, Any]) -> list[list[float]]:
        vectors = self.delegate.embed(payload)
        for vector in vectors:
            if len(vector) != self.native_dim:
                raise ValueError(
                    f"native embedding dimension is {len(vector)}, "
                    f"expected {self.native_dim}"
                )
        return [right_zero_pad(vector, self.dim) for vector in vectors]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    native_args = argparse.Namespace(
        model=MODEL_NAME,
        dim=NATIVE_DIM,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    native_app = EmbeddingApplication(native_args)
    Handler.app = PaddedEmbeddingApplication(native_app)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"Serving {MODEL_NAME} ({NATIVE_DIM}-D native, zero-padded to "
        f"{OUTPUT_DIM}-D) on http://{args.host}:{args.port}/v1",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
