from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from serve_qwen3_embedding06_2048 import (  # noqa: E402
    PaddedEmbeddingApplication,
    right_zero_pad,
)


class FakeEmbeddingApplication:
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    def embed(self, payload):
        del payload
        return [[0.6, 0.8], [-1.0, 0.0]]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def test_right_zero_pad_preserves_norm_and_cosine() -> None:
    left = [0.6, 0.8]
    right = [-1.0, 0.0]
    padded_left = right_zero_pad(left, 4)
    padded_right = right_zero_pad(right, 4)

    assert padded_left == [0.6, 0.8, 0.0, 0.0]
    assert math.sqrt(sum(value * value for value in padded_left)) == pytest.approx(1.0)
    assert cosine(padded_left, padded_right) == pytest.approx(cosine(left, right))


def test_application_validates_native_dimension_and_reports_output_dimension() -> None:
    app = PaddedEmbeddingApplication(
        FakeEmbeddingApplication(), native_dim=2, output_dim=4
    )

    assert app.dim == 4
    assert app.embed({}) == [[0.6, 0.8, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]


def test_right_zero_pad_rejects_shrinking() -> None:
    with pytest.raises(ValueError, match="cannot zero-pad"):
        right_zero_pad([1.0, 2.0, 3.0], 2)
