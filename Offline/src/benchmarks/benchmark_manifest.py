"""Validated configuration for full benchmark completeness checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkExpectation:
    """Expected shape of one complete benchmark run."""

    judge_name: str
    question_count: int
    dataset_count: int | None = None
    sample_count: int | None = None
    require_all_datasets: bool | None = None
    data_dir_name: str | None = None
    variants: dict[str, int] = field(default_factory=dict)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _optional_positive_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name=field_name)


def load_benchmark_manifest(
    path: str | Path,
    *,
    required_benchmarks: Iterable[str] = (),
) -> dict[str, BenchmarkExpectation]:
    """Load and validate an independent full-benchmark manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read benchmark manifest {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in benchmark manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("benchmark manifest must use schema_version=1")
    rows = payload.get("benchmarks")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("benchmark manifest must contain a non-empty benchmarks object")

    expectations: dict[str, BenchmarkExpectation] = {}
    for benchmark, row in rows.items():
        if not isinstance(benchmark, str) or not benchmark.strip() or not isinstance(row, dict):
            raise ValueError("each benchmark manifest entry must be a named object")
        prefix = f"benchmarks.{benchmark}"
        judge_name = row.get("judge_name")
        if not isinstance(judge_name, str) or not judge_name.strip():
            raise ValueError(f"{prefix}.judge_name must be a non-empty string")
        question_count = _positive_int(
            row.get("question_count"),
            field_name=f"{prefix}.question_count",
        )
        variants_payload = row.get("variants") or {}
        if not isinstance(variants_payload, dict):
            raise ValueError(f"{prefix}.variants must be an object")
        variants = {
            str(name): _positive_int(count, field_name=f"{prefix}.variants.{name}")
            for name, count in variants_payload.items()
        }
        if variants and sum(variants.values()) != question_count:
            raise ValueError(
                f"{prefix}.variants total {sum(variants.values())} does not match "
                f"question_count {question_count}"
            )
        require_all_datasets = row.get("require_all_datasets")
        if require_all_datasets is not None and not isinstance(require_all_datasets, bool):
            raise ValueError(f"{prefix}.require_all_datasets must be a boolean")
        data_dir_name = row.get("data_dir_name")
        if data_dir_name is not None and (
            not isinstance(data_dir_name, str) or not data_dir_name.strip()
        ):
            raise ValueError(f"{prefix}.data_dir_name must be a non-empty string")
        expectations[benchmark] = BenchmarkExpectation(
            judge_name=judge_name,
            question_count=question_count,
            dataset_count=_optional_positive_int(
                row.get("dataset_count"),
                field_name=f"{prefix}.dataset_count",
            ),
            sample_count=_optional_positive_int(
                row.get("sample_count"),
                field_name=f"{prefix}.sample_count",
            ),
            require_all_datasets=require_all_datasets,
            data_dir_name=data_dir_name,
            variants=variants,
        )

    memgallery = expectations.get("Mem-Gallery")
    if memgallery is not None and (
        memgallery.dataset_count is None
        or memgallery.require_all_datasets is None
    ):
        raise ValueError(
            "benchmarks.Mem-Gallery requires dataset_count and require_all_datasets"
        )
    worldmemarena = expectations.get("WorldMemArena")
    if worldmemarena is not None and (
        worldmemarena.sample_count is None
        or worldmemarena.data_dir_name is None
    ):
        raise ValueError(
            "benchmarks.WorldMemArena requires sample_count and data_dir_name"
        )
    h2hmem = expectations.get("H2HMEM")
    if h2hmem is not None and not h2hmem.variants:
        raise ValueError("benchmarks.H2HMEM requires variants")

    missing = sorted(set(required_benchmarks).difference(expectations))
    if missing:
        raise ValueError(f"benchmark manifest is missing required benchmarks: {missing}")
    return expectations
