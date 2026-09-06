from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.call_trace import load_call_rows, summarize_call_rows

from nltk.stem import PorterStemmer

from benchmarks.memgallery_harness.runner.answer_client import (
    build_retrieved_memory_context,
)


MEMORY_METRICS_FILENAME = "memory_metrics.json"
CALL_TRACE_FILENAME = "call_trace.jsonl"
CALL_METRICS_FILENAME = "call_metrics.json"
EFFICIENCY_METRICS_FILENAME = "efficiency_metrics.json"
RETRIEVAL_MEMORY_TOKEN_FILENAME = "retrieval_memory_token.json"
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
_PORTER_STEMMER = PorterStemmer()


def normalize_answer(text: str) -> str:
    """Match Mem-Gallery's official ``normalize_answer_universal`` logic."""
    s = str(text).lower()
    dot_placeholder = "DOTPLACEHOLDER"
    underscore_placeholder = "UNDERSCOREPLACEHOLDER"
    s = re.sub(r"(?<=\d)\.(?=\d)", dot_placeholder, s)
    s = s.replace("_", underscore_placeholder)
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = s.replace(dot_placeholder, ".")
    s = s.replace(underscore_placeholder, "_")
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = [_PORTER_STEMMER.stem(token) for token in normalize_answer(prediction).split()]
    gold = [_PORTER_STEMMER.stem(token) for token in normalize_answer(ground_truth).split()]
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def provenance_hit(source_groups: list[list[str]], clue_ids: list[str], k: int = 5) -> float:
    """HIT@k over grouped provenance: one retrieved MAU may cover several
    dialogue ids (source_dialogue_ids); a hit means any of the top-k groups
    intersects the clue set. (Replaced the flat retrieved_ids version on
    2026-08-07 when hive_mem/core/metrics.py was merged in.)"""
    if not clue_ids:
        return 0.0
    clues = set(clue_ids)
    return float(any(clues.intersection(group) for group in source_groups[:k]))


def summarize_results(results: list[dict], k: int = 5) -> dict:
    rows = []
    by_category = defaultdict(list)
    for result in results:
        row = {
            "f1": f1_score(result.get("system_answer", ""), result.get("original_answer", "")),
            "em": exact_match(result.get("system_answer", ""), result.get("original_answer", "")),
            "hit": provenance_hit(
                result.get("retrieved_source_groups", []),
                result.get("clue", []),
                k=k,
            ),
        }
        rows.append(row)
        by_category[result.get("category", "")].append(row)

    def summary(values: list[dict]) -> dict:
        count = len(values)
        return {
            "count": count,
            "f1": sum(v["f1"] for v in values) / count if count else 0.0,
            "em": sum(v["em"] for v in values) / count if count else 0.0,
            f"retrieval_hitrate@{k}": sum(v["hit"] for v in values) / count if count else 0.0,
        }

    output = summary(rows)
    output["by_category"] = {c: summary(v) for c, v in sorted(by_category.items())}
    return output


def merge_llm_judge_metrics(metrics: dict, judge_metrics: dict) -> dict:
    """Add overall and per-category Judge accuracy to benchmark metrics."""

    def merge_row(row: dict, judge_row: dict) -> dict:
        merged = {
            "f1": row.get("f1"),
            "em": row.get("em", row.get("exact_match")),
            "llm_judge": judge_row.get("accuracy"),
        }
        merged.update(
            {
                key: value
                for key, value in row.items()
                if key not in {"f1", "em", "exact_match", "llm_judge"}
            }
        )
        return merged

    merged = merge_row(
        {key: value for key, value in metrics.items() if key != "by_category"},
        judge_metrics,
    )
    if isinstance(judge_metrics.get("calls"), dict):
        calls = dict(merged.get("calls") or {})
        calls["judge"] = dict(judge_metrics["calls"])
        merged["calls"] = calls
    judge_categories = judge_metrics.get("by_category") or {}
    merged["by_category"] = {
        category: merge_row(values, judge_categories.get(category, {}))
        for category, values in (metrics.get("by_category") or {}).items()
    }
    return merged


def merge_existing_llm_judge_metrics(
    metrics: dict[str, Any],
    result_dir: str | Path,
) -> dict[str, Any]:
    """Preserve a complete Judge artifact when a finished run is resumed."""
    path = Path(result_dir) / "llm_judge_metrics.json"
    if not path.is_file():
        return metrics
    judge = json.loads(path.read_text(encoding="utf-8"))
    expected = int(metrics.get("count") or 0)
    complete = (
        int(judge.get("count", -1)) == expected
        and int(judge.get("valid_count", -1)) == expected
        and int(judge.get("judge_errors", -1)) == 0
        and not bool(judge.get("provisional", True))
    )
    return merge_llm_judge_metrics(metrics, judge) if complete else metrics


def calculate_memory_metrics(
    index_root: str | Path,
    *,
    tokenizer_name: str = "",
    sample_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate build-token usage and active-summary characters for one bank.

    New build traces carry API-reported ``llm_usage``. Historical traces are
    backfilled with the executor tokenizer using the recorded event and raw
    response, so old result directories do not require another model run.
    """
    root = Path(index_root)
    selected_samples = _normalize_sample_ids(sample_ids)
    traces = list(_iter_build_traces(root, sample_ids=selected_samples))
    if not traces:
        raise FileNotFoundError(f"No build traces found under {root / 'datasets'}")

    usage = {key: 0 for key in TOKEN_KEYS}
    missing_usage = []
    for row in traces:
        row_usage = _token_usage(row.get("llm_usage"))
        if row_usage is None:
            missing_usage.append(row)
        else:
            for key in TOKEN_KEYS:
                usage[key] += row_usage[key]

    if missing_usage:
        estimated = _estimate_trace_tokens(root, missing_usage, tokenizer_name)
        for key in TOKEN_KEYS:
            usage[key] += estimated[key]

    return {
        "memory_build_tokens": usage,
        "summary_characters": _count_summary_characters(
            root, sample_ids=selected_samples
        ),
    }


def calculate_cost_mb(
    index_root: str | Path,
    sample_ids: Iterable[str],
    *,
    input_price: float | None,
    output_price: float | None,
) -> dict[str, Any]:
    """Compute exact Memory-Bank cost, summed then divided by conversations.

    Prices are denominated in USD per million tokens.
    """

    samples = _normalize_sample_ids(sample_ids) or ()
    sample_count = len(samples)
    base = {
        "input_tokens": None,
        "output_tokens": None,
        "input_price": input_price,
        "output_price": output_price,
        "cost_sum": None,
        "num_samples": sample_count,
        "mean_per_sample": None,
        "formula": None,
        "aggregation": "sum_build_cost_divided_by_samples",
        "token_source": "provider_usage",
        "available": False,
    }
    if not samples:
        return {**base, "reason": "No evaluated samples were supplied."}
    prices = _validated_cost_prices(input_price, output_price)
    if prices is None:
        return {
            **base,
            "reason": (
                "Set cost_mb_input_price and cost_mb_output_price in "
                "configs/defaults.json."
            ),
        }
    input_coefficient, output_coefficient = prices
    root = Path(index_root)
    input_tokens = 0
    output_tokens = 0
    missing_samples: list[str] = []
    missing_usage: list[str] = []
    for sample_id in samples:
        trace_path = _build_trace_path(root / "datasets" / sample_id)
        if trace_path is None:
            missing_samples.append(sample_id)
            continue
        rows = list(_read_jsonl(trace_path))
        if not rows:
            missing_samples.append(sample_id)
            continue
        for trace_index, row in enumerate(rows, start=1):
            usage = _token_usage(row.get("llm_usage"))
            if usage is None:
                missing_usage.append(f"{sample_id}:{trace_index}")
                continue
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
    if missing_samples or missing_usage:
        details = []
        if missing_samples:
            details.append(f"missing build traces for samples={missing_samples}")
        if missing_usage:
            preview = missing_usage[:10]
            suffix = "..." if len(missing_usage) > len(preview) else ""
            details.append(f"missing exact provider usage at {preview}{suffix}")
        return {**base, "reason": "; ".join(details)}

    cost_sum = (
        input_tokens * input_coefficient
        + output_tokens * output_coefficient
    ) / 1_000_000
    mean_per_sample = cost_sum / sample_count
    formula = (
        f"(({input_tokens} / 1000000) * {input_coefficient:g} + "
        f"({output_tokens} / 1000000) * {output_coefficient:g}) / "
        f"{sample_count} = {mean_per_sample:.12g}"
    )
    return {
        **base,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price": input_coefficient,
        "output_price": output_coefficient,
        "cost_sum": cost_sum,
        "mean_per_sample": mean_per_sample,
        "formula": formula,
        "available": True,
    }


def calculate_cost_qa(
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
    input_price: float | None,
    output_price: float | None,
) -> dict[str, Any]:
    """Compute exact QA-generation cost, summed then divided by samples.

    Each row represents one benchmark question. The denominator is the number
    of unique benchmark samples, never the number of questions. Price
    prices are denominated in USD per million tokens.
    """
    sample_ids = {
        str(row.get(sample_id_field) or "").strip()
        for row in results
        if str(row.get(sample_id_field) or "").strip()
    }
    missing_sample_rows = [
        index
        for index, row in enumerate(results, start=1)
        if not str(row.get(sample_id_field) or "").strip()
    ]
    num_queries = len(results)
    num_samples = len(sample_ids)
    attempts = [
        max(int(row.get("answer_attempts") or 0), 0)
        for row in results
        if row.get("answer_attempts") is not None
    ]
    base = {
        "input_tokens": None,
        "output_tokens": None,
        "input_price": input_price,
        "output_price": output_price,
        "cost_sum": None,
        "num_queries": num_queries,
        "num_samples": num_samples,
        "mean_per_sample": None,
        "formula": None,
        "aggregation": "sum_qa_cost_divided_by_samples",
        "token_source": "provider_usage",
        "total_attempts": sum(attempts) if len(attempts) == num_queries else None,
        "retried_queries": (
            sum(value > 1 for value in attempts)
            if len(attempts) == num_queries
            else None
        ),
        "available": False,
    }
    if not results:
        return {**base, "reason": "No evaluated QA results were supplied."}
    if missing_sample_rows:
        return {
            **base,
            "reason": (
                f"Missing {sample_id_field!r} for QA result rows "
                f"{missing_sample_rows[:10]}."
            ),
        }
    prices = _validated_cost_prices(input_price, output_price)
    if prices is None:
        return {
            **base,
            "reason": (
                "Set cost_qa_input_price and cost_qa_output_price in "
                "configs/defaults.json."
            ),
        }
    input_coefficient, output_coefficient = prices

    failed_rows = [
        index
        for index, row in enumerate(results, start=1)
        if str(row.get("error") or "").strip()
    ]
    missing_usage_rows = []
    input_tokens = 0
    output_tokens = 0
    for index, row in enumerate(results, start=1):
        usage = _token_usage(row.get("answer_token_usage"))
        if usage is None:
            missing_usage_rows.append(index)
            continue
        input_tokens += usage["prompt_tokens"]
        output_tokens += usage["completion_tokens"]
    if failed_rows or missing_usage_rows:
        details = []
        if failed_rows:
            details.append(f"answer failures at QA rows {failed_rows[:10]}")
        if missing_usage_rows:
            details.append(
                "missing exact provider usage at QA rows "
                f"{missing_usage_rows[:10]}"
            )
        return {**base, "reason": "; ".join(details)}

    cost_sum = (
        input_tokens * input_coefficient
        + output_tokens * output_coefficient
    ) / 1_000_000
    mean_per_sample = cost_sum / num_samples
    formula = (
        f"(({input_tokens} / 1000000) * {input_coefficient:g} + "
        f"({output_tokens} / 1000000) * {output_coefficient:g}) / "
        f"{num_samples} = {mean_per_sample:.12g}"
    )
    return {
        **base,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price": input_coefficient,
        "output_price": output_coefficient,
        "cost_sum": cost_sum,
        "mean_per_sample": mean_per_sample,
        "formula": formula,
        "available": True,
    }


def load_model_efficiency_profile(
    config_path: str | Path,
    model: str,
) -> dict[str, Any]:
    """Load and validate one model's pricing and modeled-latency constants."""
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = (payload.get("models") or {}).get(model)
    if not isinstance(raw, dict):
        raise KeyError(f"No efficiency profile for model {model!r} in {path}")
    pricing = raw.get("pricing") or {}
    latency = raw.get("latency") or {}
    profile = {
        "model": model,
        "config_path": str(path),
        "pricing": {
            "input_per_million_usd": float(
                pricing["input_per_million_usd"]
            ),
            "output_per_million_usd": float(
                pricing["output_per_million_usd"]
            ),
            "source": str(pricing.get("source") or ""),
        },
        "latency": {
            "base_seconds": float(latency["base_seconds"]),
            "input_seconds_per_token": float(
                latency["input_seconds_per_token"]
            ),
            "output_seconds_per_token": float(
                latency["output_seconds_per_token"]
            ),
            "image_seconds": float(latency["image_seconds"]),
            "source": str(latency.get("source") or ""),
        },
    }
    numeric_values = [
        profile["pricing"]["input_per_million_usd"],
        profile["pricing"]["output_per_million_usd"],
        profile["latency"]["base_seconds"],
        profile["latency"]["input_seconds_per_token"],
        profile["latency"]["output_seconds_per_token"],
        profile["latency"]["image_seconds"],
    ]
    if any(not math.isfinite(value) or value < 0 for value in numeric_values):
        raise ValueError(f"Efficiency coefficients must be finite and nonnegative: {path}")
    return profile


def _inference_aggregate(
    *,
    num_samples: int,
    source: str,
    input_tokens: int | None,
    output_tokens: int | None,
    calls: int | None,
    image_count: int | None,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": calls,
        "image_count": image_count,
        "num_samples": num_samples,
        "source": source,
        "reason": reason,
    }


def _answer_inference_aggregate(
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
) -> dict[str, Any]:
    sample_ids = {
        str(row.get(sample_id_field) or "").strip()
        for row in results
        if str(row.get(sample_id_field) or "").strip()
    }
    input_tokens = 0
    output_tokens = 0
    calls = 0
    image_count = 0
    missing_usage: list[int] = []
    missing_counts: list[int] = []
    missing_images: list[int] = []
    for index, row in enumerate(results, start=1):
        usage = _token_usage(row.get("answer_token_usage"))
        attempts = row.get("answer_attempts")
        images_per_attempt = row.get("answer_image_count")
        if usage is None:
            missing_usage.append(index)
        else:
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
        try:
            normalized_attempts = int(attempts)
            if normalized_attempts < 1:
                raise ValueError
        except (TypeError, ValueError):
            missing_counts.append(index)
            normalized_attempts = 0
        else:
            calls += normalized_attempts
        try:
            normalized_images = int(images_per_attempt)
            if normalized_images < 0:
                raise ValueError
        except (TypeError, ValueError):
            missing_images.append(index)
        else:
            image_count += normalized_images * normalized_attempts
    reasons = []
    if missing_usage:
        reasons.append(f"missing QA usage at rows {missing_usage[:10]}")
        input_tokens = output_tokens = None
    if missing_counts:
        reasons.append(f"missing QA attempts at rows {missing_counts[:10]}")
        calls = None
    if missing_images:
        reasons.append(f"missing QA image counts at rows {missing_images[:10]}")
        image_count = None
    return _inference_aggregate(
        num_samples=len(sample_ids),
        source="answer_results",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=calls,
        image_count=image_count,
        reason="; ".join(reasons),
    )


def _call_trace_inference_aggregate(
    trace_path: str | Path,
    *,
    phase: str,
    num_samples: int,
) -> dict[str, Any]:
    path = Path(trace_path)
    if not path.is_file():
        return _inference_aggregate(
            num_samples=num_samples,
            source=f"call_trace:{phase}",
            input_tokens=None,
            output_tokens=None,
            calls=None,
            image_count=None,
            reason=f"missing call trace: {path}",
        )
    rows = [row for row in _read_jsonl(path) if row.get("phase") == phase]
    input_tokens = 0
    output_tokens = 0
    image_count = 0
    missing_usage: list[int] = []
    missing_images: list[int] = []
    for index, row in enumerate(rows, start=1):
        usage = _token_usage(row)
        if usage is None:
            missing_usage.append(index)
        else:
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
        try:
            images = int(row.get("image_count"))
            if images < 0:
                raise ValueError
        except (TypeError, ValueError):
            missing_images.append(index)
        else:
            image_count += images
    reasons = []
    if missing_usage:
        reasons.append(f"missing {phase} usage at calls {missing_usage[:10]}")
        input_tokens = output_tokens = None
    if missing_images:
        reasons.append(f"missing {phase} image counts at calls {missing_images[:10]}")
        image_count = None
    return _inference_aggregate(
        num_samples=num_samples,
        source=f"call_trace:{phase}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=len(rows),
        image_count=image_count,
        reason="; ".join(reasons),
    )


def _hivemem_build_inference_aggregate(
    index_root: str | Path,
    sample_ids: Iterable[str],
) -> dict[str, Any]:
    samples = _normalize_sample_ids(sample_ids) or ()
    input_tokens = 0
    output_tokens = 0
    calls = 0
    image_count = 0
    missing: list[str] = []
    for sample_id in samples:
        trace_path = _build_trace_path(Path(index_root) / "datasets" / sample_id)
        rows = list(_read_jsonl(trace_path)) if trace_path is not None else []
        if not rows:
            missing.append(f"{sample_id}:trace")
            continue
        for trace_index, row in enumerate(rows, start=1):
            usage = _token_usage(row.get("llm_usage"))
            counts = _validated_call_counts(
                row.get("llm_attempts"), row.get("llm_failed_attempts")
            )
            try:
                images_per_attempt = int(row.get("executor_image_count"))
                if images_per_attempt < 0:
                    raise ValueError
            except (TypeError, ValueError):
                images_per_attempt = -1
            if usage is None or counts is None or images_per_attempt < 0:
                missing.append(f"{sample_id}:{trace_index}")
                continue
            attempts, _ = counts
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
            calls += attempts
            image_count += images_per_attempt * attempts
    if missing:
        return _inference_aggregate(
            num_samples=len(samples),
            source="hivemem_build_trace",
            input_tokens=None,
            output_tokens=None,
            calls=None,
            image_count=None,
            reason=f"incomplete HiveMem build efficiency trace at {missing[:10]}",
        )
    return _inference_aggregate(
        num_samples=len(samples),
        source="hivemem_build_trace",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=calls,
        image_count=image_count,
    )


def _zero_inference_aggregate(num_samples: int, source: str) -> dict[str, Any]:
    return _inference_aggregate(
        num_samples=num_samples,
        source=source,
        input_tokens=0,
        output_tokens=0,
        calls=0,
        image_count=0,
    )


def _combine_inference_aggregates(
    *aggregates: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    sample_counts = {int(row.get("num_samples") or 0) for row in aggregates}
    reasons = [str(row.get("reason")) for row in aggregates if row.get("reason")]
    if len(sample_counts) != 1:
        reasons.append(f"component sample counts differ: {sorted(sample_counts)}")

    def total(field: str) -> int | None:
        values = [row.get(field) for row in aggregates]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    return _inference_aggregate(
        num_samples=next(iter(sample_counts)) if len(sample_counts) == 1 else 0,
        source=source,
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        calls=total("calls"),
        image_count=total("image_count"),
        reason="; ".join(reasons),
    )


def _cost_from_inference_aggregate(
    aggregate: dict[str, Any],
    profile: dict[str, Any],
    *,
    aggregation: str,
) -> dict[str, Any]:
    pricing = profile["pricing"]
    input_tokens = aggregate.get("input_tokens")
    output_tokens = aggregate.get("output_tokens")
    num_samples = int(aggregate.get("num_samples") or 0)
    base = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_price_per_million_usd": pricing["input_per_million_usd"],
        "output_price_per_million_usd": pricing["output_per_million_usd"],
        "cost_sum_usd": None,
        "cost_sum": None,
        "num_samples": num_samples,
        "mean_per_sample_usd": None,
        "mean_per_sample": None,
        "formula": None,
        "aggregation": aggregation,
        "source": aggregate.get("source"),
        "available": False,
    }
    if input_tokens is None or output_tokens is None or not num_samples:
        return {
            **base,
            "reason": aggregate.get("reason") or "incomplete token accounting",
        }
    cost_sum = (
        int(input_tokens) * pricing["input_per_million_usd"]
        + int(output_tokens) * pricing["output_per_million_usd"]
    ) / 1_000_000
    mean = cost_sum / num_samples
    return {
        **base,
        "cost_sum_usd": cost_sum,
        "cost_sum": cost_sum,
        "mean_per_sample_usd": mean,
        "mean_per_sample": mean,
        "formula": (
            f"(({input_tokens} / 1000000) * "
            f"{pricing['input_per_million_usd']:g} + "
            f"({output_tokens} / 1000000) * "
            f"{pricing['output_per_million_usd']:g}) / "
            f"{num_samples} = {mean:.12g} USD/sample"
        ),
        "available": True,
    }


def _latency_from_inference_aggregate(
    aggregate: dict[str, Any],
    profile: dict[str, Any],
    *,
    aggregation: str,
) -> dict[str, Any]:
    latency = profile["latency"]
    input_tokens = aggregate.get("input_tokens")
    output_tokens = aggregate.get("output_tokens")
    calls = aggregate.get("calls")
    image_count = aggregate.get("image_count")
    num_samples = int(aggregate.get("num_samples") or 0)
    base = {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "image_count": image_count,
        "base_seconds": latency["base_seconds"],
        "input_seconds_per_token": latency["input_seconds_per_token"],
        "output_seconds_per_token": latency["output_seconds_per_token"],
        "image_seconds": latency["image_seconds"],
        "latency_sum_seconds": None,
        "num_samples": num_samples,
        "mean_per_sample_seconds": None,
        "formula": None,
        "aggregation": aggregation,
        "source": aggregate.get("source"),
        "available": False,
    }
    if (
        input_tokens is None
        or output_tokens is None
        or calls is None
        or image_count is None
        or not num_samples
    ):
        return {
            **base,
            "reason": aggregate.get("reason") or "incomplete latency inputs",
        }
    latency_sum = (
        int(calls) * latency["base_seconds"]
        + int(input_tokens) * latency["input_seconds_per_token"]
        + int(output_tokens) * latency["output_seconds_per_token"]
        + int(image_count) * latency["image_seconds"]
    )
    mean = latency_sum / num_samples
    return {
        **base,
        "latency_sum_seconds": latency_sum,
        "mean_per_sample_seconds": mean,
        "formula": (
            f"({calls} * {latency['base_seconds']:g} + "
            f"{input_tokens} * {latency['input_seconds_per_token']:g} + "
            f"{output_tokens} * {latency['output_seconds_per_token']:g} + "
            f"{image_count} * {latency['image_seconds']:g}) / "
            f"{num_samples} = {mean:.12g} seconds/sample"
        ),
        "available": True,
    }


def write_efficiency_metrics(
    result_dir: str | Path,
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
    sample_ids: Iterable[str],
    model: str,
    config_path: str | Path,
    hivemem_index_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write modeled LLM cost and latency for MB, query-time QA, and total.

    Query-time QA includes any LLM-backed retrieval calls plus the final answer
    calls. Embedding, database, wall-clock, and Judge costs are intentionally
    excluded because their coefficients are not part of this model profile.
    """
    samples = _normalize_sample_ids(sample_ids) or ()
    profile = load_model_efficiency_profile(config_path, model)
    if hivemem_index_root is not None:
        memory_bank = _hivemem_build_inference_aggregate(
            hivemem_index_root, samples
        )
        retrieval = _zero_inference_aggregate(
            len(samples), "hivemem_non_llm_retrieval"
        )
    else:
        trace_path = Path(result_dir) / CALL_TRACE_FILENAME
        memory_bank = _call_trace_inference_aggregate(
            trace_path, phase="memory_build", num_samples=len(samples)
        )
        retrieval = _call_trace_inference_aggregate(
            trace_path, phase="retrieval", num_samples=len(samples)
        )
    answer = _answer_inference_aggregate(
        results, sample_id_field=sample_id_field
    )
    qa = _combine_inference_aggregates(
        retrieval, answer, source="retrieval_plus_answer"
    )
    total = _combine_inference_aggregates(
        memory_bank, qa, source="memory_build_plus_retrieval_plus_answer"
    )
    output = {
        "profile": profile,
        "scope": {
            "included": ["memory_build_llm", "retrieval_llm", "answer_llm"],
            "excluded": ["embedding", "database", "judge", "wall_clock"],
        },
        "cost_mb": _cost_from_inference_aggregate(
            memory_bank, profile, aggregation="sum_mb_cost_divided_by_samples"
        ),
        "cost_qa": _cost_from_inference_aggregate(
            qa, profile, aggregation="sum_retrieval_answer_cost_divided_by_samples"
        ),
        "cost_total": _cost_from_inference_aggregate(
            total, profile, aggregation="sum_total_cost_divided_by_samples"
        ),
        "latency_mb": _latency_from_inference_aggregate(
            memory_bank, profile, aggregation="sum_mb_latency_divided_by_samples"
        ),
        "latency_qa": _latency_from_inference_aggregate(
            qa,
            profile,
            aggregation="sum_retrieval_answer_latency_divided_by_samples",
        ),
        "latency_total": _latency_from_inference_aggregate(
            total, profile, aggregation="sum_total_latency_divided_by_samples"
        ),
        "components": {
            "memory_build": memory_bank,
            "retrieval": retrieval,
            "answer": answer,
        },
    }
    path = Path(result_dir) / EFFICIENCY_METRICS_FILENAME
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def calculate_calls_mb(
    index_root: str | Path | None,
    sample_ids: Iterable[str],
    *,
    unavailable_reason: str = "",
) -> dict[str, Any]:
    """Count actual Memory-Bank model invocations, including failed retries."""
    samples = _normalize_sample_ids(sample_ids) or ()
    base = _call_metric_base(len(samples), "build_calls_divided_by_samples")
    if not samples:
        return {**base, "reason": "No evaluated samples were supplied."}
    if index_root is None:
        return {
            **base,
            "reason": unavailable_reason
            or "This baseline does not expose exact Memory-Bank call counts.",
        }

    total_calls = 0
    failed_calls = 0
    missing_samples: list[str] = []
    invalid_rows: list[str] = []
    root = Path(index_root)
    for sample_id in samples:
        trace_path = _build_trace_path(root / "datasets" / sample_id)
        if trace_path is None:
            missing_samples.append(sample_id)
            continue
        rows = list(_read_jsonl(trace_path))
        if not rows:
            missing_samples.append(sample_id)
            continue
        for trace_index, row in enumerate(rows, start=1):
            counts = _validated_call_counts(
                row.get("llm_attempts"),
                row.get("llm_failed_attempts"),
            )
            if counts is None:
                invalid_rows.append(f"{sample_id}:{trace_index}")
                continue
            attempts, failures = counts
            total_calls += attempts
            failed_calls += failures
    if missing_samples or invalid_rows:
        details = []
        if missing_samples:
            details.append(f"missing build traces for samples={missing_samples}")
        if invalid_rows:
            details.append(
                "missing or invalid exact call counts at build rows "
                f"{invalid_rows[:10]}"
            )
        return {**base, "reason": "; ".join(details)}
    return _available_call_metric(
        base,
        total_calls=total_calls,
        failed_calls=failed_calls,
    )


def calculate_calls_qa(
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
) -> dict[str, Any]:
    """Count actual QA answer-model invocations, including failed retries."""
    sample_ids = {
        str(row.get(sample_id_field) or "").strip()
        for row in results
        if str(row.get(sample_id_field) or "").strip()
    }
    base = _call_metric_base(len(sample_ids), "qa_calls_divided_by_samples")
    if not results:
        return {**base, "reason": "No evaluated QA results were supplied."}
    missing_sample_rows = [
        index
        for index, row in enumerate(results, start=1)
        if not str(row.get(sample_id_field) or "").strip()
    ]
    if missing_sample_rows:
        return {
            **base,
            "reason": (
                f"Missing {sample_id_field!r} for QA result rows "
                f"{missing_sample_rows[:10]}."
            ),
        }

    total_calls = 0
    failed_calls = 0
    invalid_rows = []
    for index, row in enumerate(results, start=1):
        counts = _validated_call_counts(
            row.get("answer_attempts"),
            row.get("answer_failed_attempts"),
        )
        if counts is None:
            invalid_rows.append(index)
            continue
        attempts, failures = counts
        total_calls += attempts
        failed_calls += failures
    if invalid_rows:
        return {
            **base,
            "reason": (
                "Missing or invalid exact call counts at QA rows "
                f"{invalid_rows[:10]}."
            ),
        }
    return _available_call_metric(
        base,
        total_calls=total_calls,
        failed_calls=failed_calls,
    )


def combine_call_metrics(
    memory_bank: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any]:
    """Combine Calls-MB and Calls-QA into the requested sample-wise metric."""
    num_samples = int(qa.get("num_samples") or memory_bank.get("num_samples") or 0)
    total = {
        "memory_bank_calls": memory_bank.get("total_calls"),
        "qa_calls": qa.get("total_calls"),
        "memory_bank_failed_calls": memory_bank.get("failed_calls"),
        "qa_failed_calls": qa.get("failed_calls"),
        "total_calls": None,
        "failed_calls": None,
        "successful_calls": None,
        "num_samples": num_samples,
        "mean_per_sample": None,
        "formula": None,
        "aggregation": "build_plus_qa_calls_divided_by_samples",
        "available": False,
    }
    reasons = []
    if not memory_bank.get("available"):
        reasons.append(f"Calls-MB unavailable: {memory_bank.get('reason', 'unknown reason')}")
    if not qa.get("available"):
        reasons.append(f"Calls-QA unavailable: {qa.get('reason', 'unknown reason')}")
    if memory_bank.get("num_samples") != qa.get("num_samples"):
        reasons.append("Calls-MB and Calls-QA sample counts differ.")
    if reasons:
        total["reason"] = "; ".join(reasons)
        return {"memory_bank": memory_bank, "qa": qa, "total": total}

    mb_calls = int(memory_bank["total_calls"])
    qa_calls = int(qa["total_calls"])
    failed_calls = int(memory_bank["failed_calls"]) + int(qa["failed_calls"])
    total_calls = mb_calls + qa_calls
    mean_per_sample = total_calls / num_samples
    total.update(
        {
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "successful_calls": total_calls - failed_calls,
            "mean_per_sample": mean_per_sample,
            "formula": (
                f"({mb_calls} + {qa_calls}) / {num_samples} = "
                f"{mean_per_sample:.12g}"
            ),
            "available": True,
        }
    )
    return {"memory_bank": memory_bank, "qa": qa, "total": total}


def write_runtime_call_metrics(
    trace_paths: Iterable[str | Path],
    result_dir: str | Path,
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
    sample_ids: Iterable[str],
) -> dict[str, Any]:
    """Merge sample-local executor traces and calculate exact run-time calls.

    The primary ``total`` remains backward compatible: Memory-Bank LLM calls
    plus answer-model calls.  Retrieval-time executor calls are retained as a
    separate metric so they are visible without changing the historical
    definition of ``#Calls``.
    """
    paths = [Path(value) for value in trace_paths]
    rows = load_call_rows(paths)
    rows.extend(_result_attempt_rows(results, sample_id_field=sample_id_field))
    rows.sort(
        key=lambda row: (
            str(row.get("sample_id") or ""),
            float(row.get("started_at") or 0.0),
            str(row.get("call_id") or row.get("request_id") or ""),
        )
    )
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    trace_path = root / CALL_TRACE_FILENAME
    temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(trace_path)

    normalized_samples = _normalize_sample_ids(sample_ids) or ()
    memory_bank = summarize_call_rows(
        rows,
        phase="memory_build",
        num_samples=len(normalized_samples),
    )
    retrieval = summarize_call_rows(
        rows,
        phase="retrieval",
        num_samples=len(normalized_samples),
    )
    calls = combine_call_metrics(
        memory_bank,
        calculate_calls_qa(results, sample_id_field=sample_id_field),
    )
    calls["retrieval"] = retrieval
    metrics_path = root / CALL_METRICS_FILENAME
    metrics_path.write_text(
        json.dumps(calls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return calls


def write_judge_call_trace(
    result_dir: str | Path,
    judged: list[dict[str, Any]],
) -> None:
    """Merge idempotent per-attempt Judge rows into the unified call trace."""
    root = Path(result_dir)
    path = root / CALL_TRACE_FILENAME
    rows = []
    if path.is_file():
        rows = [row for row in _read_jsonl(path) if row.get("phase") != "judge"]
    for row in judged:
        rows.extend(
            _attempt_rows(
                phase="judge",
                sample_id=str(row.get("sample_id") or row.get("dataset") or ""),
                query_id=str(
                    row.get("uid") or row.get("question_id") or row.get("index") or ""
                ),
                attempts=row.get("judge_attempts"),
                failed_attempts=row.get("judge_failed_attempts"),
                usage=(row.get("judge") or {}).get("usage"),
                error=str((row.get("judge") or {}).get("error") or ""),
            )
        )
    rows.sort(
        key=lambda row: (
            str(row.get("phase") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("query_id") or ""),
            int(row.get("attempt") or 0),
        )
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _result_attempt_rows(
    results: list[dict[str, Any]],
    *,
    sample_id_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        rows.extend(
            _attempt_rows(
                phase="qa",
                sample_id=str(result.get(sample_id_field) or ""),
                query_id=str(result.get("query_id") or result.get("uid") or index),
                attempts=result.get("answer_attempts"),
                failed_attempts=result.get("answer_failed_attempts"),
                usage=result.get("answer_token_usage"),
                error=str(result.get("error") or ""),
                image_count=result.get("answer_image_count"),
            )
        )
    return rows


def _attempt_rows(
    *,
    phase: str,
    sample_id: str,
    query_id: str,
    attempts: Any,
    failed_attempts: Any,
    usage: Any,
    error: str,
    image_count: Any = 0,
) -> list[dict[str, Any]]:
    counts = _validated_call_counts(attempts, failed_attempts)
    if counts is None:
        return []
    total, failed = counts
    token_usage = _token_usage(usage)
    rows = []
    for attempt in range(1, total + 1):
        is_failed = attempt <= failed
        row = {
            "trace_version": 1,
            "phase": phase,
            "service": "llm",
            "sample_id": sample_id,
            "query_id": query_id,
            "call_id": f"{phase}:{query_id}:{attempt}",
            "attempt": attempt,
            "success": not is_failed,
            "failed": is_failed,
            "error": error if is_failed else "",
            "image_count": image_count,
        }
        if token_usage is not None and attempt == total:
            row.update(token_usage)
            row["usage_scope"] = "cumulative_query_attempts"
        rows.append(row)
    return rows


def write_memory_metrics(
    index_root: str | Path,
    result_dir: str | Path,
    *,
    tokenizer_name: str = "",
    sample_ids: Iterable[str] | None = None,
    cost_mb_input_price: float | None = None,
    cost_mb_output_price: float | None = None,
) -> dict[str, Any]:
    selected_samples = _normalize_sample_ids(sample_ids)
    metrics = calculate_memory_metrics(
        index_root,
        tokenizer_name=tokenizer_name,
        sample_ids=selected_samples,
    )
    if selected_samples is not None:
        metrics["cost_mb"] = calculate_cost_mb(
            index_root,
            selected_samples,
            input_price=cost_mb_input_price,
            output_price=cost_mb_output_price,
        )
    path = Path(result_dir) / MEMORY_METRICS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_snapshot_memory_metrics(
    snapshots: list[dict[str, Any]],
    result_dir: str | Path,
    *,
    sample_ids: Iterable[str] | None = None,
    cost_mb_input_price: float | None = None,
    cost_mb_output_price: float | None = None,
) -> dict[str, Any]:
    """Write comparable size metrics for non-HiveMem native backends.

    Native projects do not expose uniform provider token accounting, so build
    tokens are explicitly marked unavailable instead of being fabricated.
    """
    metrics = {
        "memory_build_tokens": None,
        "summary_characters": sum(len(str(row.get("text") or "")) for row in snapshots),
        "memory_count": len(snapshots),
    }
    samples = _normalize_sample_ids(sample_ids) or ()
    metrics["cost_mb"] = {
        "input_tokens": None,
        "output_tokens": None,
        "input_price": cost_mb_input_price,
        "output_price": cost_mb_output_price,
        "cost_sum": None,
        "num_samples": len(samples),
        "mean_per_sample": None,
        "formula": None,
        "aggregation": "sum_build_cost_divided_by_samples",
        "token_source": None,
        "available": False,
        "reason": "This baseline does not expose exact Memory-Bank provider usage.",
    }
    path = Path(result_dir) / MEMORY_METRICS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def add_memory_metrics(summary: dict, memory_metrics: dict) -> dict:
    """Order the result summary as F1, EM, Judge, memory metrics, then the rest."""
    combined = {
        "f1": summary.get("f1"),
        "em": summary.get("em", summary.get("exact_match")),
        "llm_judge": summary.get("llm_judge"),
        "memory_build_tokens": memory_metrics["memory_build_tokens"],
        "summary_characters": memory_metrics["summary_characters"],
    }
    if "cost_mb" in memory_metrics:
        combined["cost_mb"] = memory_metrics["cost_mb"]
    combined.update(
        {
            key: value
            for key, value in summary.items()
            if key not in {
                "f1",
                "em",
                "exact_match",
                "llm_judge",
                "memory_build_tokens",
                "summary_characters",
                "cost_mb",
            }
        }
    )
    categories = combined.get("by_category")
    if isinstance(categories, dict):
        combined["by_category"] = {
            category: {
                "f1": values.get("f1"),
                "em": values.get("em", values.get("exact_match")),
                "llm_judge": values.get("llm_judge"),
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"f1", "em", "exact_match", "llm_judge"}
                },
            }
            for category, values in categories.items()
        }
    return combined


def calculate_retrieval_memory_tokens(
    result_dir: str | Path,
    *,
    tokenizer_name: str = "",
    tokenizer: Any = None,
) -> dict[str, int | float]:
    """Count retrieved-memory text tokens across all QA retrieval traces.

    The count covers only the memory section rendered into the answer prompt:
    its heading, per-memory headers, memory text, graph prefixes, redaction,
    and attached-memory-image labels. It excludes the system prompt, question,
    answer completion, and visual image tokens.
    """
    root = Path(result_dir)
    trace_path = root / "retrieval_trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Retrieval trace not found: {trace_path}")

    manifest_path = root / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        model = tokenizer_name or str(manifest.get("answer_model") or "")
        tokenizer = AutoTokenizer.from_pretrained(
            _resolve_tokenizer_name(model),
            trust_remote_code=True,
        )

    append_graph_memories = str(manifest.get("graph_mode") or "").lower() == "append"
    total_tokens = 0
    query_count = 0
    for row in _read_jsonl(trace_path):
        query_count += 1
        hits = row.get("top_k") or []
        if not hits:
            continue
        memory_context = row.get("memory_context")
        if not isinstance(memory_context, str):
            memory_items = _memory_items_from_trace(
                hits,
                append_graph_memories=append_graph_memories,
            )
            memory_context, _ = build_retrieved_memory_context(
                memory_items,
                str(row.get("category") or ""),
            )
        token_ids = tokenizer.encode(memory_context, add_special_tokens=False)
        total_tokens += len(token_ids)

    return {
        "total_tokens": total_tokens,
        "query_count": query_count,
        "average_tokens_per_query": (
            round(total_tokens / query_count, 2) if query_count else 0.0
        ),
    }


def write_retrieval_memory_token(
    result_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    tokenizer_name: str = "",
    tokenizer: Any = None,
) -> dict[str, int | float]:
    metrics = calculate_retrieval_memory_tokens(
        result_dir,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
    )
    path = Path(output_dir or result_dir) / RETRIEVAL_MEMORY_TOKEN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def add_retrieval_memory_tokens(summary: dict, retrieval_metrics: dict) -> dict:
    """Place retrieval-memory token totals with the other memory metrics."""
    leading_keys = (
        "f1",
        "em",
        "llm_judge",
        "memory_build_tokens",
        "summary_characters",
    )
    combined = {
        key: summary.get(key)
        for key in leading_keys
        if key in summary
    }
    combined["retrieval_memory_tokens"] = dict(retrieval_metrics)
    combined.update(
        {
            key: value
            for key, value in summary.items()
            if key not in {*leading_keys, "retrieval_memory_tokens"}
        }
    )
    return combined


def _memory_items_from_trace(
    hits: list[dict[str, Any]],
    *,
    append_graph_memories: bool,
) -> list[dict[str, Any]]:
    """Rebuild answer-client memory items from historical retrieval traces."""
    items = []
    for hit in hits:
        source_ids = [str(value) for value in (hit.get("source_dialogue_ids") or [])]
        dialogue_id = source_ids[0] if source_ids else ""
        session_id = dialogue_id.split(":", 1)[0] if dialogue_id else ""
        image_ids = [str(value) for value in (hit.get("image_ids") or [])]
        image_paths = [str(value) for value in (hit.get("image_paths") or [])]
        image = None
        if image_paths:
            image = {
                "path": image_paths[0],
                "img_id": image_ids[0] if image_ids else "",
            }
        text = str(hit.get("content") or "")
        if append_graph_memories and str(hit.get("via") or "") == "graph":
            text = f"(related background memory) {text}"
        items.append(
            {
                "text": text,
                "image": image,
                "metadata": {
                    "session_id": session_id,
                    "dialogue_id": dialogue_id,
                    "image_id": image_ids[0] if image_ids else "",
                },
            }
        )
    return items


def _count_summary_characters(
    index_root: Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
) -> int:
    total = 0
    if sample_ids is None:
        paths = sorted((index_root / "datasets").glob("*/memories.jsonl"))
    else:
        paths = [index_root / "datasets" / sample_id / "memories.jsonl" for sample_id in sample_ids]
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            if str(row.get("status", "ACTIVE")).upper() == "ACTIVE":
                total += len(str(row.get("summary") or row.get("content") or ""))
    return total


def _iter_build_traces(
    index_root: Path,
    *,
    sample_ids: tuple[str, ...] | None = None,
):
    datasets_root = index_root / "datasets"
    dataset_dirs = (
        sorted(datasets_root.iterdir())
        if sample_ids is None
        else [datasets_root / sample_id for sample_id in sample_ids]
    )
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            continue
        path = _build_trace_path(dataset_dir)
        if path is not None:
            yield from _read_jsonl(path)


def _build_trace_path(dataset_dir: Path) -> Path | None:
    current = dataset_dir / "traces" / "build.jsonl"
    legacy = dataset_dir / "build_trace.jsonl"
    if current.exists():
        return current
    return legacy if legacy.exists() else None


def _normalize_sample_ids(
    sample_ids: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if sample_ids is None:
        return None
    return tuple(sorted({str(value).strip() for value in sample_ids if str(value).strip()}))


def _validated_cost_prices(
    input_price: float | None,
    output_price: float | None,
) -> tuple[float, float] | None:
    if input_price is None or output_price is None:
        return None
    if isinstance(input_price, bool) or isinstance(output_price, bool):
        raise ValueError("Cost prices must be non-negative numbers.")
    try:
        values = (float(input_price), float(output_price))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cost prices must be non-negative numbers.") from exc
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Cost prices must be non-negative numbers.")
    return values


def _call_metric_base(num_samples: int, aggregation: str) -> dict[str, Any]:
    return {
        "total_calls": None,
        "failed_calls": None,
        "successful_calls": None,
        "num_samples": num_samples,
        "mean_per_sample": None,
        "formula": None,
        "aggregation": aggregation,
        "available": False,
    }


def _available_call_metric(
    base: dict[str, Any],
    *,
    total_calls: int,
    failed_calls: int,
) -> dict[str, Any]:
    num_samples = int(base["num_samples"])
    mean_per_sample = total_calls / num_samples
    return {
        **base,
        "total_calls": total_calls,
        "failed_calls": failed_calls,
        "successful_calls": total_calls - failed_calls,
        "mean_per_sample": mean_per_sample,
        "formula": f"{total_calls} / {num_samples} = {mean_per_sample:.12g}",
        "available": True,
    }


def _validated_call_counts(attempts: Any, failed_attempts: Any) -> tuple[int, int] | None:
    if isinstance(attempts, bool) or isinstance(failed_attempts, bool):
        return None
    try:
        total = int(attempts)
        failed = int(failed_attempts)
    except (TypeError, ValueError):
        return None
    if total < 1 or failed < 0 or failed > total:
        return None
    return total, failed


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _token_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict) or not all(key in value for key in TOKEN_KEYS):
        return None
    return {key: int(value.get(key) or 0) for key in TOKEN_KEYS}


def _estimate_trace_tokens(
    index_root: Path,
    traces: list[dict[str, Any]],
    tokenizer_name: str,
) -> dict[str, int]:
    from transformers import AutoTokenizer
    from hive_mem.executor import MemoryExecutor

    manifest_path = index_root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    profiles_path = Path(str(manifest.get("profiles_file") or ""))
    profiles = (
        json.loads(profiles_path.read_text(encoding="utf-8"))
        if profiles_path.is_file()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        _resolve_tokenizer_name(tokenizer_name or str(manifest.get("executor_model") or "")),
        trust_remote_code=True,
    )
    prompt_builder = MemoryExecutor(llm_client=None, embedder=None)
    usage = {key: 0 for key in TOKEN_KEYS}
    for row in traces:
        event = dict(row.get("event") or {})
        prompt = prompt_builder._build_prompt(
            str(event.get("text") or ""),
            profile=str(profiles.get(str(event.get("dataset") or ""), "")),
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        completion_ids = tokenizer.encode(
            str(row.get("raw_response") or ""),
            add_special_tokens=False,
        )
        usage["prompt_tokens"] += len(prompt_ids)
        usage["completion_tokens"] += len(completion_ids)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def _resolve_tokenizer_name(model: str) -> str:
    if not model:
        raise ValueError("A tokenizer is required to estimate historical build tokens")
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    return model
