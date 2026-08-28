from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

from benchmarks.memgallery_harness.runner.answer_client import (
    build_retrieved_memory_context,
)


MEMORY_METRICS_FILENAME = "memory_metrics.json"
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
    judge_categories = judge_metrics.get("by_category") or {}
    merged["by_category"] = {
        category: merge_row(values, judge_categories.get(category, {}))
        for category, values in (metrics.get("by_category") or {}).items()
    }
    return merged


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

    Price coefficients intentionally multiply raw token counts directly. They
    are not divided by one million, matching the final-evaluation protocol.
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

    cost_sum = input_tokens * input_coefficient + output_tokens * output_coefficient
    mean_per_sample = cost_sum / sample_count
    formula = (
        f"({input_tokens} * {input_coefficient:g} + "
        f"{output_tokens} * {output_coefficient:g}) / "
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
    coefficients multiply raw provider token counts directly (no /1M).
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

    cost_sum = input_tokens * input_coefficient + output_tokens * output_coefficient
    mean_per_sample = cost_sum / num_samples
    formula = (
        f"({input_tokens} * {input_coefficient:g} + "
        f"{output_tokens} * {output_coefficient:g}) / "
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
