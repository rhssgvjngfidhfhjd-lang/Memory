from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmarks.memgallery_harness.runner.answer_client import (
    build_retrieved_memory_context,
)


MEMORY_METRICS_FILENAME = "memory_metrics.json"
RETRIEVAL_MEMORY_TOKEN_FILENAME = "retrieval_memory_token.json"
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def normalize_answer(text: str) -> str:
    s = str(text).lower()
    s = s.replace("_", " UNDERSCORE ")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = s.replace(" UNDERSCORE ", "_")
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
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
    from collections import defaultdict
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
            "exact_match": sum(v["em"] for v in values) / count if count else 0.0,
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
            "llm_judge": judge_row.get("accuracy"),
        }
        merged.update({key: value for key, value in row.items() if key != "f1"})
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
) -> dict[str, Any]:
    """Calculate build-token usage and active-summary characters for one bank.

    New build traces carry API-reported ``llm_usage``. Historical traces are
    backfilled with the executor tokenizer using the recorded event and raw
    response, so old result directories do not require another model run.
    """
    root = Path(index_root)
    traces = list(_iter_build_traces(root))
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
        "summary_characters": _count_summary_characters(root),
    }


def write_memory_metrics(
    index_root: str | Path,
    result_dir: str | Path,
    *,
    tokenizer_name: str = "",
) -> dict[str, Any]:
    metrics = calculate_memory_metrics(index_root, tokenizer_name=tokenizer_name)
    path = Path(result_dir) / MEMORY_METRICS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def add_memory_metrics(summary: dict, memory_metrics: dict) -> dict:
    """Order the result summary as F1, Judge, memory metrics, then the rest."""
    combined = {
        "f1": summary.get("f1"),
        "llm_judge": summary.get("llm_judge"),
        "memory_build_tokens": memory_metrics["memory_build_tokens"],
        "summary_characters": memory_metrics["summary_characters"],
    }
    combined.update(
        {
            key: value
            for key, value in summary.items()
            if key not in {"f1", "llm_judge", "memory_build_tokens", "summary_characters"}
        }
    )
    categories = combined.get("by_category")
    if isinstance(categories, dict):
        combined["by_category"] = {
            category: {
                "f1": values.get("f1"),
                "llm_judge": values.get("llm_judge"),
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"f1", "llm_judge"}
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


def _count_summary_characters(index_root: Path) -> int:
    total = 0
    for path in sorted((index_root / "datasets").glob("*/memories.jsonl")):
        for row in _read_jsonl(path):
            if str(row.get("status", "ACTIVE")).upper() == "ACTIVE":
                total += len(str(row.get("summary") or row.get("content") or ""))
    return total


def _iter_build_traces(index_root: Path):
    for dataset_dir in sorted((index_root / "datasets").iterdir()):
        if not dataset_dir.is_dir():
            continue
        current = dataset_dir / "traces" / "build.jsonl"
        legacy = dataset_dir / "build_trace.jsonl"
        path = current if current.exists() else legacy
        if path.exists():
            yield from _read_jsonl(path)


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
    shared_model = Path("/data/shared_models") / model.rsplit("/", 1)[-1]
    return str(shared_model) if shared_model.exists() else model
