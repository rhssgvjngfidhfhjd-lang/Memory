#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from json_repair import repair_json
from openai import OpenAI

from benchmarks.io_utils import (
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)

from benchmarks.memgallery_harness.runner.metrics import (
    MEMORY_METRICS_FILENAME,
    RETRIEVAL_MEMORY_TOKEN_FILENAME,
    add_memory_metrics,
    add_retrieval_memory_tokens,
    merge_llm_judge_metrics,
    write_memory_metrics,
    write_retrieval_memory_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_BENCHMARKS = ("memgallery", "worldmemarena", "h2hmem")

# Snapshot copied from evaluation_protocol_bundle on 2026-08-25. Keep the
# protocol text, rendering, request shape, and parsing behavior in sync with
# agentic_memrl/evaluation/{judge_protocols,run_llm_judge}.py.
EVALUATION_PROTOCOL_SNAPSHOT = "evaluation_protocol_bundle@2026-08-25"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
DEFAULT_JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_JUDGE_MAX_NEW_TOKENS = 512
JUDGE_CACHE_VERSION = "agentic_memrl.judge_cache.v2"

GRADED_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)
BINARY_SCORES = (0.0, 1.0)
WORLD_LABELS = ("Correct", "Hallucination", "Omission")


@dataclass(frozen=True)
class JudgeProtocol:
    protocol_id: str
    benchmark: str
    official_role: str
    rubric: str
    score_values: tuple[float, ...] = GRADED_SCORES
    labels: tuple[str, ...] = ()


_MEMORY_QA_JUDGE_ROLE = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "with the question-answering task."
)
_WORLD_JUDGE_ROLE = (
    "You are an **evaluation expert for AI memory system question answering**."
)

_FIVE_LEVEL_RUBRIC = """**Score 0 (Incorrect / Miss):**
- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**
- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**
- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**
- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**
- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations."""

_WORLD_RUBRIC = """### 1. Correct
* The response accurately answers the question and is **semantically equivalent** to the Reference Answer.
* No contradictions with the Reference Answer.
* Synonyms, paraphrasing, and reasonable summarization are acceptable.

### 2. Hallucination
* The response includes information that **contradicts** the Reference Answer.
* When the Reference Answer is *unknown/uncertain*, yet the response provides a specific fact.

### 3. Omission
* The response is **incomplete** compared to the Reference Answer.
* It states "don't know" or "no related memory" even though the Reference Answer supplies the answer.
* For multi-element answers, missing **any** element counts as Omission.

## Priority Rules
* Both missing info AND fabricated info -> **Hallucination**.
* No fabrication but missing info -> **Omission**.
* Fully equivalent -> **Correct**."""

PROTOCOLS = {
    "mem_gallery_answer_v1": JudgeProtocol(
        "mem_gallery_answer_v1",
        "Mem-Gallery",
        _MEMORY_QA_JUDGE_ROLE,
        _FIVE_LEVEL_RUBRIC,
    ),
    "h2hmem_answer_v1": JudgeProtocol(
        "h2hmem_answer_v1",
        "H2HMem",
        _MEMORY_QA_JUDGE_ROLE,
        _FIVE_LEVEL_RUBRIC,
    ),
    "worldmemarena_answer_v1": JudgeProtocol(
        "worldmemarena_answer_v1",
        "WorldMemArena",
        _WORLD_JUDGE_ROLE,
        _WORLD_RUBRIC,
        score_values=BINARY_SCORES,
        labels=WORLD_LABELS,
    ),
}


def get_judge_protocol(protocol_id: str) -> JudgeProtocol:
    try:
        return PROTOCOLS[str(protocol_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown judge protocol: {protocol_id!r}.") from exc


def judge_protocol_id_for_sample(
    data_source: Any,
    metadata: Mapping[str, Any] | None = None,
    ground_truth: Any = None,
) -> str:
    del metadata, ground_truth
    source = str(data_source or "").strip().casefold().replace("-", "_")
    if "worldmemarena" in source:
        return "worldmemarena_answer_v1"
    if "h2hmem" in source:
        return "h2hmem_answer_v1"
    if "mem_gallery" in source:
        return "mem_gallery_answer_v1"
    raise ValueError(f"No final-evaluation judge protocol for data_source={data_source!r}.")


def render_judge_prompt(
    protocol_id: str,
    *,
    prediction: Any,
    references: Sequence[Any],
) -> str:
    protocol = get_judge_protocol(protocol_id)
    clean_references = [
        str(reference).strip() for reference in references if str(reference).strip()
    ]
    if not clean_references:
        raise ValueError("Judge prompts require at least one non-empty ground-truth reference.")
    reference_text = (
        clean_references[0]
        if len(clean_references) == 1
        else json.dumps(clean_references, ensure_ascii=False)
    )
    prediction_text = str(prediction or "").strip()
    if protocol.labels == WORLD_LABELS:
        return f"""{protocol.official_role}
Based **only** on the provided **Reference Answer**, strictly evaluate the **accuracy** of the **Memory System Response**. Classify it as one of **Correct**, **Hallucination**, or **Omission**. Do **not** use any external knowledge or subjective inference.

# Evaluation Criteria
{protocol.rubric}

# Information
* **Reference Answer:** {reference_text}
* **Memory System Response:** {prediction_text}

# Output
```json
{{
  "reasoning": "Concise evaluation rationale",
  "evaluation_result": "Correct | Hallucination | Omission"
}}
```"""
    reasoning_placeholder = (
        "" if protocol.protocol_id == "mem_gallery_answer_v1" else "<short explanation>"
    )
    return f"""{protocol.official_role}
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.

### Scoring Rubric

{protocol.rubric}

### Input Data

Ground Truth: {reference_text}
Assistant Answer: {prediction_text}

### Output Format

Output strictly in the following JSON format:
{{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "{reasoning_placeholder}"}}"""


def validate_protocol_snapshot() -> None:
    expected = {
        "mem_gallery": "mem_gallery_answer_v1",
        "worldmemarena": "worldmemarena_answer_v1",
        "h2hmem": "h2hmem_answer_v1",
    }
    for data_source, protocol_id in expected.items():
        if judge_protocol_id_for_sample(data_source) != protocol_id:
            raise RuntimeError(f"Judge protocol dispatch drifted for {data_source}.")
        prompt = render_judge_prompt(
            protocol_id,
            prediction="PREDICTION_SENTINEL",
            references=["REFERENCE_SENTINEL"],
        )
        if "PREDICTION_SENTINEL" not in prompt or "REFERENCE_SENTINEL" not in prompt:
            raise RuntimeError(f"Judge prompt inputs drifted for {protocol_id}.")
    if PROTOCOLS["worldmemarena_answer_v1"].labels != WORLD_LABELS:
        raise RuntimeError("WorldMemArena Judge labels drifted.")
    for protocol_id in ("mem_gallery_answer_v1", "h2hmem_answer_v1"):
        if PROTOCOLS[protocol_id].score_values != GRADED_SCORES:
            raise RuntimeError(f"Judge score space drifted for {protocol_id}.")


def build_prompt(
    benchmark: str,
    rows: list[dict[str, Any]],
    template: str | None = None,
) -> str:
    """Compatibility wrapper; external prompt templates are intentionally ignored."""

    del template
    if len(rows) != 1:
        raise ValueError(f"{benchmark} Judge expects exactly one question per request")
    normalized = normalize_judge_row(benchmark, rows[0], 1)
    return render_judge_prompt(
        normalized["protocol_id"],
        prediction=normalized["prediction"],
        references=normalized["references"],
    )


@dataclass(frozen=True)
class JudgeTask:
    members: tuple[tuple[int, dict[str, Any]], ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.members)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return [row for _, row in self.members]


def load_api_keys(
    path: str | Path = "",
    key_start: int = 0,
    key_count: int = 0,
    *,
    env_name: str = "OPENROUTER_API_KEY",
) -> list[str]:
    keys: list[str] = []
    env_key = os.environ.get(env_name, "").strip()
    if env_key:
        keys.append(env_key)
    if path:
        key_path = Path(path)
        if not key_path.is_file():
            raise FileNotFoundError(f"Judge API key file not found: {key_path}")
        raw = key_path.read_text(encoding="utf-8")
        file_keys = re.findall(r"sk-or-v1-[A-Za-z0-9_-]+", raw)
        if not file_keys and "nvapi-" in raw:
            raise ValueError(
                "The configured key file contains NVIDIA nvapi-* keys, but the "
                "evaluation protocol requires an OpenRouter key (sk-or-v1-*)."
            )
        keys.extend(file_keys)
    keys = list(dict.fromkeys(keys))
    if key_start:
        keys = keys[key_start:]
    if key_count > 0:
        keys = keys[:key_count]
    if not keys:
        raise ValueError(
            f"No OpenRouter API key found. Set {env_name} or pass --key-file "
            "containing an sk-or-v1-* key."
        )
    return keys


class RoundRobinClients:
    def __init__(self, keys: list[str], base_url: str, timeout: int):
        if not keys:
            raise ValueError("No Judge API keys found")
        self.keys = keys
        self.base_url = base_url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._index = 0
        self._clients: dict[str, OpenAI] = {}

    def next_client(self) -> OpenAI:
        with self._lock:
            key = self.keys[self._index % len(self.keys)]
            self._index += 1
            client = self._clients.get(key)
            if client is None:
                client = OpenAI(
                    base_url=self.base_url,
                    api_key=key,
                    timeout=self.timeout,
                    max_retries=0,
                )
                self._clients[key] = client
            return client


def _scoring_rationale(text: str) -> str:
    match = re.search(
        r"\[Scoring Rationale\]\s*:\s*(.*?)(?:\[Score\]|\[JSON\]|$)",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _extract_json_object(text: str) -> tuple[Mapping[str, Any], bool]:
    stripped = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.IGNORECASE
    )
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("Judge response does not contain a JSON object.")
        end = stripped.rfind("}")
        candidate = stripped[start : end + 1] if end > start else stripped[start:]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as strict_error:
            try:
                value = repair_json(
                    candidate,
                    return_objects=True,
                    ensure_ascii=False,
                    skip_json_loads=True,
                )
            except (TypeError, ValueError) as repair_error:
                raise ValueError("Judge response contains irreparable JSON.") from repair_error
            if not isinstance(value, Mapping):
                raise ValueError("Repaired judge response must be a JSON object.") from strict_error
            return value, True
    if not isinstance(value, Mapping):
        raise ValueError("Judge response must be a JSON object.")
    return value, False


def _score(value: Any, allowed: tuple[float, ...]) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid judge score: {value!r}.") from exc
    for candidate in allowed:
        if abs(number - candidate) < 1e-8:
            return candidate
    raise ValueError(f"Judge score {number} is not in {allowed}.")


def _canonical_label(value: Any, allowed: tuple[str, ...]) -> str:
    normalized = str(value or "").strip().casefold()
    for candidate in allowed:
        if normalized == candidate.casefold():
            return candidate
    raise ValueError(f"Judge label {value!r} is not in {allowed}.")


def _graded_label(score: float) -> str:
    return {
        0.0: "Incorrect",
        0.25: "Poor",
        0.5: "Partial",
        0.75: "Good",
        1.0: "Correct",
    }[score]


def parse_judge_response(
    text: str,
    protocol_id: str,
) -> dict[str, Any]:
    protocol = get_judge_protocol(protocol_id)
    payload, json_repaired = _extract_json_object(text)
    if protocol.labels == WORLD_LABELS:
        reasoning = str(payload.get("reasoning") or "").strip()
        if not reasoning:
            raise ValueError("WorldMemArena judge response requires non-empty reasoning.")
        label = _canonical_label(
            payload.get("evaluation_result") or payload.get("label"), WORLD_LABELS
        )
        expected_score = 1.0 if label == "Correct" else 0.0
        supplied = payload.get("score")
        if supplied not in (None, "") and _score(supplied, BINARY_SCORES) != expected_score:
            raise ValueError("WorldMemArena judge score disagrees with its label.")
        return {
            "score": expected_score,
            "label": label,
            "reasoning": reasoning,
            "json_repaired": json_repaired,
        }
    reasoning = str(payload.get("reasoning") or "").strip()
    score = _score(payload.get("score"), GRADED_SCORES)
    return {
        "score": score,
        "label": str(payload.get("label") or _graded_label(score)).strip(),
        "reasoning": reasoning,
        "json_repaired": json_repaired,
    }


def _references_from_row(row: Mapping[str, Any]) -> list[str]:
    if "references" in row:
        raw = row.get("references")
    elif "original_answer" in row:
        raw = row.get("original_answer")
    elif "reference" in row:
        raw = row.get("reference")
    else:
        raise ValueError("Judge row is missing references/original_answer.")
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    references = [str(value).strip() for value in values if str(value).strip()]
    if not references:
        raise ValueError("Judge row requires at least one non-empty reference answer.")
    return references


def normalize_judge_row(
    benchmark: str,
    row: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    prediction_key = "system_answer" if "system_answer" in row else "prediction"
    if prediction_key not in row:
        raise ValueError(f"{benchmark} row {index} is missing system_answer/prediction.")
    dataset_source = {
        "memgallery": "mem_gallery",
        "worldmemarena": "worldmemarena",
        "h2hmem": "h2hmem",
    }[benchmark]
    conversation_id = str(
        row.get("conversation_id")
        or row.get("sample_id")
        or row.get("dataset")
        or row.get("dialogue_name")
        or ""
    ).strip()
    if not conversation_id:
        raise ValueError(f"{benchmark} row {index} is missing conversation_id/sample identity.")
    question_id = str(
        row.get("question_id") or row.get("query_id") or f"row-{index}"
    ).strip()
    references = _references_from_row(row)
    protocol_id = judge_protocol_id_for_sample(
        dataset_source,
        row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
        references,
    )
    uid = str(row.get("uid") or f"{dataset_source}:{conversation_id}:{question_id}")
    return {
        "uid": uid,
        "dataset_source": dataset_source,
        "conversation_id": conversation_id,
        "question_id": question_id,
        "prediction": str(row.get(prediction_key) or "").strip(),
        "references": references,
        "protocol_id": protocol_id,
    }


def judge_once(
    client: OpenAI,
    model: str,
    task: JudgeTask,
    benchmark: str,
    max_tokens: int,
) -> list[dict[str, Any]]:
    if len(task.members) != 1:
        raise ValueError("Evaluation protocol requires itemwise Judge requests.")
    index, source_row = task.members[0]
    normalized = normalize_judge_row(benchmark, source_row, index)
    protocol = get_judge_protocol(normalized["protocol_id"])
    prompt = render_judge_prompt(
        protocol.protocol_id,
        prediction=normalized["prediction"],
        references=normalized["references"],
    )
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": DEFAULT_JUDGE_TEMPERATURE,
        "max_completion_tokens": max(int(max_tokens), 1),
    }
    if protocol.labels == WORLD_LABELS or protocol.score_values != BINARY_SCORES:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request)
    content = str(response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Judge returned an empty response")
    parsed = parse_judge_response(content, protocol.protocol_id)
    usage = getattr(response, "usage", None)
    usage_data = usage.model_dump() if hasattr(usage, "model_dump") else None
    return [{
        "normalized": normalized,
        "score": parsed["score"],
        "label": parsed["label"],
        "reason": parsed["reasoning"],
        "raw_judge": content,
        "json_repaired": parsed["json_repaired"],
        "judge": {
            "protocol_id": protocol.protocol_id,
            "status": "complete",
            "model": model,
            "prompt": prompt,
            "score": parsed["score"],
            "label": parsed["label"],
            "reasoning": parsed["reasoning"],
            "json_repaired": parsed["json_repaired"],
            "raw_response": content,
            "usage": usage_data,
        },
    }]


def _result_record(
    benchmark: str,
    index: int,
    row: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    normalized = result.pop("normalized", None) or normalize_judge_row(
        benchmark, row, index
    )
    question = row.get("question", row.get("question_text", ""))
    return {
        "index": index,
        "benchmark": benchmark,
        "uid": normalized["uid"],
        "dataset_source": normalized["dataset_source"],
        "conversation_id": normalized["conversation_id"],
        "dataset": row.get("dataset", row.get("dialogue_name", "")),
        "sample_id": row.get("sample_id", row.get("dataset", row.get("dialogue_name", ""))),
        "dialogue_name": row.get("dialogue_name", ""),
        "session_id": row.get("session_id", row.get("session_name", "")),
        "question_id": normalized["question_id"],
        "category": row.get("category", ""),
        "question": question,
        "prediction": normalized["prediction"],
        "references": normalized["references"],
        "reference": normalized["references"][0],
        "retrieved_ids": row.get("retrieved_ids", []),
        "clue": row.get("clue", []),
        **result,
    }


def judge_with_retries(
    task: JudgeTask,
    clients: RoundRobinClients,
    model: str,
    benchmark: str,
    max_tokens: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    attempts = max(int(max_retries), 1)
    for attempt in range(attempts):
        try:
            results = judge_once(
                clients.next_client(),
                model,
                task,
                benchmark,
                max_tokens,
            )
            return [
                _result_record(benchmark, index, row, result)
                for (index, row), result in zip(task.members, results)
            ]
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    index, row = task.members[0]
    normalized = normalize_judge_row(benchmark, row, index)
    prompt = render_judge_prompt(
        normalized["protocol_id"],
        prediction=normalized["prediction"],
        references=normalized["references"],
    )
    error_text = f"{type(last_error).__name__}: {last_error}"
    error_result = {
        "normalized": normalized,
        "score": None,
        "label": "judge_error",
        "reason": error_text,
        "raw_judge": "",
        "json_repaired": False,
        "judge": {
            "protocol_id": normalized["protocol_id"],
            "status": "error",
            "model": model,
            "prompt": prompt,
            "score": None,
            "label": None,
            "reasoning": None,
            "error": error_text,
        },
    }
    return [
        _result_record(benchmark, index, row, error_result)
        for index, row in task.members
    ]


def summarize(
    judged: list[dict[str, Any]],
    model: str,
    *,
    benchmark: str = "memgallery",
    expected_count: int | None = None,
) -> dict[str, Any]:
    total = len(judged)
    expected = total if expected_count is None else expected_count
    valid = [row for row in judged if row.get("label") != "judge_error"]
    score_sum = sum(float(row["score"]) for row in valid)
    correct = sum(float(row["score"]) == 1.0 for row in valid)
    judge_errors = total - len(valid)
    return {
        "benchmark": benchmark,
        "model": model,
        "protocol_snapshot": EVALUATION_PROTOCOL_SNAPSHOT,
        "count": total,
        "valid_count": len(valid),
        "judge_errors": judge_errors,
        "correct": correct,
        "score_sum": score_sum,
        "average_score": score_sum / len(valid) if valid else None,
        # Backward-compatible QA-wise alias consumed by merge_llm_judge_metrics.
        "accuracy": score_sum / len(valid) if valid else None,
        "coverage": len(valid) / total if total else 0.0,
        "completion": total / expected if expected else 1.0,
        "provisional": judge_errors > 0 or total != expected,
    }


def checkpoint_signature(
    args: argparse.Namespace,
    results_path: Path,
) -> dict[str, Any]:
    protocol_payload = {
        protocol_id: {
            "role": protocol.official_role,
            "rubric": protocol.rubric,
            "scores": protocol.score_values,
            "labels": protocol.labels,
        }
        for protocol_id, protocol in sorted(PROTOCOLS.items())
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "results_path": str(results_path.resolve()),
        "results_sha256": sha256_file(results_path),
        "selection": {"start": args.start, "end": args.end},
        "judge": {
            "benchmark": args.benchmark,
            "base_url": args.base_url,
            "model": args.model,
            "temperature": DEFAULT_JUDGE_TEMPERATURE,
            "max_tokens": args.max_tokens,
            "cache_version": JUDGE_CACHE_VERSION,
            "protocol_snapshot": EVALUATION_PROTOCOL_SNAPSHOT,
            "protocol_sha256": protocol_hash,
        },
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def load_source_rows(path: Path, benchmark: str) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_rows(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if benchmark != "h2hmem" or not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON array in {path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(
            "H2HMem input must be a JSON array or a native object containing results[]"
        )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    dialogue_name = str(metadata.get("dialogue_name") or "")
    if not dialogue_name:
        dialogue_name = next(
            (part for part in path.parts if str(part).startswith("dialogue")),
            path.parent.name,
        )
    shared = {
        "dialogue_name": dialogue_name,
        "dataset": dialogue_name,
        "session_id": metadata.get("session_id", ""),
        "session_name": metadata.get(
            "session_dir_name", metadata.get("session_id", path.parent.name)
        ),
        "memory_type": metadata.get("memory_type", "unknown"),
        "vlm_model": metadata.get("vlm_model", "unknown"),
    }
    return [{**shared, **row} for row in results if isinstance(row, dict)]


def build_tasks(
    benchmark: str,
    members: list[tuple[int, dict[str, Any]]],
) -> list[JudgeTask]:
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return [JudgeTask(((index, row),)) for index, row in members]


def main() -> None:
    validate_protocol_snapshot()
    parser = argparse.ArgumentParser(
        description="Parallel LLM Judge for Mem-Gallery, WorldMemArena, and H2HMem."
    )
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default="memgallery",
        help="Interpret the source schema; protocol dispatch remains dataset_source-based.",
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument(
        "--key-file",
        default="",
        help="Optional file containing OpenRouter sk-or-v1-* keys.",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-count", type=int, default=0)
    parser.add_argument("--base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", "--max-retries", dest="retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--rerun-label", default="", help="Only rerun existing rows with this label, e.g. judge_error.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "defaults.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Provider/model/temperature/token defaults are protocol-controlled.
        mapping = {"judge_timeout": "timeout"}
        parser.set_defaults(**{dest: config[key] for key, dest in mapping.items() if key in config})
    args = parser.parse_args()

    if args.start < 1 or args.end < 0 or (args.end and args.end < args.start):
        parser.error("Require 1 <= --start <= --end, or use --end=0")
    if args.workers < 1 or args.retries < 1 or args.checkpoint_every < 1:
        parser.error("--workers, --retries, and --checkpoint-every must be positive")

    results_path = Path(args.results)
    if not results_path.is_file():
        raise FileNotFoundError(f"Judge results input not found: {results_path}")
    rows = load_source_rows(results_path, args.benchmark)
    end = args.end if args.end > 0 else len(rows)
    selected = [(idx, row) for idx, row in enumerate(rows, start=1) if args.start <= idx <= end]
    for index, row in selected:
        normalize_judge_row(args.benchmark, row, index)

    keys = load_api_keys(
        args.key_file,
        args.key_start,
        args.key_count,
        env_name=args.api_key_env,
    )
    clients = RoundRobinClients(keys, args.base_url, args.timeout)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    judged_path = out_dir / "llm_judge_results.json"
    metrics_path = out_dir / "llm_judge_metrics.json"
    progress_path = out_dir / "llm_judge_progress.jsonl"
    checkpoint_manifest_path = out_dir / "llm_judge_checkpoint.json"
    signature = checkpoint_signature(args, results_path)

    existing: dict[int, dict[str, Any]] = {}
    selected_indices = {idx for idx, _ in selected}
    if args.resume and checkpoint_manifest_path.exists():
        saved = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise RuntimeError(
                f"Judge checkpoint does not match the source results or settings: "
                f"{checkpoint_manifest_path}; rerun with --no-resume"
            )
        for path in (progress_path, judged_path):
            for row in load_rows(path):
                index = int(row["index"])
                if index in selected_indices:
                    existing[index] = row

    if args.rerun_label:
        pending = [
            (idx, row)
            for idx, row in selected
            if existing.get(idx, {}).get("label") == args.rerun_label
        ]
    else:
        pending = [
            (idx, row)
            for idx, row in selected
            if idx not in existing or existing[idx].get("label") == "judge_error"
        ]
    tasks = build_tasks(args.benchmark, pending)
    judged = dict(existing)

    def save_checkpoint() -> None:
        ordered_rows = [judged[index] for index in sorted(judged)]
        write_json_atomic(judged_path, ordered_rows)
        write_jsonl_atomic(progress_path, ordered_rows)
        write_json_atomic(
            metrics_path,
            summarize(
                ordered_rows,
                args.model,
                benchmark=args.benchmark,
                expected_count=len(selected),
            ),
        )
        write_json_atomic(
            checkpoint_manifest_path,
            {
                "signature": signature,
                "completed": len(ordered_rows),
                "expected": len(selected),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    completed_since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                judge_with_retries,
                task,
                clients,
                args.model,
                args.benchmark,
                args.max_tokens,
                args.retries,
            ): task.indices
            for task in tasks
        }
        for future in as_completed(futures):
            results = future.result()
            for result in results:
                judged[int(result["index"])] = result
            completed_since_checkpoint += len(results)
            if completed_since_checkpoint >= args.checkpoint_every:
                save_checkpoint()
                completed_since_checkpoint = 0
            for result in results:
                print(
                    f"[{len(judged)}/{len(selected)}] idx={result['index']} "
                    f"{result['category']} {result['label']} score={result['score']} "
                    f"reason={result['reason'][:100]}",
                    flush=True,
                )

    ordered = [judged[i] for i in sorted(judged)]
    summary = summarize(
        ordered,
        args.model,
        benchmark=args.benchmark,
        expected_count=len(selected),
    )
    save_checkpoint()
    benchmark_metrics_path = Path(args.results).parent / "metrics.json"
    if benchmark_metrics_path.exists():
        benchmark_metrics = json.loads(benchmark_metrics_path.read_text(encoding="utf-8"))
        combined = merge_llm_judge_metrics(benchmark_metrics, summary)
        source_result_dir = Path(args.results).parent
        run_manifest_path = source_result_dir / "run_manifest.json"
        run_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.exists()
            else {}
        )
        memory_metrics_path = out_dir / MEMORY_METRICS_FILENAME
        if not memory_metrics_path.exists() and run_manifest:
            try:
                write_memory_metrics(
                    run_manifest["index_root"],
                    out_dir,
                    tokenizer_name=str(run_manifest.get("memory_tokenizer") or ""),
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                print(f"memory metrics unavailable: {exc}", flush=True)
        if memory_metrics_path.exists():
            memory_metrics = json.loads(memory_metrics_path.read_text(encoding="utf-8"))
            combined = add_memory_metrics(combined, memory_metrics)

        retrieval_metrics_path = out_dir / RETRIEVAL_MEMORY_TOKEN_FILENAME
        if not retrieval_metrics_path.exists() and run_manifest:
            try:
                write_retrieval_memory_token(
                    source_result_dir,
                    out_dir,
                    tokenizer_name=str(
                        run_manifest.get("retrieval_memory_tokenizer")
                        or run_manifest.get("answer_model")
                        or ""
                    ),
                )
            except (OSError, KeyError, ValueError) as exc:
                print(f"retrieval memory token metrics unavailable: {exc}", flush=True)
        if retrieval_metrics_path.exists():
            retrieval_metrics = json.loads(
                retrieval_metrics_path.read_text(encoding="utf-8")
            )
            combined = add_retrieval_memory_tokens(combined, retrieval_metrics)
        summary_path = out_dir / "summary.json"
        if not summary["provisional"] and len(ordered) == len(selected):
            write_json_atomic(summary_path, combined)
        else:
            summary_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
