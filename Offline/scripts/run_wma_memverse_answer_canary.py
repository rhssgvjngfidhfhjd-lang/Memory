from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from benchmarks.io_utils import write_json_atomic
from benchmarks.wma_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
    build_retrieved_memory_evidence,
)
from benchmarks.wma_harness.runner.prompts import (
    build_answer_messages,
    parse_answer_response,
)


def _load_jobs(source_artifact: Path) -> list[dict[str, Any]]:
    payload = json.loads(source_artifact.read_text(encoding="utf-8"))
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    jobs = artifact.get("jobs") if isinstance(artifact, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"No WMA jobs found in {source_artifact}")
    return [dict(job) for job in jobs]


def _select_worst_visual_job(
    jobs: list[dict[str, Any]], category: str
) -> tuple[dict[str, Any], int, int]:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for job in jobs:
        if str(job.get("category") or "").upper() != category.upper():
            continue
        _, image_paths = build_retrieved_memory_context(
            list(job["memory_items"]), str(job.get("category") or "")
        )
        raw_count = sum(
            len(item.get("images") or []) for item in job["memory_items"]
        )
        candidates.append((len(image_paths), raw_count, job))
    if not candidates:
        raise RuntimeError(f"No WMA jobs found for category {category!r}")
    unique_count, raw_count, job = max(candidates, key=lambda row: row[0])
    if unique_count == 0:
        raise RuntimeError(f"Selected {category!r} job has no attached images")
    return job, raw_count, unique_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category", default="VFR")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job, raw_count, unique_count = _select_worst_visual_job(
        _load_jobs(args.source_artifact), args.category
    )
    selection = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "category": job["category"],
        "raw_image_count": raw_count,
        "unique_image_count": unique_count,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "selection.json", selection)
    print(json.dumps({"selection": selection}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    client = VLMAnswerClient(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=0.0,
        num_predict=512,
        timeout=args.timeout,
        retries=args.retries,
        reasoning_effort="minimal",
    )
    # Force the API answer path even if a source checkpoint ever gains a
    # baseline-native cached answer.
    job["native_answer"] = None
    evidence, _ = build_retrieved_memory_evidence(
        job["memory_items"], job["category"]
    )
    messages = build_answer_messages(
        question=job["question"],
        question_type=job["category"],
        memory_evidence=evidence,
    )
    started = time.time()
    try:
        response = client.answer_messages_with_usage(
            messages=messages,
            memory_items=job["memory_items"],
            category=job["category"],
        )
        answer = parse_answer_response(response.text)
        error = ""
        usage = response.usage
        attempts = response.attempts
        failed_attempts = response.failed_attempts
        image_count = response.image_count
        raw_answer = response.text
    except Exception as exc:
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
        usage = None
        attempts = args.retries + 1
        failed_attempts = attempts
        image_count = unique_count
        raw_answer = ""
    finally:
        client._session.close()

    duration_seconds = time.time() - started
    result = {
        "query_id": job["query_id"],
        "manifest_question_id": job["manifest_question_id"],
        "category": job["category"],
        "question": job["question"],
        "system_answer": answer,
        "answer_raw_response": raw_answer,
        "error": error,
        "answer_token_usage": usage,
        "answer_attempts": attempts,
        "answer_failed_attempts": failed_attempts,
        "answer_image_count": image_count,
        "answer_seconds": duration_seconds,
    }
    trace = {
        "query_id": job["query_id"],
        "answer_prompt_messages": messages,
    }
    write_json_atomic(args.output_dir / "result.json", result)
    write_json_atomic(args.output_dir / "trace.json", trace)
    summary = {
        "query_id": result["query_id"],
        "error": str(result.get("error") or ""),
        "empty_answer": not bool(str(result.get("system_answer") or "").strip()),
        "attempts": int(result.get("answer_attempts") or 0),
        "failed_attempts": int(result.get("answer_failed_attempts") or 0),
        "image_count": int(result.get("answer_image_count") or 0),
        "duration_seconds": duration_seconds,
        "usage": result.get("answer_token_usage"),
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    if summary["error"] or summary["empty_answer"] or summary["failed_attempts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
