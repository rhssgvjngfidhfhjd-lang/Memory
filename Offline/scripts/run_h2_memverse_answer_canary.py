from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any

from benchmarks.h2hmem_harness.prompts import parse_answer_response
from benchmarks.io_utils import write_json_atomic
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient


def _memory_items(trace: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for hit in trace.get("top_k", []):
        image_paths = list(hit.get("image_paths") or [])
        image_ids = list(hit.get("image_ids") or [])
        images = []
        for index, path in enumerate(image_paths):
            images.append(
                {
                    "path": str(path),
                    "img_id": str(image_ids[index]) if index < len(image_ids) else "",
                    "kind": "image",
                }
            )
        items.append(
            {
                "text": str(hit.get("content") or ""),
                "metadata": {
                    "session_id": str(hit.get("session_id") or ""),
                    "dialogue_id": ",".join(hit.get("source_dialogue_ids") or []),
                },
                "images": images,
            }
        )
    return items


def _load_candidates(
    source_dir: Path, dialogue: str, limit: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    results = json.loads((source_dir / "results.json").read_text(encoding="utf-8"))
    traces = {
        str(row["query_id"]): row
        for row in (
            json.loads(line)
            for line in (source_dir / "retrieval_trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    selected = [
        row
        for row in results
        if row.get("error")
        and row.get("dialogue_name") == dialogue
        and not row.get("question_image")
    ][:limit]
    if len(selected) != limit:
        raise RuntimeError(
            f"Requested {limit} candidates, found {len(selected)} for {dialogue}"
        )
    return [(row, traces[str(row["query_id"])]) for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dialogue", default="dialogue9")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = _load_candidates(args.source_dir, args.dialogue, args.limit)
    selected = []
    for result, trace in candidates:
        items = _memory_items(trace)
        raw_count = sum(len(item["images"]) for item in items)
        unique_count = len(
            {str(image["path"]) for item in items for image in item["images"]}
        )
        selected.append(
            {
                "query_id": result["query_id"],
                "raw_image_count": raw_count,
                "unique_image_count": unique_count,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "selection.json", selected)
    print(json.dumps({"selection": selected}, ensure_ascii=False), flush=True)
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

    def answer(candidate: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        source_result, trace = candidate
        started = time.time()
        try:
            response = client.answer_messages_with_usage(
                messages=list(trace["answer_prompt_messages"]),
                memory_items=_memory_items(trace),
                query_image=None,
                category="VR",
            )
            parsed = parse_answer_response(response.text)
            error = ""
            usage = response.usage
            attempts = response.attempts
            image_count = response.image_count
        except Exception as exc:
            parsed = ""
            error = f"{type(exc).__name__}: {exc}"
            usage = None
            attempts = args.retries + 1
            image_count = 0
        return {
            "query_id": source_result["query_id"],
            "answer": parsed,
            "error": error,
            "usage": usage,
            "attempts": attempts,
            "image_count": image_count,
            "duration_seconds": time.time() - started,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(answer, candidate): candidate for candidate in candidates}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json_atomic(args.output_dir / "results.json", rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    client._session.close()
    errors = [row for row in rows if row["error"] or not row["answer"]]
    summary = {
        "count": len(rows),
        "errors": len(errors),
        "input_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in rows),
        "output_tokens": sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in rows),
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
