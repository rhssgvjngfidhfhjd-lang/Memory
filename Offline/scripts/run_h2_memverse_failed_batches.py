from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from benchmarks.h2hmem_harness.prompts import (
    build_answer_messages,
    parse_answer_response,
)
from benchmarks.io_utils import write_json_atomic
from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)


DIALOGUE_ORDER = ("dialogue3", "dialogue9", "dialogue7")


def _load_jobs(source_dir: Path) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    checkpoint_dir = source_dir / ".checkpoint" / "samples"
    for path in sorted(checkpoint_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact = payload.get("artifact") or {}
        for job in artifact.get("jobs") or []:
            query_id = str(job.get("query_id") or "")
            if query_id:
                jobs[query_id] = dict(job)
    return jobs


def _selected_rows(
    source_dir: Path,
    selection: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = json.loads((source_dir / "results.json").read_text(encoding="utf-8"))
    if selection == "failed":
        selected = [
            row
            for row in rows
            if row.get("error") or not str(row.get("system_answer") or "").strip()
        ]
    elif selection == "missing-usage":
        selected = [row for row in rows if row.get("answer_token_usage") is None]
    else:
        raise ValueError(f"Unsupported selection: {selection}")
    order = {name: index for index, name in enumerate(DIALOGUE_ORDER)}
    selected.sort(key=lambda row: order.get(str(row.get("dialogue_name") or ""), 99))
    return rows, selected


def _batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in DIALOGUE_ORDER}
    for row in rows:
        grouped.setdefault(str(row.get("dialogue_name") or ""), []).append(row)
    batches: list[list[dict[str, Any]]] = []
    for dialogue in DIALOGUE_ORDER:
        group = grouped.get(dialogue) or []
        batches.extend(
            group[index : index + batch_size]
            for index in range(0, len(group), batch_size)
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="Maximum answer tokens; raise only for responses truncated before </answer>.",
    )
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match the benchmark's default non-thinking answer configuration.",
    )
    parser.add_argument(
        "--selection",
        choices=("failed", "missing-usage"),
        default="failed",
        help="Select failed/empty answers or successful answers missing provider usage.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.concurrency <= 0 or args.num_predict <= 0:
        raise ValueError("batch-size, concurrency, and num-predict must be positive")

    source_rows, failed = _selected_rows(args.source_dir, args.selection)
    jobs = _load_jobs(args.source_dir)
    missing = [row["query_id"] for row in failed if row["query_id"] not in jobs]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} prepared H2 job(s): {missing[:3]}")
    batches = _batches(failed, args.batch_size)
    selection = {
        "count": len(failed),
        "batch_count": len(batches),
        "batch_sizes": [len(batch) for batch in batches],
        "dialogue_counts": {
            dialogue: sum(row.get("dialogue_name") == dialogue for row in failed)
            for dialogue in DIALOGUE_ORDER
        },
        "with_query_image": sum(bool(jobs[row["query_id"]].get("query_image_payload")) for row in failed),
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
        num_predict=args.num_predict,
        timeout=args.timeout,
        retries=args.retries,
        think=args.think,
        reasoning_effort="minimal",
    )

    def answer(source_row: dict[str, Any]) -> dict[str, Any]:
        job = jobs[source_row["query_id"]]
        evidence, _ = build_retrieved_memory_evidence(
            job["memory_items"], category="VR"
        )
        messages = build_answer_messages(
            question=job["question"],
            question_type=job["category"],
            memory_evidence=evidence,
            query_images=query_image_prompt_metadata(job.get("query_image_payload")),
        )
        started = time.time()
        response = None
        try:
            response = client.answer_messages_with_usage(
                messages=messages,
                memory_items=job["memory_items"],
                query_image=job.get("query_image_payload"),
                category="VR",
            )
            raw_answer = response.text
            answer_text = parse_answer_response(raw_answer)
            error = ""
            usage = response.usage
            attempts = response.attempts
            failed_attempts = response.failed_attempts
            image_count = response.image_count
        except Exception as exc:
            raw_answer = response.text if response is not None else ""
            answer_text = ""
            error = f"{type(exc).__name__}: {exc}"
            usage = response.usage if response is not None else None
            attempts = response.attempts if response is not None else args.retries + 1
            failed_attempts = (
                response.failed_attempts if response is not None else attempts
            )
            image_count = (
                response.image_count
                if response is not None
                else client.count_answer_images(
                    job["memory_items"],
                    query_image=job.get("query_image_payload"),
                    category="VR",
                )
            )
        recovered = dict(source_row)
        recovered.update(
            {
                "system_answer": answer_text,
                "answer_raw_response": raw_answer,
                "error": error,
                "answer_seconds": time.time() - started,
                "answer_token_usage": usage,
                "answer_attempts": attempts,
                "answer_failed_attempts": failed_attempts,
                "answer_image_count": image_count,
            }
        )
        return recovered

    recovered_by_id: dict[str, dict[str, Any]] = {}
    recovered_path = args.output_dir / "recovered_results.json"
    if recovered_path.is_file():
        recovered_by_id = {
            str(row["query_id"]): row
            for row in json.loads(recovered_path.read_text(encoding="utf-8"))
        }

    try:
        for batch_index, batch in enumerate(batches, start=1):
            pending = [row for row in batch if row["query_id"] not in recovered_by_id]
            if not pending:
                continue
            print(
                json.dumps(
                    {
                        "batch_start": batch_index,
                        "size": len(pending),
                        "dialogue": pending[0].get("dialogue_name"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            batch_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {pool.submit(answer, row): row for row in pending}
                for future in as_completed(futures):
                    result = future.result()
                    batch_rows.append(result)
                    recovered_by_id[result["query_id"]] = result
                    write_json_atomic(
                        recovered_path, list(recovered_by_id.values())
                    )
                    print(
                        json.dumps(
                            {
                                "query_id": result["query_id"],
                                "error": result["error"],
                                "attempts": result["answer_attempts"],
                                "image_count": result["answer_image_count"],
                                "duration_seconds": result["answer_seconds"],
                                "usage": result["answer_token_usage"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            write_json_atomic(
                args.output_dir / f"batch_{batch_index:03d}.json", batch_rows
            )
            failed_batch = [
                row
                for row in batch_rows
                if row.get("error")
                or not str(row.get("system_answer") or "").strip()
                or int(row.get("answer_failed_attempts") or 0) > 0
            ]
            status = {
                "completed": len(recovered_by_id),
                "expected": len(failed),
                "last_batch": batch_index,
                "halted": bool(failed_batch),
                "last_batch_problem_count": len(failed_batch),
            }
            write_json_atomic(args.output_dir / "status.json", status)
            print(json.dumps({"batch_complete": status}, ensure_ascii=False), flush=True)
            if failed_batch:
                raise RuntimeError(
                    f"Batch {batch_index} had {len(failed_batch)} error/empty/retried answer(s)"
                )
    finally:
        client._session.close()

    if len(recovered_by_id) != len(failed):
        raise RuntimeError(
            f"Recovered {len(recovered_by_id)} of {len(failed)} failed answers"
        )
    merged = [recovered_by_id.get(row["query_id"], row) for row in source_rows]
    remaining = [
        row
        for row in merged
        if row.get("error") or not str(row.get("system_answer") or "").strip()
    ]
    if remaining:
        raise RuntimeError(f"Merged results still contain {len(remaining)} failures")
    write_json_atomic(args.output_dir / "merged_results.json", merged)

    targets = [
        args.source_dir / "results.json",
        args.source_dir / ".checkpoint" / "results.json",
    ]
    for target in targets:
        if not target.is_file():
            continue
        backup = target.with_name(target.stem + ".before_transport_compression.json")
        if not backup.exists():
            shutil.copy2(target, backup)
        write_json_atomic(target, merged)
    summary = {
        "recovered": len(recovered_by_id),
        "total_results": len(merged),
        "remaining_errors": 0,
        "prompt_tokens": sum(
            int((row.get("answer_token_usage") or {}).get("prompt_tokens") or 0)
            for row in recovered_by_id.values()
        ),
        "completion_tokens": sum(
            int((row.get("answer_token_usage") or {}).get("completion_tokens") or 0)
            for row in recovered_by_id.values()
        ),
    }
    write_json_atomic(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
