#!/usr/bin/env python3
"""Compare sequential executor requests across OpenAI-compatible endpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

from hive_mem.builder import load_events
from hive_mem.executor import MemoryExecutor, visual_input_uses_images
from hive_mem.llm_client import LLMClient


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_executor(base_url: str, args: argparse.Namespace) -> MemoryExecutor:
    client = LLMClient(
        model=args.model,
        api_base=base_url.rstrip("/") + "/v1",
        api_key="EMPTY",
        temperature=0.0,
        max_new_tokens=args.max_tokens,
        top_p=1.0,
        max_retries=args.retries,
        timeout=args.timeout,
    )
    # The benchmark stops after generation/parsing and deliberately excludes
    # embedding so that both vLLM layouts see exactly the same isolated work.
    return MemoryExecutor(client, embedder=None)


def request_one(
    executor: MemoryExecutor,
    event: Any,
    profile: str,
    visual_input: str,
) -> dict[str, Any]:
    image_paths = event.image_paths if visual_input_uses_images(visual_input) else []
    started = time.perf_counter()
    raw_response, actions, usage = executor.execute_with_usage(
        chunk_text=event.text,
        profile=profile,
        image_paths=image_paths,
        visual_input=visual_input,
    )
    elapsed = time.perf_counter() - started
    return {
        "source_chunk_id": event.source_chunk_id,
        "elapsed_seconds": elapsed,
        "has_image": bool(image_paths),
        "response_chars": len(raw_response),
        "actions": len(actions),
        "successful_actions": sum(action.success for action in actions),
        "parse_ok": any(action.success for action in actions),
        "usage": usage,
    }


def benchmark_endpoint(
    name: str,
    base_url: str,
    events: list[Any],
    warmup_events: list[Any],
    profiles: dict[str, str],
    barrier: threading.Barrier,
    args: argparse.Namespace,
) -> dict[str, Any]:
    executor = make_executor(base_url, args)
    for event in warmup_events:
        request_one(
            executor,
            event,
            profiles.get(event.dataset, ""),
            args.visual_input,
        )

    barrier.wait()
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=args.client_concurrency) as pool:
        pending = {}
        for index, event in enumerate(events):
            future = pool.submit(
                request_one,
                executor,
                event,
                profiles.get(event.dataset, ""),
                args.visual_input,
            )
            pending[future] = index
        for completed, future in enumerate(as_completed(pending), start=1):
            row = future.result()
            row["event_index"] = pending[future]
            rows.append(row)
            if completed % 10 == 0 or completed == len(events):
                print(
                    f"[{name}] {completed}/{len(events)} "
                    f"last={row['elapsed_seconds']:.3f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda row: row["event_index"])

    latencies = [row["elapsed_seconds"] for row in rows]
    image_latencies = [row["elapsed_seconds"] for row in rows if row["has_image"]]
    text_latencies = [row["elapsed_seconds"] for row in rows if not row["has_image"]]
    usage = {
        key: sum(int(row["usage"].get(key, 0)) for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    summary = {
        "name": name,
        "base_url": base_url,
        "dataset": args.dataset,
        "client_concurrency": args.client_concurrency,
        "requests": len(rows),
        "image_requests": len(image_latencies),
        "elapsed_seconds": elapsed,
        "requests_per_second": len(rows) / elapsed,
        "completion_tokens_per_second": usage["completion_tokens"] / elapsed,
        "total_tokens_per_second": usage["total_tokens"] / elapsed,
        "latency_mean_seconds": statistics.fmean(latencies),
        "latency_p50_seconds": percentile(latencies, 0.50),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "text_latency_mean_seconds": statistics.fmean(text_latencies),
        "image_latency_mean_seconds": statistics.fmean(image_latencies),
        "parse_failures": sum(not row["parse_ok"] for row in rows),
        "usage": usage,
    }
    return {"summary": summary, "requests": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl")
    parser.add_argument("--profiles-file", default="configs/profiles.json")
    parser.add_argument("--dataset", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--endpoint", action="append", required=True, metavar="NAME=URL")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--visual-input", default="image", choices=("image", "caption", "image_caption"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--client-concurrency", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    endpoints = []
    for value in args.endpoint:
        if "=" not in value:
            parser.error(f"invalid --endpoint {value!r}; expected NAME=URL")
        endpoints.append(tuple(value.split("=", 1)))

    all_events = load_events(args.chunks)
    events = [event for event in all_events if event.dataset == args.dataset]
    if not events:
        parser.error(f"dataset not found: {args.dataset}")
    warmup_pool = [event for event in all_events if event.dataset != args.dataset]
    warmup_text = next(event for event in warmup_pool if not event.image_paths)
    warmup_image = next(event for event in warmup_pool if event.image_paths)
    warmup_events = [warmup_text, warmup_image]
    profiles = json.loads(Path(args.profiles_file).read_text(encoding="utf-8"))

    barrier = threading.Barrier(len(endpoints))
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        futures = [
            pool.submit(
                benchmark_endpoint,
                name,
                url,
                events,
                warmup_events,
                profiles,
                barrier,
                args,
            )
            for name, url in endpoints
        ]
        results = [future.result() for future in futures]

    payload = {
        "configuration": {
            "chunks": args.chunks,
            "dataset": args.dataset,
            "visual_input": args.visual_input,
            "max_tokens": args.max_tokens,
            "client_concurrency": args.client_concurrency,
            "warmup_requests_per_endpoint": len(warmup_events),
            "endpoints": dict(endpoints),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([result["summary"] for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
