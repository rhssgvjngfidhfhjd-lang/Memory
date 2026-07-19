#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from judge_results_llm import build_prompt, extract_json


def load_nvidia_keys(path: str | Path, key_start: int = 0, key_count: int = 0) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    keys = re.findall(r'"(nvapi-[^"]+)"', raw)
    if key_start:
        keys = keys[key_start:]
    if key_count > 0:
        keys = keys[:key_count]
    return keys


class RoundRobinClients:
    def __init__(self, keys: list[str], base_url: str, timeout: int):
        if not keys:
            raise ValueError("No NVIDIA API keys found")
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


def judge_one(client: OpenAI, model: str, row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(row)}],
        temperature=0,
        max_tokens=max_tokens,
    )
    message = resp.choices[0].message
    content = (
        message.content
        or getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    parsed = extract_json(content)
    score = int(parsed.get("score", 0))
    return {
        "score": 1 if score else 0,
        "label": "correct" if score else "incorrect",
        "reason": str(parsed.get("reason", "")),
        "raw_judge": content,
    }


def judge_with_retries(
    idx: int,
    row: dict[str, Any],
    clients: RoundRobinClients,
    model: str,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            result = judge_one(clients.next_client(), model, row, max_tokens)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1 + attempt)
    else:
        result = {
            "score": 0,
            "label": "judge_error",
            "reason": last_error,
            "raw_judge": "",
        }
    return {
        "index": idx,
        "dataset": row.get("dataset", ""),
        "category": row.get("category", ""),
        "question": row.get("question", ""),
        "prediction": row.get("system_answer", ""),
        "reference": row.get("original_answer", ""),
        "retrieved_ids": row.get("retrieved_ids", []),
        "clue": row.get("clue", []),
        **result,
    }


def summarize(judged: list[dict[str, Any]], model: str) -> dict[str, Any]:
    total = len(judged)
    correct = sum(int(r["score"]) for r in judged)
    by_category: dict[str, dict[str, Any]] = {}
    for row in judged:
        cat = row["category"]
        stats = by_category.setdefault(cat, {"count": 0, "correct": 0, "judge_error": 0})
        stats["count"] += 1
        stats["correct"] += int(row["score"])
        if row.get("label") == "judge_error":
            stats["judge_error"] += 1
    for stats in by_category.values():
        stats["accuracy"] = stats["correct"] / stats["count"] if stats["count"] else 0.0
    return {
        "model": model,
        "count": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_category": dict(sorted(by_category.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel judge Mem-Gallery QA results with key round-robin.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--key-file", default="/data1/haozhen/Visual_Primitives/Offline/Nvida_api/apikey")
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--key-count", type=int, default=0)
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=0)
    parser.add_argument("--rerun-label", default="", help="Only rerun existing rows with this label, e.g. judge_error.")
    args = parser.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    end = args.end if args.end > 0 else len(rows)
    selected = [(idx, row) for idx, row in enumerate(rows, start=1) if args.start <= idx <= end]

    keys = load_nvidia_keys(args.key_file, args.key_start, args.key_count)
    clients = RoundRobinClients(keys, args.base_url, args.timeout)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    judged_path = out_dir / "llm_judge_results.json"
    metrics_path = out_dir / "llm_judge_metrics.json"
    progress_path = out_dir / "llm_judge_progress.jsonl"

    existing: dict[int, dict[str, Any]] = {}
    if judged_path.exists():
        for row in json.loads(judged_path.read_text(encoding="utf-8")):
            existing[int(row["index"])] = row

    if args.rerun_label:
        pending = [
            (idx, row)
            for idx, row in selected
            if existing.get(idx, {}).get("label") == args.rerun_label
        ]
    else:
        pending = [(idx, row) for idx, row in selected if idx not in existing]
    judged = dict(existing)
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                judge_with_retries,
                idx,
                row,
                clients,
                args.model,
                args.max_tokens,
                args.retries,
            ): idx
            for idx, row in pending
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            with lock:
                judged[idx] = result
                ordered = [judged[i] for i in sorted(judged)]
                judged_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
                metrics_path.write_text(
                    json.dumps(summarize(ordered, args.model), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"[{len(judged)}/{len(selected)}] idx={idx} {result['category']} "
                    f"{result['label']} score={result['score']} reason={result['reason'][:100]}",
                    flush=True,
                )

    ordered = [judged[i] for i in sorted(judged)]
    summary = summarize(ordered, args.model)
    judged_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
