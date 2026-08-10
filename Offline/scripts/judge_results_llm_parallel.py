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

from benchmarks.memgallery_harness.runner.metrics import (
    MEMORY_METRICS_FILENAME,
    add_memory_metrics,
    merge_llm_judge_metrics,
    write_memory_metrics,
)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_prompt(row: dict[str, Any]) -> str:
    return f"""You are evaluating a QA system for a memory benchmark.

Judge whether the predicted answer is semantically correct with respect to the reference answer and question.
Be strict about yes/no, before/after, dates, image IDs, and contradictions.
Do not require exact wording. A verbose answer can be correct if it clearly contains the reference answer and does not contradict it.
If the prediction is empty or says the opposite of the reference, mark it incorrect.

Return only JSON in this schema:
{{"score": 0 or 1, "label": "correct" or "incorrect", "reason": "short explanation"}}

Question:
{row.get("question", "")}

Reference answer:
{row.get("original_answer", "")}

Predicted answer:
{row.get("system_answer", "")}
"""


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
    config_path = Path(__file__).resolve().parents[1] / "configs" / "defaults.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mapping = {"judge_model": "model", "judge_timeout": "timeout",
                   "judge_key_file": "key_file", "judge_key_count": "key_count",
                   "judge_max_tokens": "max_tokens"}
        parser.set_defaults(**{dest: config[key] for key, dest in mapping.items() if key in config})
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
    benchmark_metrics_path = Path(args.results).parent / "metrics.json"
    if benchmark_metrics_path.exists():
        benchmark_metrics = json.loads(benchmark_metrics_path.read_text(encoding="utf-8"))
        combined = merge_llm_judge_metrics(benchmark_metrics, summary)
        memory_metrics_path = out_dir / MEMORY_METRICS_FILENAME
        if not memory_metrics_path.exists():
            run_manifest_path = Path(args.results).parent / "run_manifest.json"
            if run_manifest_path.exists():
                run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
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
        (out_dir / "summary.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
