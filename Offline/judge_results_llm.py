#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_nvidia_keys(path: str | Path) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8")
    return re.findall(r'"(nvapi-[^"]+)"', raw)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge Mem-Gallery QA results with an OpenAI-compatible LLM.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--key-file", default="/data1/haozhen/Visual_Primitives/Offline/Nvida_api/apikey")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    keys = load_nvidia_keys(args.key_file)
    if not keys:
        raise ValueError("No NVIDIA API keys found")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.results).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    judged_path = out_dir / "llm_judge_results.json"
    metrics_path = out_dir / "llm_judge_metrics.json"
    judged = []
    if judged_path.exists() and args.start > 1:
        judged = json.loads(judged_path.read_text(encoding="utf-8"))

    end = args.end if args.end > 0 else len(rows)
    for idx, row in enumerate(rows, start=1):
        if idx < args.start or idx > end:
            continue
        last_error = ""
        result = None
        for attempt in range(args.retries + 1):
            key = keys[(idx + attempt - 1) % len(keys)]
            client = OpenAI(
                base_url=args.base_url,
                api_key=key,
                timeout=args.timeout,
                max_retries=0,
            )
            try:
                result = judge_one(client, args.model, row, args.max_tokens)
                break
            except Exception as exc:
                last_error = str(exc)
                if attempt < args.retries:
                    time.sleep(1 + attempt)
        if result is None:
            result = {
                "score": 0,
                "label": "judge_error",
                "reason": last_error,
                "raw_judge": "",
            }
        judged_row = {
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
        judged.append(judged_row)
        judged_path.write_text(
            json.dumps(judged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{idx}/{len(rows)}] {judged_row['category']} {judged_row['label']} "
            f"score={judged_row['score']} reason={judged_row['reason'][:100]}",
            flush=True,
        )

    total = len(judged)
    correct = sum(int(r["score"]) for r in judged)
    by_category: dict[str, dict[str, Any]] = {}
    for row in judged:
        cat = row["category"]
        stats = by_category.setdefault(cat, {"count": 0, "correct": 0})
        stats["count"] += 1
        stats["correct"] += int(row["score"])
    for stats in by_category.values():
        stats["accuracy"] = stats["correct"] / stats["count"] if stats["count"] else 0.0

    summary = {
        "model": args.model,
        "count": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_category": dict(sorted(by_category.items())),
    }
    judged_path.write_text(
        json.dumps(judged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
