from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .metrics import summarize_results


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_WORLDMEMARENA_ROOT = PROJECT_ROOT / "WorldMemArena"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WorldMemArena result JSON.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument(
        "--official-mode", choices=("none", "metrics", "judge"), default="none"
    )
    parser.add_argument(
        "--worldmemarena-root",
        default=str(DEFAULT_WORLDMEMARENA_ROOT),
    )
    args = parser.parse_args()
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    metrics = summarize_results(rows, k=args.top_k)
    if args.official_mode != "none":
        framework_root = Path(args.worldmemarena_root)
        sys.path.insert(0, str(framework_root))
        from eval_framework.cli import _qa_record_from_dict
        from eval_framework.evaluators.qa import (
            evaluate_checkpoint_qa,
            evaluate_checkpoint_qa_metrics_only,
        )

        pipeline_path = Path(args.results).with_name("pipeline_qa.jsonl")
        pipeline_rows = [
            json.loads(line)
            for line in pipeline_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluator = (
            evaluate_checkpoint_qa_metrics_only
            if args.official_mode == "metrics"
            else evaluate_checkpoint_qa
        )
        official: list[dict[str, Any]] = [
            evaluator(_qa_record_from_dict(row)) for row in pipeline_rows
        ]
        count = len(official)
        metrics["official"] = {
            "mode": args.official_mode,
            "count": count,
            "answer_f1": (
                sum(float(row.get("answer_f1") or 0.0) for row in official) / count
                if count else 0.0
            ),
            "answer_bleu1": (
                sum(float(row.get("answer_bleu1") or 0.0) for row in official) / count
                if count else 0.0
            ),
            "correct": sum(row.get("answer_label") == "Correct" for row in official),
        }
        official_path = Path(args.output).with_name("official_qa_eval.jsonl")
        with official_path.open("w", encoding="utf-8") as handle:
            for row in official:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
