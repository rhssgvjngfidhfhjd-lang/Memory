#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.memgallery_harness.runner.metrics import (
    add_retrieval_memory_tokens,
    write_retrieval_memory_token,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill retrieved-memory text-token metrics from QA outputs."
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing retrieval_trace.jsonl and run_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; defaults to --result-dir.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Tokenizer override; defaults to answer_model in run_manifest.json.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    metrics = write_retrieval_memory_token(
        result_dir,
        output_dir,
        tokenizer_name=args.tokenizer,
    )

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        combined = add_retrieval_memory_tokens(summary, metrics)
        summary_path.write_text(
            json.dumps(combined, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
