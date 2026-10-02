#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.baseline_runtime.call_contract import audit_baseline_call_flow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--expected-qa", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_baseline_call_flow(
        args.run_dir,
        baseline=args.baseline,
        benchmark=args.benchmark,
        expected_qa=args.expected_qa,
        expected_top_k=args.top_k,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
