from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import summarize_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    metrics = summarize_results(results, k=5)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
