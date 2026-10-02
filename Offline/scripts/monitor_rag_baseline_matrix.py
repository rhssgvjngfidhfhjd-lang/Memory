#!/usr/bin/env python3
"""Print a compact status summary for one RAG matrix run prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id_prefix")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    found = False
    for profile in ("qwen3vl4b_local", "gpt5mini_api"):
        run_id = f"{args.run_id_prefix}_{profile}"
        path = args.output_root / "_runs" / run_id / "status.json"
        if not path.is_file():
            print(f"{profile}: missing {path}")
            continue
        found = True
        status = json.loads(path.read_text(encoding="utf-8"))
        jobs = status.get("jobs") or {}
        smoke = status.get("smoke") or {}
        judges = status.get("judges") or {}
        counts = lambda rows: {
            state: sum(1 for row in rows.values() if row.get("status") == state)
            for state in ("pending", "running", "retrying", "completed", "failed")
        }
        print(
            json.dumps(
                {
                    "profile": profile,
                    "phase": status.get("phase"),
                    "smoke": counts(smoke),
                    "jobs": counts(jobs),
                    "judges": counts(judges),
                    "fatal_error": status.get("fatal_error", ""),
                },
                ensure_ascii=False,
            )
        )
    if not found:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
