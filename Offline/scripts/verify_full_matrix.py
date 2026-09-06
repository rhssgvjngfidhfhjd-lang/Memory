"""Verify every artifact required by the 24-job formal experiment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_full_baseline_matrix import OUTPUT_ROOT, all_jobs, validate_job_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report incomplete jobs without returning a failing exit status.",
    )
    args = parser.parse_args()

    first, rest = all_jobs()
    passed: list[str] = []
    incomplete: list[str] = []
    invalid: list[tuple[str, str]] = []
    for job in [*first, *rest]:
        result_dir = OUTPUT_ROOT / job.benchmark / job.method
        if not (result_dir / "results.json").is_file():
            incomplete.append(job.name)
            continue
        try:
            validate_job_outputs(job, result_dir)
        except Exception as exc:
            invalid.append((job.name, f"{type(exc).__name__}: {exc}"))
        else:
            passed.append(job.name)

    print(
        f"verified={len(passed)}/24 incomplete={len(incomplete)} "
        f"invalid={len(invalid)}"
    )
    for name in passed:
        print("PASS", name)
    for name in incomplete:
        print("INCOMPLETE", name)
    for name, error in invalid:
        print("INVALID", name, error)

    if (incomplete or invalid) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
