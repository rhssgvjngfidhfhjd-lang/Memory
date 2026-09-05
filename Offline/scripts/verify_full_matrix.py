"""Verify every artifact required by the 24-job formal experiment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_full_baseline_matrix import (
    BENCHMARKS,
    DEFAULT_BENCHMARK_MANIFEST,
    OUTPUT_ROOT,
    all_jobs,
    load_benchmark_manifest,
    validate_job_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report incomplete jobs without returning a failing exit status.",
    )
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK_MANIFEST,
        help="Full-benchmark completeness manifest.",
    )
    args = parser.parse_args()
    expectations = load_benchmark_manifest(
        args.benchmark_manifest,
        required_benchmarks=BENCHMARKS,
    )

    first, rest = all_jobs()
    jobs = [*first, *rest]
    passed: list[str] = []
    incomplete: list[str] = []
    invalid: list[tuple[str, str]] = []
    for job in jobs:
        result_dir = OUTPUT_ROOT / job.benchmark / job.method
        if not (result_dir / "results.json").is_file():
            incomplete.append(job.name)
            continue
        try:
            validate_job_outputs(job, result_dir, expectations)
        except Exception as exc:
            invalid.append((job.name, f"{type(exc).__name__}: {exc}"))
        else:
            passed.append(job.name)

    print(
        f"verified={len(passed)}/{len(jobs)} incomplete={len(incomplete)} "
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
