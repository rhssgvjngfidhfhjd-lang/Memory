from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.memlens_harness.dataset import load_memlens_samples
from benchmarks.multimodal_dataset_harness.runner import (
    add_common_arguments,
    run_harness,
    select_samples,
    validate_common_arguments,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMLENS_DATA_DIR = WORKSPACE_ROOT / "MEMLENS"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MemVerse, M2A, M3-Agent, or MIRIX on MEMLENS."
    )
    add_common_arguments(parser, default_data_dir=DEFAULT_MEMLENS_DATA_DIR)
    parser.add_argument("--dataset-file", default="dataset_32k.json")
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    samples = load_memlens_samples(Path(args.data_dir), args.dataset_file)
    samples = select_samples(
        samples,
        sample_ids=args.sample_id,
        question_ids=args.question_id,
        max_qa=args.max_qa,
    )
    run_harness(
        args=args,
        benchmark="MEMLENS",
        samples=samples,
        source_paths=[sample.source_path for sample in samples],
    )


if __name__ == "__main__":
    main()
