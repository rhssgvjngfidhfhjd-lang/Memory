from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.memeye_harness.dataset import load_memeye_samples
from benchmarks.multimodal_dataset_harness.runner import (
    add_common_arguments,
    run_harness,
    select_samples,
    validate_common_arguments,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEMEYE_DATA_DIR = WORKSPACE_ROOT / "MemEye" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MemVerse, M2A, M3-Agent, or MIRIX on MemEye."
    )
    add_common_arguments(parser, default_data_dir=DEFAULT_MEMEYE_DATA_DIR)
    args = parser.parse_args()
    validate_common_arguments(parser, args)
    samples = load_memeye_samples(Path(args.data_dir))
    samples = select_samples(
        samples,
        sample_ids=args.sample_id,
        question_ids=args.question_id,
        max_qa=args.max_qa,
    )
    run_harness(
        args=args,
        benchmark="MemEye",
        samples=samples,
        source_paths=[sample.source_path for sample in samples],
    )


if __name__ == "__main__":
    main()
