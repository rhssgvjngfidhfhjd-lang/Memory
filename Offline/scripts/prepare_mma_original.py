#!/usr/bin/env python3
"""Prepare and verify the immutable official MMA source checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = OFFLINE_ROOT / ".upstream" / "mma-c0e1a12"
UPSTREAM_URL = "https://github.com/AIGeeksGroup/MMA.git"
UPSTREAM_COMMIT = "c0e1a127722edcfa4db5e71d03708cba53363000"
UPSTREAM_TREE = "4bb8b33535cb9d14b39277d953bcc80b5aec2e4c"


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def prepare(destination: Path, source: str) -> Path:
    destination = destination.expanduser().resolve()
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--no-checkout", source, str(destination))
    _git("fetch", "origin", UPSTREAM_COMMIT, cwd=destination)
    _git("checkout", "--detach", UPSTREAM_COMMIT, cwd=destination)
    commit = _git("rev-parse", "HEAD", cwd=destination)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=destination)
    dirty = _git("status", "--porcelain", cwd=destination)
    if commit != UPSTREAM_COMMIT or tree != UPSTREAM_TREE:
        raise RuntimeError(
            "MMA checkout mismatch: "
            f"expected {UPSTREAM_COMMIT}/{UPSTREAM_TREE}, got {commit}/{tree}"
        )
    if dirty:
        raise RuntimeError(f"MMA checkout is dirty:\n{dirty}")
    source_root = destination / "MMA"
    print(source_root)
    return source_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", default=UPSTREAM_URL)
    args = parser.parse_args()
    prepare(args.destination, args.source)


if __name__ == "__main__":
    main()
