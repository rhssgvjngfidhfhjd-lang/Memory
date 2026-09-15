#!/usr/bin/env python3
"""Prepare and verify the immutable official OmniSimpleMem source checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = OFFLINE_ROOT / ".upstream" / "simplemem-836ce97"
UPSTREAM_URL = "https://github.com/aiming-lab/SimpleMem.git"
UPSTREAM_COMMIT = "836ce9718f3e9cb7f93c9d7c842b47f62e177a66"
OMNI_TREE = "685109637c4c8b9a2469e695ad3dbed40762c0f2"


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

    actual_commit = _git("rev-parse", "HEAD", cwd=destination)
    actual_tree = _git("rev-parse", "HEAD:OmniSimpleMem", cwd=destination)
    dirty = _git("status", "--porcelain", cwd=destination)
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"SimpleMem checkout mismatch: expected {UPSTREAM_COMMIT}, got {actual_commit}"
        )
    if actual_tree != OMNI_TREE:
        raise RuntimeError(
            f"OmniSimpleMem tree mismatch: expected {OMNI_TREE}, got {actual_tree}"
        )
    if dirty:
        raise RuntimeError(f"SimpleMem checkout is dirty:\n{dirty}")

    source_root = destination / "OmniSimpleMem"
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
