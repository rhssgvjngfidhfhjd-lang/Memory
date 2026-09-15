#!/usr/bin/env python3
"""Prepare and verify the immutable MIRIX v0.1.1 source checkout."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = OFFLINE_ROOT / ".upstream" / "mirix-v0.1.1"
UPSTREAM_URL = "https://github.com/Mirix-AI/MIRIX.git"
UPSTREAM_COMMIT = "ac0a1f2890df5e7435c66d6c2827f34c5c4ce32d"


def _git(*args: str, cwd: Path | None = None) -> str:
    command = ["git", *(args)]
    result = subprocess.run(
        command,
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
    _git("fetch", "--tags", "origin", UPSTREAM_COMMIT, cwd=destination)
    _git("checkout", "--detach", UPSTREAM_COMMIT, cwd=destination)
    actual = _git("rev-parse", "HEAD", cwd=destination)
    dirty = _git("status", "--porcelain", cwd=destination)
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"MIRIX checkout mismatch: expected {UPSTREAM_COMMIT}, got {actual}")
    if dirty:
        raise RuntimeError(f"MIRIX checkout is dirty:\n{dirty}")
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", default=UPSTREAM_URL)
    args = parser.parse_args()
    prepare(args.destination, args.source)


if __name__ == "__main__":
    main()
