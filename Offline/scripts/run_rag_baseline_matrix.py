#!/usr/bin/env python3
"""Launch the fixed-chunk RAG baseline matrix for local Qwen and GPT-5 mini."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "rag_baseline_matrix.json"
RUNNER = ROOT / "scripts" / "run_test_baseline_matrix.py"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"NaiveRAG", "MuRAG", "UniversalRAG"}
    if set(config.get("baselines") or []) != expected:
        raise ValueError("RAG matrix must contain exactly NaiveRAG, MuRAG, UniversalRAG")
    if int(config.get("top_k") or 0) != 7:
        raise ValueError("RAG matrix requires top_k=7")
    if int(config["embedding"]["dimension"]) != 2048:
        raise ValueError("RAG matrix requires 2048-dimensional VL embeddings")
    return config


def resolve(path: str, *, config_path: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    # Matrix paths follow the existing Offline config convention.
    beside_config = (config_path.parent / value).resolve()
    if len(value.parts) == 1 and beside_config.exists():
        return beside_config
    return (ROOT / value).resolve()


def profile_command(
    config: dict[str, Any],
    *,
    config_path: Path,
    profile_name: str,
    run_id_prefix: str,
    smoke_only: bool,
    skip_smoke: bool,
) -> tuple[list[str], dict[str, str], str]:
    profile = dict(config["answer_profiles"][profile_name])
    run_id = f"{run_id_prefix}_{profile_name}"
    command = [
        sys.executable,
        str(RUNNER),
        "--run-id",
        run_id,
        "--defaults",
        str(resolve(str(profile["defaults"]), config_path=config_path)),
        "--efficiency-config",
        str(resolve(str(profile["efficiency_config"]), config_path=config_path)),
        "--split-manifest",
        str(resolve(str(config["split_manifest"]), config_path=config_path)),
        "--embedding-base-url",
        str(config["embedding"]["base_url"]),
        "--top-k",
        str(config["top_k"]),
    ]
    for baseline in config["baselines"]:
        command.extend(["--baseline", str(baseline)])
    for benchmark in config["benchmarks"]:
        command.extend(["--benchmark", str(benchmark)])
    endpoints = [str(value) for value in profile.get("endpoints") or []]
    workers = int(profile.get("workers") or len(endpoints))
    if not endpoints or workers < 1:
        raise ValueError(f"profile {profile_name} has no endpoints/workers")
    for index in range(workers):
        command.extend(["--endpoint", endpoints[index % len(endpoints)]])
    if smoke_only:
        command.append("--smoke-only")
    if skip_smoke:
        command.append("--skip-smoke")

    env = os.environ.copy()
    key_file = str(profile.get("api_key_file") or "").strip()
    if key_file:
        key_path = resolve(key_file, config_path=config_path)
        key = key_path.read_text(encoding="utf-8").strip()
        if len(key) < 20:
            raise ValueError(f"API key file is empty or invalid: {key_path}")
        env["OPENAI_API_KEY"] = key
        env["OPENROUTER_API_KEY"] = key
    else:
        env["OPENAI_API_KEY"] = "EMPTY"
    env["EMBEDDING_API_KEY"] = "EMPTY"
    return command, env, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        action="append",
        help="Run only this answer profile; repeat to select both.",
    )
    parser.add_argument(
        "--run-id-prefix",
        default=datetime.now().strftime("%m%d_rag_baselines_v1"),
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke_only and args.skip_smoke:
        parser.error("--smoke-only and --skip-smoke are mutually exclusive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id_prefix):
        parser.error("--run-id-prefix must be a safe path component")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    selected = args.profile or list(config["answer_profiles"])
    unknown = sorted(set(selected) - set(config["answer_profiles"]))
    if unknown:
        parser.error(f"unknown answer profiles: {unknown}")

    children: list[tuple[str, str, subprocess.Popen[str]]] = []
    for profile_name in selected:
        command, env, run_id = profile_command(
            config,
            config_path=config_path,
            profile_name=profile_name,
            run_id_prefix=args.run_id_prefix,
            smoke_only=args.smoke_only,
            skip_smoke=args.skip_smoke,
        )
        print(f"launching {profile_name} as {run_id}", flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env, text=True)
        children.append((profile_name, run_id, child))

    failures = []
    for profile_name, run_id, child in children:
        return_code = child.wait()
        print(f"finished {profile_name} ({run_id}) exit={return_code}", flush=True)
        if return_code:
            failures.append(f"{profile_name}:{return_code}")
    if failures:
        raise RuntimeError("RAG matrix profile failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
