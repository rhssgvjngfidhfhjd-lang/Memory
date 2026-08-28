from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = OFFLINE_ROOT / "configs" / "defaults.json"
BASELINES_PATH = OFFLINE_ROOT / "configs" / "baselines.json"
REQUIRED_RUNTIME_KEYS = (
    "answer_model",
    "answer_base_url",
    "executor_model",
    "executor_base_url",
    "embedding_model",
    "embedding_base_url",
    "embedding_dim",
    "top_k",
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_defaults(path: str | Path = DEFAULTS_PATH) -> dict[str, Any]:
    values = load_json(path)
    values.pop("_comment", None)
    return values


def load_baseline_registry(path: str | Path = BASELINES_PATH) -> dict[str, dict[str, Any]]:
    return {str(key): dict(value) for key, value in load_json(path).items()}


def resolve_source_root(entry: dict[str, Any]) -> Path:
    root = Path(str(entry["source_root"]))
    return root.resolve() if root.is_absolute() else (OFFLINE_ROOT / root).resolve()


def resolve_python(entry: dict[str, Any]) -> str:
    env_name = str(entry.get("python_env") or "")
    return os.getenv(env_name) or os.getenv("BASELINE_PYTHON") or os.sys.executable


def runtime_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_defaults()
    config.update(overrides or {})
    config.setdefault("answer_temperature", 0.0)
    config.setdefault("executor_temperature", 0.0)
    config.setdefault("embedding_base_url", "")
    config.setdefault("embedding_api_key_env", "EMBEDDING_API_KEY")
    config.setdefault("baseline_worker_timeout", 180)
    config.setdefault("baseline_strict_config", True)
    missing = [key for key in REQUIRED_RUNTIME_KEYS if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"missing baseline runtime config: {', '.join(missing)}")
    for key in ("embedding_dim", "top_k", "baseline_worker_timeout"):
        if float(config[key]) <= 0:
            raise ValueError(f"{key} must be greater than zero")
    return config
