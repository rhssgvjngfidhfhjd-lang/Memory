from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.config import (
    load_baseline_registry,
    resolve_python,
    resolve_source_root,
    runtime_config,
)
from benchmarks.baseline_runtime.protocol import BaselineAdapter


_REGISTRY = load_baseline_registry()
BASELINE_NAMES = tuple(_REGISTRY)
_ALIASES = {name.lower(): name for name in BASELINE_NAMES}
_ALIASES.update(
    {
        "hive": "HiveMem",
        "omni-simplemem": "OmniSimpleMem",
        "omnisimplemem": "OmniSimpleMem",
        "memverse-main": "MemVerse",
        "m3-agent": "M3-Agent-caption",
        "m3-agent-master": "M3-Agent-caption",
        "mma-main": "MMA",
    }
)


def canonical_name(name: str) -> str:
    try:
        return _ALIASES[name.strip().lower()]
    except KeyError as exc:
        raise KeyError(
            f"unknown baseline {name!r}; choose one of {', '.join(BASELINE_NAMES)}"
        ) from exc


def create_adapter(
    name: str,
    *,
    config_overrides: dict[str, Any] | None = None,
    in_process: bool | None = None,
) -> BaselineAdapter:
    baseline = canonical_name(name)
    entry = dict(_REGISTRY[baseline])
    config = runtime_config(config_overrides)
    local = bool(entry.get("in_process")) if in_process is None else bool(in_process)
    if local:
        return create_local_adapter(baseline, config=config, entry=entry)
    from benchmarks.baseline_runtime.process import BaselineProcess

    return BaselineProcess(
        baseline,
        entry=entry,
        config=config,
        python_executable=resolve_python(entry),
    )


def create_local_adapter(
    name: str,
    *,
    config: dict[str, Any],
    entry: dict[str, Any],
) -> BaselineAdapter:
    baseline = canonical_name(name)
    adapter_name = str(entry["adapter"])
    source_root = resolve_source_root(entry)
    if not source_root.exists():
        raise FileNotFoundError(f"baseline source directory does not exist: {source_root}")
    common = {"baseline": baseline, "source_root": source_root, "config": config}
    if adapter_name == "hivemem":
        from benchmarks.baseline_runtime.adapters.hivemem import HiveMemAdapter

        return HiveMemAdapter(**common)
    if adapter_name == "memengine":
        from benchmarks.baseline_runtime.adapters.memengine import MemEngineAdapter

        return MemEngineAdapter(**common)
    if adapter_name == "omni_simplemem":
        from benchmarks.baseline_runtime.adapters.omni_simplemem import OmniSimpleMemAdapter

        return OmniSimpleMemAdapter(**common)
    if adapter_name == "m2a":
        from benchmarks.baseline_runtime.adapters.m2a import M2AAdapter

        return M2AAdapter(**common)
    if adapter_name == "mirix_family":
        from benchmarks.baseline_runtime.adapters.mirix_family import MirixFamilyAdapter

        return MirixFamilyAdapter(**common)
    if adapter_name == "memverse":
        from benchmarks.baseline_runtime.adapters.memverse import MemVerseAdapter

        return MemVerseAdapter(**common)
    if adapter_name == "m3_agent":
        from benchmarks.baseline_runtime.adapters.m3_agent import M3AgentAdapter

        return M3AgentAdapter(**common)
    raise KeyError(f"unknown adapter type: {adapter_name}")


def baseline_entry(name: str) -> dict[str, Any]:
    return dict(_REGISTRY[canonical_name(name)])


def baseline_metadata(name: str) -> dict[str, Any]:
    baseline = canonical_name(name)
    entry = baseline_entry(baseline)
    metadata = {
        "name": baseline,
        "adapter": entry["adapter"],
        "source_root": str(resolve_source_root(entry)),
        "python_executable": resolve_python(entry),
        "in_process": bool(entry.get("in_process")),
    }
    for key in ("compatibility_mode", "audio_enabled"):
        if key in entry:
            metadata[key] = entry[key]
    return metadata
