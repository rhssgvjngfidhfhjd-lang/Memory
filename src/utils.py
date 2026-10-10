"""Shared configuration, paths, runtime layouts, and atomic file utilities.

This module uses only the standard library so data tools can import it without
initializing model or training dependencies.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


def _project_root() -> Path:
    configured = os.getenv("HIVE_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file() and parent / "src/utils.py" == module_path:
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _project_root()
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent / "benchmarks"
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
ALIASES = {
    "Mem-Gallery": "memgallery", "mem_gallery": "memgallery",
    "H2HMEM": "h2hmem", "h2hmem_dyadic": "h2hmem", "h2hmem_multiparty": "h2hmem",
    "WorldMemArena": "wma", "worldmemarena": "wma", "worldmemarena_lifelong": "wma",
}
DATASETS = ("memgallery", "h2hmem", "wma")


def benchmark_path(name: str) -> Path:
    """Locate benchmark assets in both source and installed distributions."""
    path = (BENCHMARK_ROOT / name).resolve()
    if not path.is_relative_to(BENCHMARK_ROOT.resolve()):
        raise ValueError(f"Benchmark asset escapes its root: {name}")
    return path


def resource_path(name: str) -> Path:
    """Resolve the two former resource names for existing external configs."""
    names = {
        "multimodal_split_manifest.json": "multimodal_split_manifest.json",
        "profiles.json": "memgallery_harness/profiles.json",
    }
    if name not in names:
        raise ValueError(f"Unknown legacy resource: {name}")
    return benchmark_path(names[name])


def profiles_path() -> Path:
    return benchmark_path("memgallery_harness/profiles.json")


def chunks_path(benchmark: str, variant: str = "dyadic") -> Path:
    """Locate dialogue chunks produced by the standard preparation commands."""
    if benchmark == "h2hmem" and variant not in {"dyadic", "multiparty"}:
        raise ValueError(f"Unknown H2HMEM variant: {variant}")
    names = {
        "memgallery": "chunks_no_profile.jsonl",
        "h2hmem": f"chunks_{variant}.jsonl",
        "wma": "chunks_lifelong.jsonl",
    }
    if benchmark not in names:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return PROJECT_ROOT / "data" / benchmark / "chunks" / names[benchmark]


def query_embedding_path(benchmark: str, model_name: str) -> Path:
    """Locate query vectors using the embedding model's preparation directory."""
    if benchmark not in DATASETS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    model_directory = re.sub(
        r"[^a-z0-9]+", "_", model_name.rstrip("/").split("/")[-1].lower()
    ).strip("_")
    if not model_directory:
        raise ValueError("An embedding model name is required to locate query vectors")
    path = PROJECT_ROOT / "data" / benchmark / "query_embeddings" / model_directory
    return path / "lifelong" if benchmark == "wma" else path


def validate_embedding_settings(
    parser: Any,
    model_name: Any,
    dimension: Any,
    *,
    model_flag: str = "--embedding-model",
    dimension_flag: str = "--embedding-dim",
) -> tuple[str, int]:
    """Validate embedding settings after explicit CLI overrides are applied."""
    model = model_name.strip() if isinstance(model_name, str) else ""
    if not model:
        parser.error(f"Set HIVE_EMBEDDING_MODEL or pass {model_flag}.")
    try:
        dim = int(dimension)
    except (TypeError, ValueError, OverflowError):
        parser.error(f"Set HIVE_EMBEDDING_DIM or pass {dimension_flag} with a positive integer.")
    if isinstance(dimension, bool) or dim <= 0 or isinstance(dimension, float) and dimension != dim:
        parser.error(f"HIVE_EMBEDDING_DIM/{dimension_flag} must be a positive integer.")
    return model, dim


def load_config_document(name: str, path: str | Path | None = None) -> dict[str, Any]:
    selected = project_path(path) if path is not None else CONFIG_ROOT / name
    return json.loads(selected.read_text(encoding="utf-8-sig"))


def raw_data_root() -> Path:
    value = os.getenv("HIVE_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT / "data/raw"


def output_root() -> Path:
    """Resolve the common artifact root using runtime configuration precedence."""
    return Path(load_runtime_config()["output_root"])


def dataset_root(source: str, workspace_root: str | Path | None = None) -> Path:
    name = ALIASES.get(source, source.lower())
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {source}")
    explicit = os.getenv(f"HIVE_{name.upper()}_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    base_root = raw_data_root() if workspace_root is None or os.getenv("HIVE_DATA_ROOT") else Path(workspace_root).expanduser().resolve()
    base = base_root / name
    candidates = {
        "memgallery": ("", "data", "benchmark/data"),
        "h2hmem": ("", "dataset"),
        "wma": ("", "lifelong", "WorldMemArena/lifelong", "WorldMemArena/WorldMemArena/lifelong"),
    }[name]
    marker = {"memgallery": "dialog", "h2hmem": "dyadic", "wma": "project"}[name]
    for suffix in candidates:
        candidate = base / suffix
        if (candidate / marker).exists():
            return candidate.resolve()
    return base.resolve()


def resolve_reference(value: str) -> str:
    if value.startswith("dataset://"):
        source, separator, relative = value[len("dataset://"):].partition("/")
        if not source or not separator:
            raise ValueError(f"Invalid dataset reference: {value}")
        base = dataset_root(source)
    elif value == "profile://memgallery":
        return str(profiles_path())
    elif value.startswith("benchmark://"):
        return str(benchmark_path(value[len("benchmark://"):]))
    elif value.startswith("resource://"):
        return str(resource_path(value[len("resource://"):]))
    elif value.startswith("config://"):
        base, relative = CONFIG_ROOT, value[len("config://"):]
    elif value.startswith("project://"):
        base, relative = PROJECT_ROOT, value[len("project://"):]
    else:
        return value
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"Reference escapes its root: {value}")
    return str(path)


def project_path(value: str | Path) -> Path:
    path = Path(resolve_reference(str(value))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def wma_framework_root() -> Path:
    """Locate the optional official WMA evaluator through an explicit root."""
    if os.getenv("HIVE_WMA_FRAMEWORK_ROOT"):
        return project_path(os.environ["HIVE_WMA_FRAMEWORK_ROOT"])
    data = dataset_root("wma")
    for candidate in (data, *data.parents):
        if (candidate / "eval_framework").is_dir():
            return candidate
    return raw_data_root() / "wma"


def source_reference(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        return value
    if path.is_relative_to(BENCHMARK_ROOT):
        return "benchmark://" + path.relative_to(BENCHMARK_ROOT).as_posix()
    if path.is_relative_to(CONFIG_ROOT):
        return "config://" + path.relative_to(CONFIG_ROOT).as_posix()
    roots = sorted(((name, dataset_root(name)) for name in DATASETS), key=lambda item: len(item[1].parts), reverse=True)
    for name, base in roots:
        if path.is_relative_to(base):
            return f"dataset://{name}/{path.relative_to(base).as_posix()}"
    if path.is_relative_to(PROJECT_ROOT):
        return "project://" + path.relative_to(PROJECT_ROOT).as_posix()
    return value


def portable_value(value: Any) -> Any:
    if isinstance(value, str):
        return source_reference(value)
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: portable_value(item) for key, item in value.items()}
    return value


def resolved_value(value: Any) -> Any:
    if isinstance(value, str):
        return resolve_reference(value)
    if isinstance(value, list):
        return [resolved_value(item) for item in value]
    if isinstance(value, dict):
        return {key: resolved_value(item) for key, item in value.items()}
    return value


ENV_FIELDS = {
    "HIVE_ANSWER_MODEL": "answer_model", "HIVE_ANSWER_BASE_URL": "answer_base_url",
    "HIVE_EXECUTOR_MODEL": "executor_model", "HIVE_EXECUTOR_BASE_URL": "executor_base_url",
    "HIVE_EMBEDDING_MODEL": "embedding_model", "HIVE_EMBEDDING_BASE_URL": "embedding_base_url",
    "HIVE_TOKENIZER": "retrieval_memory_tokenizer", "HIVE_OUTPUT_ROOT": "output_root",
    "HIVE_JUDGE_MODEL": "judge_model", "HIVE_JUDGE_BASE_URL": "judge_base_url",
    "HIVE_EMBEDDING_DIM": "embedding_dim", "HIVE_SAMPLE_CONCURRENCY": "sample_concurrency",
    "HIVE_EMBEDDING_REVISION": "embedding_revision",
}
PATH_FIELDS = {
    "data_dir", "query_embedding_dir", "query_cache", "profiles_file", "efficiency_config",
    "memgallery_chunks_file", "h2hmem_dyadic_chunks_file", "h2hmem_multiparty_chunks_file",
    "wma_lifelong_chunks_file", "memory_bank", "output_dir", "output_root", "split_manifest",
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def apply_environment(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for env, key in ENV_FIELDS.items():
        if env in os.environ:
            value = os.environ[env].strip()
            if not value:
                continue
            result[key] = int(value) if key == "sample_concurrency" else value
    return result


def api_key_for(role: str, explicit: str | None = None, *, env_name: str | None = None) -> str:
    """Explicit key > custom role variable > role key > common provider keys."""
    if explicit and explicit != "EMPTY":
        return explicit
    for name in (env_name, f"{role.upper()}_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        key = os.getenv(name) if name else None
        if key and key != "EMPTY":
            return key
    return "EMPTY"


def load_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    values = {
        "efficiency_config": str(CONFIG_ROOT / "defaults.json"),
        "profiles_file": str(profiles_path()),
        "memgallery_chunks_file": str(chunks_path("memgallery")),
        "h2hmem_dyadic_chunks_file": str(chunks_path("h2hmem", "dyadic")),
        "h2hmem_multiparty_chunks_file": str(chunks_path("h2hmem", "multiparty")),
        "wma_lifelong_chunks_file": str(chunks_path("wma")),
        **load_config_document("defaults.json")["runtime"],
    }
    selected = path or os.getenv("HIVE_CONFIG")
    if selected:
        document = load_config_document("defaults.json", selected)
        values = _merge(values, document.get("runtime", document))
    values = apply_environment(values)
    values.pop("_comment", None)
    for key in PATH_FIELDS:
        if isinstance(values.get(key), str) and values[key]:
            values[key] = str(project_path(values[key]))
    return values


def load_policy_config(path: str | Path | None = None, *, benchmark: str = "memgallery") -> dict[str, Any]:
    document = load_config_document("experiments.json", path)
    if "protocols" in document:
        defaults = load_config_document("defaults.json")
        values = _merge(defaults["evidence_policy"], {"ppo": defaults["ppo"]})
        values = _merge(values, document["protocols"]["ppo"][benchmark])
        if benchmark == "memgallery":
            values.setdefault("profiles_file", str(profiles_path()))
    else:
        values = document
    runtime = load_runtime_config()
    for field in ("embedding_model", "embedding_revision"):
        if runtime.get(field):
            values.setdefault(field, runtime[field])
    if "embedding_dim" in runtime:
        values.setdefault("policy", {})["embedding_dim"] = runtime["embedding_dim"]
    if "protocols" in document and runtime.get("embedding_model"):
        values.setdefault("query_cache", str(query_embedding_path(benchmark, runtime["embedding_model"])))
    values = resolved_value(values)
    for key in PATH_FIELDS:
        if isinstance(values.get(key), str) and values[key]:
            values[key] = str(project_path(values[key]))
    if values.get("evidence", {}).get("gvv_run_dir"):
        values["evidence"]["gvv_run_dir"] = str(project_path(values["evidence"]["gvv_run_dir"]))
    if output_root() != PROJECT_ROOT / "outputs":
        for key in ("memory_bank", "output_dir"):
            if not values.get(key):
                continue
            old = Path(values[key])
            if old.is_relative_to(PROJECT_ROOT / "outputs"):
                values[key] = str(output_root() / old.relative_to(PROJECT_ROOT / "outputs"))
        if values.get("evidence", {}).get("gvv_run_dir"):
            old = Path(values["evidence"]["gvv_run_dir"])
            if old.is_relative_to(PROJECT_ROOT / "outputs"):
                values["evidence"]["gvv_run_dir"] = str(output_root() / old.relative_to(PROJECT_ROOT / "outputs"))
    return values


# Run and dataset output layouts


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def from_path(cls, root: str | Path) -> "RunLayout":
        return cls(Path(root))

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / ".checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def build_manifest(self) -> Path:
        return self.root / "build_manifest.json"

    def dataset(self, name: str) -> "DatasetLayout":
        return DatasetLayout(self.datasets_dir / name)

    def checkpoint(self, name: str) -> Path:
        return self.checkpoints_dir / name


@dataclass(frozen=True)
class DatasetLayout:
    root: Path

    @property
    def vectors_dir(self) -> Path:
        return self.root / "vectors"

    @property
    def text_vectors(self) -> Path:
        return self.vectors_dir / "text.npy"

    @property
    def image_vectors(self) -> Path:
        return self.vectors_dir / "image.npy"

    @property
    def image_mask(self) -> Path:
        return self.vectors_dir / "image_mask.npy"

    @property
    def attribute_vectors(self) -> Path:
        return self.vectors_dir / "attributes.npy"

    @property
    def attributes(self) -> Path:
        return self.root / "attributes.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def build_stats(self) -> Path:
        return self.reports_dir / "build.json"

    @property
    def edges_manifest(self) -> Path:
        return self.reports_dir / "edges.json"

    @property
    def conflict_candidates(self) -> Path:
        return self.reports_dir / "conflicts.json"

    @property
    def build_trace(self) -> Path:
        return self.traces_dir / "build.jsonl"

    @property
    def edge_progress(self) -> Path:
        return self.traces_dir / "edges.jsonl"

    def existing_vector_path(self, filename: str, legacy_filename: str) -> Path:
        """Prefer the new vectors/ layout while accepting historical banks."""
        path = self.vectors_dir / filename
        return path if path.exists() else self.root / legacy_filename


# Atomic file writes and checksums

def write_text_atomic(path: str | Path, content: str) -> None:
    """Replace a text file atomically without leaving a shared ``.tmp`` file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


@contextmanager
def atomic_binary_writer(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def write_json_atomic(
    path: str | Path, payload: Any, *, indent: int = 2, trailing_newline: bool = False
) -> None:
    write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=indent) + ("\n" if trailing_newline else ""),
    )


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_data_uri(path: str | Path) -> str:
    """Upload image bytes without requiring a shared client/server filesystem."""
    source = Path(resolve_reference(str(path))).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Image not found: {source}")
    mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def file_manifest(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash existing files using resolved paths as stable manifest keys."""
    manifest: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            manifest[str(path)] = sha256_file(path)
    return dict(sorted(manifest.items()))
