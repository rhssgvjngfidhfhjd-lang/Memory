from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    NativeAnswerRequest,
    NativeAnswerResult,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from embedding.chunk_builder import Chunk


UPSTREAM_URL = "https://github.com/AIGeeksGroup/MMA.git"
UPSTREAM_COMMIT = "c0e1a127722edcfa4db5e71d03708cba53363000"
UPSTREAM_TREE = "4bb8b33535cb9d14b39277d953bcc80b5aec2e4c"
MAX_NATIVE_CANDIDATES_PER_PARTITION = 10
MAX_CONSECUTIVE_BAD_MEMORY_POINTS = 10
MAX_CONSECUTIVE_BAD_QA_POINTS = 10

_MMA_MEMORY_TABLES = (
    "episodic_memory",
    "semantic_memory",
    "procedural_memory",
    "resource_memory",
    "knowledge_vault",
)
_MMA_PRISTINE_INITIALIZATION_COUNTS = {
    "agents": 8,
    "messages": 8,
    "steps": 0,
}

_PROMPT_SHA256 = {
    "chat_agent": "f516291f6c8b06b30e5fde12a47c8a252ecfa58bae3a211f8e59c5a1c083916c",
    "core_memory_agent": "de19c09315dca5023bfd51a42342cc290722140156a06e0714648752048fef4a",
    "episodic_memory_agent": "ffb108177d2838ebe8b3fd0302b5911fca8be7e509df11e9f5afcd68dd9588ac",
    "knowledge_vault_agent": "4bdb5ee62eee663044bb0035ef0d215281e444d4803356251bcd62acce8060ee",
    "meta_memory_agent": "de048ac81c73f71933a747a684b6cacac8a070554ae2405fdeb8f54bfd5cedee",
    "procedural_memory_agent": "4887e4ef155074ca88c2b32821eb0671597594159f85d9b595e3430a03a0a05a",
    "resource_memory_agent": "685bc2c31813e8aaf2760c1be553618141984a2ba7214c9f0ea8ef841bb0d5d1",
    "semantic_memory_agent": "99fe1fcb67252b3693970c0f11f6d0e5187be8e6ee0f7363259b1205a0980c72",
}

_PARTITIONS: tuple[dict[str, Any], ...] = (
    {
        "manager": "episodic_memory_manager",
        "state": "episodic_memory_agent_state",
        "method": "list_episodic_memory",
        "partition": "episodic_memory",
        "search_field": "details",
        "embedding_field": "details_embedding",
    },
    {
        "manager": "semantic_memory_manager",
        "state": "semantic_memory_agent_state",
        "method": "list_semantic_items",
        "partition": "semantic_memory",
        "search_field": "details",
        "embedding_field": "details_embedding",
    },
    {
        "manager": "procedural_memory_manager",
        "state": "procedural_memory_agent_state",
        "method": "list_procedures",
        "partition": "procedural_memory",
        "search_field": "summary",
        "embedding_field": "summary_embedding",
    },
    {
        "manager": "resource_memory_manager",
        "state": "resource_memory_agent_state",
        "method": "list_resources",
        "partition": "resource_memory",
        "search_field": "summary",
        "embedding_field": "summary_embedding",
    },
    {
        "manager": "knowledge_vault_manager",
        "state": "knowledge_vault_agent_state",
        "method": "list_knowledge",
        "partition": "knowledge_vault_memory",
        "search_field": "caption",
        "embedding_field": "caption_embedding",
        "sensitivity": ["low", "medium"],
    },
)


class MMAAnswerContractError(RuntimeError):
    pass


class MMAConsecutiveBadPointError(RuntimeError):
    """Ten independent baseline requests failed without an intervening success."""


def is_mma_consecutive_bad_point_error(exc: BaseException) -> bool:
    """Recognize the limit signal locally or after worker serialization."""
    return (
        isinstance(exc, MMAConsecutiveBadPointError)
        or type(exc).__name__ == "MMAConsecutiveBadPointError"
        or "MMAConsecutiveBadPointError:" in str(exc)
    )


def quarantine_pristine_mma_state(state_dir: Path) -> Path | None:
    """Preserve and isolate a proven-empty pre-checkpoint MMA initialization.

    A prior process can be interrupted after MMA creates its agents and schema
    but before the first source session is ingested. Such a directory has no
    resumable checkpoint and cannot be reused safely. We never delete it: only
    the exact official pristine shape is moved aside, while any memory row,
    step, unexpected initialization count, WAL, or resume metadata fails shut.
    """
    state_dir = Path(state_dir)
    if not state_dir.is_dir():
        return None
    if (state_dir / ".offline_mma_resume.json").exists() or (
        state_dir / ".resume"
    ).exists():
        return None
    database = state_dir / "sqlite.db"
    if not database.is_file() or any(
        Path(str(database) + suffix).exists()
        for suffix in ("-wal", "-shm", "-journal")
    ):
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                return None
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            required = set(_MMA_MEMORY_TABLES) | set(
                _MMA_PRISTINE_INITIALIZATION_COUNTS
            )
            if not required.issubset(tables):
                return None
            for table in _MMA_MEMORY_TABLES:
                if connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]:
                    return None
            for table, expected in _MMA_PRISTINE_INITIALIZATION_COUNTS.items():
                actual = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
                if actual != expected:
                    return None
        finally:
            connection.close()
    except sqlite3.Error:
        return None

    candidate = state_dir.with_name(f"{state_dir.name}.orphaned_pristine")
    sequence = 1
    while candidate.exists():
        candidate = state_dir.with_name(
            f"{state_dir.name}.orphaned_pristine.{sequence}"
        )
        sequence += 1
    os.replace(state_dir, candidate)
    return candidate


class MMANativeAgentFailure(RuntimeError):
    """A root native failure already recorded on MMA's shared message queue."""

    def __init__(self, failures: list[dict[str, str]]) -> None:
        self.failures = tuple(dict(row) for row in failures)
        super().__init__(
            "MMA native memory agent failure was returned to the Meta Agent: "
            + json.dumps(failures, ensure_ascii=False)
        )


_UNSAFE_NATIVE_TOOL_ERROR_NAMES = frozenset(
    {
        # ORM lifecycle failures can occur after an earlier commit. Retrying the
        # memory tool is unsafe because append/merge operations may be repeated.
        "DetachedInstanceError",
        "ObjectDeletedError",
        "StaleDataError",
        "PendingRollbackError",
        "ResourceClosedError",
        "UnboundExecutionError",
        # SQL/driver errors do not expose whether a multi-commit MMA manager
        # failed before or after its first commit, so they are also fail-closed.
        "DBAPIError",
        "DatabaseError",
        "IntegrityError",
        "OperationalError",
        "InternalError",
        "ProgrammingError",
        "InterfaceError",
        "StatementError",
    }
)


class MMAOriginalAdapter(BaselineAdapter):
    """Strict bridge to MMA's original Agent/Manager implementation.

    The bridge controls only experiment I/O: benchmark-native observations are
    accumulated with MMA's original 20-message limit, residual observations are
    absorbed at session boundaries, and batch provenance is retained alongside
    isolated SQLite state and the final global Top-K handoff. Memory decisions
    and writes remain with MMA's original Meta/Memory Agents and Managers.
    """

    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        if baseline != "MMA":
            raise ValueError(f"MMAOriginalAdapter only supports MMA, got {baseline!r}")
        self.baseline = baseline
        self.source_root = Path(source_root).resolve()
        self.config = dict(config)
        self.backend: Any = None
        self.provenance = ProvenanceIndex()
        self._retrievals: dict[str, RetrievalResult] = {}
        self._ingested_chunks = 0
        self._pending_chunks: list[Chunk] = []
        self._pending_memory_fingerprints: dict[str, str] | None = None
        self._absorption_batches = 0
        self._absorption_batch_size = 20
        self._source_commit = ""
        self._source_tree = ""
        self._sample_id = ""
        self._state_dir: Path | None = None
        self._completed_session_ids: list[str] = []
        self._bad_memory_points: dict[str, dict[str, str]] = {}
        self._bad_retrieval_points: dict[str, list[str]] = {}
        self._consecutive_bad_memory_points = 0
        self._bad_qa_points: dict[str, str] = {}
        self._consecutive_bad_qa_points = 0

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self._verify_official_source()
        resume_payload = self._load_resume_checkpoint(sample_id, state_dir)
        if resume_payload is None and state_dir.exists():
            if bool(self.config.get("mma_resume_enabled", False)) and any(
                state_dir.iterdir()
            ):
                quarantined = quarantine_pristine_mma_state(state_dir)
                if quarantined is None:
                    raise RuntimeError(
                        "MMA resume checkpoint is missing or incompatible; refusing "
                        f"to delete existing state: {state_dir}"
                    )
                print(
                    "[mma-resume] preserved pristine pre-checkpoint state at "
                    f"{quarantined}",
                    file=sys.stderr,
                    flush=True,
                )
            elif state_dir.exists():
                shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if resume_payload is not None:
            self._restore_resume_database(state_dir, resume_payload)
        temp_dir = state_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        native_config = state_dir / "mma.config"
        native_config.write_text(
            "[defaults]\n"
            "preset = mma_chat\n"
            "persona = sam_pov\n"
            "human = basic\n\n"
            "[archival_storage]\n"
            f"type = sqlite\npath = {state_dir}\n\n"
            "[recall_storage]\n"
            f"type = sqlite\npath = {state_dir}\n\n"
            "[metadata_storage]\n"
            f"type = sqlite\npath = {state_dir}\n\n"
            "[version]\nmma_version = 0.1.0\n",
            encoding="utf-8",
        )
        os.environ["MMA_CONFIG_PATH"] = str(native_config)
        os.environ["MMA_DIR"] = str(state_dir)
        os.environ["MMA_IMAGES_DIR"] = str(state_dir / "images")
        os.environ["TMPDIR"] = str(temp_dir)
        os.environ["SQLITE_TMPDIR"] = str(temp_dir)
        os.environ["OPENAI_API_BASE"] = str(self.config["executor_base_url"])
        os.environ["OPENAI_BASE_URL"] = str(self.config["executor_base_url"])
        os.environ.setdefault(
            "OPENAI_API_KEY", str(self.config.get("executor_api_key") or "EMPTY")
        )

        if not bool(self.config.get("executor_native_tool_calls", False)):
            raise ValueError(
                "strict MMA requires executor_native_tool_calls=true; textual "
                "tool-call recovery is forbidden"
            )

        self._ensure_package_importable()
        self._install_embedding_dimension_compatibility()
        self._install_sqlalchemy_session_compatibility()
        constants = importlib.import_module("mma.agent.app_constants")
        wrapper_module = importlib.import_module("mma.agent.agent_wrapper")
        model = str(self.config["executor_model"])
        if model not in constants.OPENAI_MODELS:
            constants.OPENAI_MODELS = [*constants.OPENAI_MODELS, model]
            wrapper_module.OPENAI_MODELS = constants.OPENAI_MODELS

        self._install_configured_embedding_transport()
        self._install_local_image_transport()
        self._install_native_tool_call_validation()
        self._install_output_length_classification()
        self._install_strict_message_queue()
        llm, embedding = self._model_configs()
        agent_config = state_dir / "mma-agent.yaml"
        agent_config.write_text(
            json.dumps(
                {
                    "agent_name": f"mma_{sample_id}_{uuid.uuid4().hex[:8]}",
                    "model_name": model,
                }
            ),
            encoding="utf-8",
        )
        agent_module = importlib.import_module("mma.agent")
        with self._patched_defaults(llm, embedding):
            self.backend = agent_module.AgentWrapper(str(agent_config))
        self._apply_model_config(llm, embedding)

        accumulator = self.backend.temp_message_accumulator
        if int(accumulator.temporary_message_limit) != 20:
            raise RuntimeError(
                "MMA original TEMPORARY_MESSAGE_LIMIT changed unexpectedly: "
                f"{accumulator.temporary_message_limit}"
            )
        self._absorption_batch_size = int(
            self.config.get("mma_native_batch_size") or 20
        )
        if not 1 <= self._absorption_batch_size <= 20:
            raise ValueError(
                "mma_native_batch_size must be between 1 and MMA's original "
                f"limit 20, got {self._absorption_batch_size}"
            )
        accumulator.temporary_message_limit = self._absorption_batch_size
        self.provenance.clear()
        self._retrievals.clear()
        self._ingested_chunks = 0
        self._pending_chunks.clear()
        self._pending_memory_fingerprints = None
        self._bad_memory_points.clear()
        self._bad_retrieval_points.clear()
        self._consecutive_bad_memory_points = 0
        self._bad_qa_points.clear()
        self._consecutive_bad_qa_points = 0
        self._sample_id = str(sample_id)
        self._state_dir = state_dir
        if resume_payload is None:
            self._absorption_batches = 0
            self._completed_session_ids = []
        else:
            self._absorption_batches = int(
                resume_payload.get("absorption_batches") or 0
            )
            self._ingested_chunks = int(resume_payload.get("ingested_chunks") or 0)
            self._completed_session_ids = [
                str(value)
                for value in resume_payload.get("completed_session_ids") or []
            ]
            self.provenance.restore_rows(
                dict(resume_payload.get("provenance") or {})
            )

    def completed_session_ids(self) -> tuple[str, ...]:
        """Sessions durably restored from the last consistent SQLite snapshot."""
        return tuple(self._completed_session_ids)

    def filter_completed_session_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Drop only the contiguous session prefix restored by ``reset``."""
        completed = self.completed_session_ids()
        if not completed:
            return chunks
        ordered_sessions: list[str] = []
        for chunk in chunks:
            session_id = str(chunk.metadata.get("session_id") or "")
            if not session_id:
                raise RuntimeError("MMA resumable chunk is missing session_id")
            if not ordered_sessions or ordered_sessions[-1] != session_id:
                ordered_sessions.append(session_id)
        if tuple(ordered_sessions[: len(completed)]) != completed:
            raise RuntimeError(
                "MMA resume checkpoint is not a contiguous source-session prefix: "
                f"checkpoint={list(completed)}, source={ordered_sessions}"
            )
        completed_set = set(completed)
        pending = [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("session_id") or "") not in completed_set
        ]
        if any(
            str(chunk.metadata.get("session_id") or "") in completed_set
            for chunk in pending
        ):
            raise RuntimeError("MMA resume filtering left a completed session pending")
        return pending

    def _resume_manifest_path(self, state_dir: Path) -> Path:
        return state_dir / ".offline_mma_resume.json"

    def _resume_signature(self) -> str:
        return str(self.config.get("mma_resume_signature") or "")

    def _compatible_resume_signatures(self) -> set[str]:
        signatures = {self._resume_signature()}
        configured = self.config.get("mma_resume_compatible_signatures") or []
        if isinstance(configured, (list, tuple, set)):
            signatures.update(str(value) for value in configured if value)
        return signatures

    def _load_resume_checkpoint(
        self, sample_id: str, state_dir: Path
    ) -> dict[str, Any] | None:
        if not bool(self.config.get("mma_resume_enabled", False)):
            return None
        manifest_path = self._resume_manifest_path(state_dir)
        if not manifest_path.is_file():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("version") != 1
            or payload.get("sample_id") != str(sample_id)
            or payload.get("signature") not in self._compatible_resume_signatures()
        ):
            return None
        snapshot_name = str(payload.get("sqlite_snapshot") or "")
        if not snapshot_name or Path(snapshot_name).name != snapshot_name:
            return None
        snapshot_path = state_dir / ".resume" / snapshot_name
        if not snapshot_path.is_file():
            return None
        return payload

    def _restore_resume_database(
        self, state_dir: Path, payload: dict[str, Any]
    ) -> None:
        snapshot = state_dir / ".resume" / str(payload["sqlite_snapshot"])
        target = state_dir / "sqlite.db"
        temporary = state_dir / ".sqlite.db.restore"
        shutil.copyfile(snapshot, temporary)
        os.replace(temporary, target)
        for suffix in ("-wal", "-shm", "-journal"):
            stale = Path(str(target) + suffix)
            if stale.exists():
                stale.unlink()

    def _checkpoint_completed_session(self, session_id: str) -> None:
        if not bool(self.config.get("mma_resume_enabled", False)):
            return
        session_id = str(session_id)
        if session_id in self._completed_session_ids:
            return
        if self._state_dir is None or not self._sample_id:
            raise RuntimeError("MMA resume checkpoint requested before reset")
        database = self._state_dir / "sqlite.db"
        if not database.is_file():
            raise RuntimeError(f"MMA SQLite database is missing: {database}")
        resume_dir = self._state_dir / ".resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        sequence = len(self._completed_session_ids) + 1
        snapshot_name = f"sqlite.session-{sequence:06d}.db"
        snapshot = resume_dir / snapshot_name
        temporary = resume_dir / f".{snapshot_name}.tmp"
        if temporary.exists():
            temporary.unlink()
        source = sqlite3.connect(str(database))
        destination = sqlite3.connect(str(temporary))
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        os.replace(temporary, snapshot)
        completed = [*self._completed_session_ids, session_id]
        payload = {
            "version": 1,
            "sample_id": self._sample_id,
            "signature": self._resume_signature(),
            "sqlite_snapshot": snapshot_name,
            "completed_session_ids": completed,
            "ingested_chunks": self._ingested_chunks,
            "absorption_batches": self._absorption_batches,
            "provenance": self.provenance.export_rows(),
        }
        manifest = self._resume_manifest_path(self._state_dir)
        manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
        manifest_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest)
        self._completed_session_ids = completed
        for old_snapshot in resume_dir.glob("sqlite.session-*.db"):
            if old_snapshot.name != snapshot_name:
                old_snapshot.unlink()

    def _verify_official_source(self) -> None:
        repo = self.source_root.parent
        if not (repo / ".git").is_dir():
            raise RuntimeError(
                f"official MMA checkout is missing at {repo}; run "
                "Offline/scripts/prepare_mma_original.py"
            )
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        if head != UPSTREAM_COMMIT or tree != UPSTREAM_TREE or dirty:
            raise RuntimeError(
                "MMA source must be the clean official checkout "
                f"commit={UPSTREAM_COMMIT} tree={UPSTREAM_TREE}; "
                f"found commit={head} tree={tree} dirty={bool(dirty)}"
            )
        self._source_commit = head
        self._source_tree = tree

    def _ensure_package_importable(self) -> None:
        loaded = sys.modules.get("mma")
        if loaded is not None:
            loaded_path = Path(str(getattr(loaded, "__file__", ""))).resolve()
            expected = (self.source_root / "MMA" / "__init__.py").resolve()
            if loaded_path != expected:
                raise ImportError(
                    f"another mma package is already loaded from {loaded_path}; "
                    f"expected {expected}"
                )
            return
        package_dir = self.source_root / "MMA"
        init_path = package_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "mma", init_path, submodule_search_locations=[str(package_dir)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load MMA package from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["mma"] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop("mma", None)
            raise

    def _model_configs(self) -> tuple[Any, Any]:
        package = importlib.import_module("mma")
        model = str(self.config["executor_model"])
        endpoint = str(self.config["executor_base_url"])
        provider = _provider_name(endpoint)
        llm = package.LLMConfig(
            model=model,
            model_endpoint_type="openai",
            model_endpoint=endpoint,
            model_wrapper=None,
            handle=f"{provider}/{model}",
            context_window=int(
                self.config.get("mirix_context_window")
                or self.config.get("context_window")
                or 131072
            ),
            temperature=float(self.config.get("executor_temperature") or 0.0),
            max_tokens=int(
                self.config.get("executor_max_tokens")
                or self.config.get("num_predict")
                or 512
            ),
        )
        embedding = package.EmbeddingConfig(
            embedding_model=str(self.config["embedding_model"]),
            embedding_endpoint_type="openai",
            embedding_endpoint=str(self.config["embedding_base_url"]),
            embedding_dim=int(self.config["embedding_dim"]),
            embedding_chunk_size=300,
        )
        return llm, embedding

    @contextmanager
    def _patched_defaults(self, llm: Any, embedding: Any) -> Iterator[None]:
        package = importlib.import_module("mma")
        original_llm = package.LLMConfig.__dict__["default_config"]
        original_embedding = package.EmbeddingConfig.__dict__["default_config"]
        package.LLMConfig.default_config = classmethod(lambda cls, *args, **kwargs: llm)
        package.EmbeddingConfig.default_config = classmethod(
            lambda cls, *args, **kwargs: embedding
        )
        try:
            yield
        finally:
            package.LLMConfig.default_config = original_llm
            package.EmbeddingConfig.default_config = original_embedding

    def _apply_model_config(self, llm: Any, embedding: Any) -> None:
        self.backend.client.set_default_llm_config(llm)
        self.backend.client.set_default_embedding_config(embedding)
        state_by_id = {}
        for state in self.backend.client.list_agents():
            updated = self.backend.client.update_agent(
                agent_id=state.id,
                llm_config=llm,
                embedding_config=embedding,
            )
            state_by_id[str(state.id)] = updated
        for name, state in vars(self.backend.agent_states).items():
            state_id = str(getattr(state, "id", ""))
            if state_id in state_by_id:
                setattr(self.backend.agent_states, name, state_by_id[state_id])

    def _install_local_image_transport(self) -> None:
        module = importlib.import_module("mma.agent.temporary_message_accumulator")
        accumulator_class = module.TemporaryMessageAccumulator
        if getattr(accumulator_class, "_offline_mma_local_images", False):
            return
        original_build = accumulator_class._build_memory_message
        original_absorb = accumulator_class.absorb_content_into_memory
        prefix = "offline_database_image:"

        def build(accumulator: Any, ready: Any, voice: Any) -> Any:
            converted = []
            for timestamp, item in ready:
                item_copy = dict(item)
                references = []
                for reference in list(item_copy.get("image_uris") or []):
                    if isinstance(reference, str):
                        image_path = Path(reference).resolve()
                        if not image_path.is_file():
                            raise FileNotFoundError(
                                f"MMA observation image not found: {image_path}"
                            )
                        metadata = accumulator.client._save_image_from_file_uri(
                            str(image_path)
                        )
                        references.append(
                            SimpleNamespace(uri=prefix + str(metadata.id))
                        )
                    else:
                        references.append(reference)
                item_copy["image_uris"] = references
                converted.append((timestamp, item_copy))
            result = original_build(accumulator, converted, voice)
            for part in result:
                if part.get("type") != "google_cloud_file_uri":
                    continue
                uri = str(part.get("google_cloud_file_uri") or "")
                if uri.startswith(prefix):
                    part.clear()
                    part.update(
                        {"type": "database_image_id", "image_id": uri[len(prefix) :]}
                    )
            return result

        def absorb(accumulator: Any, agent_states: Any, ready_messages: Any = None) -> Any:
            if ready_messages is None and not accumulator.needs_upload:
                queued = copy.deepcopy(list(accumulator.temporary_messages))
                if any(item.get("image_uris") for _, item in queued):
                    ready_messages = queued
            return original_absorb(
                accumulator, agent_states, ready_messages=ready_messages
            )

        accumulator_class._build_memory_message = build
        accumulator_class.absorb_content_into_memory = absorb
        accumulator_class._offline_mma_local_images = True

    def _install_configured_embedding_transport(self) -> None:
        """Route the configured model name to the OpenAI-compatible endpoint.

        MMA's committed OpenAI branch constructs ``OpenAIEmbedding`` without
        its ``EmbeddingConfig.embedding_model`` and therefore silently sends
        ``text-embedding-ada-002``. This transport bridge does not change the
        embedding or retrieval algorithm; it makes the experiment's declared
        model/endpoint/dimension reach the provider.
        """
        module = importlib.import_module("mma.embeddings")
        if getattr(module, "_offline_mma_configured_embedding", False):
            return
        original = module.embedding_model

        def embedding_model(config: Any, user_id: Any = None) -> Any:
            if str(config.embedding_endpoint_type) != "openai":
                return original(config, user_id)
            from llama_index.embeddings.openai import OpenAIEmbedding

            additional = {"user_id": user_id} if user_id else {}
            return OpenAIEmbedding(
                model_name=str(config.embedding_model),
                api_base=str(config.embedding_endpoint),
                api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                dimensions=int(config.embedding_dim),
                additional_kwargs=additional,
            )

        module.embedding_model = embedding_model
        module._offline_mma_configured_embedding = True
        agent_module = importlib.import_module("mma.agent.agent")
        agent_module.embedding_model = embedding_model
        embedding_utils = importlib.import_module("mma.services.embedding_utils")
        embedding_utils.embedding_model = embedding_model
        embedding_utils._EMBED_MODEL_CACHE.clear()
        original_prepare = embedding_utils.prepare_embeddings_from_config

        def prepare_embeddings_from_config(*args: Any, **kwargs: Any) -> Any:
            return _prepare_embeddings_from_config_compat(
                original_prepare, *args, **kwargs
            )

        embedding_utils.prepare_embeddings_from_config = prepare_embeddings_from_config
        for name in (
            "agent_manager",
            "utils",
            "semantic_memory_manager",
            "episodic_memory_manager",
            "procedural_memory_manager",
            "resource_memory_manager",
            "knowledge_vault_manager",
        ):
            native = importlib.import_module(f"mma.services.{name}")
            if hasattr(native, "embedding_model"):
                native.embedding_model = embedding_model

    def _install_embedding_dimension_compatibility(self) -> None:
        """Use the configured native embedding width throughout fresh MMA state.

        Upstream defaults ``MAX_EMBEDDING_DIM`` to 4096 and pads newly-created
        memory records to that width even when the configured embedding model
        emits 2048 values. Some update paths assign fresh embeddings directly,
        bypassing the schema padding and later mixing 4096- and 2048-wide
        vectors. Install the experiment's declared dimension before MMA imports
        its ORM, schema, query, and SQLite-distance modules. Every formal sample
        starts with a fresh database, as required when changing this constant.
        """
        configured = int(self.config["embedding_dim"])
        constants = importlib.import_module("mma.constants")
        upstream_max = int(
            getattr(constants, "_offline_mma_upstream_max_embedding_dim", 4096)
        )
        if not 1 <= configured <= upstream_max:
            raise ValueError(
                "MMA embedding_dim must be between 1 and its upstream maximum "
                f"{upstream_max}, got {configured}"
            )
        installed = getattr(
            constants, "_offline_mma_storage_embedding_dim", None
        )
        if installed is not None and int(installed) != configured:
            raise RuntimeError(
                "MMA embedding dimension cannot change inside one worker: "
                f"installed={installed}, requested={configured}"
            )
        constants._offline_mma_upstream_max_embedding_dim = upstream_max
        constants._offline_mma_storage_embedding_dim = configured
        constants.MAX_EMBEDDING_DIM = configured

        # ``from mma.constants import MAX_EMBEDDING_DIM`` copies the value into
        # modules that may already be loaded. Most MMA memory modules load only
        # after this bridge, but synchronize any early imports defensively.
        for name, module in tuple(sys.modules.items()):
            if name == "mma" or name.startswith("mma."):
                if hasattr(module, "MAX_EMBEDDING_DIM"):
                    setattr(module, "MAX_EMBEDDING_DIM", configured)

    def _install_strict_message_queue(self) -> None:
        """Preserve queue ordering while surfacing native Agent failures."""
        module = importlib.import_module("mma.agent.message_queue")
        queue_class = module.MessageQueue
        if not getattr(queue_class, "_offline_mma_strict_errors", False):

            def send_message_in_queue(
                queue: Any,
                client: Any,
                agent_id: str,
                kwargs: dict[str, Any],
                agent_type: str = "chat",
            ) -> tuple[Any, str]:
                message_id = uuid.uuid4()
                with queue._message_queue_lock:
                    queue.message_queue[message_id] = {
                        "kwargs": kwargs,
                        "started": False,
                        "finished": False,
                        "type": agent_type,
                    }
                while not queue._check_if_earlier_requests_are_finished(message_id):
                    time.sleep(0.1)
                with queue._message_queue_lock:
                    queue.message_queue[message_id]["started"] = True
                try:
                    try:
                        response = client.send_message(
                            agent_id=agent_id,
                            role="user",
                            **queue.message_queue[message_id]["kwargs"],
                        )
                    except MMANativeAgentFailure:
                        # Another concurrent Memory Agent already recorded the
                        # root failure. Propagate it without recursively adding
                        # the same failure at every queue/Meta Agent boundary.
                        raise
                    except Exception as exc:
                        _record_native_agent_failure(queue, agent_type, exc)
                        raise
                    if response == "ERROR":
                        error = RuntimeError(
                            f"MMA {agent_type} returned the upstream ERROR sentinel"
                        )
                        _record_native_agent_failure(queue, agent_type, error)
                        raise error
                    return response, agent_type
                finally:
                    with queue._message_queue_lock:
                        row = queue.message_queue.get(message_id)
                        if row is not None:
                            row["finished"] = True
                            del queue.message_queue[message_id]

            queue_class.send_message_in_queue = send_message_in_queue
            queue_class._offline_mma_strict_errors = True

        # MMA's execute_tool_and_persist_state converts exceptions raised by a
        # nested Memory Agent into a successful-looking tool string. Check the
        # shared queue after each Meta Agent step so the outer chaining loop
        # cannot issue another round of paid calls after a nested failure.
        agent_module = importlib.import_module("mma.agent.agent")
        agent_class = agent_module.Agent
        if not getattr(agent_class, "_offline_mma_strict_tool_errors", False):
            original_execute_tool = agent_class.execute_tool_and_persist_state

            def execute_tool_and_persist_state(
                agent: Any, *args: Any, **kwargs: Any
            ) -> Any:
                response = original_execute_tool(agent, *args, **kwargs)
                if _is_unsafe_native_tool_error(response):
                    queue = getattr(
                        agent, "_offline_mma_active_message_queue", None
                    )
                    if queue is not None:
                        function_name = str(
                            args[0] if args else kwargs.get("function_name", "unknown")
                        )
                        _record_native_agent_failure(
                            queue,
                            str(getattr(agent.agent_state, "name", function_name)),
                            RuntimeError(str(response)),
                            tool_name=function_name,
                        )
                        # Abort inside the current inner_step. Waiting for the
                        # step wrapper would allow later tool calls from the
                        # same assistant response to run after an unsafe,
                        # potentially partially committed database failure.
                        _raise_native_agent_failures_from_queue(queue, clear=False)
                return response

            agent_class.execute_tool_and_persist_state = execute_tool_and_persist_state
            agent_class._offline_mma_strict_tool_errors = True

        if not getattr(agent_class, "_offline_mma_strict_nested_errors", False):
            original_inner_step = agent_class.inner_step

            def inner_step(agent: Any, *args: Any, **kwargs: Any) -> Any:
                queue = kwargs.get("message_queue")
                missing = object()
                previous = getattr(
                    agent, "_offline_mma_active_message_queue", missing
                )
                agent._offline_mma_active_message_queue = queue
                try:
                    response = original_inner_step(agent, *args, **kwargs)
                finally:
                    if previous is missing:
                        delattr(agent, "_offline_mma_active_message_queue")
                    else:
                        agent._offline_mma_active_message_queue = previous
                _raise_native_agent_failures_from_queue(queue, clear=False)
                return response

            agent_class.inner_step = inner_step
            agent_class._offline_mma_strict_nested_errors = True

    def _install_sqlalchemy_session_compatibility(self) -> None:
        """Give MMA's Managers sole ownership of their Session context.

        The pinned Managers enter the generator-based ``db_context()``, which
        yields a Session without entering the Session context, and then pass
        that same Session to ORM helpers which enter ``with db_session``.  The
        helper exit therefore closes/expunges a Session the Manager still owns.
        Wrap ``db_context`` so the Manager enters the Session first, then use a
        re-entrant Session to suppress only nested helper exits.  The outermost
        Manager exit retains SQLAlchemy's original close behavior. All original
        ORM helper bodies, commits, refreshes, and exception propagation remain.
        """
        module = importlib.import_module("mma.server.server")
        session_factory = module.SessionLocal
        session_class = session_factory.class_
        if not getattr(session_class, "_offline_mma_manager_owned_context", False):

            class MMAManagerOwnedSession(session_class):
                _offline_mma_manager_owned_context = True

                def __enter__(session: Any) -> Any:
                    depth = int(
                        getattr(session, "_offline_mma_context_depth", 0)
                    )
                    session._offline_mma_context_depth = depth + 1
                    try:
                        return super().__enter__()
                    except Exception:
                        session._offline_mma_context_depth = depth
                        raise

                def __exit__(
                    session: Any,
                    exception_type: Any,
                    exception: Any,
                    traceback: Any,
                ) -> Any:
                    depth = int(
                        getattr(session, "_offline_mma_context_depth", 0)
                    )
                    if depth < 1:
                        raise RuntimeError("MMA Session context depth underflow")
                    remaining = depth - 1
                    session._offline_mma_context_depth = remaining
                    if remaining:
                        # Returning False preserves exception propagation while
                        # leaving cleanup to the Manager's outer context.
                        return False
                    try:
                        return super().__exit__(
                            exception_type, exception, traceback
                        )
                    finally:
                        session._offline_mma_context_depth = 0

            MMAManagerOwnedSession.__name__ = "MMAManagerOwnedSession"
            MMAManagerOwnedSession.__qualname__ = "MMAManagerOwnedSession"
            session_factory.class_ = MMAManagerOwnedSession

        manager_context = getattr(module, "db_context", None)
        if manager_context is None:
            raise RuntimeError("MMA server module is missing db_context")
        if not getattr(
            manager_context, "_offline_mma_manager_owned_context", False
        ):

            @contextmanager
            def manager_owned_db_context() -> Iterator[Any]:
                with session_factory() as session:
                    yield session

            manager_owned_db_context._offline_mma_manager_owned_context = True
            module.db_context = manager_owned_db_context

        # Restore SQLAlchemy's official/default commit-expiration semantics. An
        # active Manager-owned Session can refresh expired fields normally.
        session_factory.configure(expire_on_commit=True)
        if getattr(session_factory, "kw", {}).get("expire_on_commit") is not True:
            raise RuntimeError(
                "failed to restore MMA SessionLocal expire_on_commit=True"
            )

    def _install_native_tool_call_validation(self) -> None:
        """Reject provider text that only resembles a native tool call.

        Some OpenAI-compatible vLLM parsers leave ``<tool_call>`` JSON in the
        assistant text while returning an empty ``tool_calls`` field. Parsing
        or repairing that text in the adapter would change MMA's behavior, so
        strict reproduction treats it as a provider incompatibility instead.
        """
        module = importlib.import_module("mma.llm_api.openai_client")
        client_class = module.OpenAIClient
        if getattr(client_class, "_offline_mma_native_tool_validation", False):
            return
        original = client_class.convert_response_to_chat_completion

        def convert_response_to_chat_completion(
            client: Any, response_data: dict[str, Any], input_messages: list[Any]
        ) -> Any:
            response = original(client, response_data, input_messages)
            _reject_unstructured_native_tool_calls(response)
            return response

        client_class.convert_response_to_chat_completion = (
            convert_response_to_chat_completion
        )
        client_class._offline_mma_native_tool_validation = True

    def _install_output_length_classification(self) -> None:
        """Do not send output-budget exhaustion through context summarization."""
        module = importlib.import_module("mma.agent.agent")
        if getattr(module, "_offline_mma_output_length_classification", False):
            return
        original = module.is_context_overflow_error

        def is_context_overflow_error(exception: Exception) -> bool:
            if _is_output_length_exhaustion(exception):
                return False
            return bool(original(exception))

        module.is_context_overflow_error = is_context_overflow_error
        module._offline_mma_output_length_classification = True

    def ingest(self, chunk: Chunk) -> None:
        for raw_path in chunk.images:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"MMA observation image not found: {path}")
        accumulator = self.backend.temp_message_accumulator
        queued_before = len(accumulator.temporary_messages)
        if queued_before != len(self._pending_chunks):
            raise RuntimeError(
                "MMA accumulator/provenance queue mismatch before ingest: "
                f"native={queued_before}, provenance={len(self._pending_chunks)}"
            )
        if not self._pending_chunks:
            self._pending_memory_fingerprints = self._memory_fingerprints()
        self._pending_chunks.append(chunk)
        kwargs: dict[str, Any] = {
            "message": chunk.text,
            "image_uris": [str(Path(path).resolve()) for path in chunk.images] or None,
            "memorizing": True,
            "force_absorb_content": False,
            "delete_after_upload": False,
            "async_upload": False,
        }
        timestamp = str(chunk.metadata.get("timestamp") or "").strip()
        if timestamp:
            kwargs["specific_timestamps"] = [_mma_observation_timestamp(timestamp)]
        try:
            self.backend.send_message(**kwargs)
            self._raise_native_agent_failures()
        except Exception:
            # Keep the adapter-side batch aligned with whatever remains in the
            # native accumulator. The harness will stop on the original error.
            native_count = len(accumulator.temporary_messages)
            self._pending_chunks = self._pending_chunks[:native_count]
            if not self._pending_chunks:
                self._pending_memory_fingerprints = None
            raise
        queued_after = len(accumulator.temporary_messages)
        if queued_after not in {0, len(self._pending_chunks)}:
            raise RuntimeError(
                "MMA accumulator consumed only part of a provenance batch: "
                f"native={queued_after}, provenance={len(self._pending_chunks)}"
            )
        if queued_after == 0:
            self._register_absorbed_batch()
        self._ingested_chunks += 1

    def end_session(self, session_id: str) -> None:
        accumulator = self.backend.temp_message_accumulator
        queued_count = len(accumulator.temporary_messages)
        if queued_count != len(self._pending_chunks):
            raise RuntimeError(
                "MMA accumulator/provenance queue mismatch at session boundary: "
                f"native={queued_count}, provenance={len(self._pending_chunks)}"
            )
        pending_sessions = {
            str(chunk.metadata.get("session_id") or "")
            for chunk in self._pending_chunks
            if str(chunk.metadata.get("session_id") or "")
        }
        if pending_sessions and pending_sessions != {str(session_id)}:
            raise RuntimeError(
                "MMA session-tail absorption received mixed or mismatched sessions: "
                f"expected={session_id!r}, pending={sorted(pending_sessions)}"
            )
        if not self._pending_chunks:
            self._checkpoint_completed_session(session_id)
            return
        accumulator.absorb_content_into_memory(self.backend.agent_states)
        self._raise_native_agent_failures()
        if accumulator.temporary_messages:
            raise RuntimeError(
                "MMA session-tail absorption left observations in the accumulator"
            )
        self._register_absorbed_batch()
        self._checkpoint_completed_session(session_id)

    def _register_absorbed_batch(self) -> None:
        if not self._pending_chunks or self._pending_memory_fingerprints is None:
            raise RuntimeError("MMA absorbed content without a tracked provenance batch")
        before = self._pending_memory_fingerprints
        chunks = list(self._pending_chunks)
        current = self._memory_rows()
        for row in current:
            if before.get(row["memory_id"]) == row["fingerprint"]:
                continue
            for chunk in chunks:
                self.provenance.register(row["memory_id"], chunk)
        self._pending_chunks.clear()
        self._pending_memory_fingerprints = None
        self._absorption_batches += 1

    def _raise_native_agent_failures(self) -> None:
        queue = getattr(self.backend, "message_queue", None)
        _raise_native_agent_failures_from_queue(queue, clear=True)

    def _memory_fingerprints(self) -> dict[str, str]:
        return {row["memory_id"]: row["fingerprint"] for row in self._memory_rows()}

    def _memory_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        server = self.backend.client.server
        for spec in _PARTITIONS:
            manager = getattr(server, spec["manager"])
            state = getattr(self.backend.agent_states, spec["state"])
            values = getattr(manager, spec["method"])(agent_state=state, limit=None) or []
            for value in values:
                raw_id = str(getattr(value, "id", "")).strip()
                if not raw_id:
                    raise RuntimeError(
                        f"MMA {spec['partition']} returned a row without an id"
                    )
                structured = _structured_memory(value, spec["partition"])
                encoded = json.dumps(
                    structured, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                rows.append(
                    {
                        "memory_id": f"{spec['manager']}:{raw_id}",
                        "partition": spec["partition"],
                        "text": encoded,
                        "structured": structured,
                        "raw": value,
                        "fingerprint": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    }
                )
        return rows

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self._pending_chunks:
            raise RuntimeError(
                "MMA retrieval requested with unabsorbed observations; "
                "the harness must end the current session/checkpoint first"
            )
        if request.top_k < 0 or request.top_k > 7:
            raise ValueError(f"MMA final handoff requires 0 <= top_k <= 7, got {request.top_k}")
        if request.query_id in self._retrievals:
            raise ValueError(f"duplicate MMA retrieval query_id: {request.query_id}")
        query_embedding = self._query_embedding(request)
        padded_embedding = [*query_embedding]
        if len(padded_embedding) > 4096:
            raise ValueError(
                f"MMA query embedding exceeds MAX_EMBEDDING_DIM: {len(padded_embedding)}"
            )
        padded_embedding.extend([0.0] * (4096 - len(padded_embedding)))
        timezone_str = self.backend.client.server.user_manager.get_user_by_id(
            self.backend.client.user.id
        ).timezone

        candidates: list[tuple[float, str, RetrievedMemory]] = []
        candidate_counts: dict[str, int] = {}
        skipped_bad_points: list[dict[str, Any]] = []
        server = self.backend.client.server
        for spec in _PARTITIONS:
            manager = getattr(server, spec["manager"])
            state = getattr(self.backend.agent_states, spec["state"])
            kwargs: dict[str, Any] = {
                "agent_state": state,
                "embedded_text": padded_embedding,
                "query": request.text,
                "search_field": spec["search_field"],
                "search_method": "embedding",
                "limit": MAX_NATIVE_CANDIDATES_PER_PARTITION,
                "timezone_str": timezone_str,
            }
            if "sensitivity" in spec:
                kwargs["sensitivity"] = list(spec["sensitivity"])
            hits = getattr(manager, spec["method"])(**kwargs) or []
            candidate_counts[spec["partition"]] = len(hits)
            for hit_index, hit in enumerate(hits):
                raw_id = str(getattr(hit, "id", "")).strip()
                if not raw_id:
                    bad_id = (
                        f"{spec['manager']}:missing-id:"
                        f"{request.query_id}:{hit_index}"
                    )
                    self._record_bad_memory_row(
                        bad_id,
                        partition=str(spec["partition"]),
                        reason="retrieval row has no id",
                    )
                    skipped_bad_points.append(
                        {
                            "memory_id": bad_id,
                            "partition": spec["partition"],
                            "reason": "missing_id",
                        }
                    )
                    continue
                memory_id = f"{spec['manager']}:{raw_id}"
                if not self.provenance.visible(memory_id, request.visible_session_ids):
                    continue
                score = _cosine_similarity(
                    query_embedding, getattr(hit, spec["embedding_field"], None)
                )
                if score is None:
                    reason = f"invalid {spec['embedding_field']}"
                    self._record_bad_memory_row(
                        memory_id,
                        partition=str(spec["partition"]),
                        reason=reason,
                    )
                    skipped_bad_points.append(
                        {
                            "memory_id": memory_id,
                            "partition": spec["partition"],
                            "reason": reason,
                        }
                    )
                    continue
                source = self.provenance.get(memory_id)
                structured = _structured_memory(hit, spec["partition"])
                item = RetrievedMemory(
                    memory_id=memory_id,
                    text=json.dumps(
                        structured,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    score=score,
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    metadata={
                        "partition": spec["partition"],
                        "structured_memory": structured,
                        "via": "mma_original_manager_embedding",
                    },
                )
                candidates.append((score, memory_id, item))

        if skipped_bad_points:
            consecutive_bad_retrievals = self._record_bad_retrieval_point(
                request.query_id, skipped_bad_points
            )
        else:
            self._consecutive_bad_memory_points = 0
            consecutive_bad_retrievals = 0

        candidates.sort(key=lambda row: (-row[0], row[1]))
        selected: list[RetrievedMemory] = []
        selected_ids: set[str] = set()
        for _, memory_id, item in candidates:
            if memory_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(memory_id)
            if len(selected) >= request.top_k:
                break
        result = RetrievalResult(
            items=selected,
            trace={
                "baseline": "MMA",
                "via": "mma_original_managers_global_top7_handoff",
                "ranking": "cosine_over_original_manager_embedding_candidates",
                "candidate_limit_per_partition": MAX_NATIVE_CANDIDATES_PER_PARTITION,
                "candidate_counts": candidate_counts,
                "skipped_bad_memory_points": skipped_bad_points,
                "skipped_bad_memory_point_count": len(skipped_bad_points),
                "unique_bad_memory_point_count": len(self._bad_memory_points),
                "unique_bad_retrieval_point_count": len(
                    self._bad_retrieval_points
                ),
                "consecutive_bad_retrieval_points": consecutive_bad_retrievals,
                "consecutive_bad_memory_point_limit": (
                    MAX_CONSECUTIVE_BAD_MEMORY_POINTS
                ),
                "requested_top_k": request.top_k,
                "returned_memories": len(selected),
                "category": request.category,
                "qa_prompt_applied": False,
                "structured_evidence": True,
                "provenance_required": True,
            },
        )
        self._retrievals[request.query_id] = result
        return result

    def _record_bad_memory_row(
        self,
        memory_id: str,
        *,
        partition: str,
        reason: str,
    ) -> None:
        """Audit one malformed native row without treating rows as requests."""
        bad_points = getattr(self, "_bad_memory_points", None)
        if bad_points is None:
            bad_points = {}
            self._bad_memory_points = bad_points
        if memory_id in bad_points:
            return
        bad_points[memory_id] = {
            "partition": str(partition),
            "reason": str(reason),
        }
        print(
            "[mma-bad-memory-row] "
            + json.dumps(
                {
                    "sample_id": str(getattr(self, "_sample_id", "")),
                    "memory_id": memory_id,
                    "partition": partition,
                    "reason": reason,
                    "action": "skip_without_repair_or_rank_fallback",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def _record_bad_retrieval_point(
        self,
        query_id: str,
        skipped_rows: list[dict[str, Any]],
    ) -> int:
        """Count one malformed retrieval request, irrespective of row count."""
        bad_retrievals = getattr(self, "_bad_retrieval_points", None)
        if bad_retrievals is None:
            bad_retrievals = {}
            self._bad_retrieval_points = bad_retrievals
        if query_id in bad_retrievals:
            return int(getattr(self, "_consecutive_bad_memory_points", 0))
        memory_ids = [str(row.get("memory_id") or "") for row in skipped_rows]
        bad_retrievals[query_id] = memory_ids
        consecutive = int(
            getattr(self, "_consecutive_bad_memory_points", 0)
        ) + 1
        self._consecutive_bad_memory_points = consecutive
        print(
            "[mma-bad-point] "
            + json.dumps(
                {
                    "sample_id": str(getattr(self, "_sample_id", "")),
                    "query_id": query_id,
                    "malformed_memory_rows": len(skipped_rows),
                    "memory_ids": memory_ids,
                    "consecutive": consecutive,
                    "limit": MAX_CONSECUTIVE_BAD_MEMORY_POINTS,
                    "action": "skip_rows_and_continue_without_repair",
                    "counting_unit": "independent_retrieval_request",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if consecutive >= MAX_CONSECUTIVE_BAD_MEMORY_POINTS:
            raise MMAConsecutiveBadPointError(
                "MMA produced "
                f"{consecutive} consecutive malformed retrieval requests; "
                "stopping this sample without repairing baseline output"
            )
        return consecutive

    def _query_embedding(self, request: RetrievalRequest) -> list[float]:
        if request.query_vector is not None:
            values = [float(value) for value in request.query_vector]
        else:
            module = importlib.import_module("mma.embeddings")
            state = self.backend.agent_states.agent_state
            values = [
                float(value)
                for value in module.embedding_model(
                    state.embedding_config
                ).get_text_embedding(request.text)
            ]
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError("MMA query embedding is empty or non-finite")
        return values

    def answer_with_memory(self, request: NativeAnswerRequest) -> NativeAnswerResult:
        selected = self._retrievals.pop(request.query_id, None)
        if selected is None:
            raise KeyError(
                f"MMA answer has no preceding retrieval: {request.query_id}"
            )
        if request.top_k > 7 or len(request.retrieval.items) > request.top_k:
            raise ValueError("MMA final answer received more than the allowed Top-7")
        expected_ids = [item.memory_id for item in selected.items]
        actual_ids = [item.memory_id for item in request.retrieval.items]
        if actual_ids != expected_ids:
            raise ValueError(
                "MMA final answer evidence differs from the retrieval handoff: "
                f"expected={expected_ids}, actual={actual_ids}"
            )

        client = self.backend.client
        chat_state = self.backend.agent_states.agent_state
        saved = _capture_chat_state(client, chat_state.id)
        memory_context = _selected_memory_context(selected.items, request.messages)
        max_attempts = int(self.config.get("retries") or 0) + 1
        cumulative_usage: dict[str, Any] = {}
        last_error: Exception | None = None

        def continue_after_bad_point(
            exc: Exception, *, attempts: int
        ) -> NativeAnswerResult:
            consecutive = self._record_bad_qa_point(request.query_id, exc)
            if consecutive >= MAX_CONSECUTIVE_BAD_QA_POINTS:
                raise MMAConsecutiveBadPointError(
                    "MMA produced "
                    f"{consecutive} consecutive failed QA points; "
                    "stopping this sample without repairing baseline output"
                ) from exc
            return NativeAnswerResult(
                text="",
                attempts=attempts,
                failed_attempts=attempts,
                image_count=0,
                usage=cumulative_usage,
                trace={
                    "via": "mma_original_chat_agent",
                    "global_top_k": request.top_k,
                    "retrieved_count": len(selected.items),
                    "retrieved_memory_ids": expected_ids,
                    "bad_qa_point": True,
                    "bad_qa_reason": str(exc),
                    "consecutive_bad_qa_points": consecutive,
                    "consecutive_bad_qa_point_limit": (
                        MAX_CONSECUTIVE_BAD_QA_POINTS
                    ),
                    "action": "record_empty_answer_and_continue",
                    "format_retries": max(0, attempts - 1),
                },
                retrieval=selected,
            )

        for attempt in range(1, max_attempts + 1):
            response = None
            try:
                with _fixed_chat_memory_prompt(memory_context):
                    response, attached_images = _send_native_benchmark_messages(
                        client,
                        chat_state.id,
                        request.messages,
                        request.query_image,
                        selected.items,
                    )
                cumulative_usage = _sum_usage(
                    cumulative_usage,
                    _model_dump(getattr(response, "usage", None)) or {},
                )
                text = _require_answer_block(_extract_chat_answer(response))
                if not text or text == "ERROR":
                    raise MMAAnswerContractError(
                        "MMA Chat Agent did not return a final answer"
                    )
                self._consecutive_bad_qa_points = 0
                return NativeAnswerResult(
                    text=text,
                    attempts=attempt,
                    failed_attempts=attempt - 1,
                    image_count=attached_images,
                    usage=cumulative_usage,
                    trace={
                        "via": "mma_original_chat_agent",
                        "global_top_k": request.top_k,
                        "retrieved_count": len(selected.items),
                        "retrieved_memory_ids": expected_ids,
                        "structured_evidence": True,
                        "provenance_preserved": True,
                        "qa_prompt_applied_stage": "final_answer_only",
                        "original_chat_system_prompt": True,
                        "memory_tool_scope": "original_unmodified_chat_tools",
                        "chat_tool_calls": _response_tool_names(response),
                        "format_retries": attempt - 1,
                    },
                    retrieval=selected,
                )
            except MMAAnswerContractError as exc:
                last_error = exc
                if attempt == max_attempts:
                    return continue_after_bad_point(exc, attempts=attempt)
            except Exception as exc:
                if not _is_output_length_exhaustion(exc):
                    raise
                # The upstream client has already consumed its configured
                # retries when it emits this terminal length exception. Count
                # the whole exhausted request as one bad QA point, not three.
                last_error = exc
                return continue_after_bad_point(exc, attempts=max_attempts)
            finally:
                _restore_chat_state(client, chat_state.id, saved)
        raise RuntimeError("MMA Chat Agent retry loop ended unexpectedly") from last_error

    def _record_bad_qa_point(self, query_id: str, exc: Exception) -> int:
        bad_points = getattr(self, "_bad_qa_points", None)
        if bad_points is None:
            bad_points = {}
            self._bad_qa_points = bad_points
        if query_id in bad_points:
            return int(getattr(self, "_consecutive_bad_qa_points", 0))
        bad_points[query_id] = f"{type(exc).__name__}: {exc}"
        consecutive = int(getattr(self, "_consecutive_bad_qa_points", 0)) + 1
        self._consecutive_bad_qa_points = consecutive
        print(
            "[mma-qa-bad-point] "
            + json.dumps(
                {
                    "sample_id": str(getattr(self, "_sample_id", "")),
                    "query_id": query_id,
                    "reason": str(exc),
                    "consecutive": consecutive,
                    "limit": MAX_CONSECUTIVE_BAD_QA_POINTS,
                    "action": "record_empty_answer_and_continue",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return consecutive

    def snapshot(self) -> list[MemoryRecord]:
        if self._pending_chunks:
            raise RuntimeError("MMA snapshot requested with unabsorbed observations")
        records: list[MemoryRecord] = []
        for row in self._memory_rows():
            source = self.provenance.get(row["memory_id"])
            records.append(
                MemoryRecord(
                    memory_id=row["memory_id"],
                    text=row["text"],
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    backend_type=f"mma_{row['partition']}",
                    metadata={
                        "partition": row["partition"],
                        "structured_memory": row["structured"],
                    },
                )
            )
        return records

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "mma",
            "baseline": "MMA",
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "native_memory_build": True,
            "native_chat_answer": True,
            "native_meta_memory_agent": True,
            "native_memory_agents": True,
            "native_memory_managers": True,
            "native_absorption": "original_accumulator_with_session_tail_absorption",
            "native_absorption_batch": self._absorption_batch_size,
            "session_tail_absorption": True,
            "batch_provenance": "all_source_observations_in_absorption_batch",
            "completed_absorption_batches": self._absorption_batches,
            "output_length_exhaustion_triggers_summarizer": False,
            "sqlalchemy_expire_on_commit": True,
            "sqlalchemy_session_context": "reentrant_manager_owned",
            "mma_upstream_max_embedding_dim": 4096,
            "mma_storage_embedding_dim": int(self.config["embedding_dim"]),
            "unsafe_database_tool_errors": "fail_closed_first_root_cause",
            "recoverable_tool_validation_errors": "native_agent_retry",
            "original_temporary_message_limit": 20,
            "fixed_global_top_k": 7,
            "candidate_limit_per_partition": MAX_NATIVE_CANDIDATES_PER_PARTITION,
            "bad_memory_point_policy": "skip_without_repair_or_rank_fallback",
            "bad_memory_point_counting_unit": "independent_retrieval_request",
            "consecutive_bad_memory_point_limit": (
                MAX_CONSECUTIVE_BAD_MEMORY_POINTS
            ),
            "unique_bad_memory_point_count": len(self._bad_memory_points),
            "unique_bad_retrieval_point_count": len(self._bad_retrieval_points),
            "bad_qa_point_policy": "record_empty_answer_and_continue",
            "consecutive_bad_qa_point_limit": MAX_CONSECUTIVE_BAD_QA_POINTS,
            "unique_bad_qa_point_count": len(self._bad_qa_points),
            "qa_prompt_stage": "final_answer_only",
            "semantic_fallback_on_error": False,
            "direct_insert": False,
            "source_commit": self._source_commit or UPSTREAM_COMMIT,
            "source_tree": self._source_tree or UPSTREAM_TREE,
        }

    def close(self) -> None:
        self._retrievals.clear()
        self._pending_chunks.clear()
        self._pending_memory_fingerprints = None
        self._bad_memory_points.clear()
        self._bad_retrieval_points.clear()
        self._consecutive_bad_memory_points = 0
        self._bad_qa_points.clear()
        self._consecutive_bad_qa_points = 0
        self.backend = None
        self._state_dir = None


def mma_conformance_manifest(*, answer_prompt_sha256: str) -> dict[str, Any]:
    prompt_root = (
        Path(__file__).resolve().parents[4]
        / ".upstream"
        / "mma-c0e1a12"
        / "MMA"
        / "MMA"
        / "prompts"
        / "system"
    )
    actual = {
        name: hashlib.sha256(
            (prompt_root / f"{name}.txt").read_bytes()
        ).hexdigest()
        for name in _PROMPT_SHA256
    }
    return {
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_tree": UPSTREAM_TREE,
        "internal_prompt_sha256": {
            name: {"expected": expected, "actual": actual[name]}
            for name, expected in _PROMPT_SHA256.items()
        },
        "internal_prompt_hash_method": "SHA256 of exact official system prompt bytes",
        "answer_prompt_sha256": answer_prompt_sha256,
        "approved_experiment_bridges": [
            "benchmark-native observations accumulated by MMA's original 20-message limit",
            "session-boundary absorption of the original accumulator tail",
            "isolated SQLite path and model endpoint configuration",
            "local image to MMA database-image transport",
            "re-entrant SQLAlchemy Session contexts preserving Manager ownership",
            "fresh-database MMA storage dimension aligned to the configured embedding model",
            "first-root fail-closed handling for unsafe native database tool errors",
            "malformed native memory points logged and skipped without repair or rank fallback",
            "memory-id provenance sidecar",
            "global structured Top-7 handoff to original Chat Agent",
            "benchmark QA messages supplied only at final answer",
        ],
        "forbidden_paths": {
            "semantic_fallback": False,
            "direct_insert": False,
            "textual_tool_call_repair": False,
            "empty_message_flush": False,
            "frozen_retrieval": False,
        },
    }


def _provider_name(endpoint: str) -> str:
    return (
        "vllm"
        if str(endpoint).startswith(("http://127.0.0.1", "http://localhost"))
        else "openai"
    )


def _is_output_length_exhaustion(exception: Exception) -> bool:
    return (
        "maximum context length exceeded or generated content is too long"
        in str(exception)
    )


def _reject_unstructured_native_tool_calls(response: Any) -> None:
    """Fail when a provider emits tool-call markup as ordinary assistant text."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    for choice in choices or []:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            continue
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(message, dict):
            content = message.get("content", content)
            tool_calls = message.get("tool_calls", tool_calls)
        if "<tool_call>" in str(content or "") and not tool_calls:
            raise RuntimeError(
                "MMA provider returned <tool_call> markup as assistant text "
                "without native tool_calls; textual parsing and repair are forbidden"
            )


def _structured_memory(value: Any, partition: str) -> dict[str, Any]:
    raw = _model_dump(value)
    if raw is None:
        raw = {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key, None))
        }
    structured: dict[str, Any] = {"memory_type": partition}
    embedding_fields: dict[str, int] = {}
    for key, item in raw.items():
        if key.endswith("_embedding"):
            vector = _vector_values(item)
            if vector:
                embedding_fields[key] = len(vector)
            continue
        if key == "embedding_config":
            continue
        structured[key] = _json_value(item)
    structured["embedding_fields"] = embedding_fields
    return structured


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    dumped = _model_dump(value)
    return _json_value(dumped) if dumped is not None else str(value)


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return None


def _sum_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = result.get(key, 0) + value
        elif key not in result:
            result[key] = value
    return result


def _vector_values(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _cosine_similarity(left: Any, right: Any) -> float | None:
    left_values = _vector_values(left)
    right_values = _vector_values(right)
    if not left_values or not right_values:
        return None
    size = min(len(left_values), len(right_values))
    left_values = left_values[:size]
    right_values = right_values[:size]
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left_values, right_values)) / (
        left_norm * right_norm
    )


def _selected_memory_context(
    items: list[RetrievedMemory], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {
        "episodic_memory": [],
        "semantic_memory": [],
        "procedural_memory": [],
        "resource_memory": [],
        "knowledge_vault_memory": [],
    }
    for item in items:
        partition = str(item.metadata.get("partition") or "")
        if partition not in grouped:
            raise ValueError(f"unknown MMA memory partition in Top-7: {partition!r}")
        grouped[partition].append(item.text)
    question = "\n".join(
        str(row.get("content") or "")
        for row in messages
        if str(row.get("role") or "") == "user"
    ).strip()
    knowledge = grouped["knowledge_vault_memory"]
    semantic = [*grouped["semantic_memory"], *knowledge]
    return {
        "key_words": question,
        "episodic": ["", "\n".join(grouped["episodic_memory"])],
        "semantic": "\n".join(semantic),
        "procedural": "\n".join(grouped["procedural_memory"]),
        "resource": "\n".join(grouped["resource_memory"]),
        "knowledge_vault": "\n".join(knowledge),
    }


def _prepare_embeddings_from_config_compat(
    original: Any, *args: Any, **kwargs: Any
) -> Any:
    """Bridge the two helper contracts shipped together in the pinned commit.

    Most Managers call ``prepare_embeddings_from_config(cfg, texts)`` and
    expect ``(embeddings, cfg)``. Episodic Manager calls the same function
    with ``embedding_config=`` plus ``existing_embeddings=`` and expects a
    flattened mapping whose keys end in ``_embedding``. Both call forms are
    present in the unmodified official tree.
    """
    if "embedding_config" not in kwargs and "existing_embeddings" not in kwargs:
        return original(*args, **kwargs)

    call_kwargs = dict(kwargs)
    config = call_kwargs.pop("embedding_config", args[0] if args else None)
    texts = call_kwargs.pop("texts", args[1] if len(args) > 1 else None)
    existing = dict(call_kwargs.pop("existing_embeddings", None) or {})
    if call_kwargs:
        unexpected = ", ".join(sorted(call_kwargs))
        raise TypeError(f"unexpected embedding compatibility arguments: {unexpected}")
    if texts is None:
        raise TypeError("missing required embedding texts")

    generated, _ = original(config, texts)
    merged = dict(existing)
    for field, value in generated.items():
        key = field if str(field).endswith("_embedding") else f"{field}_embedding"
        if merged.get(key) is None:
            merged[key] = value
    return merged


def _mma_observation_timestamp(value: str) -> str:
    """Represent a source date in MMA's required second-resolution format."""
    timestamp = str(value or "").strip()
    if len(timestamp) == 10:
        try:
            date.fromisoformat(timestamp)
        except ValueError:
            pass
        else:
            return f"{timestamp} 00:00:00"
    return timestamp


def _is_unsafe_native_tool_error(response: Any) -> bool:
    """Classify only tool errors for which an automatic retry may duplicate data."""
    text = str(response)
    if not text.startswith("Error executing function "):
        return False
    return any(
        f": {exception_name}:" in text
        for exception_name in _UNSAFE_NATIVE_TOOL_ERROR_NAMES
    )


def _record_native_agent_failure(
    queue: Any,
    agent_type: str,
    exception: Exception,
    *,
    tool_name: str | None = None,
) -> None:
    with queue._message_queue_lock:
        failures = getattr(queue, "_offline_mma_native_failures", None)
        if failures is None:
            failures = []
            queue._offline_mma_native_failures = failures
        # One native root failure causes all concurrent Agent calls to unwind.
        # Keep only that first root cause instead of recording propagation noise.
        if failures:
            return
        row = {
            "agent_type": str(agent_type),
            "error": f"{type(exception).__name__}: {exception}",
        }
        if tool_name is not None:
            row["tool_name"] = str(tool_name)
        failures.append(row)


def _raise_native_agent_failures_from_queue(
    queue: Any | None, *, clear: bool
) -> None:
    if queue is None:
        return
    with queue._message_queue_lock:
        failures = list(
            getattr(queue, "_offline_mma_native_failures", None) or []
        )
        if clear:
            queue._offline_mma_native_failures = []
    if failures:
        raise MMANativeAgentFailure(failures)


@contextmanager
def _fixed_chat_memory_prompt(memory_context: dict[str, Any]) -> Iterator[None]:
    module = importlib.import_module("mma.agent.agent")
    agent_class = module.Agent
    original = agent_class.build_system_prompt_with_memories

    def build(agent: Any, raw_system: str, topics: Any = None, retrieved_memories: Any = None) -> Any:
        if str(getattr(agent.agent_state, "name", "")) != "chat_agent":
            return original(agent, raw_system, topics, retrieved_memories)
        return original(
            agent,
            raw_system,
            topics=topics,
            retrieved_memories=dict(memory_context),
        )

    agent_class.build_system_prompt_with_memories = build
    try:
        yield
    finally:
        agent_class.build_system_prompt_with_memories = original


def _capture_chat_state(client: Any, agent_id: str) -> dict[str, Any]:
    messages = client.get_in_context_messages(agent_id)
    actor = client.server.user_manager.get_user_by_id(client.user.id)
    state = client.server.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor)
    return {
        "message_ids": [str(row.id) for row in messages],
        "topic": getattr(state, "topic", None),
    }


def _restore_chat_state(client: Any, agent_id: str, saved: dict[str, Any]) -> None:
    try:
        server = client.server
        actor = server.user_manager.get_user_by_id(client.user.id)
        server.agent_manager.set_in_context_messages(
            agent_id=agent_id,
            message_ids=list(saved["message_ids"]),
            actor=actor,
        )
        server.message_manager.delete_detached_messages_for_agent(
            agent_id=agent_id, actor=actor
        )
        update_module = importlib.import_module("mma.schemas.agent")
        server.agent_manager.update_agent(
            agent_id=agent_id,
            agent_update=update_module.UpdateAgent(topic=saved.get("topic")),
            actor=actor,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to restore MMA Chat Agent history/topic after QA"
        ) from exc


def _send_native_benchmark_messages(
    client: Any,
    agent_id: str,
    messages: list[dict[str, Any]],
    query_image: str | None,
    memories: list[RetrievedMemory],
) -> tuple[Any, int]:
    message_module = importlib.import_module("mma.schemas.message")
    enum_module = importlib.import_module("mma.schemas.enums")
    content_module = importlib.import_module("mma.schemas.mma_message_content")
    response_module = importlib.import_module("mma.schemas.mma_response")
    prompt_text = "\n".join(str(row.get("content") or "") for row in messages)
    attach_memory_images = "Attached memory " in prompt_text
    image_paths: list[str] = []
    if attach_memory_images:
        for memory in memories:
            for path in memory.image_paths:
                resolved = str(Path(path).resolve())
                if resolved not in image_paths:
                    image_paths.append(resolved)
    if query_image:
        resolved_query = str(Path(query_image).resolve())
        if resolved_query not in image_paths:
            image_paths.append(resolved_query)

    packed = []
    for index, row in enumerate(messages):
        role = str(row.get("role") or "user")
        content: list[Any] = [
            content_module.TextContent(text=str(row.get("content") or ""))
        ]
        if role == "user" and index == len(messages) - 1:
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"MMA answer image not found: {path}")
                metadata = client._save_image_from_file_uri(path)
                content.append(
                    content_module.ImageContent(image_id=metadata.id, detail="auto")
                )
        packed.append(
            message_module.MessageCreate(
                role=enum_module.MessageRole(role), content=content
            )
        )

    client.interface.clear()
    server = client.server
    usage = server.send_messages(
        actor=server.user_manager.get_user_by_id(client.user.id),
        agent_id=agent_id,
        input_messages=packed,
        interface=client.interface,
        force_response=True,
        chaining=True,
    )
    mma_messages = []
    for event in client.interface.to_list():
        mma_messages.extend(event.to_mma_message())
    return response_module.MMAResponse(messages=mma_messages, usage=usage), len(image_paths)


def _extract_chat_answer(response: Any) -> str:
    messages = list(getattr(response, "messages", None) or [])
    for message in reversed(messages):
        tool_call = getattr(message, "tool_call", None)
        if tool_call is None and isinstance(message, dict):
            tool_call = message.get("tool_call")
        if tool_call is None:
            continue
        name = getattr(tool_call, "name", None)
        arguments = getattr(tool_call, "arguments", None)
        if isinstance(tool_call, dict):
            name = tool_call.get("name", name)
            arguments = tool_call.get("arguments", arguments)
        if name and str(name) != "send_message":
            continue
        payload = arguments if isinstance(arguments, dict) else None
        if payload is None and isinstance(arguments, str):
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if payload and payload.get("message"):
            return str(payload["message"])
    return ""


def _require_answer_block(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return stripped
    if "<answer>" not in stripped or "</answer>" not in stripped:
        raise MMAAnswerContractError(
            "MMA Chat Agent violated the benchmark answer-tag contract; "
            "response rewriting is forbidden"
        )
    return stripped


def _response_tool_names(response: Any) -> list[str]:
    names = []
    for message in list(getattr(response, "messages", None) or []):
        tool_call = getattr(message, "tool_call", None)
        if tool_call is None and isinstance(message, dict):
            tool_call = message.get("tool_call")
        name = getattr(tool_call, "name", None)
        if isinstance(tool_call, dict):
            name = tool_call.get("name", name)
        if name:
            names.append(str(name))
    return names
