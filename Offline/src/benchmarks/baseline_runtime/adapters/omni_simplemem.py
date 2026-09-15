from __future__ import annotations

import base64
import io
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import httpx

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from embedding.chunk_builder import Chunk


UPSTREAM_COMMIT = "836ce9718f3e9cb7f93c9d7c842b47f62e177a66"
UPSTREAM_TREE = "685109637c4c8b9a2469e695ad3dbed40762c0f2"
FIXED_TOP_K = 7
OFFLINE_ROOT = Path(__file__).resolve().parents[4]


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


class _RemoteQwenEmbeddingTransport:
    """Transport-only bridge to the experiment's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimension = dimension
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.calls = 0
        if not self.base_url:
            raise ValueError("OmniSimpleMem embedding_base_url is required")

    @staticmethod
    def _image_data_uri(image: Any) -> str:
        from PIL import Image

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"OmniSimpleMem image does not exist: {path}")
            pil_image = Image.open(path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        elif hasattr(image, "__array__"):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            raise TypeError(f"unsupported OmniSimpleMem image type: {type(image)!r}")

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=95)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _embed(self, *, text: str = "", image: Any | None = None) -> list[float]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_uri(image)},
                }
            )
        content.append({"type": "text", "text": text or " "})
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Represent the input for retrieval.",
                            }
                        ],
                    },
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}],
                    },
                ],
                "encoding_format": "float",
                "continue_final_message": True,
                "add_special_tokens": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            values = response.json()["data"][0]["embedding"]
            vector = [float(value) for value in values]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("OmniSimpleMem embedding response has invalid schema") from exc
        if len(vector) != self.dimension:
            raise RuntimeError(
                "OmniSimpleMem embedding dimension mismatch: "
                f"expected {self.dimension}, got {len(vector)}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("OmniSimpleMem embedding response has invalid norm")
        self.calls += 1
        return [value / norm for value in vector]

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text=str(text))

    def embed_image(self, image: Any) -> list[float]:
        return self._embed(image=image)


class OmniSimpleMemAdapter(BaselineAdapter):
    """Thin Offline protocol bridge over an immutable official Omni core."""

    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root.resolve()
        self.config = dict(config)
        self.backend: Any = None
        self.provenance = ProvenanceIndex()
        self._current_session = ""
        self._original_cwd: Path | None = None
        self._embedding_transport: _RemoteQwenEmbeddingTransport | None = None
        self._last_native_top_k: int | None = None
        self._last_native_modality_filter: str | None = None
        self._ingest_stats = {"stored": 0, "skipped": 0, "failed": 0}
        self._verify_upstream()
        if int(self.config.get("top_k", FIXED_TOP_K)) != FIXED_TOP_K:
            raise ValueError(
                f"OmniSimpleMem reproduction requires fixed top_k={FIXED_TOP_K}"
            )
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))

    def _verify_upstream(self) -> None:
        repository = self.source_root.parent
        if not (repository / ".git").is_dir():
            raise RuntimeError(
                "OmniSimpleMem must run from the prepared immutable official checkout: "
                f"{self.source_root}"
            )
        actual_commit = _git("rev-parse", "HEAD", cwd=repository)
        actual_tree = _git("rev-parse", "HEAD:OmniSimpleMem", cwd=repository)
        dirty = _git("status", "--porcelain", cwd=repository)
        if actual_commit != UPSTREAM_COMMIT or actual_tree != UPSTREAM_TREE or dirty:
            raise RuntimeError(
                "OmniSimpleMem upstream verification failed: "
                f"commit={actual_commit}, tree={actual_tree}, dirty={bool(dirty)}"
            )

    def _executor_api_key(self) -> str:
        direct = str(self.config.get("executor_api_key") or "").strip()
        if direct:
            return direct
        configured = self.config.get("omni_executor_api_key_file")
        path = (
            Path(str(configured)).expanduser()
            if configured
            else OFFLINE_ROOT.parent / "Nvida_api" / "Openrouter_api"
        )
        if not path.is_absolute():
            path = (OFFLINE_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OmniSimpleMem executor API key file missing: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"OmniSimpleMem executor API key file is empty: {path}")
        return key

    def _build_config(self) -> Any:
        from omni_memory import OmniMemoryConfig

        config = OmniMemoryConfig.create_default()
        embedding_model = str(self.config["embedding_model"])
        embedding_dim = int(self.config["embedding_dim"])
        config.embedding.model_name = embedding_model
        config.embedding.embedding_dim = embedding_dim
        config.embedding.visual_embedding_model = embedding_model
        config.embedding.visual_embedding_dim = embedding_dim
        config.retrieval.default_top_k = FIXED_TOP_K
        config.llm.api_base_url = str(self.config["executor_base_url"])
        config.llm.api_key = self._executor_api_key()
        config.llm.temperature = float(self.config.get("executor_temperature") or 0.0)
        config.llm.max_tokens = int(self.config.get("executor_max_tokens") or 512)
        config.set_unified_model(str(self.config["executor_model"]))
        return config

    def _install_embedding_transport(self) -> None:
        assert self.backend is not None
        transport = _RemoteQwenEmbeddingTransport(
            base_url=str(self.config["embedding_base_url"]),
            model=str(self.config["embedding_model"]),
            dimension=int(self.config["embedding_dim"]),
            api_key=os.getenv(str(self.config.get("embedding_api_key_env") or ""), ""),
            timeout=float(self.config.get("request_timeout") or 180),
        )
        self._embedding_transport = transport

        def text_embedding(_processor: Any, data: Any) -> list[float]:
            return transport.embed_text(str(data)[:8000])

        def image_embedding(_processor: Any, data: Any) -> list[float]:
            return transport.embed_image(data)

        self.backend.text_processor.generate_embedding = MethodType(
            text_embedding, self.backend.text_processor
        )
        self.backend.image_processor.generate_embedding = MethodType(
            image_embedding, self.backend.image_processor
        )
        self.backend.retriever._embedding_service = transport

    def _install_fixed_top_k_policy(self) -> None:
        assert self.backend is not None
        original = self.backend.query_processor.determine_retrieval_strategy

        def fixed_strategy(_processor: Any, parsed: Any) -> dict[str, Any]:
            strategy = dict(original(parsed))
            self._last_native_top_k = int(strategy.get("top_k", FIXED_TOP_K))
            strategy["top_k"] = FIXED_TOP_K
            modality_filter = strategy.get("modality_filter")
            self._last_native_modality_filter = (
                str(getattr(modality_filter, "value", modality_filter))
                if modality_filter is not None
                else None
            )
            # Qwen3-VL provides one shared 2048-dimensional embedding space for
            # all indexed benchmark evidence.  Upstream modality hints are found
            # with substring matches (for example, "clipboard" -> "clip"/VIDEO
            # and "context" -> "text"/TEXT), then applied as hard filters after
            # vector candidates have already been selected.  The benchmark index
            # also has no AUDIO or VIDEO MAUs, so those false positives can erase
            # every candidate.  Preserve the native hint in the trace, ranking,
            # graph traversal, and fixed Top-7, but search the unified MAU pool.
            if modality_filter is not None:
                strategy["modality_filter"] = None
            return strategy

        self.backend.query_processor.determine_retrieval_strategy = MethodType(
            fixed_strategy, self.backend.query_processor
        )

    def _restore_cwd(self) -> None:
        if self._original_cwd is not None:
            os.chdir(self._original_cwd)
            self._original_cwd = None

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self._restore_cwd()
        state_dir = state_dir.resolve()
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        # Official KnowledgeGraph uses a cwd-relative default path. Isolating the
        # worker cwd keeps that unchanged behavior sample-local.
        self._original_cwd = Path.cwd()
        os.chdir(state_dir)
        from omni_memory import OmniMemoryOrchestrator

        self.backend = OmniMemoryOrchestrator(
            config=self._build_config(), data_dir=str(state_dir)
        )
        self._install_embedding_transport()
        self._install_fixed_top_k_policy()
        self.provenance.clear()
        self._current_session = ""
        self._last_native_top_k = None
        self._last_native_modality_filter = None
        self._ingest_stats = {"stored": 0, "skipped": 0, "failed": 0}

    @staticmethod
    def _tags(chunk: Chunk) -> list[str]:
        metadata = chunk.metadata
        dialogue_id = str(metadata.get("dialogue_id") or chunk.chunk_id)
        session_id = str(metadata.get("session_id") or "")
        tags = [f"dialogue_id:{dialogue_id}", f"session_id:{session_id}"]
        timestamp = str(metadata.get("timestamp") or metadata.get("date") or "")
        if timestamp:
            tags.append(f"timestamp:{timestamp}")
        for image_id in metadata.get("image_ids") or []:
            if str(image_id):
                tags.append(f"image_id:{image_id}")
        return tags

    def _record_processing_result(
        self, result: Any, chunk: Chunk, *, operation: str
    ) -> None:
        if result.success and result.mau is not None:
            self.provenance.register(str(result.mau.id), chunk)
            self._ingest_stats["stored"] += 1
            return
        if bool(getattr(result, "skipped", False)):
            self._ingest_stats["skipped"] += 1
            return
        self._ingest_stats["failed"] += 1
        raise RuntimeError(
            f"OmniSimpleMem {operation} failed: {getattr(result, 'error', None) or 'unknown error'}"
        )

    def ingest(self, chunk: Chunk) -> None:
        if self.backend is None:
            raise RuntimeError("OmniSimpleMem adapter has not been reset")
        session_id = str(chunk.metadata.get("session_id") or "")
        if session_id != self._current_session:
            if self._current_session:
                self.backend.end_session()
            self.backend.start_session(session_id or None)
            self._current_session = session_id

        tags = self._tags(chunk)
        images = [str(path) for path in chunk.images if str(path)]
        for path in images:
            if not Path(path).is_file():
                raise FileNotFoundError(f"OmniSimpleMem image does not exist: {path}")

        text_result = self.backend.add_text(
            chunk.text,
            session_id=session_id or None,
            tags=tags,
        )
        self._record_processing_result(text_result, chunk, operation="add_text")
        for image in images:
            result = self.backend.add_image(
                image,
                session_id=session_id or None,
                tags=tags,
            )
            self._record_processing_result(result, chunk, operation="add_image")

    def end_session(self, session_id: str) -> None:
        if self.backend is not None and (
            not session_id or session_id == self._current_session
        ):
            self.backend.end_session()
            self._current_session = ""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self.backend is None:
            raise RuntimeError("OmniSimpleMem adapter has not been reset")
        if request.top_k != FIXED_TOP_K:
            raise ValueError(
                f"OmniSimpleMem reproduction requires request top_k={FIXED_TOP_K}"
            )
        embedding_calls_before = (
            self._embedding_transport.calls if self._embedding_transport else 0
        )
        native = self.backend.query(
            request.text,
            top_k=FIXED_TOP_K,
            auto_expand=False,
            benchmark_safe=True,
        )
        items: list[RetrievedMemory] = []
        for row in native.items:
            memory_id = str(row.get("id") or row.get("memory_id") or "")
            if not memory_id:
                raise RuntimeError("OmniSimpleMem returned a retrieval row without an id")
            if request.visible_session_ids and not self.provenance.visible(
                memory_id, request.visible_session_ids
            ):
                continue
            provenance = self.provenance.get(memory_id)
            details = row.get("details")
            full_text = details.get("full_text", "") if isinstance(details, dict) else ""
            items.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    text=str(full_text or row.get("summary") or row.get("text") or ""),
                    score=(
                        float(row["score"]) if row.get("score") is not None else None
                    ),
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(
                        provenance.get("source_dialogue_ids") or []
                    ),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=list(provenance.get("image_paths") or []),
                    metadata={
                        **dict(row.get("metadata") or {}),
                        "modality_type": row.get("modality_type"),
                        "has_raw_data": bool(row.get("has_raw_data")),
                    },
                )
            )
            if len(items) == FIXED_TOP_K:
                break
        graph_result = getattr(native, "graph_entities", None)
        graph_entity_count = len(getattr(graph_result, "entities", []) or []) + len(
            getattr(graph_result, "related_entities", []) or []
        )
        graph_mau_count = len(getattr(graph_result, "mau_ids", []) or [])
        return RetrievalResult(
            items=items,
            trace={
                "baseline": self.baseline,
                "via": "OmniMemoryOrchestrator.query",
                "official_benchmark_entry": False,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_tree": UPSTREAM_TREE,
                "requested_top_k": FIXED_TOP_K,
                "native_dynamic_top_k_before_override": self._last_native_top_k,
                "effective_top_k": FIXED_TOP_K,
                "native_modality_filter_before_override": (
                    self._last_native_modality_filter
                ),
                "effective_modality_filter": None,
                "modality_hard_filter_disabled": True,
                "visual_hard_filter_disabled": True,
                "native_returned": len(native.items),
                "returned": len(items),
                "native_level": str(getattr(native, "level", "")),
                "native_total_candidates": int(
                    getattr(native, "total_candidates", 0) or 0
                ),
                "graph_entity_count": graph_entity_count,
                "graph_mau_count": graph_mau_count,
                "parametric_answer_count": len(
                    getattr(native, "parametric_answers", []) or []
                ),
                "embedding_model": str(self.config["embedding_model"]),
                "embedding_base_url": str(self.config["embedding_base_url"]),
                "query_embedding_calls": (
                    (self._embedding_transport.calls if self._embedding_transport else 0)
                    - embedding_calls_before
                ),
                "ingest": dict(self._ingest_stats),
            },
        )

    def snapshot(self) -> list[MemoryRecord]:
        if self.backend is None:
            return []
        records = []
        for mau in self.backend.mau_store.get_active(limit=1_000_000):
            provenance = self.provenance.get(str(mau.id))
            native = mau.to_dict()
            details = native.get("details")
            full_text = details.get("full_text", "") if isinstance(details, dict) else ""
            records.append(
                MemoryRecord(
                    memory_id=str(mau.id),
                    text=str(full_text or mau.summary or ""),
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(
                        provenance.get("source_dialogue_ids") or []
                    ),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=list(provenance.get("image_paths") or []),
                    backend_type=f"omni_mau:{mau.modality_type.value}",
                    metadata={
                        "upstream_commit": UPSTREAM_COMMIT,
                        "upstream_tree": UPSTREAM_TREE,
                        "native": {
                            key: value
                            for key, value in native.items()
                            if key != "embedding"
                        },
                        "embedding_dim": len(mau.embedding or []),
                    },
                )
            )
        return records

    def close(self) -> None:
        try:
            if self.backend is not None:
                self.backend.close()
                self.backend = None
        finally:
            self._restore_cwd()
            self._current_session = ""

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "omni_simplemem",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "official_benchmark_entry": False,
            "answer_path": "benchmark_qa_prompt",
            "orchestrator_answer_used": False,
            "fixed_top_k": FIXED_TOP_K,
            "modality_hard_filter_disabled": True,
            "visual_hard_filter_disabled": True,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_tree": UPSTREAM_TREE,
            "embedding_transport": "openai_compatible_qwen_vl",
            "ingest_path": "OmniMemoryOrchestrator.add_text+add_image",
            "upstream_add_multimodal_used": False,
        }
