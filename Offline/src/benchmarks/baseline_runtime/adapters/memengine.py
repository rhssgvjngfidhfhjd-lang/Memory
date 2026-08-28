from __future__ import annotations

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from embedding.chunk_builder import Chunk


class MemEngineAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.memory: Any = None
        self.sample_id = ""
        self._chunks: list[Chunk] = []
        embedding_base_url = str(config.get("embedding_base_url") or "")
        if embedding_base_url:
            os.environ["OPENAI_EMBEDDING_BASE_URL"] = embedding_base_url
            os.environ["QWEN_VL_EMBED_BASE_URL"] = embedding_base_url
        self._bootstrap()

    def _bootstrap(self) -> None:
        baselines_root = self.source_root.parents[1]
        for path in (baselines_root, self.source_root.parent):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        if "memengine" not in sys.modules:
            module = types.ModuleType("memengine")
            module.__path__ = [str(self.source_root)]  # type: ignore[attr-defined]
            module.__package__ = "memengine"
            sys.modules["memengine"] = module
        for child in ("config", "function", "memory", "operation", "utils"):
            name = f"memengine.{child}"
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = [str(self.source_root / child)]  # type: ignore[attr-defined]
                module.__package__ = name
                sys.modules[name] = module
                setattr(sys.modules["memengine"], child, module)
        function_pkg = sys.modules["memengine.function"]
        for module_name in (
            "Encoder", "Retrieval", "LLM", "Judge", "Summarizer", "Truncation",
            "Trigger", "Utilization", "MultiModalEncoder", "MultiModalRetrieval",
            "ConceptExtractor", "ConceptBasedRetrieval",
        ):
            try:
                module = importlib.import_module(f"memengine.function.{module_name}")
            except ImportError as exc:
                raise ImportError(
                    f"MemEngine dependency missing while loading {module_name}: {exc}"
                ) from exc
            for key, value in vars(module).items():
                if not key.startswith("_") and not hasattr(function_pkg, key):
                    setattr(function_pkg, key, value)

    def _native_config(self) -> Any:
        defaults = importlib.import_module("default_config.DefaultMemoryConfig")
        raw = copy.deepcopy(defaults.DEFAULT_AUGUSTUSMEMORY)
        self._apply_common_config(raw)
        config_cls = importlib.import_module("memengine.config.Config").MemoryConfig
        return config_cls(raw)

    def _apply_common_config(self, value: Any) -> None:
        if isinstance(value, dict):
            if "topk" in value:
                value["topk"] = int(self.config.get("top_k") or value["topk"])
            method = str(value.get("method") or "")
            if method in {"APILLM", "OpenAILLM"}:
                value.update(
                    {
                        "name": self.config["executor_model"],
                        "base_url": self.config["executor_base_url"],
                        "api_key": os.getenv("OPENAI_API_KEY") or "EMPTY",
                        "temperature": float(self.config.get("executor_temperature") or 0.0),
                    }
                )
            if method in {"LMEncoder", "OpenAIEncoder", "GMEEncoder"}:
                if method == "LMEncoder":
                    value["method"] = "OpenAIEncoder"
                value.update(
                    {
                        "name": self.config["embedding_model"],
                        "path": self.config["embedding_model"],
                        "dimension": int(self.config["embedding_dim"]),
                        "base_url": self.config.get("embedding_base_url") or "",
                        "api_key": os.getenv(str(self.config.get("embedding_api_key_env") or "")) or "EMPTY",
                    }
                )
            if method in {"LMTruncation", "MMLMTruncation"}:
                # The shared executor is an API service and may not have a local
                # Hugging Face tokenizer checkout in the worker environment.
                value["mode"] = "word"
            for nested in value.values():
                self._apply_common_config(nested)
        elif isinstance(value, list):
            for nested in value:
                self._apply_common_config(nested)

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del state_dir
        module = importlib.import_module(f"memengine.memory.{self.baseline}")
        self.memory = getattr(module, self.baseline)(self._native_config())
        self.sample_id = sample_id
        self._chunks = []

    def ingest(self, chunk: Chunk) -> None:
        if self.memory is None:
            raise RuntimeError("MemEngine adapter has not been reset")
        observation = dict(chunk.metadata)
        observation["text"] = chunk.text
        observation["image"] = chunk.images[0] if chunk.images else ""
        observation["images"] = list(chunk.images)
        observation["source_dialogue_ids"] = [
            str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
        ]
        self.memory.store(observation)
        self._chunks.append(chunk)

    def end_session(self, session_id: str) -> None:
        del session_id

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        query: dict[str, Any] = {"text": request.text, "category": request.category}
        if request.query_image:
            query["image"] = request.query_image
        value = self.memory.recall(query)
        text = value if isinstance(value, str) else str(value)
        source_ids = self._last_retrieved_source_ids()
        session_id = ""
        image_ids: list[str] = []
        image_paths: list[str] = []
        for chunk in self._chunks:
            dialogue_id = str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
            if dialogue_id in source_ids:
                session_id = session_id or str(chunk.metadata.get("session_id") or "")
                image_ids.extend(str(x) for x in chunk.metadata.get("image_ids") or [])
                image_paths.extend(chunk.images)
        item = RetrievedMemory(
            memory_id=f"{self.baseline}:context:{request.query_id}",
            text=text,
            score=None,
            session_id=session_id,
            source_dialogue_ids=source_ids,
            image_ids=list(dict.fromkeys(image_ids)),
            image_paths=list(dict.fromkeys(image_paths)),
            metadata={"aggregated_context": True},
        )
        return RetrievalResult(
            items=[] if not text or text == "None" else [item],
            trace={"baseline": self.baseline, "via": "native_recall"},
        )

    def _last_retrieved_source_ids(self) -> list[str]:
        ids = getattr(getattr(self.memory, "recall_op", None), "last_retrieved_ids", None)
        if not ids:
            return []
        result = []
        snapshots = self._storage_rows()
        for value in ids:
            try:
                row = snapshots[int(value)]
            except (IndexError, TypeError, ValueError):
                continue
            result.extend(str(x) for x in row.get("source_dialogue_ids") or [])
        return list(dict.fromkeys(result))

    def _storage_rows(self) -> list[dict[str, Any]]:
        storages = []
        for name in ("recall_storage", "archival_storage", "contextual_memory", "storage"):
            value = getattr(self.memory, name, None)
            if value is not None and value not in storages:
                storages.append(value)
        for value in (getattr(self.memory, "main_context", None) or {}).values():
            if hasattr(value, "get_all_memory_in_order") and value not in storages:
                storages.append(value)
        rows: list[dict[str, Any]] = []
        for storage in storages:
            if hasattr(storage, "get_all_memory_in_order"):
                values = storage.get_all_memory_in_order() or []
            elif hasattr(storage, "memory_list"):
                values = storage.memory_list
            elif hasattr(storage, "memory_node"):
                values = storage.memory_node.values()
            elif hasattr(storage, "node_id_to_memory"):
                values = storage.node_id_to_memory.values()
            elif hasattr(storage, "node"):
                values = storage.node.values()
            else:
                values = []
            rows.extend(dict(row) for row in values if isinstance(row, dict))
        return rows

    def snapshot(self) -> list[MemoryRecord]:
        records = []
        for index, row in enumerate(self._storage_rows()):
            source_ids = [str(x) for x in row.get("source_dialogue_ids") or []]
            records.append(
                MemoryRecord(
                    memory_id=f"{self.baseline}:{index}",
                    text=str(row.get("text") or row.get("summary") or row),
                    session_id=str(row.get("session_id") or ""),
                    source_dialogue_ids=source_ids,
                    image_ids=[str(x) for x in row.get("image_ids") or []],
                    image_paths=[str(x) for x in row.get("images") or ([row["image"]] if row.get("image") else [])],
                    backend_type="memengine",
                    metadata={key: value for key, value in row.items() if key not in {"text", "image", "images"}},
                )
            )
        return records

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "memengine",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": False,
        }
