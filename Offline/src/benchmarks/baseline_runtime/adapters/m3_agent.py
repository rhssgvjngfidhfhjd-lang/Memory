from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.baseline_runtime.openai_compat import embed_texts
from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from embedding.chunk_builder import Chunk


class M3AgentAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.graph: Any = None
        self._node_sources: dict[int, dict[str, Any]] = {}
        self._clip_id = 0
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id, state_dir
        old_cwd = Path.cwd()
        try:
            os.chdir(self.source_root)
            if "mmagent" not in sys.modules:
                package = types.ModuleType("mmagent")
                package.__path__ = [str(self.source_root / "mmagent")]  # type: ignore[attr-defined]
                package.__package__ = "mmagent"
                sys.modules["mmagent"] = package
            if "mmagent.memory_processing" not in sys.modules:
                processing = types.ModuleType("mmagent.memory_processing")
                processing.parse_video_caption = lambda graph, caption: []
                sys.modules["mmagent.memory_processing"] = processing
            VideoGraph = importlib.import_module("mmagent.videograph").VideoGraph

            config_path = self.source_root / "configs" / "memory_config.json"
            graph_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.graph = VideoGraph(**graph_config)
        finally:
            os.chdir(old_cwd)
        self._node_sources = {}
        self._clip_id = 0

    def ingest(self, chunk: Chunk) -> None:
        self._clip_id += 1
        vector = embed_texts([chunk.text], self.config)[0]
        node_id = self.graph.add_text_node(
            {"contents": [chunk.text], "embeddings": [vector]},
            self._clip_id,
            "episodic",
        )
        self._node_sources[node_id] = {
            "session_id": str(chunk.metadata.get("session_id") or ""),
            "source_dialogue_ids": [str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)],
            "image_ids": [str(x) for x in chunk.metadata.get("image_ids") or []],
            "image_paths": list(chunk.images),
        }

    def end_session(self, session_id: str) -> None:
        del session_id

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        vector = embed_texts([request.text], self.config)[0]
        ranked = self.graph.search_text_nodes(np.asarray([vector]), mode="max")
        items = []
        visible = set(request.visible_session_ids)
        for node_id, score in ranked:
            source = self._node_sources.get(int(node_id), {})
            session_id = str(source.get("session_id") or "")
            if visible and session_id not in visible:
                continue
            node = self.graph.nodes[node_id]
            items.append(
                RetrievedMemory(
                    memory_id=f"m3:{node_id}",
                    text="\n".join(str(x) for x in node.metadata.get("contents") or []),
                    score=float(score),
                    session_id=session_id,
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    metadata={"clip_id": node.metadata.get("timestamp"), "node_type": node.type},
                )
            )
            if len(items) >= request.top_k:
                break
        return RetrievalResult(items=items, trace={"baseline": self.baseline, "via": "m3_graph"})

    def snapshot(self) -> list[MemoryRecord]:
        records = []
        for node_id in self.graph.text_nodes:
            node = self.graph.nodes[node_id]
            source = self._node_sources.get(int(node_id), {})
            records.append(
                MemoryRecord(
                    memory_id=f"m3:{node_id}",
                    text="\n".join(str(x) for x in node.metadata.get("contents") or []),
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    backend_type=f"m3_{node.type}",
                    metadata={"clip_id": node.metadata.get("timestamp")},
                )
            )
        return records

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "m3_agent",
            "baseline": self.baseline,
            "available": True,
            "compatibility_mode": "dialogue_round_as_clip",
            "audio_enabled": False,
            "supports_images": False,
            "supports_session_filter": True,
        }
