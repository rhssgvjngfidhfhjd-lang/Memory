"""Adapter for the HiveMem baseline.

Drives the insert-only pipeline from ``Offline/src/hive_mem`` online against
the WMA runner protocol: turns are buffered into dialogue rounds, and each
completed round is converted by the executor into one or more MAUs.

The pipeline code is imported from an external checkout (config
``baselines.AgentMem.pipeline_root`` / env ``AGENTMEM_ROOT``);
import failures are reported via ``get_capabilities()["available"] = False``
instead of crashing the CLI.

``mm_mode`` mirrors the pipeline's Mem-Gallery a/b/c ablation:

* ``text``    — dialogue text only (image_id kept, captions stripped)
* ``caption`` — text + ``image_caption:`` lines
* ``image``   — same as caption, plus image paths in metadata and
  ``RetrievalItem.image_path`` for the answer-stage VLM.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from eval_framework.datasets.schemas import (
    MemoryDeltaRecord,
    MemorySnapshotRecord,
    NormalizedTurn,
    RetrievalItem,
    RetrievalRecord,
)
from eval_framework.memory_adapters.base import MemoryAdapter

_DEFAULT_PIPELINE_ROOT = "/data/haozhen/Memory/Offline"

# One embedding model per process: sample workers run in threads, and a
# per-adapter Qwen3-VL-Embedding-2B instance (~5 GB) per thread would not fit
# next to the vLLM servers already resident on the GPUs.
_EMBEDDER_LOCK = threading.Lock()
_SHARED_EMBEDDER: Any = None


def _param(baseline_name: str, key: str, default: Any) -> Any:
    from eval_framework.config import resolve_baseline_param

    return resolve_baseline_param(baseline_name, key, default)


def _ensure_pipeline_on_path(pipeline_root: str) -> None:
    root = Path(pipeline_root).resolve()
    source_root = root / "src"
    if not (source_root / "hive_mem").is_dir():
        raise FileNotFoundError(
            f"src/hive_mem/ not found under {root}. Set "
            "baselines.HiveMem.pipeline_root in config.yaml or the "
            "HIVEMEM_ROOT env var."
        )
    s = str(source_root)
    if s not in sys.path:
        sys.path.insert(0, s)


class _LockedEmbedder:
    """Serialize GPU forward passes of the shared embedder across threads."""

    def __init__(self, inner: Any):
        self._inner = inner
        self.expected_dim = inner.expected_dim

    def embed_texts(self, texts: Any, mode: str = "context") -> Any:
        with _EMBEDDER_LOCK:
            return self._inner.embed_texts(texts, mode=mode)

    def embed_images(self, image_paths: Any) -> Any:
        with _EMBEDDER_LOCK:
            return self._inner.embed_images(image_paths)


def _get_shared_embedder(model_name: str, device: str, expected_dim: int) -> _LockedEmbedder:
    global _SHARED_EMBEDDER
    with _EMBEDDER_LOCK:
        if _SHARED_EMBEDDER is None:
            from embedding.qwen3_text_embedding import create_memory_embedder

            _SHARED_EMBEDDER = create_memory_embedder(
                model_name=model_name,
                device=device,
                expected_dim=expected_dim,
            )
    return _LockedEmbedder(_SHARED_EMBEDDER)


class HiveMemAdapter(MemoryAdapter):
    """HiveMem driven online by the WMA eval runner."""

    def __init__(self, baseline_name: str = "HiveMem") -> None:
        self.baseline_name = baseline_name
        self._integration_error: str | None = None

        self.mm_mode = str(_param(baseline_name, "mm_mode", "image")).strip().lower()
        if self.mm_mode not in {"text", "caption", "image"}:
            raise ValueError(f"invalid mm_mode: {self.mm_mode!r}")
        self.chunk_mode = str(_param(baseline_name, "chunk_mode", "round")).strip().lower()
        if self.chunk_mode not in {"round", "turn", "session"}:
            raise ValueError(f"invalid chunk_mode: {self.chunk_mode!r}")
        pipeline_root = str(
            os.getenv("HIVEMEM_ROOT")
            or os.getenv("AGENTMEM_ROOT")
            or _param(baseline_name, "pipeline_root", _DEFAULT_PIPELINE_ROOT)
        )
        try:
            _ensure_pipeline_on_path(pipeline_root)
            from hive_mem.executor import MemoryExecutor
            from hive_mem.llm_client import LLMClient
            from hive_mem.mau import MAUBank
        except Exception as exc:  # noqa: BLE001 — surface via capabilities
            self._integration_error = f"{type(exc).__name__}: {exc}"
            return

        self._memory_bank_cls = MAUBank
        self.operation_names = ["insert"]

        api_key_env = str(_param(baseline_name, "executor_api_key_env", ""))
        api_key = (os.getenv(api_key_env) if api_key_env else None) or "EMPTY"
        backend = LLMClient(
            model=str(_param(baseline_name, "executor_model", "Qwen/Qwen3-VL-4B-Instruct")),
            api_base=str(_param(baseline_name, "executor_base_url", "http://127.0.0.1:8000/v1")),
            api_key=api_key,
            temperature=float(_param(baseline_name, "executor_temperature", 0.0)),
            max_new_tokens=int(_param(baseline_name, "executor_max_new_tokens", 1024)),
            max_retries=int(_param(baseline_name, "executor_retries", 3)),
            timeout=int(_param(baseline_name, "executor_timeout", 120)),
        )
        device = str(
            _param(baseline_name, "embedder_device", "")
            or os.getenv("QWEN3VL_EMBEDDING_DEVICE", "")
            or "auto"
        )
        self._embedder = _get_shared_embedder(
            model_name=str(_param(baseline_name, "embedder_model", "Qwen/Qwen3-VL-Embedding-2B")),
            device=None if device == "auto" else device,
            expected_dim=int(_param(baseline_name, "embedder_dim", 2048)),
        )
        self._executor = MemoryExecutor(backend, self._embedder)

        self._bank = MAUBank()
        self._round_buffer: list[NormalizedTurn] = []
        self._round_counters: dict[str, int] = {}
        self._prev_contents: dict[str, str] = {}
        self._action_counts: dict[str, int] = {}
        self._parse_failures = 0
        self._rounds_processed = 0

    # ─── MemoryAdapter API ─────────────────────────────────────────────

    def reset(self) -> None:
        if self._integration_error:
            return
        self._bank = self._memory_bank_cls()
        self._round_buffer = []
        self._round_counters = {}
        self._prev_contents = {}
        self._action_counts = {}
        self._parse_failures = 0
        self._rounds_processed = 0

    def ingest_turn(self, turn: NormalizedTurn) -> None:
        if self.chunk_mode == "turn":
            self._round_buffer.append(turn)
            self._flush_round()
            return
        if (
            self.chunk_mode == "round"
            and turn.role == "user"
            and any(t.role != "user" for t in self._round_buffer)
        ):
            self._flush_round()
        if self._round_buffer and turn.session_id != self._round_buffer[-1].session_id:
            self._flush_round()
        self._round_buffer.append(turn)

    def end_session(self, session_id: str) -> None:
        self._flush_round()

    def snapshot_memories(self) -> list[MemorySnapshotRecord]:
        out: list[MemorySnapshotRecord] = []
        for item in self._bank.memories:
            out.append(
                MemorySnapshotRecord(
                    memory_id=item.memory_id,
                    text=item.content,
                    session_id=str(item.metadata.get("session_id", "")),
                    status="active",
                    source=self.baseline_name,
                    raw_backend_id=item.memory_id,
                    raw_backend_type="hivemem",
                    metadata=dict(item.metadata),
                )
            )
        return out

    def export_memory_delta(self, session_id: str) -> list[MemoryDeltaRecord]:
        current = {item.memory_id: item.content for item in self._bank.memories}
        deltas: list[MemoryDeltaRecord] = []
        for memory_id, content in current.items():
            if memory_id not in self._prev_contents:
                op = "add"
            elif self._prev_contents[memory_id] != content:
                op = "update"
            else:
                continue
            deltas.append(
                MemoryDeltaRecord(
                    session_id=session_id,
                    op=op,
                    text=content,
                    raw_backend_id=memory_id,
                    metadata={"baseline": self.baseline_name},
                )
            )
        for memory_id, content in self._prev_contents.items():
            if memory_id not in current:
                deltas.append(
                    MemoryDeltaRecord(
                        session_id=session_id,
                        op="suppress",
                        text=content,
                        raw_backend_id=memory_id,
                        metadata={"baseline": self.baseline_name},
                    )
                )
        self._prev_contents = current
        return deltas

    def retrieve(self, query: str, top_k: int, **_: Any) -> RetrievalRecord:
        import numpy as np

        items: list[RetrievalItem] = []
        memories = self._bank.memories
        if memories:
            query_vec = np.asarray(
                self._embedder.embed_texts(str(query), mode="query"), dtype=np.float32
            ).reshape(1, -1)
            query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8)
            matrix = np.vstack([m.embedding for m in memories]).astype(np.float32)
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
            scores = (query_vec @ matrix.T)[0]
            order = np.argsort(scores)[::-1][: max(int(top_k), 0)]
            for rank, index in enumerate(order.tolist()):
                memory = memories[index]
                image_path: str | None = None
                if self.mm_mode == "image":
                    paths = memory.metadata.get("image_paths") or []
                    image_path = str(paths[0]) if paths else None
                items.append(
                    RetrievalItem(
                        rank=rank,
                        memory_id=memory.memory_id,
                        text=memory.content,
                        score=float(scores[index]),
                        raw_backend_id=memory.memory_id,
                        image_path=image_path,
                    )
                )
        return RetrievalRecord(
            query=query,
            top_k=top_k,
            items=items,
            raw_trace={
                "baseline": self.baseline_name,
                "mm_mode": self.mm_mode,
                "memory_count": len(memories),
                "rounds_processed": self._rounds_processed,
                "action_counts": dict(self._action_counts),
                "parse_failures": self._parse_failures,
            },
        )

    def get_capabilities(self) -> dict[str, Any]:
        caps: dict[str, Any] = {
            "backend": "hivemem",
            "baseline": self.baseline_name,
            "available": self._integration_error is None,
            "delta_granularity": "per_session",
            "snapshot_mode": "full",
            "mm_mode": self.mm_mode,
            "chunk_mode": self.chunk_mode,
            "operations": getattr(self, "operation_names", []),
        }
        if self._integration_error:
            caps["integration_error"] = self._integration_error
        return caps

    # ─── internals ─────────────────────────────────────────────────────

    def _flush_round(self) -> None:
        turns = self._round_buffer
        self._round_buffer = []
        if not turns:
            return
        chunk_text = self._format_chunk(turns)
        if not chunk_text.strip():
            return
        metadata = self._round_metadata(turns)

        _, actions = self._executor.execute(
            chunk_text=chunk_text,
            image_paths=metadata.get("image_paths") or [],
            visual_input="image" if self.mm_mode == "image" else "caption",
        )
        self._executor.apply_to_memory_bank(
            actions,
            self._bank,
            event_metadata=metadata,
        )
        if not any(action.success and action.memory_content.strip() for action in actions):
            fallback_text = self._executor.prepare_chunk_text(
                chunk_text,
                "image" if self.mm_mode == "image" else "caption",
            )
            embedding = self._embedder.embed_texts(fallback_text, mode="context")
            self._bank.add_memory(
                fallback_text,
                embedding,
                metadata={**metadata, "source": "fallback_insert"},
            )
        self._rounds_processed += 1
        for action in actions:
            key = "INSERT"
            self._action_counts[key] = self._action_counts.get(key, 0) + 1
            if not action.success:
                self._parse_failures += 1

    def _format_chunk(self, turns: list[NormalizedTurn]) -> str:
        lines: list[str] = []
        for turn in turns:
            stamp = f" [{turn.timestamp}]" if turn.timestamp else ""
            lines.append(f"[{turn.session_id}]{stamp} {turn.role}: {turn.text}")
            for att in turn.attachments:
                if att.image_id:
                    lines.append(f"image_id: {att.image_id}")
                if self.mm_mode in {"caption", "image"} and att.caption:
                    lines.append(f"image_caption: {att.caption}")
        return "\n".join(lines)

    def _round_metadata(self, turns: list[NormalizedTurn]) -> dict[str, Any]:
        first = turns[0]
        session_id = first.session_id
        round_id = self._round_counters.get(session_id, 0)
        self._round_counters[session_id] = round_id + 1
        dialogue_id = f"{first.sample_id}_{session_id}_r{round_id:03d}"

        image_ids: list[str] = []
        image_paths: list[str] = []
        image_captions: list[str] = []
        for turn in turns:
            for att in turn.attachments:
                if att.image_id:
                    image_ids.append(att.image_id)
                if self.mm_mode == "image" and att.file_path:
                    image_paths.append(att.file_path)
                if self.mm_mode in {"caption", "image"} and att.caption:
                    image_captions.append(att.caption)
        return {
            "dataset": first.sample_id,
            "session_id": session_id,
            "round_id": round_id,
            "dialogue_id": dialogue_id,
            "source_dialogue_ids": [dialogue_id],
            "source_chunk_ids": [dialogue_id],
            "timestamp": first.timestamp or "",
            "image_id": image_ids[0] if image_ids else "",
            "image_ids": image_ids,
            "image_paths": image_paths,
            "image_captions": image_captions,
        }


# Keep the historical class name importable for old configs and result scripts.
AgentMemAdapter = HiveMemAdapter
