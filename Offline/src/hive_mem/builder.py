from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from benchmarks.io_utils import write_json_atomic
from hive_mem.executor import (
    MemoryExecutor,
    normalize_visual_input,
    visual_input_uses_images,
)
from hive_mem.mau import MAUBank
from hive_mem.output_layout import DatasetLayout
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MemoryEvent:
    text: str
    dataset: str
    dialogue_id: str
    session_id: str
    round_id: int
    source_chunk_id: str
    date: str = ""
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)
    # Multi-round chunks (token-packed): all covered dialogue/chunk ids for
    # retrieval attribution; empty -> single-round chunk.
    dialogue_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "dialogue_id": self.dialogue_id,
            "date": self.date,
            "source_dialogue_ids": self.dialogue_ids or [self.dialogue_id],
            "source_chunk_ids": self.chunk_ids or [self.source_chunk_id],
            "image_id": self.image_ids[0] if self.image_ids else "",
            "image_ids": self.image_ids,
            "image_paths": self.image_paths,
            "image_captions": self.image_captions,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_events(path: str | Path, dataset: str | None = None) -> list[MemoryEvent]:
    events = []
    for row in read_jsonl(path):
        metadata = dict(row.get("metadata") or {})
        event_dataset = str(metadata.get("dataset", ""))
        if dataset and event_dataset != dataset:
            continue
        events.append(
            MemoryEvent(
                text=str(row.get("text", "")).strip(),
                dataset=event_dataset,
                dialogue_id=str(metadata.get("dialogue_id", "")),
                session_id=str(metadata.get("session_id", "")),
                round_id=int(metadata.get("round_id", 0)),
                source_chunk_id=str(row.get("chunk_id", "")),
                date=str(metadata.get("date") or metadata.get("timestamp") or ""),
                image_ids=_strings(metadata.get("image_ids") or [metadata.get("image_id", "")]),
                image_paths=_strings(row.get("images") or []),
                image_captions=_strings(
                    metadata.get("image_captions") or [metadata.get("image_caption", "")]
                ),
                dialogue_ids=_strings(metadata.get("source_dialogue_ids") or []),
                chunk_ids=_strings(metadata.get("source_chunk_ids") or []),
            )
        )
    events.sort(key=lambda item: (item.dataset, _session_key(item.session_id), item.round_id, item.source_chunk_id))
    return events


def _strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value and str(value)))


def _session_key(session_id: str) -> tuple[int, str]:
    digits = "".join(character for character in session_id if character.isdigit())
    return (int(digits) if digits else 0, session_id)



class MAUBuilder:
    def __init__(self, llm_client, embedder):
        self.executor = MemoryExecutor(llm_client, embedder)
        self.embedder = embedder

    def build(
        self,
        events: Iterable[MemoryEvent],
        output_dir: str | Path,
        *,
        checkpoint_dir: str | Path | None = None,
        resume: bool = True,
        checkpoint_every: int = 1,
        max_events: int = 0,
        build_image_vectors: bool = False,
        profile: str = "",
        executor_visual_input: str = "image",
        executor_concurrency: int = 1,
        checkpoint_signature: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        executor_visual_input = normalize_visual_input(executor_visual_input)
        executor_concurrency = int(executor_concurrency)
        if executor_concurrency < 1:
            raise ValueError("executor_concurrency must be at least 1")
        checkpoint_every = int(checkpoint_every)
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        checkpoint_signature = dict(checkpoint_signature or {})
        all_events = list(events)
        event_list = all_events[:max_events] if max_events else all_events
        output_layout = DatasetLayout(Path(output_dir))
        output_dir = output_layout.root
        output_dir.mkdir(parents=True, exist_ok=True)
        output_layout.reports_dir.mkdir(parents=True, exist_ok=True)
        output_layout.traces_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else output_dir / ".checkpoint"
        trace_path = output_layout.build_trace
        state_path = checkpoint_dir / "builder_state.json"

        start_index = 0
        if resume and state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            checkpoint_visual_input = str(
                state.get("executor_visual_input") or "caption"
            )
            if checkpoint_visual_input != executor_visual_input:
                raise ValueError(
                    "Checkpoint executor visual input mismatch: "
                    f"{checkpoint_visual_input!r} != {executor_visual_input!r}. "
                    "Use the original mode or restart with --no-resume."
                )
            if not build_signatures_compatible(
                state.get("signature"), checkpoint_signature
            ):
                raise ValueError(
                    "Checkpoint build signature does not match the current inputs/config. "
                    "Use the original settings or restart with --no-resume."
                )
            start_index = int(state.get("next_event_index", 0))
            if start_index < 0 or start_index > len(event_list):
                raise ValueError(
                    f"Checkpoint next_event_index {start_index} is outside "
                    f"the current event range 0..{len(event_list)}"
                )
            bank = MAUBank.load(checkpoint_dir)
            _truncate_trace(trace_path, start_index)
        else:
            bank = MAUBank()
            if trace_path.exists():
                trace_path.unlink()

        memory_items = 0
        parse_failures = 0
        fallback_inserts = 0
        executor_image_requests = 0
        started = time.time()
        pool: ThreadPoolExecutor | None = None
        futures: dict[int, Future] = {}
        try:
            if executor_concurrency > 1:
                pool = ThreadPoolExecutor(
                    max_workers=executor_concurrency,
                    thread_name_prefix="memory-executor",
                )
                futures = {
                    event_index: pool.submit(
                        self._execute_event,
                        event_list[event_index],
                        profile=profile,
                        executor_visual_input=executor_visual_input,
                    )
                    for event_index in range(start_index, len(event_list))
                }

            # Executor calls may finish out of order, but every state mutation is
            # committed in event order. This keeps memories, vectors, traces and
            # resume checkpoints aligned and reproducible.
            for event_index in range(start_index, len(event_list)):
                event = event_list[event_index]
                if pool is None:
                    executor_images, raw_response, actions, llm_usage, llm_call_stats = self._execute_event(
                        event,
                        profile=profile,
                        executor_visual_input=executor_visual_input,
                    )
                else:
                    executor_images, raw_response, actions, llm_usage, llm_call_stats = futures[
                        event_index
                    ].result()
                executor_image_requests += int(bool(executor_images))
                self.executor.apply_to_memory_bank(
                    actions,
                    bank,
                    event_metadata=event.metadata,
                )
                used_fallback = False
                if not any(action.success for action in actions):
                    fallback_text = self.executor.prepare_chunk_text(
                        event.text,
                        executor_visual_input,
                    )
                    embedding = self.embedder.embed_texts(fallback_text, mode="context")
                    bank.add_memory(
                        fallback_text,
                        embedding,
                        metadata={**event.metadata, "source": "fallback_insert"},
                    )
                    fallback_inserts += 1
                    used_fallback = True
                for action in actions:
                    if action.success:
                        memory_items += 1
                    else:
                        parse_failures += 1
                trace = {
                    "event_index": event_index,
                    "event": event.to_dict(),

                    "raw_response": raw_response,
                    "actions": [action.to_dict() for action in actions],
                    "executor_visual_input": executor_visual_input,
                    "executor_image_count": len(executor_images),
                    "fallback_insert": used_fallback,
                    "memory_count_after": len(bank),
                }
                if llm_usage:
                    trace["llm_usage"] = llm_usage
                if llm_call_stats:
                    trace["llm_attempts"] = llm_call_stats["attempts"]
                    trace["llm_failed_attempts"] = llm_call_stats["failed_attempts"]
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                if (event_index + 1) % checkpoint_every == 0 or event_index + 1 == len(event_list):
                    bank.save(checkpoint_dir)
                    write_json_atomic(
                        state_path,
                        {
                            "next_event_index": event_index + 1,
                            "executor_visual_input": executor_visual_input,
                            "signature": checkpoint_signature,
                        },
                    )
        finally:
            if pool is not None:
                for future in futures.values():
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)

        bank.save(output_dir)
        image_vector_count = 0
        if build_image_vectors:
            output_layout.vectors_dir.mkdir(parents=True, exist_ok=True)
            image_vectors = np.zeros((len(bank), self.embedder.expected_dim), dtype=np.float32)
            image_mask = np.zeros(len(bank), dtype=np.bool_)
            # Several memories extracted from the same dialogue round retain the
            # same image paths. Encode each distinct image set once and reuse the
            # normalized vector instead of repeating identical GPU work.
            image_vector_cache: dict[tuple[str, ...], np.ndarray] = {}
            for index, memory in enumerate(bank.memories):
                paths = memory.metadata.get("image_paths", [])
                if not paths:
                    continue
                cache_key = tuple(str(path) for path in paths)
                vector = image_vector_cache.get(cache_key)
                if vector is None:
                    vectors = self.embedder.embed_images(paths)
                    if not len(vectors):
                        continue
                    vector = vectors.mean(axis=0)
                    vector = vector / (np.linalg.norm(vector) + 1e-8)
                    image_vector_cache[cache_key] = vector
                image_vectors[index] = vector
                image_mask[index] = True
                image_vector_count += 1
            np.save(output_layout.image_vectors, image_vectors)
            np.save(output_layout.image_mask, image_mask)
        stats = {
            "input_events": len(event_list),
            "processed_this_run": max(0, len(event_list) - start_index),
            "final_memories": len(bank),
            "compression_ratio": len(bank) / len(event_list) if event_list else 0.0,
            "memory_items_this_run": memory_items,
            "parse_failures_this_run": parse_failures,
            "fallback_inserts_this_run": fallback_inserts,
            "executor_visual_input": executor_visual_input,
            "executor_concurrency": executor_concurrency,
            "executor_image_requests_this_run": executor_image_requests,
            "elapsed_seconds_this_run": time.time() - started,
            "image_vector_memories": image_vector_count,
            "build_signature": checkpoint_signature,
        }
        write_json_atomic(output_layout.build_stats, stats)
        # A completed build no longer needs its duplicate checkpoint copy.
        # Crashed and deliberately partial (--max-events) builds retain it.
        completed_all_events = not max_events or max_events >= len(all_events)
        if completed_all_events and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            if checkpoint_dir.parent.name == ".checkpoints":
                try:
                    checkpoint_dir.parent.rmdir()
                except OSError:
                    pass
        return stats
    def _execute_event(
        self,
        event: MemoryEvent,
        *,
        profile: str,
        executor_visual_input: str,
    ):
        """Run the stateless executor step; callers serialize all bank mutations."""
        executor_images = (
            event.image_paths
            if visual_input_uses_images(executor_visual_input)
            else []
        )
        raw_response, actions, llm_usage, llm_call_stats = self.executor.execute_with_usage(
            chunk_text=event.text,
            profile=profile,
            image_paths=executor_images,
            visual_input=executor_visual_input,
        )
        return executor_images, raw_response, actions, llm_usage, llm_call_stats


def build_signatures_compatible(
    stored: dict[str, Any] | None, current: dict[str, Any] | None
) -> bool:
    """Allow resume when only the port of an equivalent loopback service moved."""
    left = dict(stored or {})
    right = dict(current or {})
    endpoints = ("executor_base_url", "embedding_base_url")
    endpoint_pairs = [(left.pop(key, ""), right.pop(key, "")) for key in endpoints]
    return left == right and all(
        _equivalent_loopback_endpoint(old, new) for old, new in endpoint_pairs
    )


def _equivalent_loopback_endpoint(left: Any, right: Any) -> bool:
    first = str(left or "").rstrip("/")
    second = str(right or "").rstrip("/")
    if first == second:
        return True
    if not first or not second:
        return False
    first_url = urlparse(first)
    second_url = urlparse(second)
    loopback = {"localhost", "127.0.0.1", "::1"}
    return (
        first_url.scheme == second_url.scheme
        and (first_url.hostname or "").lower() in loopback
        and (second_url.hostname or "").lower() in loopback
        and first_url.path.rstrip("/") == second_url.path.rstrip("/")
    )


def _truncate_trace(trace_path: Path, keep_before_index: int) -> None:
    if not trace_path.exists():
        return
    rows = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event_index", 0) < keep_before_index:
                rows.append(row)
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
