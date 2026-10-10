"""Memory construction: event loading, resumable builds, and command entry point."""
from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import threading
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np

from embedding.backends import OpenAIMemoryEmbedder, create_memory_embedder
from .utils import (
    DatasetLayout,
    PROJECT_ROOT,
    RunLayout,
    atomic_binary_writer,
    load_runtime_config,
    output_root,
    resolved_value,
    validate_embedding_settings,
    write_json_atomic,
    write_text_atomic,
)
from .memory import (
    MemoryExecutor,
    normalize_visual_input,
    visual_input_uses_images,
    MemoryBank,
    iter_node_attributes,
    serialize_attribute,
    EXECUTOR_PROMPT_SCHEMA_VERSION,
    EXECUTOR_VISUAL_INPUTS,
    MEMORY_RESPONSE_FORMAT,
)


# Executor model client

@dataclass(frozen=True)
class GenerationResponse:
    text: str
    usage: dict[str, int]
    attempts: int = 1
    failed_attempts: int = 0

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> str:
        raise NotImplementedError


class LLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        timeout: int = 60,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str = "",
    ):
        keys = _normalize_api_keys(api_key)
        if not keys:
            raise ValueError("api=True requires at least one api_key.")

        self.model = model
        self.api_base = api_base
        self.api_keys = keys
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = retry_sleep
        self.timeout = timeout
        self.response_format = response_format
        self.reasoning_effort = str(reasoning_effort).strip()
        self._client_cache = {}
        self._lock = threading.Lock()
        self._key_index = 0

    def generate(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> str:
        return self.generate_with_usage(prompt, image_paths=image_paths).text

    def generate_with_usage(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> GenerationResponse:
        user_content = _build_user_content(prompt, image_paths)
        last_error = None
        cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage_is_exact = True
        for attempt in range(self.max_retries):
            client = self._next_client()
            try:
                request = dict(
                    model=self.model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                )
                if self.response_format is not None:
                    request["response_format"] = self.response_format
                if self.reasoning_effort:
                    request["extra_body"] = {
                        "reasoning": {"effort": self.reasoning_effort}
                    }
                completion = client.chat.completions.create(**request)
                message = completion.choices[0].message.content
                usage = _normalize_usage(completion.usage)
                if usage:
                    cumulative_usage = _sum_usage(cumulative_usage, usage)
                else:
                    usage_is_exact = False
                return GenerationResponse(
                    text=(message or "").strip(),
                    usage=cumulative_usage if usage_is_exact else {},
                    attempts=attempt + 1,
                    failed_attempts=attempt,
                )
            except Exception as exc:
                last_error = exc
                # Provider usage is unavailable for an exception, so an
                # eventual success cannot claim exact cumulative token cost.
                usage_is_exact = False
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_sleep)
        raise RuntimeError(f"API generation failed after {self.max_retries} attempts: {last_error}")

    def _next_client(self):
        with self._lock:
            api_key = self.api_keys[self._key_index % len(self.api_keys)]
            self._key_index += 1
            client = self._client_cache.get(api_key)
            if client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "API mode requires the 'openai' package to be installed."
                    ) from exc
                client = OpenAI(
                    base_url=self.api_base,
                    api_key=api_key,
                    # Keep all retries in generate_with_usage so every actual
                    # API invocation is visible to Calls metrics.
                    max_retries=0,
                    timeout=self.timeout,
                )
                self._client_cache[api_key] = client
            return client


def _normalize_usage(usage: Any) -> dict[str, int]:
    """Return the three build-token counters used by result metrics."""
    if usage is None:
        return {}

    def value(name: str) -> int:
        raw = usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)
        return int(raw or 0)

    prompt_tokens = value("prompt_tokens")
    completion_tokens = value("completion_tokens")
    total_tokens = value("total_tokens") or prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _sum_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        key: int(left.get(key) or 0) + int(right.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _normalize_api_keys(api_key) -> List[str]:
    if api_key is None:
        return []
    if isinstance(api_key, str):
        key = api_key.strip()
        return [key] if key else []
    if isinstance(api_key, Sequence):
        return [str(key).strip() for key in api_key if str(key).strip()]
    raise ValueError("api_key must be a string or a list of strings.")


def _build_user_content(
    prompt: str,
    image_paths: Sequence[str] | None,
) -> str | list[dict[str, Any]]:
    """Build an OpenAI-compatible user message without altering text-only calls."""
    paths = [Path(path) for path in (image_paths or []) if str(path).strip()]
    if not paths:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(path)},
        }
        for path in paths
    )
    return content


def _encode_image_data_url(path: Path) -> str:
    """Encode the source image bytes directly; build-time input is not resized."""
    if not path.is_file():
        raise FileNotFoundError(f"Build image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# Memory build orchestration

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
        return [resolved_value(json.loads(line)) for line in handle if line.strip()]


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



class MemoryEpisodeBuilder:
    """Construct memory episodes from interaction chunks and persist their memory bank."""

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
        event_ids = [str(event.source_chunk_id) for event in all_events]
        if any(not identifier.strip() for identifier in event_ids) or len(set(event_ids)) != len(event_ids):
            raise ValueError("Build events require nonempty, unique source_chunk_id values")
        if max_events < 0:
            raise ValueError("max_events cannot be negative")
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
            bank, state = MemoryBank.load_checkpoint(checkpoint_dir)
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
            for event, memory in zip(event_list[:start_index], bank.memories):
                if memory.id != event.source_chunk_id and event.source_chunk_id not in memory.metadata.get("source_chunk_ids", []):
                    raise ValueError("Checkpoint memory IDs do not match the committed input events")
            _truncate_trace(trace_path, start_index)
        else:
            bank = MemoryBank()
            if not resume:
                # An explicit restart discards the old commit pointer without
                # reading a possibly corrupt generation. Publish and validate
                # the first new generation before pruning the abandoned ones.
                state_path.unlink(missing_ok=True)
                if checkpoint_dir.is_dir():
                    _sync_directory(checkpoint_dir)
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
                    raw_chunk=event.text,
                    node_id=event.source_chunk_id,
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
                        raw_chunk=event.text,
                        memory_id=event.source_chunk_id or None,
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
                    _save_builder_checkpoint(
                        bank,
                        checkpoint_dir,
                        {
                            "next_event_index": event_index + 1,
                            "executor_visual_input": executor_visual_input,
                            "signature": checkpoint_signature,
                        },
                        trace_path=trace_path,
                    )
        finally:
            if pool is not None:
                for future in futures.values():
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)

        bank.save(output_dir)
        attributes = sorted(
            {
                attribute
                for memory in bank.memories
                for values in (memory.textual_anchors, memory.visual_anchors)
                for attribute in iter_node_attributes(values)
            }
        )
        attribute_texts = [serialize_attribute(attribute) for attribute in attributes]
        if attribute_texts:
            attribute_vectors = np.asarray(
                self.embedder.embed_texts(attribute_texts, mode="context"),
                dtype=np.float32,
            )
            if attribute_vectors.ndim == 1:
                attribute_vectors = attribute_vectors.reshape(1, -1)
        else:
            embedding_dim = (
                int(bank.memories[0].embedding.size) if bank.memories else 0
            )
            attribute_vectors = np.zeros((0, embedding_dim), dtype=np.float32)
        if attribute_vectors.shape[0] != len(attributes):
            raise ValueError(
                "Attribute/vector count mismatch: "
                f"{len(attributes)} vs {attribute_vectors.shape[0]}"
            )
        write_json_atomic(
            output_layout.attributes,
            [
                {"attribute": key, "value": value, "text": text}
                for (key, value), text in zip(attributes, attribute_texts)
            ],
        )
        with atomic_binary_writer(output_layout.attribute_vectors) as handle:
            np.save(handle, attribute_vectors)
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
            with atomic_binary_writer(output_layout.image_vectors) as handle:
                np.save(handle, image_vectors)
            with atomic_binary_writer(output_layout.image_mask) as handle:
                np.save(handle, image_mask)
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
            "unique_attributes": len(attributes),
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
    # Historical checkpoints did not record a revision; an explicit revision
    # must still reject them instead of claiming that their weights were pinned.
    left.setdefault("embedding_revision", "")
    right.setdefault("embedding_revision", "")
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
        if keep_before_index:
            raise ValueError("Checkpoint trace is missing committed events")
        return
    rows = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) == keep_before_index:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Checkpoint trace is corrupt within its committed event range") from error
            if not isinstance(row, dict) or row.get("event_index") != len(rows):
                raise ValueError("Checkpoint trace does not contain the committed events in order")
            rows.append(row)
    if len(rows) != keep_before_index:
        raise ValueError("Checkpoint trace is missing committed events")
    write_text_atomic(trace_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_checkpoint_generation(bank: MemoryBank, checkpoint_dir: Path, state: dict[str, Any]) -> str:
    staging = Path(tempfile.mkdtemp(prefix=".generation-", dir=checkpoint_dir))
    generation = f"generation_{uuid4().hex}"
    destination = checkpoint_dir / generation
    try:
        bank.save(staging)
        write_json_atomic(staging / "builder_state.json", state)
        MemoryBank.load_checkpoint(staging)
        _sync_directory(staging / "vectors")
        _sync_directory(staging)
        os.replace(staging, destination)
        _sync_directory(checkpoint_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return generation


def _save_builder_checkpoint(
    bank: MemoryBank,
    checkpoint_dir: Path,
    state: dict[str, Any],
    *,
    trace_path: Path,
) -> None:
    """Publish a complete immutable checkpoint before replacing its commit pointer."""
    if state["next_event_index"] != len(bank):
        raise ValueError("Checkpoint event boundary does not match its memory count")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    if (checkpoint_dir / "builder_state.json").is_file():
        old_bank, old_state = MemoryBank.load_checkpoint(checkpoint_dir)
        previous = old_state.get("generation")
        if previous is None:
            previous = _publish_checkpoint_generation(old_bank, checkpoint_dir, old_state)
    with trace_path.open("rb") as trace:
        os.fsync(trace.fileno())
    generation = _publish_checkpoint_generation(bank, checkpoint_dir, state)
    write_json_atomic(
        checkpoint_dir / "builder_state.json",
        {**state, "generation": generation, "previous_generation": previous},
    )
    _sync_directory(checkpoint_dir)
    # Never delete either committed generation when a save fails before this point.
    keep = {generation, previous}
    for directory in checkpoint_dir.glob("generation_*"):
        if directory.is_dir() and directory.name not in keep:
            shutil.rmtree(directory, ignore_errors=True)


# Memory build command

def apply_config_defaults(
    parser: argparse.ArgumentParser,
    *,
    allowed_keys: set[str] | None = None,
) -> None:
    """Overlay the packaged runtime configuration onto argparse defaults.
    CLI arguments still take precedence; unknown keys are ignored."""
    from .utils import load_runtime_config
    config = load_runtime_config()
    known = {action.dest for action in parser._actions}
    if allowed_keys is not None:
        known &= allowed_keys
    parser.set_defaults(
        **{k: v for k, v in config.items() if not k.startswith("_") and k in known}
    )


def completed_dataset_stats(
    dataset_layout: DatasetLayout,
    checkpoint_dir: Path,
    *,
    expected_events: int,
    expected_dim: int,
    expected_executor_visual_input: str = "image",
    expected_signature: dict | None = None,
) -> dict | None:
    """Return stats only for a complete build made with compatible settings."""
    if (checkpoint_dir / "builder_state.json").exists():
        return None
    memories_path = dataset_layout.root / "memories.jsonl"
    if not dataset_layout.build_stats.exists() or not memories_path.exists():
        return None
    try:
        stats = json.loads(dataset_layout.build_stats.read_text(encoding="utf-8"))
        if not isinstance(stats, dict):
            return None
        bank = MemoryBank.load(dataset_layout.root)
        memory_count = len(bank)
        vectors = np.load(dataset_layout.text_vectors, mmap_mode="r", allow_pickle=False)
        attributes = json.loads(dataset_layout.attributes.read_text(encoding="utf-8"))
        attribute_vectors = np.load(dataset_layout.attribute_vectors, mmap_mode="r", allow_pickle=False)
        image_vectors = np.load(dataset_layout.image_vectors, mmap_mode="r", allow_pickle=False)
        image_mask = np.load(dataset_layout.image_mask, mmap_mode="r", allow_pickle=False)
        if not isinstance(attributes, list) or any(not isinstance(row, dict) for row in attributes):
            return None
        if vectors.shape != (memory_count, int(expected_dim)):
            return None
        if attribute_vectors.shape != (len(attributes), int(expected_dim)):
            return None
        if image_vectors.shape != vectors.shape or image_mask.shape != (memory_count,) or image_mask.dtype != np.bool_:
            return None
        if any(not np.isfinite(matrix).all() for matrix in (vectors, attribute_vectors, image_vectors)):
            return None
        if int(stats.get("input_events", -1)) != int(expected_events) or int(stats.get("final_memories", -1)) != memory_count:
            return None
        # Builds made before this field existed used captions and no raw images.
        if str(stats.get("executor_visual_input") or "caption") != expected_executor_visual_input:
            return None
        if expected_signature is not None and not build_signatures_compatible(stats.get("build_signature"), expected_signature):
            return None
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return {**stats, "skipped_complete": True}


def build_signature(args: argparse.Namespace, dataset: str, events, profile: str) -> dict:
    event_payload = json.dumps(
        [event.to_dict() for event in events],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 2,
        "executor_prompt_schema_version": EXECUTOR_PROMPT_SCHEMA_VERSION,
        "dataset": dataset,
        "events_sha256": hashlib.sha256(event_payload).hexdigest(),
        "profile_sha256": hashlib.sha256(profile.encode("utf-8")).hexdigest(),
        "mode": args.mode,
        "executor_model": args.executor_model,
        "executor_base_url": args.executor_base_url,
        "executor_max_tokens": args.executor_max_tokens,
        "executor_reasoning_effort": args.executor_reasoning_effort,
        "executor_visual_input": args.executor_visual_input,
        "embedding_model": args.embedding_model,
        "embedding_revision": getattr(args, "embedding_revision", ""),
        "embedding_base_url": args.embedding_base_url,
        "embedding_dim": args.embedding_dim,
        "dtype": args.dtype,
        "benchmark": args.benchmark,
        "build_image_vectors": True,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal HiVe_mem memories.")
    parser.add_argument("--benchmark", choices=("memgallery", "h2hmem", "wma"), default="memgallery")
    parser.add_argument("--mode", default="c", choices=["c"], help="Multimodal memory construction.")
    parser.add_argument("--chunks", default="")
    parser.add_argument("--dataset", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument(
        "--output-root",
        default="",
        help="Run directory; datasets are written directly under <run>/datasets.",
    )
    parser.add_argument("--executor-model", default="")
    parser.add_argument("--executor-base-url", default="")
    parser.add_argument("--executor-api-key", default="EMPTY")
    parser.add_argument("--executor-api-key-env", default="")
    parser.add_argument("--executor-reasoning-effort", default="")
    parser.add_argument("--executor-max-tokens", type=int, default=512)
    parser.add_argument("--executor-timeout", type=int, default=180)
    parser.add_argument("--executor-retries", type=int, default=2)
    parser.add_argument(
        "--executor-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent executor LLM requests (results commit in event order).",
    )
    parser.add_argument(
        "--executor-visual-input",
        choices=EXECUTOR_VISUAL_INPUTS,
        default="image",
        help="Use original images when extracting multimodal memory attributes.",
    )
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-revision", default="", help="Optional Hugging Face revision for locally loaded embedding weights.")
    embedding_backend = parser.add_mutually_exclusive_group()
    embedding_backend.add_argument("--embedding-base-url", default="")
    embedding_backend.add_argument(
        "--local-embedding", action="store_true",
        help="Load the embedding model locally instead of using an API service.",
    )
    parser.add_argument("--embedding-api-key", default="EMPTY")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--profiles-file",
        default="",
        help="JSON file mapping dataset name -> profile_summary text. When set, the "
        "profile is injected into the executor prompt per dataset (use with "
        "profile-free chunks).",
    )
    apply_config_defaults(parser, allowed_keys={action.dest for action in parser._actions} - {"output_root"})
    args = parser.parse_args()
    args.executor_base_url = args.executor_base_url.strip()
    args.embedding_base_url = args.embedding_base_url.strip()
    if not args.executor_model or not args.executor_base_url:
        parser.error("Set HIVE_EXECUTOR_MODEL/HIVE_EXECUTOR_BASE_URL or pass --executor-model/--executor-base-url.")
    if args.local_embedding:
        args.embedding_base_url = ""
    elif not args.embedding_base_url:
        parser.error("Set HIVE_EMBEDDING_BASE_URL or pass --embedding-base-url; use --local-embedding to load the model locally.")
    args.embedding_model, args.embedding_dim = validate_embedding_settings(
        parser, args.embedding_model, args.embedding_dim
    )
    args.embedding_revision = str(args.embedding_revision or "").strip()

    profiles: dict[str, str] = {}
    if args.benchmark == "memgallery" and args.profiles_file:
        profiles = json.loads(Path(args.profiles_file).read_text(encoding="utf-8"))

    runtime = load_runtime_config()
    if args.chunks:
        chunk_paths = [args.chunks]
    else:
        keys = {
            "memgallery": ("memgallery_chunks_file",),
            "h2hmem": ("h2hmem_dyadic_chunks_file", "h2hmem_multiparty_chunks_file"),
            "wma": ("wma_lifelong_chunks_file",),
        }[args.benchmark]
        chunk_paths = [runtime[key] for key in keys]
    if args.benchmark != "memgallery":
        profiles = {}
        explicit_dataset = any(value == "--dataset" or value.startswith("--dataset=") for value in sys.argv[1:])
        if not explicit_dataset:
            args.all_datasets = True
    args.output_root = args.output_root or str(output_root() / "memory" / args.benchmark)
    layout = RunLayout.from_path(args.output_root)
    all_events = [event for path in chunk_paths for event in load_events(path)]
    all_events.sort(key=lambda item: (item.dataset, _session_key(item.session_id), item.round_id, item.source_chunk_id))
    datasets = sorted({event.dataset for event in all_events}) if args.all_datasets else [args.dataset]
    from .utils import api_key_for

    executor_api_key = api_key_for("executor", args.executor_api_key)
    if args.executor_api_key_env:
        executor_api_key = os.environ.get(args.executor_api_key_env, "").strip()
        if not executor_api_key:
            parser.error(
                f"Environment variable {args.executor_api_key_env!r} is empty"
            )
    llm_client = LLMClient(
        model=args.executor_model,
        api_base=args.executor_base_url,
        api_key=executor_api_key,
        temperature=0.0,
        max_new_tokens=args.executor_max_tokens,
        max_retries=args.executor_retries + 1,
        timeout=args.executor_timeout,
        response_format=MEMORY_RESPONSE_FORMAT,
        reasoning_effort=args.executor_reasoning_effort,
    )
    if args.embedding_base_url:
        embedder = OpenAIMemoryEmbedder(
            base_url=args.embedding_base_url,
            model_name=args.embedding_model,
            expected_dim=args.embedding_dim,
            api_key=api_key_for("embedding", args.embedding_api_key),
            timeout=args.executor_timeout,
        )
    else:
        embedder = create_memory_embedder(
            model_name=args.embedding_model,
            device=args.device,
            expected_dim=args.embedding_dim,
            dtype=args.dtype,
            revision=args.embedding_revision or None,
        )
    if not embedder.supports_images:
        raise ValueError("Multimodal HiVe_mem construction requires an image-capable embedding model")
    builder = MemoryEpisodeBuilder(llm_client, embedder)
    summaries = {}
    for dataset in datasets:
        events = [event for event in all_events if event.dataset == dataset]
        if not events:
            raise ValueError(f"No input events found for dataset {dataset!r}")
        profile = profiles.get(dataset, "")
        signature = build_signature(args, dataset, events, profile)
        dataset_layout = layout.dataset(dataset)
        output_dir = dataset_layout.root
        if not args.no_resume and not args.max_events:
            existing = completed_dataset_stats(
                dataset_layout,
                layout.checkpoint(dataset),
                expected_events=len(events),
                expected_dim=args.embedding_dim,
                expected_executor_visual_input=args.executor_visual_input,
                expected_signature=signature,
            )
            if existing is not None:
                summaries[dataset] = existing
                print(json.dumps({dataset: existing}, ensure_ascii=False))
                continue
        summaries[dataset] = builder.build(
            events,
            output_dir,
            checkpoint_dir=layout.checkpoint(dataset),
            resume=not args.no_resume,
            checkpoint_every=args.checkpoint_every,
            max_events=args.max_events,
            build_image_vectors=True,
            profile=profile,
            executor_visual_input=args.executor_visual_input,
            executor_concurrency=args.executor_concurrency,
            checkpoint_signature=signature,
        )
        print(json.dumps({dataset: summaries[dataset]}, ensure_ascii=False))
    manifest_path = layout.build_manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    public_manifest = {
        key: value
        for key, value in vars(args).items()
        if key not in {"executor_api_key", "embedding_api_key"}
    }
    public_manifest["chunks"] = [str(Path(path).resolve()) for path in chunk_paths]
    write_json_atomic(manifest_path, public_manifest)
    # H2HMEM builds dyadic and multiparty banks into the same root. Preserve
    # both build configurations instead of letting the second overwrite the
    # only audit record.
    for chunks_path in chunk_paths:
        write_json_atomic(
            layout.root / f"build_manifest.{Path(chunks_path).stem}.json",
            public_manifest,
        )


if __name__ == "__main__":
    main()
