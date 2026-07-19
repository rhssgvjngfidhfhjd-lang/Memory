from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from simple_memory_pipeline.executor import MemoryExecutor
from simple_memory_pipeline.memory_bank import MemoryBank
from simple_memory_pipeline.operations import get_default_operations

from .input_loader import MemoryEvent


class MemGalleryMemoryBuilder:
    def __init__(self, backend, embedder, top_k: int = 5, operations=None):
        self.executor = MemoryExecutor(backend, embedder)
        self.embedder = embedder
        self.operations = list(operations) if operations else get_default_operations()
        self.top_k = int(top_k)
        allowed = {operation.update_type for operation in self.operations}
        self.allow_insert = "insert" in allowed
        self.allow_noop = "noop" in allowed

    def build(
        self,
        events: Iterable[MemoryEvent],
        output_dir: str | Path,
        *,
        resume: bool = True,
        checkpoint_every: int = 1,
        max_events: int = 0,
        build_image_vectors: bool = False,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoint"
        trace_path = output_dir / "build_trace.jsonl"
        state_path = checkpoint_dir / "builder_state.json"
        event_list = list(events)
        if max_events:
            event_list = event_list[:max_events]

        start_index = 0
        if resume and state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            start_index = int(state.get("next_event_index", 0))
            bank = MemoryBank.load(checkpoint_dir)
            _truncate_trace(trace_path, start_index)
        else:
            bank = MemoryBank()
            if trace_path.exists():
                trace_path.unlink()

        action_counts = Counter()
        parse_failures = 0
        fallback_inserts = 0
        started = time.time()
        for event_index in range(start_index, len(event_list)):
            event = event_list[event_index]
            query_embedding = self.embedder.embed_texts(event.text, mode="query")
            retrieved_memories, retrieved_indices = bank.retrieve(query_embedding, top_k=self.top_k)
            retrieved_memory_ids = [bank.memories[index].memory_id for index in retrieved_indices]
            raw_response, actions = self.executor.execute(
                operations=self.operations,
                chunk_text=event.text,
                retrieved_memories=retrieved_memories,
            )
            self.executor.apply_to_memory_bank(
                actions,
                bank,
                retrieved_indices,
                event_metadata=event.metadata,
            )
            used_fallback = False
            if (
                self.allow_insert
                and not self.allow_noop
                and not any(action.success for action in actions)
            ):
                embedding = self.embedder.embed_texts(event.text, mode="context")
                bank.add_memory(
                    event.text,
                    embedding,
                    metadata={**event.metadata, "source": "fallback_insert"},
                )
                fallback_inserts += 1
                used_fallback = True
            bank.step()
            for action in actions:
                action_counts[action.action_type] += 1
                if not action.success:
                    parse_failures += 1
            trace = {
                "event_index": event_index,
                "event": event.to_dict(),
                "retrieved_memory_ids": retrieved_memory_ids,
                "raw_response": raw_response,
                "actions": [action.to_dict() for action in actions],
                "fallback_insert": used_fallback,
                "memory_count_after": len(bank),
            }
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if (event_index + 1) % checkpoint_every == 0 or event_index + 1 == len(event_list):
                bank.save(checkpoint_dir)
                state_path.write_text(
                    json.dumps({"next_event_index": event_index + 1}, indent=2),
                    encoding="utf-8",
                )

        bank.save(output_dir)
        image_vector_count = 0
        if build_image_vectors:
            image_vectors = np.zeros((len(bank), self.embedder.expected_dim), dtype=np.float32)
            image_mask = np.zeros(len(bank), dtype=np.bool_)
            for index, memory in enumerate(bank.memories):
                paths = memory.metadata.get("image_paths", [])
                if not paths:
                    continue
                vectors = self.embedder.embed_images(paths)
                if len(vectors):
                    vector = vectors.mean(axis=0)
                    image_vectors[index] = vector / (np.linalg.norm(vector) + 1e-8)
                    image_mask[index] = True
                    image_vector_count += 1
            np.save(output_dir / "image_vectors.npy", image_vectors)
            np.save(output_dir / "image_mask.npy", image_mask)
        stats = {
            "input_events": len(event_list),
            "processed_this_run": max(0, len(event_list) - start_index),
            "final_memories": len(bank),
            "compression_ratio": len(bank) / len(event_list) if event_list else 0.0,
            "action_counts_this_run": dict(action_counts),
            "parse_failures_this_run": parse_failures,
            "fallback_inserts_this_run": fallback_inserts,
            "enabled_operations": sorted({operation.update_type for operation in self.operations}),
            "elapsed_seconds_this_run": time.time() - started,
            "image_vector_memories": image_vector_count,
        }
        (output_dir / "build_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return stats


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
