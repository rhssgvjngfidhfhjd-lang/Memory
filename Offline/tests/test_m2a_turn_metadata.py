from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from embedding.chunk_builder import (
    Chunk,
    batch_wma_rounds_for_m2a,
    balance_wma_chunks,
    build_chunks_from_data,
    build_h2h_chunks_from_data,
    build_wma_chunks_from_data,
)
from benchmarks.baseline_runtime.adapters.m2a import M2AAdapter


def test_memgallery_m2a_turns_preserve_utterances_and_assign_round_input_to_user(
    tmp_path: Path,
) -> None:
    payload = {
        "character_profile": {"name": "Alice"},
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2025-01-01",
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "original  user\ntext",
                        "assistant": "original assistant text",
                        "input_image": ["../image/sample/one.png"],
                        "image_id": ["D1:IMG_001"],
                        "image_caption": ["caption"],
                    }
                ],
            }
        ],
    }

    chunk = build_chunks_from_data(payload, tmp_path, "sample")[0]
    turns = chunk.metadata["m2a_turns"]

    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert [turn["speaker"] for turn in turns] == ["Alice", "assistant"]
    assert turns[0]["text"] == "original  user\ntext"
    assert turns[1]["text"] == "original assistant text"
    assert turns[0]["images"] == [str((tmp_path / "image/sample/one.png").resolve())]
    assert turns[1]["images"] == []
    assert chunk.metadata["m2a_speakers"] == ["Alice", "assistant"]
    assert chunk.metadata["image_scope"] == "round_input"
    assert chunk.metadata["image_assignment"] == "user"


def test_wma_m2a_turns_keep_each_messages_own_timestamp_and_attachments(
    tmp_path: Path,
) -> None:
    sample_path = tmp_path / "lifelong" / "personal" / "sample.json"
    payload = {
        "sample_id": "sample",
        "sessions": [
            {
                "_v2_session_id": "S00",
                "dialogue": [
                    {
                        "role": "user",
                        "content": "user text",
                        "timestamp": "2025-01-01 10:00",
                        "attachments": [
                            {"file_path": "images/user.png", "image_id": "u"}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "assistant text",
                        "timestamp": "2025-01-01 10:01",
                        "attachments": [
                            {"file_path": "images/assistant.png", "image_id": "a"}
                        ],
                    },
                ],
            }
        ],
    }

    round_chunk = build_wma_chunks_from_data(
        payload, tmp_path, sample_path=sample_path
    )[0]
    turns = round_chunk.metadata["m2a_turns"]

    assert [turn["role"] for turn in turns] == ["user", "assistant"]
    assert [turn["text"] for turn in turns] == ["user text", "assistant text"]
    assert [turn["timestamp"] for turn in turns] == [
        "2025-01-01 10:00",
        "2025-01-01 10:01",
    ]
    assert turns[0]["images"] == [
        str((sample_path.parent / "images/user.png").resolve())
    ]
    assert turns[1]["images"] == [
        str((sample_path.parent / "images/assistant.png").resolve())
    ]

    balanced = balance_wma_chunks([round_chunk], target_tokens=512)[0]
    assert balanced.metadata["m2a_turns"] == turns
    assert balanced.metadata["m2a_speakers"] == ["user", "assistant"]


def test_wma_m2a_four_round_batches_stay_in_session_and_split_on_images(
    tmp_path: Path,
) -> None:
    def round_chunk(number: int, *, session: str = "S00", image: bool = False) -> Chunk:
        path = str(tmp_path / f"image-{number}.png")
        dialogue_id = f"{session}:R{number:04d}"
        turns = [
            {
                "turn_id": f"sample:{dialogue_id}:user",
                "source_dialogue_id": dialogue_id,
                "role": "user",
                "speaker": "user",
                "text": f"user {number}",
                "timestamp": f"2025-01-{number:02d} 10:00",
                "images": [path] if image else [],
            },
            {
                "turn_id": f"sample:{dialogue_id}:assistant",
                "source_dialogue_id": dialogue_id,
                "role": "assistant",
                "speaker": "assistant",
                "text": f"assistant {number}",
                "timestamp": f"2025-01-{number:02d} 10:01",
                "images": [],
            },
        ]
        return Chunk(
            chunk_id=f"sample:{dialogue_id}",
            text=f"session: {session}\nround: {dialogue_id}",
            images=[path] if image else [],
            metadata={
                "dataset": "sample",
                "session_id": session,
                "dialogue_id": dialogue_id,
                "round_id": number,
                "image_ids": [f"image-{number}"] if image else [],
                "image_captions": [],
                "m2a_turns": turns,
                "m2a_speakers": ["user", "assistant"],
            },
        )

    rounds = [
        round_chunk(1),
        round_chunk(2, image=True),
        round_chunk(3),
        round_chunk(4, image=True),
        round_chunk(5),
        round_chunk(1, session="S01"),
    ]
    batches = batch_wma_rounds_for_m2a(rounds, rounds_per_batch=4)

    assert [row.metadata["round_count"] for row in batches] == [3, 2, 1]
    assert [row.metadata["session_id"] for row in batches] == ["S00", "S00", "S01"]
    assert [row.metadata["m2a_source_turn_count"] for row in batches] == [6, 4, 2]
    assert all(row.metadata["m2a_ingest_mode"] == "batched_rounds" for row in batches)
    assert sum(len(row.metadata["m2a_turns"]) for row in batches) == 12


def test_m2a_batched_rounds_use_one_chat_call_and_keep_source_provenance(
    tmp_path: Path,
) -> None:
    source_image = str(tmp_path / "image.png")
    turns = [
        {
            "turn_id": "sample:S00:R0001:user",
            "source_dialogue_id": "S00:R0001",
            "role": "user",
            "speaker": "user",
            "text": "first question",
            "timestamp": "2025-01-01 10:00",
            "images": [source_image],
        },
        {
            "turn_id": "sample:S00:R0001:assistant",
            "source_dialogue_id": "S00:R0001",
            "role": "assistant",
            "speaker": "assistant",
            "text": "first answer",
            "timestamp": "2025-01-01 10:01",
            "images": [],
        },
        {
            "turn_id": "sample:S00:R0002:user",
            "source_dialogue_id": "S00:R0002",
            "role": "user",
            "speaker": "user",
            "text": "second question",
            "timestamp": "2025-01-01 10:02",
            "images": [],
        },
        {
            "turn_id": "sample:S00:R0002:assistant",
            "source_dialogue_id": "S00:R0002",
            "role": "assistant",
            "speaker": "assistant",
            "text": "second answer",
            "timestamp": "2025-01-01 10:03",
            "images": [],
        },
    ]
    chunk = Chunk(
        chunk_id="sample:S00:C0001",
        text="unused shared chunk text",
        images=[source_image],
        metadata={
            "dataset": "sample",
            "session_id": "S00",
            "dialogue_id": "S00:R0001..S00:R0002",
            "round_count": 2,
            "image_ids": ["image-1"],
            "m2a_turns": turns,
            "m2a_speakers": ["user", "assistant"],
            "m2a_ingest_mode": "batched_rounds",
        },
    )

    class FakeAgent:
        def __init__(self) -> None:
            self.calls = []
            self.raw_messages = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            self.raw_messages.append(SimpleNamespace(msg_id=1))

        def pop_tool_budget_events(self):
            return []

    agent = FakeAgent()
    adapter = M2AAdapter(baseline="M2A", source_root=tmp_path, config={})
    adapter.backend = SimpleNamespace()
    adapter.state_dir = tmp_path
    adapter._eval_wrapper = SimpleNamespace(
        _format_time=lambda value: value,
        cur_time=None,
    )
    adapter._ingest_agent = agent
    adapter._initialized = True
    adapter._execution_trace_path = tmp_path / "trace.jsonl"
    adapter._memory_fingerprints = lambda: {}

    adapter.ingest(chunk)

    assert len(agent.calls) == 1
    assert agent.calls[0]["user_image_path_or_url"] == source_image
    assert "first question" in agent.calls[0]["user_text"]
    assert "second answer" in agent.calls[0]["user_text"]
    assert adapter._raw_sources[1]["source_dialogue_ids"] == [
        "S00:R0001",
        "S00:R0002",
    ]
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert trace["ingest_mode"] == "batched_rounds"
    assert trace["source_turn_count"] == 4


def test_h2h_m2a_turns_do_not_merge_consecutive_same_speaker_utterances(
    tmp_path: Path,
) -> None:
    session_path = (
        tmp_path
        / "dyadic"
        / "dialogue1"
        / "scenes"
        / "session1"
        / "session.json"
    )
    payload = {
        "session_id": "native-session",
        "timeline_date": "2025-02-02",
        "dialogue": [
            {"role": "Alice", "content": {"text": "first", "image": "1.png"}},
            {"role": "Alice", "content": {"text": "second", "image": ""}},
            {"role": "Bob", "content": {"text": "third", "image": "2.png"}},
        ],
    }

    chunk = build_h2h_chunks_from_data(
        payload,
        session_path=session_path,
        variant="dyadic",
        conversation_id="dialogue1",
    )[0]
    turns = chunk.metadata["m2a_turns"]

    assert [turn["turn_id"].rsplit(":", 1)[-1] for turn in turns] == [
        "T0001",
        "T0002",
        "T0003",
    ]
    assert [turn["speaker"] for turn in turns] == ["Alice", "Alice", "Bob"]
    assert [turn["text"] for turn in turns] == ["first", "second", "third"]
    assert turns[0]["images"] == [str((session_path.parent / "image/1.png").resolve())]
    assert turns[1]["images"] == []
    assert turns[2]["images"] == [str((session_path.parent / "image/2.png").resolve())]
    assert chunk.metadata["m2a_speakers"] == ["Alice", "Bob"]
