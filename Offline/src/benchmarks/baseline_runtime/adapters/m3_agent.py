from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import json
import mimetypes
import os
import pickle
import re
import shutil
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from benchmarks.baseline_runtime.openai_compat import embed_texts
from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from embedding.chunk_builder import Chunk


M3_UPSTREAM_URL = "https://github.com/ByteDance-Seed/m3-agent"
M3_UPSTREAM_COMMIT = "0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c"
M3_UPSTREAM_TREE = "af5aab0f5883ba4bf97209d10cc46b3837a9f94b"
M3_NATIVE_SEARCH_TOP_K = 2
M3_CHARACTER_SEARCH_TOP_K = 20
M3_TOTAL_ROUNDS = 5
M3_RETRIEVAL_THRESHOLD = 0.5
M3_PROMPT_SHA256 = {
    "prompt_generate_memory_with_ids_sft": "75dd3021b9dc951e6b6f8c158f006eeafbc302045ad4bdce7b10ce44b3b687dc",
    "control_system_prompt": "6f334258d44951e39d64add4dc9c8e6fa7b312bbe6b6337df5480a91fde7f6fe",
    "control_instruction": "d09f2de7ec4ec0aa8169c3211d231eed958e342857544f9c32b1920073e5c4bf",
}
M3_APPROVED_SOURCE_PATCHES = [
    {
        "file": "mmagent/memory_processing_qwen.py",
        "function": "generate_all_memories",
        "change": 'episodic output key "video_descriptions" -> "video_description"',
        "reason": "Align the parser with the unchanged official Qwen memorization prompt schema.",
        "approved_by_user": True,
    }
]
M3_DEVIATIONS = [
    "One original benchmark dialogue round is mapped to one temporally ordered M3 clip observation.",
    "The non-video benchmarks provide dialogue text and general images, so official face detection, speaker diarization, face/voice nodes, and character equivalence are not fabricated.",
    "The configured Qwen-VL endpoint replaces the official memorization/control checkpoints; the official Qwen SFT memorization prompt and official Control prompt literals are unchanged.",
    "With user approval, the vendored Qwen memorization parser key is corrected from video_descriptions to the prompt-defined video_description.",
    "The configured Qwen3-VL embedding endpoint replaces the official text-embedding-3-large endpoint without changing VideoGraph insertion or retrieval scoring.",
    "Native Control searches retain Top-2 per round; only the accumulated memories handed to the unchanged benchmark QA prompt are capped to Top-7.",
    "Every Control round without a discovered clip explicitly requires Search and forbids answering from prior knowledge; an ignored rule is deterministically converted to Search, and the final round is forced to Answer only after a clip has been discovered.",
    "The API bridge constrains memorization output to six concise items per field and requests the prompt-defined JSON schema so it fits the configured output budget reliably.",
]
M3_MEMORY_OUTPUT_CONSTRAINT = (
    "Output-budget constraint: return at most 6 video_description items and at most "
    "6 high_level_conclusions items. Keep each item concise (no more than 40 words) "
    "and return only the complete JSON object."
)
M3_EMPTY_KNOWLEDGE_SEARCH_RULE = (
    "No memory evidence has been retrieved yet. You must choose Action: [Search]; "
    "it is the only valid action in this round. Even if you believe you know the answer, Action: "
    "[Answer] is invalid until a search has retrieved at least one memory clip. "
    "You must not answer from prior knowledge."
)
M3_MEMORY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "m3_memory_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "video_description": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "high_level_conclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["video_description", "high_level_conclusions"],
            "additionalProperties": False,
        },
    },
}


class M3AgentAdapter(BaselineAdapter):
    """Official M3 memory graph and Control loop over dialogue observations.

    The bridge adapts only the unavailable video/audio observation boundary. It
    does not replace M3's episodic/semantic memory construction, VideoGraph,
    clip-wise search, or iterative Control protocol.
    """

    _ACTION_PATTERN = re.compile(r"Action: \[(.*)\].*Content: (.*)", re.DOTALL)

    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.graph: Any = None
        self.state_dir: Path | None = None
        self.sample_id = ""
        self._clip_id = 0
        self._clip_sources: dict[int, dict[str, Any]] = {}
        self._execution_trace_path: Path | None = None
        self._clip_manifest_path: Path | None = None
        self._memory_generation_dir: Path | None = None
        self._reuse_source_state_dir: Path | None = None
        self._reuse_manifest_rows: list[dict[str, Any]] = []
        self._reuse_full_graph: Any = None
        self._reuse_active_clip_id = -1
        self._requests = requests.Session()
        if self._is_local_endpoint:
            self._requests.trust_env = False
        self._load_official_modules()

    @property
    def _is_local_endpoint(self) -> bool:
        url = str(self.config.get("executor_base_url") or "")
        return "127.0.0.1" in url or "localhost" in url or "[::1]" in url

    def _load_official_modules(self) -> None:
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        old_cwd = Path.cwd()
        try:
            os.chdir(self.source_root)
            # The official package initializer eagerly imports face and voice
            # pipelines (and their checkpoints) even when callers only need
            # the graph. Dialogue observations deliberately mark those video
            # branches not applicable, so load the unchanged core submodules
            # without executing that eager initializer.
            if "mmagent" not in sys.modules:
                package = types.ModuleType("mmagent")
                package.__path__ = [str(self.source_root / "mmagent")]  # type: ignore[attr-defined]
                package.__package__ = "mmagent"
                sys.modules["mmagent"] = package
            if "mmagent.utils" not in sys.modules:
                package = types.ModuleType("mmagent.utils")
                package.__path__ = [str(self.source_root / "mmagent" / "utils")]  # type: ignore[attr-defined]
                package.__package__ = "mmagent.utils"
                sys.modules["mmagent.utils"] = package
            prompts = importlib.import_module("mmagent.prompts")
            general = importlib.import_module("mmagent.utils.general")
            memory_processing = importlib.import_module("mmagent.memory_processing")
            VideoGraph = importlib.import_module("mmagent.videograph").VideoGraph
            retrieve = importlib.import_module("mmagent.retrieve")
        finally:
            os.chdir(old_cwd)
        self._VideoGraph = VideoGraph
        self._memory_processing = memory_processing
        self._general = general
        self._prompts = prompts
        self._retrieve = retrieve
        # API transport is the only substituted dependency. Both official
        # memory insertion and official retrieval call these module globals.
        self._memory_processing.parallel_get_embedding = self._parallel_get_embedding
        self._retrieve.parallel_get_embedding = self._parallel_get_embedding
        self._retrieve.get_embedding_with_retry = self._get_embedding_with_retry
        self._control_system_prompt = self._control_literal("system_prompt")
        self._control_instruction = self._control_literal("instruction")
        self._assert_prompt_hashes()

    def _control_literal(self, name: str) -> str:
        path = self.source_root / "m3_agent" / "control.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id != name:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
            if isinstance(node.value, ast.JoinedStr) and all(
                isinstance(value, ast.Constant) and isinstance(value.value, str)
                for value in node.value.values
            ):
                return "".join(str(value.value) for value in node.value.values)
            raise RuntimeError(f"M3 Control {name} is no longer a literal prompt")
        raise RuntimeError(f"M3 Control prompt not found: {name}")

    def _assert_prompt_hashes(self) -> None:
        actual = self._runtime_prompt_hashes()
        mismatches = {
            name: {"expected": expected, "actual": actual.get(name, "")}
            for name, expected in M3_PROMPT_SHA256.items()
            if actual.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"M3 official prompt hash mismatch: {mismatches}")

    def _runtime_prompt_hashes(self) -> dict[str, str]:
        return {
            "prompt_generate_memory_with_ids_sft": _sha256_text(
                str(self._prompts.prompt_generate_memory_with_ids_sft)
            ),
            "control_system_prompt": _sha256_text(self._control_system_prompt),
            "control_instruction": _sha256_text(self._control_instruction),
        }

    def reset(self, sample_id: str, state_dir: Path) -> None:
        resolved = state_dir.expanduser().resolve()
        offline_root = Path(__file__).resolve().parents[4]
        if resolved in {Path("/"), offline_root, offline_root.parent} or resolved == resolved.parent:
            raise ValueError(f"refusing unsafe M3 state directory: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)
        self.state_dir = resolved
        self.sample_id = str(sample_id)
        self._execution_trace_path = resolved / "m3_execution_trace.jsonl"
        self._clip_manifest_path = resolved / "clip_manifest.jsonl"
        self._memory_generation_dir = resolved / "memory_generation"
        self._memory_generation_dir.mkdir()
        self._reuse_source_state_dir = None
        self._reuse_manifest_rows = []
        self._reuse_full_graph = None
        self._reuse_active_clip_id = -1
        reuse_state = str(self.config.get("m3_reuse_sample_state") or "").strip()
        if reuse_state:
            self._reset_from_reused_state(Path(reuse_state))
            return
        graph_config = json.loads(
            (self.source_root / "configs" / "memory_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.graph = self._VideoGraph(**graph_config)
        self._clip_id = 0
        self._clip_sources = {}
        self._trace_event(
            component="M3AgentAdapter",
            action="reset",
            sample_id=self.sample_id,
            upstream_url=M3_UPSTREAM_URL,
            upstream_commit=M3_UPSTREAM_COMMIT,
            upstream_tree=M3_UPSTREAM_TREE,
        )

    def _reset_from_reused_state(self, source_state: Path) -> None:
        assert self.state_dir is not None
        source = source_state.expanduser().resolve()
        if source == self.state_dir or source in self.state_dir.parents:
            raise ValueError(
                f"M3 reuse source must be separate from new state directory: {source}"
            )
        graph_path = source / "memory_graph.pkl"
        manifest_path = source / "clip_manifest.jsonl"
        conformance_path = source / "m3_conformance.json"
        missing = [
            str(path)
            for path in (graph_path, manifest_path, conformance_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"M3 reusable state is incomplete: {missing}")
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"M3 reusable clip manifest is empty: {manifest_path}")
        for expected_clip_id, row in enumerate(rows, start=1):
            if int(row.get("clip_id") or 0) != expected_clip_id:
                raise ValueError(
                    "M3 reusable clip manifest must contain contiguous clip IDs; "
                    f"expected {expected_clip_id}, got {row.get('clip_id')!r}"
                )
            if str(row.get("sample_id") or "") != self.sample_id:
                raise ValueError(
                    "M3 reusable state sample mismatch: "
                    f"expected {self.sample_id!r}, got {row.get('sample_id')!r}"
                )
        with graph_path.open("rb") as handle:
            full_graph = pickle.load(handle)
        graph_clips = {int(value) for value in full_graph.text_nodes_by_clip}
        manifest_clips = set(range(1, len(rows) + 1))
        if not graph_clips.issubset(manifest_clips):
            raise ValueError(
                "M3 reusable graph references clips outside its manifest: "
                f"{sorted(graph_clips - manifest_clips)[:10]}"
            )
        self._reuse_source_state_dir = source
        self._reuse_manifest_rows = rows
        self._reuse_full_graph = full_graph
        self.graph = full_graph
        self._clip_id = 0
        self._clip_sources = {}
        target_graph = self.state_dir / "memory_graph.pkl"
        target_graph.symlink_to(graph_path)
        provenance = {
            "schema_version": 1,
            "read_only_reuse": True,
            "source_state_dir": str(source),
            "source_artifact_sha256": {
                "memory_graph.pkl": _sha256_file(graph_path),
                "clip_manifest.jsonl": _sha256_file(manifest_path),
                "m3_conformance.json": _sha256_file(conformance_path),
            },
            "expected_clip_count": len(rows),
        }
        (self.state_dir / "reused_state.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._trace_event(
            component="M3AgentAdapter",
            action="reset_reused_graph",
            sample_id=self.sample_id,
            source_state_dir=str(source),
            expected_clip_count=len(rows),
            source_artifact_sha256=provenance["source_artifact_sha256"],
        )

    def ingest(self, chunk: Chunk) -> None:
        if self.graph is None or self.state_dir is None:
            raise RuntimeError("M3 adapter has not been reset")
        observation = chunk.metadata.get("m3_observation")
        if not isinstance(observation, dict):
            raise ValueError(
                f"M3 strict ingest requires metadata.m3_observation: {chunk.chunk_id}"
            )
        if observation.get("input_mode") != "dialogue_round_as_clip":
            raise ValueError(f"invalid M3 observation mode: {observation.get('input_mode')!r}")
        turns = observation.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"M3 observation has no source turns: {chunk.chunk_id}")
        if self._reuse_source_state_dir is not None:
            self._ingest_reused_chunk(chunk, observation)
            return
        self._clip_id += 1
        clip_id = self._clip_id
        source = self._source_record(chunk, observation, clip_id)
        for path in source["image_paths"]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"M3 source image does not exist: {path}")

        messages = self._memory_messages(chunk, source)
        (
            raw,
            usage,
            transport_attempts,
            generation_attempts,
            finish_reason,
            episodic,
            semantic,
            rejected,
        ) = self._memory_completion(messages, clip_id)

        before_nodes = set(self.graph.nodes)
        self._memory_processing.process_memories(
            self.graph, episodic, clip_id, type="episodic"
        )
        episodic_nodes = sorted(set(self.graph.nodes) - before_nodes)
        before_nodes = set(self.graph.nodes)
        self._memory_processing.process_memories(
            self.graph, semantic, clip_id, type="semantic"
        )
        semantic_nodes = sorted(set(self.graph.nodes) - before_nodes)
        self._clip_sources[clip_id] = source
        self._append_jsonl(self._clip_manifest_path, source)
        generation = {
            "schema_version": 1,
            "sample_id": self.sample_id,
            "clip_id": clip_id,
            "source": source,
            "model": str(self.config["executor_model"]),
            "prompt_name": "prompt_generate_memory_with_ids_sft",
            "prompt_sha256": self._runtime_prompt_hashes()[
                "prompt_generate_memory_with_ids_sft"
            ],
            "raw_output_schema": {
                "episodic": "video_description",
                "semantic": "high_level_conclusions",
            },
            "raw_response": raw,
            "episodic_memory": episodic,
            "semantic_memory": semantic,
            "memory_empty": not episodic and not semantic,
            "rejected_unsupported_feature_memories": rejected,
            "episodic_node_ids": episodic_nodes,
            "semantic_node_ids": semantic_nodes,
            "usage": usage,
            "attempts": transport_attempts,
            "generation_attempts": generation_attempts,
            "finish_reason": finish_reason,
        }
        assert self._memory_generation_dir is not None
        (self._memory_generation_dir / f"clip_{clip_id:06d}.json").write_text(
            json.dumps(generation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._trace_event(
            component="M3Memorization",
            action="generate_and_insert",
            clip_id=clip_id,
            dialogue_id=source["dialogue_id"],
            session_id=source["session_id"],
            image_ids=source["image_ids"],
            episodic_count=len(episodic),
            semantic_count=len(semantic),
            memory_empty=not episodic and not semantic,
            rejected_unsupported_feature_count=len(rejected),
            rejected_unsupported_feature_memories=rejected,
            episodic_node_ids=episodic_nodes,
            semantic_node_ids=semantic_nodes,
            generation_attempts=generation_attempts,
            finish_reason=finish_reason,
            prompt_sha256=generation["prompt_sha256"],
        )

    def _ingest_reused_chunk(
        self, chunk: Chunk, observation: dict[str, Any]
    ) -> None:
        clip_id = self._clip_id + 1
        if clip_id > len(self._reuse_manifest_rows):
            raise ValueError(
                f"M3 rebuilt input has more clips than reused state ({clip_id})"
            )
        actual = self._source_record(chunk, observation, clip_id)
        expected = self._reuse_manifest_rows[clip_id - 1]
        differences = _source_record_differences(expected, actual)
        if differences:
            raise ValueError(
                f"M3 reused clip {clip_id} does not match current input: {differences}"
            )
        for path in actual["image_paths"]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"M3 source image does not exist: {path}")
        self._clip_id = clip_id
        # Preserve the current run's staged paths while retaining the exact old
        # graph contents. Retrieval provenance therefore points at live inputs.
        self._clip_sources[clip_id] = actual
        self._append_jsonl(self._clip_manifest_path, actual)
        self._reuse_active_clip_id = -1
        self._trace_event(
            component="M3Memorization",
            action="reuse_validated_clip",
            clip_id=clip_id,
            dialogue_id=actual["dialogue_id"],
            session_id=actual["session_id"],
            source_state_dir=str(self._reuse_source_state_dir),
        )

    def _source_record(
        self, chunk: Chunk, observation: dict[str, Any], clip_id: int
    ) -> dict[str, Any]:
        images = [row for row in observation.get("images") or [] if isinstance(row, dict)]
        source_ids = [
            str(value)
            for value in observation.get("source_dialogue_ids") or []
            if value
        ]
        dialogue_id = str(observation.get("dialogue_id") or chunk.chunk_id)
        if not source_ids:
            source_ids = [dialogue_id]
        return {
            "schema_version": 1,
            "sample_id": self.sample_id,
            "clip_id": clip_id,
            "benchmark": str(observation.get("benchmark") or ""),
            "dataset": str(observation.get("dataset") or ""),
            "session_id": str(observation.get("session_id") or ""),
            "dialogue_id": dialogue_id,
            "source_dialogue_ids": source_ids,
            "timestamp": str(observation.get("timestamp") or ""),
            "turns": list(observation.get("turns") or []),
            "image_ids": [str(row.get("image_id") or "") for row in images if row.get("image_id")],
            "image_paths": [str(row.get("path") or "") for row in images if row.get("path")],
            "image_captions": [str(row.get("caption") or "") for row in images if row.get("caption")],
            "input_mode": "dialogue_round_as_clip",
            "face_nodes_applicable": False,
            "voice_nodes_applicable": False,
            "character_equivalence_applicable": False,
        }

    def _memory_messages(
        self, chunk: Chunk, source: dict[str, Any]
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": str(self._prompts.prompt_generate_memory_with_ids_sft),
            },
            {"type": "text", "text": M3_MEMORY_OUTPUT_CONSTRAINT},
            {
                "type": "text",
                "text": (
                    "Source annotation for the current clip, replacing unavailable "
                    "video/audio bytes. Treat it as the complete observable content. "
                    "Face and voice features are not applicable; do not invent feature "
                    "IDs or unobserved audiovisual details. The following fields are "
                    "source annotations, not generated memory.\n\n"
                    + chunk.text
                ),
            }
        ]
        for index, path in enumerate(source["image_paths"]):
            image_id = source["image_ids"][index] if index < len(source["image_ids"]) else ""
            caption = (
                source["image_captions"][index]
                if index < len(source["image_captions"])
                else ""
            )
            content.append(
                {
                    "type": "text",
                    "text": f"Source image ID: {image_id}\nSource image annotation: {caption}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(Path(path)), "detail": "high"},
                }
            )
        # Same order and role shape as official Qwen generate_all_memories:
        # prompt first, then the observable clip context, in one user message.
        return [{"role": "user", "content": content}]

    def _memory_completion(
        self, messages: list[dict[str, Any]], clip_id: int
    ) -> tuple[
        str,
        dict[str, Any] | None,
        int,
        int,
        str,
        list[str],
        list[str],
        list[dict[str, str]],
    ]:
        """Generate and validate one memory object, retrying malformed generations."""
        retries = max(0, int(self.config.get("retries") or 0))
        total_transport_attempts = 0
        last_error: Exception | None = None
        for generation_attempt in range(1, retries + 2):
            attempt_messages = json.loads(json.dumps(messages))
            if generation_attempt > 1:
                attempt_messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": (
                            "Retry because the previous generation was incomplete or invalid. "
                            "Be shorter and return one complete JSON object within the output budget."
                        ),
                    }
                )
            raw, usage, transport_attempts, metadata = self._chat_completion(
                attempt_messages,
                response_format=M3_MEMORY_RESPONSE_FORMAT,
            )
            total_transport_attempts += transport_attempts
            finish_reason = str(metadata.get("finish_reason") or "")
            try:
                if finish_reason == "length":
                    raise ValueError("M3 memory response reached the output-token limit")
                memories = self._general.validate_and_fix_json(raw)
                if memories is None:
                    raise ValueError("M3 memory response is not valid JSON")
                episodic, semantic, rejected = self._validate_memories(
                    memories, clip_id
                )
                return (
                    raw,
                    usage,
                    total_transport_attempts,
                    generation_attempt,
                    finish_reason,
                    episodic,
                    semantic,
                    rejected,
                )
            except (TypeError, ValueError) as exc:
                last_error = exc
                self._write_failed_memory_response(
                    clip_id=clip_id,
                    generation_attempt=generation_attempt,
                    raw=raw,
                    usage=usage,
                    finish_reason=finish_reason,
                    error=str(exc),
                )
                if generation_attempt <= retries:
                    time.sleep(min(2**generation_attempt, 5))
        raise ValueError(
            f"M3 clip {clip_id} memory generation failed after {retries + 1} "
            f"attempts: {last_error}"
        )

    def _write_failed_memory_response(
        self,
        *,
        clip_id: int,
        generation_attempt: int,
        raw: str,
        usage: dict[str, Any] | None,
        finish_reason: str,
        error: str,
    ) -> None:
        if self.state_dir is None:
            return
        directory = self.state_dir / "memory_generation_failures"
        directory.mkdir(exist_ok=True)
        path = directory / (
            f"clip_{clip_id:06d}_attempt_{generation_attempt:02d}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": self.sample_id,
                    "clip_id": clip_id,
                    "generation_attempt": generation_attempt,
                    "executor_max_tokens": int(
                        self.config.get("executor_max_tokens")
                        or self.config.get("num_predict")
                        or 512
                    ),
                    "finish_reason": finish_reason,
                    "error": error,
                    "raw_response": raw,
                    "usage": usage,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _validate_memories(
        memories: Any, clip_id: int
    ) -> tuple[list[str], list[str], list[dict[str, str]]]:
        if not isinstance(memories, dict):
            raise ValueError(f"M3 clip {clip_id} memory response is not an object")
        if set(memories) != {"video_description", "high_level_conclusions"}:
            raise ValueError(
                f"M3 clip {clip_id} memory keys must be exactly "
                f"video_description/high_level_conclusions; "
                f"got {sorted(memories)}"
            )
        episodic = memories["video_description"]
        semantic = memories["high_level_conclusions"]
        normalized: dict[str, list[str]] = {}
        rejected: list[dict[str, str]] = []
        unsupported_feature = re.compile(
            r"<(?:face|voice|character)_[^<>]+>", re.IGNORECASE
        )
        for name, values in (
            ("video_description", episodic),
            ("high_level_conclusions", semantic),
        ):
            if not isinstance(values, list):
                raise ValueError(f"M3 clip {clip_id} {name} must be a list")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"M3 clip {clip_id} {name} contains a non-string/empty item")
            accepted: list[str] = []
            for value in values:
                text = value.strip()
                if unsupported_feature.search(text):
                    rejected.append(
                        {
                            "source_field": name,
                            "text": text,
                            "reason": "feature IDs are unavailable for dialogue observations",
                        }
                    )
                else:
                    accepted.append(text)
            normalized[name] = accepted
        return (
            normalized["video_description"],
            normalized["high_level_conclusions"],
            rejected,
        )

    def end_session(self, session_id: str) -> None:
        if self.graph is None:
            raise RuntimeError("M3 adapter has not been reset")
        if self._reuse_source_state_dir is None:
            self.graph.refresh_equivalences()
            self._persist_graph()
        self._trace_event(
            component="VideoGraph",
            action="end_session",
            session_id=session_id,
            node_counts=self._node_counts(),
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self.graph is None:
            raise RuntimeError("M3 memory must be built before retrieval")
        self._activate_reused_graph()
        self.graph.refresh_equivalences()
        conversations: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._control_system_prompt.format(question=request.text),
            },
            {
                "role": "user",
                "content": "Searched knowledge: {}",
            },
        ]
        current_clips: list[int] = []
        discovered: list[dict[str, Any]] = []
        discovered_ids: set[int] = set()
        rounds: list[dict[str, Any]] = []
        agent_answer = ""

        for round_index in range(M3_TOTAL_ROUNDS):
            conversations[-1]["content"] += self._control_instruction
            if not discovered:
                conversations[-1]["content"] += (
                    "\n" + M3_EMPTY_KNOWLEDGE_SEARCH_RULE
                )
            if round_index == M3_TOTAL_ROUNDS - 1 and discovered:
                conversations[-1]["content"] += (
                    "\n(The Action of this round must be [Answer]. If there is "
                    "insufficient information, you can make reasonable guesses.)"
                )
            raw, usage, attempts, _metadata = self._chat_completion(conversations)
            conversations.append({"role": "assistant", "content": raw})
            match = self._ACTION_PATTERN.search(raw.split("</think>")[-1])
            parse_fallback = match is None
            if match:
                action = match.group(1)
                content = match.group(2)
            else:
                # Preserve the official Control behavior, but expose it in trace.
                action = "Search"
                content = None
            requested_action = action
            forced_search_without_evidence = action == "Answer" and not discovered
            if forced_search_without_evidence:
                # Control occasionally ignores the explicit empty-evidence rule.
                # Its proposed answer may be useful as a semantic search query,
                # but it is never accepted as evidence or handed to the final QA
                # agent. Only clips returned by the official search are eligible.
                action = "Search"
                content = str(content or request.text)
            round_trace: dict[str, Any] = {
                "round": round_index + 1,
                "raw_response": raw,
                "action": action,
                "requested_action": requested_action,
                "forced_search_without_evidence": forced_search_without_evidence,
                "content": content,
                "parse_fallback": parse_fallback,
                "usage": usage,
                "attempts": attempts,
                "native_search_top_k": None,
                "returned_clips": [],
            }
            if action == "Answer":
                agent_answer = str(content or "")
                rounds.append(round_trace)
                break

            new_memories: dict[str, list[str]] = {}
            clip_scores: dict[int, float] = {}
            if content:
                before = list(current_clips)
                if "character id" in content:
                    round_trace["native_search_top_k"] = M3_CHARACTER_SEARCH_TOP_K
                    new_memories, _, clip_scores = self._retrieve.search(
                        self.graph,
                        content,
                        [],
                        mem_wise=True,
                        topk=M3_CHARACTER_SEARCH_TOP_K,
                    )
                    ranked_clip_ids = [_clip_number(value) for value in new_memories]
                else:
                    round_trace["native_search_top_k"] = M3_NATIVE_SEARCH_TOP_K
                    new_memories, current_clips, clip_scores = self._retrieve.search(
                        self.graph,
                        content,
                        current_clips,
                        threshold=M3_RETRIEVAL_THRESHOLD,
                        topk=M3_NATIVE_SEARCH_TOP_K,
                    )
                    ranked_clip_ids = current_clips[len(before) :]
                visible = set(request.visible_session_ids)
                for clip_id in ranked_clip_ids:
                    if clip_id in discovered_ids:
                        continue
                    source = self._clip_sources.get(clip_id, {})
                    if visible and str(source.get("session_id") or "") not in visible:
                        continue
                    key = f"CLIP_{clip_id}"
                    values = list(new_memories.get(key) or self._clip_memory_texts(clip_id))
                    discovered.append(
                        {
                            "clip_id": clip_id,
                            "memories": values,
                            "score": float(clip_scores[clip_id]) if clip_id in clip_scores else None,
                            "query": content,
                            "round": round_index + 1,
                        }
                    )
                    discovered_ids.add(clip_id)
            round_trace["returned_clips"] = [
                {"clip_id": row["clip_id"], "memories": row["memories"]}
                for row in discovered
                if row["round"] == round_index + 1
            ]
            rounds.append(round_trace)
            search_result = "Searched knowledge: " + json.dumps(
                new_memories, ensure_ascii=False
            ).encode("utf-8", "ignore").decode("utf-8")
            if not new_memories:
                search_result += (
                    "\n(The search result is empty. Please try searching from another perspective.)"
                )
            conversations.append({"role": "user", "content": search_result})

        handoff = discovered[: request.top_k]
        items = [self._retrieved_clip(row) for row in handoff]
        trace = {
            "baseline": self.baseline,
            "via": "m3_official_control_search",
            "upstream_commit": M3_UPSTREAM_COMMIT,
            "control_prompt_sha256": {
                "system": self._runtime_prompt_hashes()["control_system_prompt"],
                "instruction": self._runtime_prompt_hashes()["control_instruction"],
            },
            "native_round_limit": M3_TOTAL_ROUNDS,
            "native_search_top_k": M3_NATIVE_SEARCH_TOP_K,
            "character_search_top_k": M3_CHARACTER_SEARCH_TOP_K,
            "threshold": M3_RETRIEVAL_THRESHOLD,
            "rounds": rounds,
            "agent_answer": agent_answer,
            "candidate_clip_ids": [row["clip_id"] for row in discovered],
            "candidate_count": len(discovered),
            "handoff_cap": request.top_k,
            "final_clip_ids": [row["clip_id"] for row in handoff],
            "final_memory_count": len(items),
            "handoff_underfilled": len(items) < request.top_k,
            "handoff_top_up_used": False,
            "forced_search_without_evidence_count": sum(
                bool(row.get("forced_search_without_evidence")) for row in rounds
            ),
        }
        self._trace_event(
            component="M3Control",
            action="retrieve",
            query_id=request.query_id,
            rounds=rounds,
            agent_answer=agent_answer,
            candidate_clip_ids=trace["candidate_clip_ids"],
            final_clip_ids=trace["final_clip_ids"],
            handoff_cap=request.top_k,
            handoff_top_up_used=False,
        )
        return RetrievalResult(items=items, trace=trace)

    def _retrieved_clip(self, row: dict[str, Any]) -> RetrievedMemory:
        clip_id = int(row["clip_id"])
        source = self._clip_sources.get(clip_id, {})
        structured = self._clip_structured_memories(clip_id)
        text = f"CLIP_{clip_id}\n" + "\n".join(
            f"[{item['type']}] {item['text']}" for item in structured
        )
        return RetrievedMemory(
            memory_id=f"m3:clip:{clip_id}",
            text=text,
            score=row.get("score"),
            session_id=str(source.get("session_id") or ""),
            source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
            image_ids=list(source.get("image_ids") or []),
            image_paths=list(source.get("image_paths") or []),
            metadata={
                "via": "m3_official_control_search",
                "clip_id": clip_id,
                "timestamp": source.get("timestamp", ""),
                "agent_query": row.get("query", ""),
                "agent_round": row.get("round"),
                "structured_memories": structured,
            },
        )

    def _clip_structured_memories(self, clip_id: int) -> list[dict[str, Any]]:
        return [
            {
                "node_id": int(node_id),
                "type": str(self.graph.nodes[node_id].type),
                "text": str(self.graph.nodes[node_id].metadata["contents"][0]),
            }
            for node_id in self.graph.text_nodes_by_clip.get(clip_id, [])
        ]

    def _clip_memory_texts(self, clip_id: int) -> list[str]:
        return [row["text"] for row in self._clip_structured_memories(clip_id)]

    def snapshot(self) -> list[MemoryRecord]:
        if self.graph is None:
            return []
        self._activate_reused_graph()
        self._persist_graph()
        records: list[MemoryRecord] = []
        for node_id in self.graph.text_nodes:
            node = self.graph.nodes[node_id]
            clip_id = int(node.metadata["timestamp"])
            source = self._clip_sources.get(clip_id, {})
            records.append(
                MemoryRecord(
                    memory_id=f"m3:node:{node_id}",
                    text="\n".join(str(value) for value in node.metadata.get("contents") or []),
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    backend_type=f"m3_{node.type}",
                    metadata={
                        "clip_id": clip_id,
                        "timestamp": source.get("timestamp", ""),
                        "input_mode": "dialogue_round_as_clip",
                    },
                )
            )
        self._write_conformance()
        return records

    def _parallel_get_embedding(
        self, _model: str, texts: list[str], timeout: int = 15
    ) -> tuple[list[list[float]], int]:
        del timeout
        if not texts:
            return [], 0
        return embed_texts([str(value) for value in texts], self.config), 0

    def _get_embedding_with_retry(
        self, _model: str, text: str, timeout: int = 15
    ) -> tuple[list[float], int]:
        del timeout
        return embed_texts([str(text)], self.config)[0], 0

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None, int, dict[str, Any]]:
        base_url = str(self.config["executor_base_url"]).rstrip("/")
        model = str(self.config["executor_model"])
        timeout = int(self.config.get("request_timeout") or 180)
        retries = max(0, int(self.config.get("retries") or 0))
        api_key = str(
            self.config.get("executor_api_key")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY"
        )
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(self.config.get("executor_temperature") or 0.0),
            "max_tokens": int(
                self.config.get("executor_max_tokens")
                or self.config.get("num_predict")
                or 512
            ),
        }
        if response_format is not None:
            payload["response_format"] = response_format
        last_error: Exception | None = None
        for attempt in range(1, retries + 2):
            try:
                response = self._requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    timeout=timeout,
                )
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    body = (response.text or "").strip()
                    raise RuntimeError(f"M3 executor HTTP error: {exc}; body={body[:2000]}") from exc
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError("M3 executor returned no choices")
                choice = choices[0]
                content = choice.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("M3 executor returned empty content")
                return (
                    content,
                    body.get("usage"),
                    attempt,
                    {
                        "finish_reason": choice.get("finish_reason"),
                        "native_finish_reason": choice.get("native_finish_reason"),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - surfaced after configured retries
                last_error = exc
                if attempt <= retries:
                    time.sleep(min(2**attempt, 5))
        raise RuntimeError(f"M3 executor failed after {retries + 1} attempts: {last_error}")

    def _persist_graph(self) -> None:
        if self.graph is None or self.state_dir is None:
            return
        if self._reuse_source_state_dir is not None:
            self._write_conformance()
            return
        with (self.state_dir / "memory_graph.pkl").open("wb") as handle:
            pickle.dump(self.graph, handle)
        self._write_conformance()

    def _write_conformance(self) -> None:
        if self.state_dir is None:
            return
        payload = m3_conformance_manifest(
            benchmark="",
            answer_prompt_sha256="",
            source_root=self.source_root,
        )
        payload.update(
            {
                "sample_id": self.sample_id,
                "clip_count": self._clip_id,
                "node_counts": self._node_counts(),
                "artifacts": {
                    "memory_graph": str(self.state_dir / "memory_graph.pkl"),
                    "clip_manifest": str(self.state_dir / "clip_manifest.jsonl"),
                    "memory_generation_dir": str(self.state_dir / "memory_generation"),
                    "execution_trace": str(self.state_dir / "m3_execution_trace.jsonl"),
                },
            }
        )
        if self._reuse_source_state_dir is not None:
            payload["reused_state"] = {
                "read_only": True,
                "source_state_dir": str(self._reuse_source_state_dir),
                "source_artifact_sha256": {
                    name: _sha256_file(self._reuse_source_state_dir / name)
                    for name in (
                        "memory_graph.pkl",
                        "clip_manifest.jsonl",
                        "m3_conformance.json",
                    )
                },
                "validated_clip_count": self._clip_id,
                "expected_clip_count": len(self._reuse_manifest_rows),
            }
        (self.state_dir / "m3_conformance.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _node_counts(self) -> dict[str, int]:
        counts = {"img": 0, "voice": 0, "episodic": 0, "semantic": 0}
        if self.graph is not None:
            for node in self.graph.nodes.values():
                counts[str(node.type)] = counts.get(str(node.type), 0) + 1
        return counts

    def _activate_reused_graph(self) -> None:
        # A few focused protocol tests construct the adapter with ``__new__``;
        # absent reuse fields therefore mean the ordinary non-reuse path.
        if getattr(self, "_reuse_source_state_dir", None) is None:
            return
        if self._clip_id < 1:
            raise RuntimeError("M3 reused graph cannot be queried before any visible clip")
        if self._clip_id > len(self._reuse_manifest_rows):
            raise RuntimeError("M3 reused graph visible clip count is invalid")
        if self._reuse_active_clip_id == self._clip_id:
            return
        if self._clip_id == len(self._reuse_manifest_rows):
            self.graph = self._reuse_full_graph
        else:
            self.graph = pickle.loads(pickle.dumps(self._reuse_full_graph))
            self.graph.truncate_memory_by_clip(self._clip_id, refresh=False)
        self._reuse_active_clip_id = self._clip_id
        self._trace_event(
            component="VideoGraph",
            action="activate_reused_prefix",
            visible_clip_count=self._clip_id,
            full_clip_count=len(self._reuse_manifest_rows),
            truncated=self._clip_id < len(self._reuse_manifest_rows),
            node_counts=self._node_counts(),
        )

    def _trace_event(self, *, component: str, action: str, **fields: Any) -> None:
        if self._execution_trace_path is None:
            return
        self._append_jsonl(
            self._execution_trace_path,
            {
                "event": action,
                "component": component,
                "action": action,
                "recorded_at": datetime.now().astimezone().isoformat(),
                **fields,
            },
        )

    @staticmethod
    def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.graph is not None and self.state_dir is not None:
            self._persist_graph()
        self.graph = None
        self._requests.close()

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "m3_agent",
            "baseline": self.baseline,
            "available": True,
            "upstream_url": M3_UPSTREAM_URL,
            "upstream_commit": M3_UPSTREAM_COMMIT,
            "upstream_tree": M3_UPSTREAM_TREE,
            "compatibility_mode": "official-memory-graph-control-dialogue-observation",
            "audio_enabled": False,
            "supports_images": True,
            "supports_session_filter": True,
            "native_search_top_k": M3_NATIVE_SEARCH_TOP_K,
            "handoff_top_k": int(self.config.get("top_k") or 7),
            "adapter_silent_fallback": False,
        }


def m3_conformance_manifest(
    benchmark: str,
    *,
    answer_prompt_sha256: str,
    source_root: Path | None = None,
) -> dict[str, Any]:
    root = source_root or Path(__file__).resolve().parents[4] / "baselines" / "m3-agent-master"
    prompt_hashes = _prompt_hashes_from_source(root)
    source_files = {
        "memorization_entry": root / "m3_agent" / "memorization_memory_graphs.py",
        "qwen_memorization": root / "mmagent" / "memory_processing_qwen.py",
        "control_entry": root / "m3_agent" / "control.py",
        "video_graph": root / "mmagent" / "videograph.py",
        "retrieval": root / "mmagent" / "retrieve.py",
        "prompts": root / "mmagent" / "prompts.py",
    }
    return {
        "upstream_url": M3_UPSTREAM_URL,
        "upstream_commit": M3_UPSTREAM_COMMIT,
        "upstream_tree": M3_UPSTREAM_TREE,
        "benchmark": benchmark,
        "official_core_used": True,
        "official_core_patched": True,
        "approved_source_patches": list(M3_APPROVED_SOURCE_PATCHES),
        "protocol_bridge": "benchmark dialogue round to M3 clip observation",
        "shared_fixed_chunks": False,
        "deviations": list(M3_DEVIATIONS),
        "native_search_top_k": M3_NATIVE_SEARCH_TOP_K,
        "character_search_top_k": M3_CHARACTER_SEARCH_TOP_K,
        "native_round_limit": M3_TOTAL_ROUNDS,
        "handoff_top_k": 7,
        "handoff_top_up_used": False,
        "empty_knowledge_search_rule": {
            "text": M3_EMPTY_KNOWLEDGE_SEARCH_RULE,
            "sha256": _sha256_text(M3_EMPTY_KNOWLEDGE_SEARCH_RULE),
            "applies_while_handoff_empty": True,
            "final_answer_requires_discovered_clip": True,
            "answer_without_clip_effective_action": "Search",
            "answer_content_used_only_as_search_query": True,
            "approved_by_user": True,
        },
        "answer_path": "unchanged benchmark QA prompt",
        "answer_prompt_sha256": answer_prompt_sha256,
        "face_nodes_applicable": False,
        "voice_nodes_applicable": False,
        "character_equivalence_applicable": False,
        "adapter_silent_fallback": False,
        "internal_prompt_sha256": {
            name: {"expected": expected, "actual": prompt_hashes.get(name, "")}
            for name, expected in M3_PROMPT_SHA256.items()
        },
        "official_source_sha256": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
    }


def _prompt_hashes_from_source(root: Path) -> dict[str, str]:
    prompts_module = ast.parse((root / "mmagent" / "prompts.py").read_text(encoding="utf-8"))
    control_module = ast.parse((root / "m3_agent" / "control.py").read_text(encoding="utf-8"))
    qwen_memory = _ast_string_assignment(
        prompts_module, "prompt_generate_memory_with_ids_sft"
    )
    return {
        "prompt_generate_memory_with_ids_sft": _sha256_text(qwen_memory),
        "control_system_prompt": _sha256_text(
            _ast_string_assignment(control_module, "system_prompt")
        ),
        "control_instruction": _sha256_text(
            _ast_string_assignment(control_module, "instruction")
        ),
    }


def _ast_string_assignment(module: ast.Module, name: str) -> str:
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
        if isinstance(node.value, ast.JoinedStr) and all(
            isinstance(value, ast.Constant) and isinstance(value.value, str)
            for value in node.value.values
        ):
            return "".join(str(value.value) for value in node.value.values)
        raise RuntimeError(f"M3 prompt {name} is no longer a string literal")
    raise RuntimeError(f"M3 prompt not found: {name}")


def _strict_json(raw: str) -> Any:
    text = raw.strip().strip("```json").strip("```python").strip("```").strip()
    return json.loads(text)


def _clip_number(value: str) -> int:
    match = re.fullmatch(r"CLIP_(\d+)", str(value))
    if match is None:
        raise ValueError(f"invalid M3 clip key: {value!r}")
    return int(match.group(1))


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record_differences(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    """Compare source identity, allowing staged image paths with equal bytes."""
    ignored = {"image_paths"}
    differences = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in sorted((set(expected) | set(actual)) - ignored)
        if expected.get(key) != actual.get(key)
    }
    expected_paths = [Path(value) for value in expected.get("image_paths") or []]
    actual_paths = [Path(value) for value in actual.get("image_paths") or []]
    if len(expected_paths) != len(actual_paths):
        differences["image_paths"] = {
            "expected_count": len(expected_paths),
            "actual_count": len(actual_paths),
        }
    else:
        mismatches = []
        for index, (old_path, new_path) in enumerate(
            zip(expected_paths, actual_paths, strict=True)
        ):
            if not old_path.is_file() or not new_path.is_file():
                mismatches.append(
                    {
                        "index": index,
                        "expected_exists": old_path.is_file(),
                        "actual_exists": new_path.is_file(),
                    }
                )
            elif _sha256_file(old_path) != _sha256_file(new_path):
                mismatches.append({"index": index, "content_sha256_mismatch": True})
        if mismatches:
            differences["image_paths"] = mismatches
    return differences
