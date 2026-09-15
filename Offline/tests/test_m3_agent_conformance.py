from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.baseline_runtime.adapters.m3_agent import (
    M3AgentAdapter,
    M3_EMPTY_KNOWLEDGE_SEARCH_RULE,
    M3_MEMORY_OUTPUT_CONSTRAINT,
    M3_PROMPT_SHA256,
    m3_conformance_manifest,
)
from benchmarks.memgallery_harness.runner.prompts import prompt_sha256
from benchmarks.baseline_runtime.protocol import RetrievalRequest, RetrievedMemory
from embedding.chunk_builder import Chunk


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
M3_ROOT = OFFLINE_ROOT / "baselines" / "m3-agent-master"


class M3AgentConformanceTest(unittest.TestCase):
    def test_initial_control_turn_requires_search_without_memory_evidence(self) -> None:
        self.assertIn("must choose Action: [Search]", M3_EMPTY_KNOWLEDGE_SEARCH_RULE)
        self.assertIn(
            "must not answer from prior knowledge", M3_EMPTY_KNOWLEDGE_SEARCH_RULE
        )

    @staticmethod
    def _control_test_adapter(responses: list[str], search_results: list[tuple]) -> tuple:
        adapter = object.__new__(M3AgentAdapter)
        adapter.baseline = "M3-Agent-caption"
        adapter.graph = SimpleNamespace(refresh_equivalences=lambda: None)
        adapter._control_system_prompt = "Question: {question}"
        adapter._control_instruction = "\nChoose an action."
        adapter._clip_sources = {1: {"session_id": "S00"}}
        adapter._execution_trace_path = None
        seen_messages: list[str] = []
        response_iter = iter(responses)
        search_iter = iter(search_results)

        def fake_completion(messages, *, response_format=None):
            del response_format
            seen_messages.append(str(messages[-1]["content"]))
            return next(response_iter), {}, 1, {"finish_reason": "stop"}

        def fake_search(*args, **kwargs):
            del args, kwargs
            return next(search_iter)

        adapter._chat_completion = fake_completion
        adapter._retrieve = SimpleNamespace(search=fake_search)
        adapter._retrieved_clip = lambda row: RetrievedMemory(
            memory_id=f"m3:clip:{row['clip_id']}", text="memory evidence"
        )
        adapter._runtime_prompt_hashes = lambda: {
            "control_system_prompt": "system",
            "control_instruction": "instruction",
        }
        adapter._trace_event = lambda **kwargs: None
        return adapter, seen_messages

    def test_control_keeps_searching_after_an_empty_search(self) -> None:
        adapter, seen = self._control_test_adapter(
            [
                "Action: [Search]\nContent: first query",
                "Action: [Search]\nContent: second query",
                "Action: [Answer]\nContent: done",
            ],
            [
                ({}, [], {}),
                ({"CLIP_1": ["memory evidence"]}, [1], {1: 0.9}),
            ],
        )
        result = adapter.retrieve(RetrievalRequest(query_id="q", text="question"))
        self.assertEqual([row["action"] for row in result.trace["rounds"]], ["Search", "Search", "Answer"])
        self.assertEqual([item.memory_id for item in result.items], ["m3:clip:1"])
        self.assertIn(M3_EMPTY_KNOWLEDGE_SEARCH_RULE, seen[0])
        self.assertIn(M3_EMPTY_KNOWLEDGE_SEARCH_RULE, seen[1])
        self.assertNotIn(M3_EMPTY_KNOWLEDGE_SEARCH_RULE, seen[2])

    def test_final_control_round_does_not_force_answer_without_evidence(self) -> None:
        adapter, seen = self._control_test_adapter(
            ["Action: [Search]\nContent: another query"] * 5,
            [({}, [], {})] * 5,
        )
        result = adapter.retrieve(RetrievalRequest(query_id="q", text="question"))
        self.assertEqual([row["action"] for row in result.trace["rounds"]], ["Search"] * 5)
        self.assertEqual(result.items, [])
        self.assertEqual(len(seen), 5)
        self.assertIn(M3_EMPTY_KNOWLEDGE_SEARCH_RULE, seen[-1])
        self.assertNotIn("must be [Answer]", seen[-1])

    def test_control_answer_without_evidence_is_forced_through_real_search(self) -> None:
        adapter, _seen = self._control_test_adapter(
            [
                "Action: [Answer]\nContent: morning phone contact field",
                "Action: [Answer]\nContent: grounded answer",
            ],
            [({"CLIP_1": ["memory evidence"]}, [1], {1: 0.9})],
        )
        result = adapter.retrieve(RetrievalRequest(query_id="q", text="question"))
        first = result.trace["rounds"][0]
        self.assertEqual(first["requested_action"], "Answer")
        self.assertEqual(first["action"], "Search")
        self.assertTrue(first["forced_search_without_evidence"])
        self.assertEqual(result.trace["forced_search_without_evidence_count"], 1)
        self.assertEqual([item.memory_id for item in result.items], ["m3:clip:1"])
        self.assertEqual(result.trace["agent_answer"], "grounded answer")

    def test_official_qwen_parser_uses_prompt_schema_key(self) -> None:
        path = M3_ROOT / "mmagent" / "memory_processing_qwen.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        generate = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_all_memories"
        )
        epi_keys = [
            node.value.value
            for node in ast.walk(generate)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "epi_key" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        self.assertEqual(epi_keys, ["video_description"])

    def test_internal_prompts_match_official_literals(self) -> None:
        manifest = m3_conformance_manifest(
            "memgallery", answer_prompt_sha256=prompt_sha256(), source_root=M3_ROOT
        )
        self.assertTrue(manifest["official_core_patched"])
        self.assertEqual(manifest["answer_path"], "unchanged benchmark QA prompt")
        self.assertEqual(manifest["answer_prompt_sha256"], prompt_sha256())
        self.assertTrue(
            manifest["empty_knowledge_search_rule"]["applies_while_handoff_empty"]
        )
        self.assertTrue(
            manifest["empty_knowledge_search_rule"][
                "final_answer_requires_discovered_clip"
            ]
        )
        self.assertEqual(
            manifest["empty_knowledge_search_rule"]["text"],
            M3_EMPTY_KNOWLEDGE_SEARCH_RULE,
        )
        for name, expected in M3_PROMPT_SHA256.items():
            self.assertEqual(manifest["internal_prompt_sha256"][name]["actual"], expected)

    def test_qwen_prompt_precedes_dialogue_observation_without_system_prompt(self) -> None:
        prompt = "OFFICIAL_QWEN_MEMORY_PROMPT"
        adapter = object.__new__(M3AgentAdapter)
        adapter._prompts = SimpleNamespace(prompt_generate_memory_with_ids_sft=prompt)
        messages = adapter._memory_messages(
            Chunk(chunk_id="d1", text="user: hello"),
            {"image_paths": [], "image_ids": [], "image_captions": []},
        )
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertEqual(messages[0]["content"][0]["text"], prompt)
        self.assertEqual(
            messages[0]["content"][1]["text"], M3_MEMORY_OUTPUT_CONSTRAINT
        )
        self.assertIn("user: hello", messages[0]["content"][2]["text"])

    def test_memory_generation_retries_a_length_truncated_response(self) -> None:
        adapter = object.__new__(M3AgentAdapter)
        adapter.config = {"retries": 1, "executor_max_tokens": 1024}
        adapter.sample_id = "sample"
        adapter._general = SimpleNamespace(validate_and_fix_json=json.loads)
        calls: list[dict] = []
        responses = iter(
            [
                ('{"video_description": ["cut off', {"completion_tokens": 1024}, 1, {"finish_reason": "length"}),
                (
                    json.dumps(
                        {
                            "video_description": ["A concise event."],
                            "high_level_conclusions": ["A concise inference."],
                        }
                    ),
                    {"completion_tokens": 32},
                    1,
                    {"finish_reason": "stop"},
                ),
            ]
        )

        def fake_completion(messages, *, response_format=None):
            calls.append({"messages": messages, "response_format": response_format})
            return next(responses)

        adapter._chat_completion = fake_completion
        with tempfile.TemporaryDirectory() as directory:
            adapter.state_dir = Path(directory)
            with patch("benchmarks.baseline_runtime.adapters.m3_agent.time.sleep"):
                result = adapter._memory_completion(
                    [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}],
                    5,
                )
            failure = (
                Path(directory)
                / "memory_generation_failures"
                / "clip_000005_attempt_01.json"
            )
            self.assertTrue(failure.is_file())
            self.assertEqual(json.loads(failure.read_text())["finish_reason"], "length")

        self.assertEqual(result[3], 2)
        self.assertEqual(result[4], "stop")
        self.assertEqual(result[5], ["A concise event."])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["response_format"]["type"], "json_schema")
        self.assertIn("Retry because", calls[1]["messages"][0]["content"][-1]["text"])

    def test_unsupported_feature_ids_are_rejected_with_provenance(self) -> None:
        episodic, semantic, rejected = M3AgentAdapter._validate_memories(
            {
                "video_description": ["Alice started pottery."],
                "high_level_conclusions": [
                    "Alice is learning a new skill.",
                    "Equivalence: <face_1>, <voice_1>",
                    "Equivalence: <face_Alice>, <voice_assistant>",
                ],
            },
            1,
        )
        self.assertEqual(episodic, ["Alice started pottery."])
        self.assertEqual(semantic, ["Alice is learning a new skill."])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all(row["source_field"] == "high_level_conclusions" for row in rejected)
        )
        self.assertIn("<face_Alice>", rejected[1]["text"])

    def test_empty_grounded_memory_is_preserved_without_fallback(self) -> None:
        episodic, semantic, rejected = M3AgentAdapter._validate_memories(
            {
                "video_description": [],
                "high_level_conclusions": ["Equivalence: <face_1>, <voice_1>"],
            },
            1,
        )
        self.assertEqual(episodic, [])
        self.assertEqual(semantic, [])
        self.assertEqual(len(rejected), 1)


if __name__ == "__main__":
    unittest.main()
