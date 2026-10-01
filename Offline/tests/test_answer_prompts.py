from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from benchmarks.answer_response import (
    AnswerFormatError,
    parse_answer_block,
    recover_unique_answer_block,
)
from benchmarks.h2hmem_harness.eval_h2hmem import answer_conversation_job
from benchmarks.h2hmem_harness import prompts as h2_prompts
from benchmarks.memeye_harness import prompts as memeye_prompts
from benchmarks.memlens_harness import prompts as memlens_prompts
from benchmarks.memgallery_harness.eval_memgallery import answer_dataset_job
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from benchmarks.memgallery_harness.runner import prompts as memgallery_prompts
from benchmarks.wma_harness.eval_wma import answer_job as answer_wma_job
from benchmarks.wma_harness.runner import prompts as wma_prompts
from benchmarks.zero_hit import ZERO_HIT_PROMPT_MARKER


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _reference_module():
    path = WORKSPACE_ROOT / "answer_prompts.py"
    spec = importlib.util.spec_from_file_location("reference_answer_prompts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CustomPromptParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = _reference_module()

    def assert_prompt_parity(self, source, module, metadata, evidence):
        expected = self.reference.build_benchmark_answer_messages(
            dataset_source=source,
            sample_metadata=metadata,
            memory_evidence=evidence,
        )
        actual = module.build_answer_messages(
            question=metadata["question"],
            question_type=metadata.get("question_type", ""),
            query_images=metadata.get("query_images"),
            memory_evidence=evidence,
            **(
                {"question_date": metadata.get("question_date", "")}
                if module in (memeye_prompts, memlens_prompts)
                else {}
            ),
        )
        self.assertEqual(actual, expected)

    def test_memeye_messages_match_reference(self):
        self.assert_prompt_parity(
            "memeye",
            memeye_prompts,
            {
                "question": "Who is shown in the question image?",
                "question_type": "X3/Y2",
                "query_images": [
                    {"id": "avatars/P01_marcus.png", "caption": "Marcus"}
                ],
            },
            ["first memory", "second memory"],
        )

    def test_memlens_messages_match_reference_and_include_question_date(self):
        self.assert_prompt_parity(
            "memlens",
            memlens_prompts,
            {
                "question": "What happened before this question?",
                "question_type": "temporal_reasoning",
                "question_date": "2024/05/31 (Fri) 07:58",
                "query_images": [
                    {"id": "needle/image.jpg", "caption": "A calendar"}
                ],
            },
            ["first memory", "second memory"],
        )

    def test_memgallery_messages_match_reference_for_all_special_types(self):
        for question_type in ("FR", "CD", "VS", "AR"):
            with self.subTest(question_type=question_type):
                self.assert_prompt_parity(
                    "mem_gallery",
                    memgallery_prompts,
                    {
                        "question": "What happened?",
                        "question_type": question_type,
                        "query_images": [{"id": "Q_IMG_1", "caption": "A park"}],
                    },
                    ["first memory", "second memory"],
                )

    def test_h2hmem_messages_match_reference_for_every_question_type(self):
        question_types = (
            *h2_prompts.INSTRUCTIONS,
            *h2_prompts.QUESTION_TYPE_ALIASES,
        )
        for question_type in question_types:
            with self.subTest(question_type=question_type):
                self.assert_prompt_parity(
                    "h2hmem",
                    h2_prompts,
                    {
                        "question": "What happened?",
                        "question_type": question_type,
                        "query_images": [{"id": "Q_IMG_1", "caption": "A park"}],
                    },
                    ["first memory", "second memory"],
                )

    def test_h2hmem_causal_inference_uses_causal_instruction(self):
        messages = h2_prompts.build_answer_messages(
            question="What caused the change?",
            question_type="Multimodal Causal Inference",
            query_images=None,
            memory_evidence=["text and visual evidence"],
        )
        self.assertIn("causal reasoning", messages[0]["content"])

    def test_wma_messages_match_reference(self):
        self.assert_prompt_parity(
            "worldmemarena",
            wma_prompts,
            {
                "question": "When did it happen?",
                "question_type": "TR",
                "query_images": [{"id": "Q_IMG_1", "caption": "A calendar"}],
            },
            ["first memory", "second memory"],
        )

    def test_empty_evidence_is_opt_in_and_omits_memory_section(self):
        cases = (
            (memgallery_prompts, "FR", "Conversation memory:"),
            (h2_prompts, "Unimodal Precise Recall", "Conversation memory:"),
            (wma_prompts, "TR", "Retrieved memories:"),
        )
        for module, question_type, evidence_heading in cases:
            kwargs = {
                "question": "What happened?",
                "question_type": question_type,
                "query_images": [{"id": "Q_IMG_1", "caption": "A park"}],
                "memory_evidence": [],
            }
            with self.subTest(module=module.__name__, mode="strict"):
                with self.assertRaisesRegex(ValueError, "at least one non-empty"):
                    module.build_answer_messages(**kwargs)
            with self.subTest(module=module.__name__, mode="ppo-empty"):
                messages = module.build_answer_messages(
                    **kwargs, allow_empty_evidence=True
                )
                self.assertIn("No conversation-memory evidence was selected", messages[0]["content"])
                self.assertNotIn(evidence_heading, messages[1]["content"])
                self.assertIn("Question Image:", messages[1]["content"])
                self.assertIn("Question: What happened?", messages[1]["content"])
                self.assertEqual(module.PPO_EMPTY_PROMPT_VERSION, "ppo-empty-evidence-20260911-v1")
                self.assertEqual(len(module.ppo_empty_prompt_sha256()), 64)

    def test_retired_answer_prompts_are_absent_from_production_code(self):
        retired_fragments = (
            "You answer questions using only the supplied retrieved memories",
            "Answer the following H2HMem question from the retrieved conversation",
            "You are an intelligent memory assistant. Answer the user's question",
            "Your task is to answer the question about the conversation between",
        )
        production_files = [
            *sorted((WORKSPACE_ROOT / "Offline" / "src").rglob("*.py")),
            *sorted((WORKSPACE_ROOT / "Offline" / "scripts").glob("*.py")),
        ]
        for path in production_files:
            source = path.read_text(encoding="utf-8")
            for fragment in retired_fragments:
                self.assertNotIn(fragment, source, str(path))


class AnswerContractTest(unittest.TestCase):
    def test_extracts_one_nonempty_answer_block(self):
        self.assertEqual(parse_answer_block("  <answer>Paris</answer>\n"), "Paris")

    def test_rejects_missing_empty_or_multiple_blocks(self):
        for raw in (
            "Paris",
            "<answer></answer>",
            "prefix <answer>Paris</answer>",
            "<answer>Paris</answer><answer>London</answer>",
        ):
            with self.subTest(raw=raw), self.assertRaises(AnswerFormatError):
                parse_answer_block(raw)

    def test_recovery_accepts_only_one_embedded_nonempty_block(self):
        self.assertEqual(
            recover_unique_answer_block("prefix <answer>Paris</answer> suffix"),
            "Paris",
        )
        for raw in (
            "Paris",
            "<answer></answer>",
            "<answer>Paris</answer><answer>London</answer>",
        ):
            with self.subTest(raw=raw), self.assertRaises(AnswerFormatError):
                recover_unique_answer_block(raw)


class PrebuiltMessageClientTest(unittest.TestCase):
    def test_final_answer_rejects_internal_agent_history(self):
        client = VLMAnswerClient(retries=0)
        invalid_cases = (
            [{"role": "user", "content": "Question only"}],
            [
                {"role": "system", "content": "QA prompt"},
                {"role": "assistant", "content": "Internal agent answer"},
                {"role": "user", "content": "Question"},
            ],
            [
                {"role": "system", "content": "Internal memory-agent prompt"},
                {"role": "system", "content": "QA prompt"},
            ],
        )
        for messages in invalid_cases:
            with self.subTest(messages=messages), self.assertRaisesRegex(
                ValueError, "Final answer isolation"
            ):
                client.answer_messages_with_usage(messages=messages, memory_items=[])

    def test_context_capacity_truncation_retries_with_smaller_images(self):
        class ContextLimitedClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(num_predict=512, retries=0)
                self.payloads = []

            def _post_json(self, _url, payload):
                self.payloads.append(payload)
                if len(self.payloads) == 1:
                    return {
                        "choices": [
                            {
                                "message": {"content": "<answer>Par"},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 32760,
                            "completion_tokens": 8,
                            "total_tokens": 32768,
                        },
                    }
                return {
                    "choices": [
                        {
                            "message": {"content": "<answer>Paris</answer>"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20000,
                        "completion_tokens": 3,
                        "total_tokens": 20003,
                    },
                }

        client = ContextLimitedClient()
        response = client.answer_messages_with_usage(
            messages=[
                {"role": "system", "content": "Return an answer block."},
                {"role": "user", "content": "Where?"},
            ],
            memory_items=[],
        )

        self.assertEqual(response.text, "<answer>Paris</answer>")
        self.assertEqual(len(client.payloads), 2)
        self.assertEqual(response.attempts, 1)
        self.assertEqual(response.failed_attempts, 0)
        self.assertEqual(response.usage["prompt_tokens"], 52760)
        self.assertEqual(response.usage["completion_tokens"], 11)

    def test_format_retry_adds_repetition_penalty_without_changing_messages(self):
        class RetryingClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(retries=1)
                self.payloads = []

            def _post_json(self, _url, payload):
                self.payloads.append(payload)
                content = "malformed" if len(self.payloads) == 1 else "<answer>Paris</answer>"
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }

        messages = [
            {"role": "system", "content": "Return an answer block."},
            {"role": "user", "content": "Where?"},
        ]
        client = RetryingClient()
        response = client.answer_messages_with_usage(messages=messages, memory_items=[])

        self.assertEqual(response.text, "<answer>Paris</answer>")
        self.assertNotIn("repetition_penalty", client.payloads[0])
        self.assertEqual(client.payloads[1]["repetition_penalty"], 1.05)
        self.assertEqual(client.payloads[0]["messages"], client.payloads[1]["messages"])

    def test_openrouter_format_retry_uses_json_schema_without_changing_messages(self):
        class RetryingClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(base_url="https://openrouter.ai/api/v1", retries=1)
                self.payloads = []

            def _post_json(self, _url, payload):
                self.payloads.append(payload)
                content = "Paris" if len(self.payloads) == 1 else '{"answer":"Paris"}'
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }

        messages = [
            {"role": "system", "content": "Return an answer block."},
            {"role": "user", "content": "Where?"},
        ]
        client = RetryingClient()
        response = client.answer_messages_with_usage(messages=messages, memory_items=[])

        self.assertEqual(response.text, "<answer>Paris</answer>")
        self.assertIn("structured_outputs", client.payloads[0])
        self.assertNotIn("response_format", client.payloads[0])
        self.assertNotIn("structured_outputs", client.payloads[1])
        self.assertEqual(
            client.payloads[1]["response_format"]["json_schema"]["name"],
            "benchmark_answer",
        )
        self.assertNotIn("provider", client.payloads[1])
        self.assertEqual(client.payloads[0]["messages"], client.payloads[1]["messages"])

    def test_exhausted_format_retries_recover_one_embedded_block(self):
        class RecoveringClient(VLMAnswerClient):
            def _post_json(self, _url, _payload):
                return {
                    "choices": [
                        {"message": {"content": "prefix <answer>Paris</answer> suffix"}}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }

        client = RecoveringClient(retries=2)
        response = client.answer_messages_with_usage(
            messages=[
                {"role": "system", "content": "Return an answer block."},
                {"role": "user", "content": "Where?"},
            ],
            memory_items=[],
        )
        self.assertEqual(response.text, "<answer>Paris</answer>")
        self.assertEqual(response.raw_text, "prefix <answer>Paris</answer> suffix")
        self.assertEqual(response.attempts, 3)
        self.assertEqual(response.failed_attempts, 2)
        self.assertEqual(response.usage["total_tokens"], 39)

    def test_attaching_images_does_not_modify_prompt_text(self):
        class CapturingClient(VLMAnswerClient):
            def _post_json(self, _url, payload):
                self.payload = payload
                return {
                    "choices": [{"message": {"content": "<answer>Paris</answer>"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (8, 8), color="red").save(image_path)
            messages = memgallery_prompts.build_answer_messages(
                question="Where?",
                question_type="VS",
                memory_evidence=["Image evidence"],
                query_images=[{"id": "query.png", "caption": "red"}],
            )
            client = CapturingClient()
            response = client.answer_messages_with_usage(
                messages=messages,
                memory_items=[
                    {
                        "text": "Image evidence",
                        "metadata": {"image_id": "D1:IMG_001"},
                        "images": [
                            {"path": str(image_path), "img_id": "D1:IMG_001", "kind": "image"}
                        ],
                    }
                ],
                query_image={"path": str(image_path), "img_id": "query.png"},
                category="VS",
            )

        self.assertEqual(response.image_count, 2)
        self.assertEqual(client.payload["messages"][0], messages[0])
        self.assertEqual(
            client.payload["structured_outputs"],
            {"regex": r"<answer>[^<]+</answer>"},
        )
        user_content = client.payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["text"], messages[1]["content"])
        self.assertEqual(
            [item["type"] for item in user_content],
            ["text", "image_url", "image_url"],
        )


class HarnessAnswerJobTest(unittest.TestCase):
    class FakeClient:
        retries = 2

        def __init__(self, answer):
            self.answer = answer

        def answer_messages_with_usage(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                text=f"<answer>{self.answer}</answer>",
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                attempts=1,
                failed_attempts=0,
                image_count=0,
            )

        @staticmethod
        def count_answer_images(*_args, **_kwargs):
            return 0

    def test_memgallery_job_ignores_frozen_old_prompt(self):
        client = self.FakeClient("Paris")
        result, trace = answer_dataset_job(
            client,
            {
                "query_id": "q1",
                "manifest_question_id": "travel_q0000",
                "dataset": "travel",
                "sample_id": "Alice",
                "qa_index": 1,
                "question": "Where did she go?",
                "category": "FR",
                "question_prompt": "RETIRED PROMPT",
                "system_prompt": "RETIRED SYSTEM",
                "query_image": None,
                "original_answer": "Paris",
                "retrieved_ids": ["m1"],
                "retrieved_source_groups": [["R1"]],
                "clue": ["R1"],
                "memory_items": [
                    {"text": "Alice went to Paris.", "metadata": {"session_id": "S1"}}
                ],
                "retrieval_top_k": [],
            },
            allow_answer_errors=False,
        )
        self.assertEqual(result["system_answer"], "Paris")
        self.assertEqual(result["answer_raw_response"], "<answer>Paris</answer>")
        self.assertNotIn("RETIRED", str(client.request["messages"]))
        self.assertEqual(trace["answer_prompt_messages"], client.request["messages"])

    def test_memgallery_zero_hit_uses_prompt_marker_without_memory(self):
        client = self.FakeClient("Unknown")
        result, trace = answer_dataset_job(
            client,
            {
                "query_id": "q-empty",
                "manifest_question_id": "travel_q0001",
                "dataset": "travel",
                "sample_id": "Alice",
                "qa_index": 2,
                "question": "Where did she go?",
                "category": "FR",
                "query_image": None,
                "original_answer": "",
                "retrieved_ids": [],
                "retrieved_source_groups": [],
                "clue": [],
                "memory_items": [],
                "retrieval_top_k": [],
            },
            allow_answer_errors=False,
        )
        self.assertTrue(result["zero_hit_prompt_marker_used"])
        self.assertTrue(trace["zero_hit_prompt_marker_used"])
        self.assertEqual(client.request["memory_items"], [])
        self.assertIn(ZERO_HIT_PROMPT_MARKER, str(client.request["messages"]))

    def test_h2hmem_job_uses_custom_messages_and_parses_tags(self):
        client = self.FakeClient("Almond")
        result, trace = answer_conversation_job(
            client,
            {
                "uid": "q1",
                "query_id": "q1",
                "manifest_question_id": "manifest-q1",
                "conversation_id": "dialogue1",
                "session_id": "session2",
                "question": "What is the cat's name?",
                "category": "Unimodal Precise Recall",
                "memory_items": [
                    {"text": "The cat's name is Almond.", "metadata": {"session_id": "session2"}}
                ],
                "retrieval_top_k": [{"memory_id": "m1", "source_dialogue_ids": ["R1"]}],
                "query_image_payload": None,
            },
        )
        self.assertEqual(result["system_answer"], "Almond")
        self.assertEqual(trace["answer_prompt_messages"], client.request["messages"])

    def test_h2hmem_zero_hit_uses_prompt_marker_without_memory(self):
        client = self.FakeClient("Not mentioned")
        result, trace = answer_conversation_job(
            client,
            {
                "uid": "q-empty",
                "query_id": "q-empty",
                "manifest_question_id": "manifest-empty",
                "conversation_id": "dialogue1",
                "session_id": "session2",
                "question": "What was mentioned?",
                "category": "Unimodal Precise Recall",
                "memory_items": [],
                "retrieval_top_k": [],
                "query_image_payload": None,
            },
        )
        self.assertTrue(result["zero_hit_prompt_marker_used"])
        self.assertTrue(trace["zero_hit_prompt_marker_used"])
        self.assertEqual(client.request["memory_items"], [])
        self.assertIn(ZERO_HIT_PROMPT_MARKER, str(client.request["messages"]))

    def test_wma_job_uses_custom_messages_and_parses_tags(self):
        client = self.FakeClient("2024")
        result, trace = answer_wma_job(
            client,
            {
                "query_id": "sample::QA00::1",
                "manifest_question_id": "sample:QA00:Q001",
                "sample_id": "sample",
                "checkpoint_id": "QA00",
                "covered_sessions": ["S00"],
                "visible_sessions": ["S00"],
                "question": "When was the trip?",
                "category": "TR",
                "memory_items": [
                    {"text": "The trip was in 2024.", "metadata": {"session_id": "S00"}}
                ],
                "retrieval_top_k": [
                    {"memory_id": "m1", "source_dialogue_ids": ["R1"], "session_id": "S00"}
                ],
            },
        )
        self.assertEqual(result["system_answer"], "2024")
        self.assertEqual(trace["answer_prompt_messages"], client.request["messages"])

    def test_wma_zero_hit_uses_prompt_marker_without_memory(self):
        client = self.FakeClient("Unknown")
        result, trace = answer_wma_job(
            client,
            {
                "query_id": "sample::QA00::2",
                "manifest_question_id": "sample:QA00:Q002",
                "sample_id": "sample",
                "checkpoint_id": "QA00",
                "covered_sessions": ["S00"],
                "visible_sessions": ["S00"],
                "question": "What happened?",
                "category": "TR",
                "memory_items": [],
                "retrieval_top_k": [],
            },
        )
        self.assertTrue(result["zero_hit_prompt_marker_used"])
        self.assertTrue(trace["zero_hit_prompt_marker_used"])
        self.assertEqual(client.request["memory_items"], [])
        self.assertIn(ZERO_HIT_PROMPT_MARKER, str(client.request["messages"]))


if __name__ == "__main__":
    unittest.main()
