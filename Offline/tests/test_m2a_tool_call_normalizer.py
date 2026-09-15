from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from langchain_core.messages import AIMessage


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "M2A"
    / "agent"
    / "agents"
    / "tool_call_normalizer.py"
)
SPEC = importlib.util.spec_from_file_location("m2a_tool_call_normalizer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
normalize_qwen_tool_calls = MODULE.normalize_qwen_tool_calls
cap_tool_calls_to_budget = MODULE.cap_tool_calls_to_budget


class M2AToolCallNormalizerTests(unittest.TestCase):
    def test_valid_textual_tool_call_is_promoted_without_repair_marker(self):
        response = AIMessage(
            content='<tool_call>{"name":"finish_memory_update","arguments":{}}</tool_call>'
        )

        normalized = normalize_qwen_tool_calls(response)

        self.assertEqual(normalized.tool_calls[0]["name"], "finish_memory_update")
        self.assertEqual(normalized.tool_calls[0]["args"], {})
        self.assertNotIn("qwen_tool_call_repairs", normalized.additional_kwargs)

    def test_surface_syntax_is_repaired_and_audited(self):
        response = AIMessage(
            content='<tool_call>{"name":"finish_memory_update" "arguments":{}}</tool_call>'
        )

        normalized = normalize_qwen_tool_calls(response)

        self.assertEqual(normalized.tool_calls[0]["name"], "finish_memory_update")
        self.assertEqual(normalized.tool_calls[0]["args"], {})
        repairs = normalized.additional_kwargs["qwen_tool_call_repairs"]
        self.assertEqual(repairs[0]["tool_call_id"], normalized.tool_calls[0]["id"])
        self.assertEqual(len(repairs[0]["raw_sha256"]), 64)

    def test_repaired_payload_still_requires_exact_m2a_schema(self):
        response = AIMessage(
            content='<tool_call>{"name":"finish_memory_update" "unexpected":1}</tool_call>'
        )

        with self.assertRaisesRegex(ValueError, "exactly 'name' and 'arguments'"):
            normalize_qwen_tool_calls(response)

    def test_mixed_assistant_content_is_preserved_and_audited(self):
        response = AIMessage(
            content=(
                "I will search memory first.\n"
                '<tool_call>{"name":"finish_memory_update","arguments":{}}</tool_call>'
            )
        )

        normalized = normalize_qwen_tool_calls(response)

        self.assertEqual(normalized.content, "I will search memory first.")
        self.assertEqual(normalized.tool_calls[0]["name"], "finish_memory_update")
        audit = normalized.response_metadata["qwen_mixed_tool_content"]
        self.assertEqual(audit["characters"], len(normalized.content))
        self.assertEqual(len(audit["sha256"]), 64)

    def test_budget_cap_trims_structured_and_raw_tool_calls(self):
        response = AIMessage(
            content="",
            tool_calls=[
                {"name": "first", "args": {}, "id": "call-1"},
                {"name": "second", "args": {}, "id": "call-2"},
            ],
            additional_kwargs={
                "tool_calls": [
                    {"id": "call-1", "type": "function"},
                    {"id": "call-2", "type": "function"},
                ]
            },
        )

        capped = cap_tool_calls_to_budget(response, 1)

        self.assertEqual([call["id"] for call in capped.tool_calls], ["call-1"])
        self.assertEqual(
            [call["id"] for call in capped.additional_kwargs["tool_calls"]],
            ["call-1"],
        )

    def test_budget_cap_rejects_negative_limit(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cap_tool_calls_to_budget(AIMessage(content="done"), -1)


if __name__ == "__main__":
    unittest.main()
