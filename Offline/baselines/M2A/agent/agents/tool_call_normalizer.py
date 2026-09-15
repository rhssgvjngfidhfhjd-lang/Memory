"""Compatibility helpers for OpenAI-compatible Qwen tool-call responses."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from json_repair import repair_json
from langchain_core.messages import AIMessage


_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _load_tool_json(raw: str, *, field: str) -> tuple[Any, bool]:
    """Parse provider-rendered JSON, repairing syntax without changing schema.

    Qwen occasionally emits the intended tool payload with a missing comma,
    colon, quote, or trailing delimiter.  ``json_repair`` is restricted to
    this provider-compatibility boundary; the caller still validates the
    complete M2A tool schema before any tool can execute.
    """

    try:
        return json.loads(raw), False
    except json.JSONDecodeError as exc:
        try:
            repaired = repair_json(raw, return_objects=True)
        except (TypeError, ValueError) as repair_exc:
            raise ValueError(
                f"Malformed Qwen tool call: {field} is not valid JSON"
            ) from repair_exc
        if repaired in (None, ""):
            raise ValueError(
                f"Malformed Qwen tool call: {field} is not valid JSON"
            ) from exc
        return repaired, True


def normalize_qwen_tool_calls(response: AIMessage) -> AIMessage:
    """Turn strict Qwen ``<tool_call>`` text into LangChain tool calls.

    Some vLLM Qwen endpoints return a valid tool request in ``content`` while
    leaving ``AIMessage.tool_calls`` empty. Native structured tool calls and
    ordinary assistant text are returned unchanged. OpenAI-compatible messages
    may carry both assistant content and tool calls, so text outside tagged JSON
    objects is preserved and audited. Surface syntax can be repaired, but the
    strict M2A payload schema is always validated before execution.
    """

    if response.tool_calls:
        return response

    content = response.content
    if not isinstance(content, str):
        return response
    if "<tool_call" not in content and "</tool_call" not in content:
        return response

    matches = list(_TOOL_CALL_TAG.finditer(content))
    if not matches:
        raise ValueError("Malformed Qwen tool call: missing closing </tool_call> tag")

    unmatched = _TOOL_CALL_TAG.sub("", content).strip()

    tool_calls: list[dict[str, Any]] = []
    repairs: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        raw_payload = match.group(1)
        payload, payload_repaired = _load_tool_json(raw_payload, field="body")

        if not isinstance(payload, dict):
            raise ValueError("Malformed Qwen tool call: body must be a JSON object")
        if set(payload) != {"name", "arguments"}:
            raise ValueError(
                "Malformed Qwen tool call: expected exactly 'name' and 'arguments'"
            )

        name = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(name, str) or not name:
            raise ValueError("Malformed Qwen tool call: 'name' must be a non-empty string")
        if isinstance(arguments, str):
            arguments, arguments_repaired = _load_tool_json(
                arguments, field="string 'arguments'"
            )
        else:
            arguments_repaired = False
        if not isinstance(arguments, dict):
            raise ValueError("Malformed Qwen tool call: 'arguments' must be a JSON object")

        digest = hashlib.sha256(
            f"{index}:{raw_payload}".encode("utf-8")
        ).hexdigest()[:24]
        if payload_repaired or arguments_repaired:
            repairs.append(
                {
                    "tool_call_id": f"qwen_tool_call_{digest}",
                    "raw_sha256": hashlib.sha256(
                        raw_payload.encode("utf-8")
                    ).hexdigest(),
                }
            )
        tool_calls.append(
            {
                "name": name,
                "args": arguments,
                "id": f"qwen_tool_call_{digest}",
                "type": "tool_call",
            }
        )

    update: dict[str, Any] = {"content": unmatched, "tool_calls": tool_calls}
    if repairs:
        update["additional_kwargs"] = {
            **response.additional_kwargs,
            "qwen_tool_call_repairs": repairs,
        }
    if unmatched:
        update["response_metadata"] = {
            **response.response_metadata,
            "qwen_mixed_tool_content": {
                "sha256": hashlib.sha256(unmatched.encode("utf-8")).hexdigest(),
                "characters": len(unmatched),
            },
        }
    return response.model_copy(update=update)


def cap_tool_calls_to_budget(response: AIMessage, max_calls: int) -> AIMessage:
    """Keep only calls that fit the current hard iteration budget.

    Qwen-compatible servers may return multiple calls despite
    ``parallel_tool_calls=False``. Trimming before the assistant message enters
    graph state keeps the tool-message protocol valid and guarantees that calls
    beyond the original M2A budget cannot execute.
    """

    if max_calls < 0:
        raise ValueError("max_calls must be non-negative")
    if len(response.tool_calls) <= max_calls:
        return response

    additional_kwargs = dict(response.additional_kwargs)
    raw_tool_calls = additional_kwargs.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        additional_kwargs["tool_calls"] = raw_tool_calls[:max_calls]
    return response.model_copy(
        update={
            "tool_calls": response.tool_calls[:max_calls],
            "additional_kwargs": additional_kwargs,
        }
    )
