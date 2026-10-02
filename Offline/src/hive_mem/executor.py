"""One-call, one-node chunk summarization and attribute extraction."""

import json
import re
from dataclasses import dataclass
from typing import List, Sequence, cast

import numpy as np
from json_repair import repair_json

from .entity_schema import (
    MAX_ATTRIBUTE_RECORDS,
    MAX_ATTRIBUTE_VALUES,
    TEXT_ATTRIBUTE_KEYS,
    VISUAL_ATTRIBUTE_KEYS,
    attribute_prompt_block,
    normalize_entities,
    normalize_attributes,
    parse_entities_payload,
)


EXECUTOR_VISUAL_INPUTS = ("image", "caption", "image_caption")
EXECUTOR_PROMPT_SCHEMA_VERSION = 6


def _attribute_schema(keys: Sequence[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "attribute": {"type": "string", "enum": list(keys)},
            "value": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_ATTRIBUTE_VALUES,
            },
        },
        # OpenAI strict structured outputs require every declared property to
        # appear in ``required``. An unknown entity is represented by "".
        "required": ["entity", "attribute", "value"],
        "additionalProperties": False,
    }


MEMORY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "hivemem_chunk_node",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "Ti": {
                    "type": "array",
                    "items": _attribute_schema(TEXT_ATTRIBUTE_KEYS),
                    "maxItems": MAX_ATTRIBUTE_RECORDS,
                },
                "Vi": {
                    "type": "array",
                    "items": _attribute_schema(VISUAL_ATTRIBUTE_KEYS),
                    "maxItems": MAX_ATTRIBUTE_RECORDS,
                },
            },
            "required": ["summary", "Ti", "Vi"],
            "additionalProperties": False,
        },
    },
}



@dataclass
class ExecutionResult:
    success: bool
    memory_content: str = ""
    reasoning: str = ""
    # Entities extracted by the same LLM call that produced memory_content.
    entities: List[dict] | None = None
    text_attributes: List[dict] | None = None
    visual_attributes: List[dict] | None = None

    def to_dict(self):
        return {
            "success": self.success,
            "memory_content": self.memory_content,
            "reasoning": self.reasoning,
            "entities": self.entities,
            "Ti": self.text_attributes,
            "Vi": self.visual_attributes,
        }


class MemoryExecutor:
    def __init__(self, llm_client, embedder):
        self.llm_client = llm_client
        self.embedder = embedder

    def execute(
        self,
        chunk_text: str,
        profile: str = "",
        *,
        image_paths: Sequence[str] | None = None,
        visual_input: str = "image",
    ):
        raw_response, results, _, _ = self.execute_with_usage(
            chunk_text,
            profile=profile,
            image_paths=image_paths,
            visual_input=visual_input,
        )
        return raw_response, results

    def execute_with_usage(
        self,
        chunk_text: str,
        profile: str = "",
        *,
        image_paths: Sequence[str] | None = None,
        visual_input: str = "image",
    ):
        visual_input = normalize_visual_input(visual_input)
        selected_images = (
            [str(path) for path in (image_paths or []) if str(path).strip()]
            if visual_input_uses_images(visual_input)
            else []
        )
        prepared_chunk = self.prepare_chunk_text(chunk_text, visual_input)
        prompt = self._build_prompt(
            prepared_chunk,
            profile=profile,
            has_images=bool(selected_images),
        )
        generate_with_usage = getattr(self.llm_client, "generate_with_usage", None)
        if callable(generate_with_usage):
            response = (
                generate_with_usage(prompt, image_paths=selected_images)
                if selected_images
                else generate_with_usage(prompt)
            )
            raw_response = response.text
            usage = dict(response.usage)
            call_stats = {
                "attempts": int(getattr(response, "attempts", 1)),
                "failed_attempts": int(getattr(response, "failed_attempts", 0)),
            }
        else:
            raw_response = (
                self.llm_client.generate(prompt, image_paths=selected_images)
                if selected_images
                else self.llm_client.generate(prompt)
            )
            usage = {}
            call_stats = {}
        results = self._parse_response(raw_response)
        if not selected_images:
            for result in results:
                result.visual_attributes = []
        return raw_response, results, usage, call_stats

    @staticmethod
    def prepare_chunk_text(chunk_text: str, visual_input: str = "image") -> str:
        """Keep captions unless raw-image-only mode explicitly removes them."""
        if normalize_visual_input(visual_input) != "image":
            return str(chunk_text)

        lines = []
        for line in str(chunk_text).splitlines():
            stripped = line.lstrip()
            if stripped.lower().startswith("image_caption:"):
                continue
            if stripped.lower().startswith("previous_round_summary:"):
                line = re.sub(
                    r";\s*image_caption\b.*$",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ).rstrip()
            lines.append(line)
        return "\n".join(lines)

    def apply_to_memory_bank(
        self,
        results: List[ExecutionResult],
        memory_bank,
        event_metadata=None,
        raw_chunk: str = "",
        node_id: str = "",
    ):  #加到之前的memorybank里
        # Each successful memory item becomes a MAU directly; the dedup/merge
        # machinery was removed 2026-08-06 together with build-time retrieval.
        if not results:
            return

        valid_inserts = [
            result
            for result in results
            if result.success and result.memory_content.strip()
        ]
        insert_embeddings = self._embed_contents(
            [result.memory_content for result in valid_inserts]
        )
        for result, embedding in zip(valid_inserts, insert_embeddings):
            metadata = {**dict(event_metadata or {}), "source": "insert"}
            memory_bank.add_memory(
                result.memory_content.strip(), embedding, metadata=metadata,
                entities=result.entities or [],
                text_attributes=result.text_attributes or [],
                visual_attributes=result.visual_attributes or [],
                raw_chunk=raw_chunk,
                memory_id=node_id or None,
            )

    def _build_prompt(
        self,
        chunk_text: str,
        profile: str = "",
        *,
        has_images: bool = False,
    ) -> str:
        profile_name = self._profile_name(profile)
        profile_block = (
            "### User Profile\n"
            "Use this name only to resolve references to the user. No other profile "
            "facts are provided or permitted as summary/Ti evidence.\n"
            f"name: {profile_name}\n\n"
        ) if profile_name else ""
        image_block = (
            "### Attached Images\n"
            "The original images associated with this chunk are attached. Inspect "
            "the images directly. Do not guess information that is not visibly "
            "supported.\n\n"
        ) if has_images else ""

        return (
            "### Role\n"
            "You convert one conversation chunk and its attached images into exactly "
            "one structured memory record.\n\n"

            "### Task\n"
            "Produce exactly one complete summary for the entire chunk, together with:\n"
            "- Ti: attributes explicitly stated in the chunk text.\n"
            "- Vi: attributes directly observable in the attached original images.\n\n"
            "The summary may use relevant information from both the chunk text and "
            "the attached images. Ti and Vi must strictly follow their respective "
            "sources. Ti and Vi are sparse retrieval properties, not exhaustive "
            "scene graphs: include only concrete, distinctive properties useful for "
            "linking this chunk to another chunk.\n\n"
            + profile_block +

            "### Current Chunk\n"
            f"{chunk_text}\n\n"
            f"{image_block}"

            "### Output Format\n"
            "Output exactly one JSON object and nothing else:\n"
            '{"summary":"<one complete summary of the entire chunk>",'
            '"Ti":[{"entity":"<optional entity name>","attribute":"<allowed textual attribute>","value":["<one or more values>"]}],'
            '"Vi":[{"entity":"<optional visible entity>","attribute":"<allowed visual attribute>","value":["<one or more values>"]}]}\n\n'
            "Do not output a JSON array, multiple JSON objects, multiple summaries, "
            "MEMORY_ITEM: lines, ENTITIES: lines, Markdown, or explanatory text.\n\n"

            "### Summary Rules\n"
            "- Write exactly one non-empty summary for the entire chunk.\n"
            "- Combine all useful facts into this single summary, even when the chunk "
            "contains multiple facts or topics.\n"
            "- Never split the chunk into multiple memory items or summaries.\n"
            "- Preserve names, numbers, dates, decisions, reasons, and image details "
            "needed to answer future questions.\n"
            "- Do not add unsupported information or unrelated profile details.\n"
            '- When the chunk has a non-empty date, begin the summary with "On <session date>, ".\n'
            "- Resolve relative dates when the chunk provides enough information.\n\n"

            "### Ti Rules\n"
            "- Ti may use only information explicitly stated in the chunk text.\n"
            "- Do not use an attached image to add or complete a Ti attribute.\n"
            "- Each attribute must use a key from the closed Ti attribute set.\n"
            "- The entity field is optional metadata and does not participate in graph matching.\n"
            "- Always return value as a JSON array of strings, including for one value.\n"
            f"- Each value array may contain at most {MAX_ATTRIBUTE_VALUES} distinct values.\n"
            "- Do not infer attributes that the text does not state.\n\n"
            "- A Ti value must be supported by words in Current Chunk itself; facts "
            "found only in User Profile are forbidden.\n"
            "- Do not create attributes for conversation mechanics or metadata such "
            "as user, assistant, session, round, dialogue, speaker, or conversation partner.\n"
            "- Do not turn a question or uncertain suggestion into an asserted attribute.\n\n"
            f"- Ti must contain at most {MAX_ATTRIBUTE_RECORDS} objects. Prefer the most "
            "specific and distinctive facts.\n"
            "- Combine multiple values for the same entity and attribute into one object.\n"
            "- Never repeat or paraphrase the same fact under multiple keys or values.\n\n"

            "### Vi Rules\n"
            "- Vi may use only information directly observable in the attached images.\n"
            "- Do not use the chunk text, image caption, or profile to add or complete a Vi attribute.\n"
            '- If no image is attached, output "Vi": [].\n'
            "- Each attribute must use a key from the closed Vi attribute set.\n"
            "- The entity field is optional metadata and does not participate in graph matching.\n"
            "- Always return value as a JSON array of strings, including for one value.\n"
            f"- Each value array may contain at most {MAX_ATTRIBUTE_VALUES} distinct values.\n"
            "- Do not infer occupation, ownership, preference, causality, identity, date, "
            "or other facts that cannot be determined visually.\n\n"
            f"- Vi must contain at most {MAX_ATTRIBUTE_RECORDS} objects. Prefer the most "
            "specific and distinctive visible facts.\n"
            "- Combine multiple values for the same entity and attribute into one object.\n"
            "- Never repeat or paraphrase the same visible fact under multiple keys or values.\n\n"
            + attribute_prompt_block()
            + "\n\n"

            "### Cross-source Rule\n"
            "If the same attribute is independently supported by both the text and the "
            "image, include it once in Ti and once in Vi. Do not copy an attribute from "
            "one source into the other.\n\n"

            "### Final Requirement\n"
            "Return exactly one valid JSON object containing exactly one summary, Ti, and Vi.\n"
        )

    @staticmethod
    def _profile_name(profile: str) -> str:
        match = re.search(r"(?:^|;)\s*name\s*:\s*([^;\n]+)", str(profile), re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _parse_response(self, response: str) -> List[ExecutionResult]:
        response = self._normalize_response(response)
        json_results = self._parse_json_response(response)
        if json_results:
            return json_results[:1]
        # Read old checkpoints and scripted fixtures without allowing a new
        # model response to create more than one node for a chunk.
        pattern = re.compile(r"(?im)^MEMORY[_ ]ITEM\s*(?::|=|-)")
        matches = list(pattern.finditer(response))

        if not matches:
            return [
                ExecutionResult(
                    success=False,
                    reasoning="No valid summary/Ti/Vi JSON object found in response.",
                )
            ]
        block_end = matches[1].start() if len(matches) > 1 else len(response)
        return [self._parse_single_action(response[matches[0].start():block_end].strip())]

    def _normalize_response(self, response: str) -> str:
        text = str(response or "").replace("\r\n", "\n").strip()
        if text.startswith("```") and text.endswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if "\n" in text:
                    first_line, remainder = text.split("\n", 1)
                    if first_line.strip().lower() in {"json", "text"}:
                        text = remainder
        return text.strip()

    def _parse_json_response(self, response: str) -> List[ExecutionResult]:
        """Salvage fallback: the model occasionally ignores the MEMORY_ITEM
        line format and emits JSON objects like
        [{"memory_item": "...", "entities": [...]}] instead. Without this,
        such responses would degrade into the builder's raw-chunk fallback."""
        stripped = response.strip()
        if stripped.startswith("["):
            start = response.find("[")
            end = response.rfind("]")
        else:
            start = response.find("{")
            end = response.rfind("}")
        if start == -1 or end == -1 or end < start:
            return []
        try:
            repaired = cast(str, repair_json(response[start:end + 1]))
            payload = json.loads(repaired)
        except Exception:
            return []

        if not isinstance(payload, dict):
            return []
        content = str(
            payload.get("summary", payload.get("memory_item", payload.get("MEMORY_ITEM", "")))
        ).strip()
        if not content:
            return []
        return [
            ExecutionResult(
                success=True,
                memory_content=content,
                entities=normalize_entities(
                    payload.get("entities", payload.get("ENTITIES")) or []
                ),
                text_attributes=normalize_attributes(payload.get("Ti"), visual=False),
                visual_attributes=normalize_attributes(payload.get("Vi"), visual=True),
            )
        ]

    def _parse_single_action(self, block: str) -> ExecutionResult:
        content_match = re.search(
            r"MEMORY[_ ]ITEM\s*(?::|=|-)?\s*(.+?)(?=\n\s*ENTITIES\b|$)",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not content_match:
            return ExecutionResult(
                success=False,
                reasoning="Memory item block is missing MEMORY_ITEM text.",
            )
        entities: List[dict] = []
        entities_match = re.search(
            r"ENTITIES\s*(?::|=|-)?\s*(\[.*)",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if entities_match:
            parsed = parse_entities_payload(entities_match.group(1))
            if parsed is not None:
                entities = normalize_entities(parsed)
        return ExecutionResult(
            success=True,
            memory_content=content_match.group(1).strip(),
            entities=entities,
        )

    def _embed_contents(self, contents: List[str]) -> np.ndarray:
        if not contents:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self.embedder.embed_texts(contents, mode="context")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

def normalize_visual_input(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode not in EXECUTOR_VISUAL_INPUTS:
        raise ValueError(
            f"Unknown executor visual input {value!r}; expected one of "
            f"{', '.join(EXECUTOR_VISUAL_INPUTS)}"
        )
    return mode


def visual_input_uses_images(value: str) -> bool:
    """Return whether an executor visual-input mode attaches original images."""
    return normalize_visual_input(value) in {"image", "image_caption"}
