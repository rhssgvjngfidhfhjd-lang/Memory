"""Insert-only memory executor.

One LLM call per chunk produces memory items (MEMORY_ITEM + ENTITIES lines).
The UPDATE/DELETE/NOOP machinery was removed on 2026-08-06, and the MAUBank
mutation/query helpers (supersede_memory etc.) on 2026-08-07 — restore from
git for mutable-memory experiments.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from json_repair import repair_json

from .entity_schema import (
    normalize_entities,
    ontology_prompt_block,
    parse_entities_payload,
)


EXECUTOR_VISUAL_INPUTS = ("image", "caption", "image_caption")



@dataclass
class ExecutionResult:
    success: bool
    memory_content: str = ""
    reasoning: str = ""
    # Entities extracted by the same LLM call that produced memory_content.
    entities: List[dict] = None

    def to_dict(self):
        return {
            "success": self.success,
            "memory_content": self.memory_content,
            "reasoning": self.reasoning,
            "entities": self.entities,
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
        raw_response, results, _ = self.execute_with_usage(
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
        else:
            raw_response = (
                self.llm_client.generate(prompt, image_paths=selected_images)
                if selected_images
                else self.llm_client.generate(prompt)
            )
            usage = {}
        results = self._parse_response(raw_response)
        return raw_response, results, usage

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
            )

    # The single build prompt. The MEMORY_ITEM rules encode three
    # empirically-motivated fixes from the 2026-08 evaluation rounds: explicit
    # date anchoring (TR/MR), no persona boilerplate (embedding dilution),
    # completeness over ENTITIES.
    def _build_prompt(
        self,
        chunk_text: str,
        profile: str = "",
        *,
        has_images: bool = False,
    ) -> str:
        profile_block = (
            "### User Profile\n"
            "Background reference for resolving who/what names refer to; "
            "do NOT copy profile facts into memories.\n"
            f"{profile}\n\n"
        ) if profile else ""
        image_block = (
            "### Attached Image\n"
            "The original image referenced by the image_id in the current chunk is "
            "attached. Inspect the image itself and preserve visual details that may "
            "be needed to answer future questions; do not guess details that are not "
            "visible.\n\n"
        ) if has_images else ""

        subject_rule = (
            '- The "user" in the chunk is the person described in the profile: always '
            'refer to them by name (e.g. "Julian said ..."), never as "the user".\n'
            if profile
            else '- No named profile is available: refer to the subject as "the user" or '
            '"the agent" unless the chunk explicitly provides a name.\n'
        )

        return (
            "### Role\n"
            "You summarize conversation chunks into memory items and, in the same "
            "pass, extract each item's entities and attributes.\n\n"

            "### Task\n"
            "Turn the current chunk into standalone memory items, each able to "
            "answer future questions on its own.\n\n"
            + profile_block +

            "### Current Chunk\n"
            f"{chunk_text}\n\n"
            f"{image_block}"

            "### Output Format\n"
            "Output one memory item per independent fact, written as a MEMORY_ITEM line "
            "plus an ENTITIES line, and nothing else.\n"
            "MEMORY_ITEM: <concise but complete memory>\n"
            'ENTITIES: <single-line JSON array: [{"name": "...", "type": "...", '
            '"attributes": {"key": "value or [values]"}}]>\n\n'

            "### Memory Item Rules\n"
            + subject_rule +
            "- Split unrelated facts into separate memory items; one topic per item.\n"
            '- When the chunk has a non-empty date, every MEMORY_ITEM MUST begin '
            'with "On <session date>, " (e.g. "On 2024-06-17, Julian ..."); if '
            'the date field is empty, do not invent a date or add an "On" prefix. '
            'Resolve relative time ("last week") to absolute dates when possible.\n'
            '- Refer to people by bare name only. WRONG: "Julian Vance, a 31-year-old '
            'UX strategist focused on emerging tech, asked ..." RIGHT: "Julian asked '
            '...". Never copy age/occupation/personality from the profile into a '
            "memory unless this chunk's conversation is itself about that background.\n"
            "- Preserve the specifics needed to answer questions later: entities, "
            "numbers, dates, decisions and their reasons, and image references (image IDs).\n"
            "- Skip pure greetings and filler.\n"
            "- MEMORY_ITEM completeness has priority: never shorten it to make room "
            "for ENTITIES.\n\n"

            "### Entities Rules\n"
            '- Every entity object MUST contain "name" and "type"; an entity without '
            "a name is discarded, e.g. "
            '{"name": "Lumi", "type": "ANIMAL", "attributes": {"breed": "Maltese"}}.\n'
            "- Resolve pronouns to canonical names; never include the user or the "
            "assistant as entities.\n"
            "- Fill only attribute keys the chunk explicitly states, restricted to "
            "the allowed keys below.\n\n"
            "### Entity Ontology\n"
            + ontology_prompt_block()
            + "\n\n"

            "### Final Requirement\n"
            "Output only memory items. Do not add prose outside them. "
            "You MUST output at least one memory item that captures the content of "
            "the current chunk.\n"
        )

    def _parse_response(self, response: str) -> List[ExecutionResult]:
        response = self._normalize_response(response)
        pattern = re.compile(r"(?im)^MEMORY[_ ]ITEM\s*(?::|=|-)")
        matches = list(pattern.finditer(response))

        if not matches:
            json_results = self._parse_json_response(response)
            if json_results:
                return json_results
            return [
                ExecutionResult(
                    success=False,
                    reasoning="No MEMORY_ITEM block found in response.",
                )
            ]

        results: List[ExecutionResult] = []
        for index, match in enumerate(matches):
            block_start = match.start()
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
            block = response[block_start:block_end].strip()
            results.append(self._parse_single_action(block))
        return results

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
            payload = json.loads(repair_json(response[start:end + 1]))
        except Exception:
            return []

        items = payload if isinstance(payload, list) else [payload]
        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("memory_item", item.get("MEMORY_ITEM", ""))).strip()
            if content:
                results.append(
                    ExecutionResult(
                        success=True,
                        memory_content=content,
                        entities=normalize_entities(
                            item.get("entities", item.get("ENTITIES")) or []
                        ),
                    )
                )
        return results

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

    @staticmethod
    def _normalize_memory_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()


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
