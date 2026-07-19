import json
import logging
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from json_repair import repair_json


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    action_type: str
    success: bool
    memory_index: int = -1
    memory_content: str = ""
    reasoning: str = ""

    def to_dict(self):
        return {
            "action_type": self.action_type,
            "success": self.success,
            "memory_index": self.memory_index,
            "memory_content": self.memory_content,
            "reasoning": self.reasoning,
        }


class MemoryExecutor:
    def __init__(self, backend, embedder):
        self.backend = backend
        self.embedder = embedder

    def execute(self, operations: Sequence[object], chunk_text: str, retrieved_memories: List[str]):
        prompt = self._build_prompt(operations, chunk_text, retrieved_memories)
        raw_response = self.backend.generate(prompt)
        results = self._parse_response(raw_response, len(retrieved_memories))
        allowed = {str(operation.update_type).strip().upper() for operation in operations}
        filtered = []
        for result in results:
            if result.success and result.action_type not in allowed:
                filtered.append(
                    ExecutionResult(
                        action_type=result.action_type,
                        success=False,
                        memory_index=result.memory_index,
                        memory_content=result.memory_content,
                        reasoning=f"Action type {result.action_type} is not enabled.",
                    )
                )
            else:
                filtered.append(result)
        return raw_response, filtered

    def apply_to_memory_bank(
        self,
        results: List[ExecutionResult],
        memory_bank,
        retrieved_indices: List[int],
        event_metadata=None,
    ):
        if not results:
            return

        inserts = [result for result in results if result.success and result.action_type == "INSERT"]
        updates = [result for result in results if result.success and result.action_type == "UPDATE"]
        deletes = [result for result in results if result.success and result.action_type == "DELETE"]

        valid_updates = [result for result in updates if result.memory_content.strip()]
        valid_inserts = [result for result in inserts if result.memory_content.strip()]
        embedding_targets = [result.memory_content for result in valid_updates + valid_inserts]
        embedded_contents = self._embed_contents(embedding_targets)
        update_embeddings = embedded_contents[:len(valid_updates)]
        insert_embeddings = embedded_contents[len(valid_updates):]

        for result, embedding in zip(valid_updates, update_embeddings):
            if not 0 <= result.memory_index < len(retrieved_indices):
                continue
            actual_index = retrieved_indices[result.memory_index]
            memory_bank.update_memory(
                actual_index,
                result.memory_content,
                embedding,
                metadata=event_metadata,
            )

        delete_targets = []
        for result in deletes:
            if 0 <= result.memory_index < len(retrieved_indices):
                delete_targets.append(retrieved_indices[result.memory_index])
        for actual_index in sorted(set(delete_targets), reverse=True):
            memory_bank.delete_memory(actual_index)

        index_by_text = {
            self._normalize_memory_text(item.content): index
            for index, item in enumerate(getattr(memory_bank, "memories", []))
            if getattr(item, "content", "").strip()
        }
        for result, embedding in zip(valid_inserts, insert_embeddings):
            content = result.memory_content.strip()
            normalized = self._normalize_memory_text(content)
            if not normalized:
                continue
            existing_index = index_by_text.get(normalized)
            if existing_index is not None:
                existing_item = memory_bank.memories[existing_index]
                memory_bank.update_memory(
                    existing_index,
                    existing_item.content,
                    existing_item.embedding,
                    metadata=event_metadata,
                )
                continue
            metadata = {**dict(event_metadata or {}), "source": "insert"}
            memory_bank.add_memory(content, embedding, metadata=metadata)
            index_by_text[normalized] = len(memory_bank.memories) - 1

    def _build_prompt(self, operations: Sequence[object], chunk_text: str, retrieved_memories: List[str]) -> str:
        if retrieved_memories:
            memory_text = "\n".join(f"{index}. {memory}" for index, memory in enumerate(retrieved_memories))
        else:
            memory_text = "(No existing memories retrieved)"

        operation_blocks = []
        for index, operation in enumerate(operations, start=1):
            operation_blocks.append(
                "\n".join(
                    [
                        f"[Operation {index}] {operation.name}",
                        f"Description: {operation.description}",
                        f"Allowed action: {str(operation.update_type).upper()}",
                        "Instructions:",
                        str(operation.instruction_template).strip(),
                    ]
                )
            )

        operations_text = "\n\n".join(operation_blocks)
        allowed = {str(operation.update_type).strip().upper() for operation in operations}

        format_blocks = []
        if "INSERT" in allowed:
            format_blocks.append(
                "INSERT block:\n"
                "ACTION: INSERT\n"
                "MEMORY_ITEM: <concise but complete memory>"
            )
        if "UPDATE" in allowed:
            format_blocks.append(
                "UPDATE block:\n"
                "ACTION: UPDATE\n"
                "MEMORY_INDEX: <0-based index>\n"
                "UPDATED_MEMORY: <concise but complete updated memory>"
            )
        if "DELETE" in allowed:
            format_blocks.append(
                "DELETE block:\n"
                "ACTION: DELETE\n"
                "MEMORY_INDEX: <0-based index>"
            )
        if "NOOP" in allowed:
            format_blocks.append(
                "NOOP block:\n"
                "ACTION: NOOP"
            )

        rules = ["- Use only the allowed action types from the selected operations."]
        if "UPDATE" in allowed or "DELETE" in allowed:
            rules.append("- MEMORY_INDEX is 0-based and must point into the retrieved memories list above.")
        rules.append("- Output only action blocks. Do not add prose outside the blocks.")
        if "NOOP" in allowed:
            rules.append("- If no memory change is needed, output a NOOP block.")
        elif "INSERT" in allowed:
            rules.append(
                "- You MUST output at least one INSERT block that captures the content of the current chunk."
            )
        else:
            rules.append("- You must output at least one action block.")

        return (
            "You are a memory management executor. Read the current chunk and the retrieved memories, "
            "then decide what memory actions to emit.\n\n"
            "Current Chunk:\n"
            f"{chunk_text}\n\n"
            "Retrieved Memories (0-based index):\n"
            f"{memory_text}\n\n"
            "Selected Operations:\n"
            f"{operations_text}\n\n"
            "Rules:\n"
            + "\n".join(rules)
            + "\n\n"
            "Output format:\n\n"
            + "\n\n".join(format_blocks)
            + "\n"
        )

    def _parse_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
        response = self._normalize_response(response)
        pattern = re.compile(r"(?im)^ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b")
        matches = list(pattern.finditer(response))

        if not matches:
            json_results = self._parse_json_response(response, num_retrieved)
            if json_results:
                return json_results
            if re.fullmatch(r"(?is)\s*noop[.!]?\s*", response):
                return [ExecutionResult(action_type="NOOP", success=True)]
            return [
                ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="Failed to parse ACTION from response.",
                )
            ]

        results: List[ExecutionResult] = []
        for index, match in enumerate(matches):
            block_start = match.start()
            block_end = matches[index + 1].start() if index + 1 < len(matches) else len(response)
            block = response[block_start:block_end].strip()
            parsed = self._parse_single_action(block, num_retrieved)
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
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

    def _parse_json_response(self, response: str, num_retrieved: int) -> List[ExecutionResult]:
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

        if isinstance(payload, dict):
            items = payload.get("actions", [payload])
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        results = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", item.get("ACTION", ""))).strip().upper()
            if action == "INSERT":
                content = str(item.get("memory_item", item.get("MEMORY_ITEM", ""))).strip()
                if content:
                    results.append(
                        ExecutionResult(
                            action_type="INSERT",
                            success=True,
                            memory_content=content,
                        )
                    )
            elif action == "UPDATE":
                content = str(item.get("updated_memory", item.get("UPDATED_MEMORY", ""))).strip()
                index_value = item.get("memory_index", item.get("MEMORY_INDEX"))
                try:
                    memory_index = int(index_value)
                except Exception:
                    continue
                if content and 0 <= memory_index < num_retrieved:
                    results.append(
                        ExecutionResult(
                            action_type="UPDATE",
                            success=True,
                            memory_index=memory_index,
                            memory_content=content,
                        )
                    )
            elif action == "DELETE":
                index_value = item.get("memory_index", item.get("MEMORY_INDEX"))
                try:
                    memory_index = int(index_value)
                except Exception:
                    continue
                if 0 <= memory_index < num_retrieved:
                    results.append(
                        ExecutionResult(
                            action_type="DELETE",
                            success=True,
                            memory_index=memory_index,
                        )
                    )
            elif action == "NOOP":
                results.append(ExecutionResult(action_type="NOOP", success=True))
        return results

    def _parse_single_action(self, block: str, num_retrieved: int):
        action_match = re.search(
            r"ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b",
            block,
            re.IGNORECASE,
        )
        if not action_match:
            return ExecutionResult(action_type="NOOP", success=False, reasoning="Missing ACTION line.")

        action_type = action_match.group(1).upper()

        if action_type == "NOOP":
            return ExecutionResult(action_type="NOOP", success=True)

        if action_type == "INSERT":
            content_match = re.search(
                r"MEMORY[_ ]ITEM\s*(?::|=|-)?\s*(.+)$",
                block,
                re.IGNORECASE | re.DOTALL,
            )
            if not content_match:
                return ExecutionResult(
                    action_type="NOOP",
                    success=False,
                    reasoning="INSERT block is missing MEMORY_ITEM.",
                )
            return ExecutionResult(
                action_type="INSERT",
                success=True,
                memory_content=content_match.group(1).strip(),
            )

        index_match = re.search(
            r"MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)",
            block,
            re.IGNORECASE,
        )
        if not index_match:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning=f"{action_type} block is missing MEMORY_INDEX.",
            )

        memory_index = int(index_match.group(1))
        if not 0 <= memory_index < num_retrieved:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning=f"MEMORY_INDEX {memory_index} out of range [0, {num_retrieved}).",
            )

        if action_type == "DELETE":
            return ExecutionResult(action_type="DELETE", success=True, memory_index=memory_index)

        content_match = re.search(
            r"UPDATED[_ ]MEMORY\s*(?::|=|-)?\s*(.+)$",
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if not content_match:
            return ExecutionResult(
                action_type="NOOP",
                success=False,
                reasoning="UPDATE block is missing UPDATED_MEMORY.",
            )
        return ExecutionResult(
            action_type="UPDATE",
            success=True,
            memory_index=memory_index,
            memory_content=content_match.group(1).strip(),
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
