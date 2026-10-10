"""Modality-aware anchor schemas, memory episodes, and chunk extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast
from uuid import uuid4

import numpy as np
from json_repair import repair_json

from .utils import portable_value, resolved_value, DatasetLayout
from .utils import atomic_binary_writer, write_text_atomic


# Attribute rules and normalization

ENTITY_TYPES = ("PERSON", "ANIMAL", "OBJECT", "PLACE", "ORGANIZATION", "EVENT")

# Closed per-type attribute-key ontology. Keys are lowercase.
ATTRIBUTE_KEYS: Dict[str, tuple] = {
    "PERSON": ("relation", "preference", "occupation", "trait", "age"),
    "ANIMAL": ("species", "breed", "owner", "trait", "appearance", "skill", "status"),
    "OBJECT": ("category", "color", "style", "material", "owner", "status", "use"),
    "PLACE": ("kind", "location", "feature"),
    "ORGANIZATION": ("kind", "location", "role"),
    "EVENT": ("date", "location", "participants", "status"),
}

VISUAL_ATTRIBUTE_KEYS = (
    "color",
    "appearance",
    "shape",
    "material",
    "texture",
    "pattern",
    "count",
    "visible_state",
    "action",
    "pose",
    "position",
    "spatial_relation",
    "ocr_text",
)

MAX_ATTRIBUTE_RECORDS = 16
MAX_ATTRIBUTE_VALUES = 8

TEXT_ATTRIBUTE_KEYS = tuple(
    dict.fromkeys(
        key
        for keys in ATTRIBUTE_KEYS.values()
        for key in keys
    )
) + tuple(
    key
    for key in VISUAL_ATTRIBUTE_KEYS
    if key not in {item for keys in ATTRIBUTE_KEYS.values() for item in keys}
)

ATTRIBUTE_KEY_ALIASES = {
    "coat color": "color",
    "coat-color": "color",
    "coat_color": "color",
}

ATTRIBUTE_VALUE_ALIASES = {
    "seated": "sitting",
}

EMPTY_ATTRIBUTE_VALUES = {"n/a", "none", "not applicable", "not specified", "unknown"}

_IRREGULAR_SINGULARS = {
    "children": "child",
    "feet": "foot",
    "geese": "goose",
    "men": "man",
    "mice": "mouse",
    "people": "person",
    "teeth": "tooth",
    "women": "woman",
}

PRONOUN_NAMES = {
    "it", "they", "them", "he", "she", "him", "her", "this", "that", "there",
    "user", "the user", "assistant", "the assistant", "someone", "something",
    "它", "他", "她", "他们", "她们", "它们", "这", "那", "这里", "那里",
    "用户", "助手", "某人", "有人",
}


def ontology_prompt_block() -> str:
    """Human-readable ontology description for LLM prompts."""
    lines = [
        "Entity types and their allowed attribute keys (fill only keys that the",
        "text explicitly states; omit everything you would have to guess):",
    ]
    for entity_type in ENTITY_TYPES:
        keys = ", ".join(ATTRIBUTE_KEYS[entity_type])
        lines.append(f"- {entity_type}: {keys}")
    return "\n".join(lines)


def attribute_prompt_block() -> str:
    """Closed Ti/Vi key lists used verbatim by the executor prompt."""
    return (
        "Allowed Ti attributes:\n"
        + ", ".join(TEXT_ATTRIBUTE_KEYS)
        + "\n\nAllowed Vi attributes:\n"
        + ", ".join(VISUAL_ATTRIBUTE_KEYS)
    )


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _singularize_last_word(value: str) -> str:
    """Apply a small deterministic English singularization rule set.

    Attribute equality must not depend on model calls or optional corpora.  The
    rules intentionally target ordinary concrete nouns and leave ambiguous
    endings such as ``glass``, ``status`` and ``analysis`` unchanged.
    """
    match = re.search(r"([a-z]+)$", value)
    if not match:
        return value
    word = match.group(1)
    singular = _IRREGULAR_SINGULARS.get(word)
    if singular is None:
        if len(word) > 3 and word.endswith("ies"):
            singular = word[:-3] + "y"
        elif len(word) > 4 and word.endswith(("sses", "shes", "ches", "xes", "zes")):
            singular = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            singular = word[:-1]
        else:
            singular = word
    return value[: match.start(1)] + singular


def normalize_attribute_pair(attribute: Any, value: Any) -> tuple[str, str] | None:
    """Return the canonical graph property ``(attribute, value)``."""
    key = _normalize_text(attribute).replace(" ", "_")
    key = ATTRIBUTE_KEY_ALIASES.get(key, key)
    normalized_value = _normalize_text(value)
    normalized_value = ATTRIBUTE_VALUE_ALIASES.get(normalized_value, normalized_value)
    normalized_value = _singularize_last_word(normalized_value)
    if not key or not normalized_value or normalized_value in EMPTY_ATTRIBUTE_VALUES:
        return None
    return key, normalized_value


def normalize_attributes(raw: Any, *, visual: bool) -> List[Dict[str, Any]]:
    """Validate VLM attributes and keep grouped list-valued JSON records.

    ``entity`` is provenance only.  It is deliberately excluded from the
    canonical pair used by intersections, document frequency and embeddings.
    """
    if not isinstance(raw, list):
        return []
    allowed = set(VISUAL_ATTRIBUTE_KEYS if visual else TEXT_ATTRIBUTE_KEYS)
    grouped: Dict[tuple[str, str], list[str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        values = item.get("value")
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        for value in values:
            pair = normalize_attribute_pair(item.get("attribute"), value)
            if pair is None or pair[0] not in allowed:
                continue
            bucket = grouped.setdefault((entity, pair[0]), [])
            if pair[1] not in bucket and len(bucket) < MAX_ATTRIBUTE_VALUES:
                bucket.append(pair[1])
    return [
        {**({"entity": entity} if entity else {}), "attribute": key, "value": values}
        for (entity, key), values in grouped.items()
    ][:MAX_ATTRIBUTE_RECORDS]


def iter_node_attributes(raw: Any):
    """Yield unique canonical pairs from a node's grouped Ti or Vi list."""
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        values = item.get("value")
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        for value in values:
            pair = normalize_attribute_pair(item.get("attribute"), value)
            if pair is not None and pair not in seen:
                seen.add(pair)
                yield pair


def serialize_attribute(attribute: tuple[str, str]) -> str:
    return f"{attribute[0]}: {attribute[1]}"


def _normalize_value(value: Any) -> Optional[Any]:
    """Attribute values are a non-empty string or a list of them."""
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
        items = list(dict.fromkeys(items))
        if not items:
            return None
        return items if len(items) > 1 else items[0]
    text = str(value).strip()
    return text or None


def normalize_entities(raw: Any) -> List[Dict[str, Any]]:
    """Validate/clean an extracted entity list against the ontology.

    Enforces: known type, concrete name (no pronouns, >=2 chars), attribute
    keys restricted to the entity type's whitelist, per-memory (name, type)
    dedup. Unknown keys and empty values are dropped silently."""
    if not isinstance(raw, list):
        return []
    kept: List[Dict[str, Any]] = []
    seen = set()
    for entity in raw:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name", "")).strip()
        entity_type = str(entity.get("type", "")).strip().upper()
        if entity_type not in ENTITY_TYPES:
            continue
        if len(name) < 2 or name.lower() in PRONOUN_NAMES:
            continue
        key = (name.lower(), entity_type)
        if key in seen:
            continue
        seen.add(key)
        cleaned: Dict[str, Any] = {"name": name, "type": entity_type}
        aliases = [
            str(alias).strip()
            for alias in (entity.get("aliases") or [])
            if str(alias).strip() and str(alias).strip().lower() != name.lower()
        ]
        if aliases:
            cleaned["aliases"] = list(dict.fromkeys(aliases))
        attributes_raw = entity.get("attributes")
        # Legacy flat schema: {"attribute": ..., "value": ...}
        if not isinstance(attributes_raw, dict) and entity.get("attribute") and entity.get("value"):
            attributes_raw = {str(entity["attribute"]): entity["value"]}
        attributes: Dict[str, Any] = {}
        if isinstance(attributes_raw, dict):
            allowed = ATTRIBUTE_KEYS[entity_type]
            for attr_key, attr_value in attributes_raw.items():
                normalized_key = str(attr_key).strip().lower()
                if normalized_key not in allowed:
                    continue
                value = _normalize_value(attr_value)
                if value is not None:
                    attributes[normalized_key] = value
        if attributes:
            cleaned["attributes"] = attributes
        kept.append(cleaned)
    return kept


def parse_entities_payload(text: str) -> Optional[List[Dict[str, Any]]]:
    """Extract the first JSON array from an LLM response (json_repair
    fallback). Returns None when unusable — callers keep the memory and
    store an empty entity list instead of failing the insert."""
    text = str(text or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            data = json.loads(repair_json(payload))
        except Exception:
            return None
    if not isinstance(data, list):
        return None
    return [item for item in data if isinstance(item, dict)]


def iter_attribute_items(entity: Dict[str, Any]):
    """Yield (key, value_str) pairs from an entity's ``attributes`` dict,
    flattening list values."""
    attributes = entity.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    if str(item).strip():
                        yield str(key).lower(), str(item).strip()
            elif str(value).strip():
                yield str(key).lower(), str(value).strip()


# Memory episodes and bank persistence

class ModalityType(str, Enum):
    """text-only vs image-bearing memory. Any other stored value is a data
    bug and must fail loudly (ModalityType(...) raises ValueError)."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class MemoryEpisode:
    """A memory episode with a compact summary, raw text, and modality-aware anchors."""

    summary: str
    embedding: np.ndarray
    raw_chunk: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional entity metadata; the affinity graph uses textual and visual anchors.
    entities: List[Dict[str, Any]] = field(default_factory=list)
    textual_anchors: List[Dict[str, Any]] = field(default_factory=list)
    visual_anchors: List[Dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"memory_{int(time.time() * 1000)}_{uuid4().hex[:8]}")
    modality_type: ModalityType = ModalityType.TEXT
    # Reserved link fields; cross-episode affinities are stored in the graph manifest.
    links: Dict[str, Any] = field(
        default_factory=lambda: {"prev": None, "next": None, "related": []}
    )
    # Retrieval filters out episodes whose status is not ACTIVE.
    status: str = "ACTIVE"

    @property
    def evidence_text(self) -> str:
        """The original chunk returned to the answer model."""
        return self.raw_chunk or self.summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a memory episode with the paper's Ti/Vi anchor fields.

        The embedding is intentionally omitted — vectors live in the
        dataset's ``vectors/`` directory.
        """
        return {
            "id": self.id,
            "modality_type": self.modality_type.value,
            "summary": self.summary,
            "chunk": self.raw_chunk,
            "Ti": self.textual_anchors,
            "Vi": self.visual_anchors,
            "entities": self.entities,
            "status": self.status,
            "metadata": portable_value(self.metadata),
            "links": self.links,
        }

class MemoryBank:
    """Store and persist a collection of memory episodes."""

    def __init__(self):
        self.memories: List[MemoryEpisode] = []

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        textual_anchors: Optional[List[Dict[str, Any]]] = None,
        visual_anchors: Optional[List[Dict[str, Any]]] = None,
        raw_chunk: str = "",
        memory_id: str | None = None,
    ):
        normalized_metadata = _normalize_metadata(metadata)
        image_paths = normalized_metadata.get("image_paths", [])
        modality_type = (
            ModalityType.MULTIMODAL if image_paths else ModalityType.TEXT
        )
        self.memories.append(
            MemoryEpisode(
                summary=str(content).strip(),
                embedding=np.asarray(embedding, dtype=np.float32),
                raw_chunk=str(raw_chunk),
                metadata=normalized_metadata,
                entities=[e for e in (entities or []) if isinstance(e, dict)],  # Keep dictionary entity records.
                textual_anchors=[e for e in (textual_anchors or []) if isinstance(e, dict)],
                visual_anchors=[e for e in (visual_anchors or []) if isinstance(e, dict)],
                **({"id": str(memory_id)} if memory_id else {}),
                modality_type=modality_type,
            )
        )

    def save(self, directory: str | Path) -> None:
        _validate_memory_ids(self.memories)
        directory = Path(directory)
        layout = DatasetLayout(directory)
        directory.mkdir(parents=True, exist_ok=True)
        layout.vectors_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(
            directory / "memories.jsonl",
            "".join(
                json.dumps(item.to_dict(), ensure_ascii=False) + "\n"
                for item in self.memories
            ),
        )
        vectors = (
            np.vstack([item.embedding for item in self.memories]).astype(np.float32)
            if self.memories
            else np.zeros((0, 0), dtype=np.float32)
        )
        with atomic_binary_writer(layout.text_vectors) as handle:
            np.save(handle, vectors)

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryBank":
        directory = Path(directory)
        state_path = directory / "builder_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("Checkpoint state must be a JSON object")
            if "generation" in state:
                bank, _ = cls.load_checkpoint(directory)
                return bank
        return cls._load_files(directory)

    @classmethod
    def load_checkpoint(cls, directory: str | Path) -> tuple["MemoryBank", dict[str, Any]]:
        """Read one committed generation, accepting historical flat checkpoints."""
        directory = Path(directory)
        pointer = json.loads((directory / "builder_state.json").read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise ValueError("Checkpoint state must be a JSON object")
        if "generation" not in pointer:
            bank = cls._load_files(directory)
            _validate_checkpoint_boundary(bank, pointer)
            return bank, pointer
        errors = []
        names = [pointer.get("generation"), pointer.get("previous_generation")]
        for position, name in enumerate(names):
            if not name:
                continue
            try:
                generation_dir = _checkpoint_generation_path(directory, name)
                state = json.loads((generation_dir / "builder_state.json").read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    raise ValueError("Generation state must be a JSON object")
                if position == 0:
                    expected = {key: value for key, value in pointer.items() if key not in {"generation", "previous_generation"}}
                    if state != expected:
                        raise ValueError("Generation state does not match the commit pointer")
                bank = cls._load_files(generation_dir)
                _validate_checkpoint_boundary(bank, state)
            except (OSError, ValueError, TypeError) as error:
                errors.append(f"{name}: {error}")
                continue
            if position:
                warnings.warn("Current checkpoint is invalid; restoring its previous committed generation", RuntimeWarning, stacklevel=2)
            return bank, {**state, "generation": name}
        raise ValueError("No valid committed checkpoint generation: " + "; ".join(errors))

    @classmethod
    def _load_files(cls, directory: str | Path) -> "MemoryBank":
        directory = Path(directory)
        layout = DatasetLayout(directory)
        bank = cls()
        with (directory / "memories.jsonl").open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Memory records must be JSON objects")
        identifiers = [row.get("id") for row in rows]
        if (
            any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("Persisted memory episode IDs must be nonempty, unique strings")
        vectors_path = layout.existing_vector_path("text.npy", "vectors.npy")
        if not vectors_path.exists():
            # Fail loudly: silently deriving vectors from rows would poison
            # retrieval with empty embeddings (rows carry no vectors by design).
            raise FileNotFoundError(f"Missing {vectors_path} next to memories.jsonl")
        vectors = np.load(vectors_path, allow_pickle=False)
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError(f"Memory vectors must be a 2-D matrix, got {getattr(vectors, 'shape', None)}")
        if not np.isfinite(vectors).all():
            raise ValueError(f"Memory vectors contain NaN or Inf: {vectors_path}")
        if len(rows) != len(vectors):
            raise ValueError(f"Memory/vector count mismatch: {len(rows)} vs {len(vectors)}")
        for row, vector in zip(rows, vectors):
            summary = row.get("summary", "")
            memory_id = row["id"]
            metadata = _normalize_metadata(row.get("metadata"))
            item = MemoryEpisode(
                id=memory_id,
                modality_type=ModalityType(
                    row.get(
                        "modality_type",
                        "multimodal" if metadata.get("image_paths") else "text",
                    )
                ),
                summary=str(summary).strip(),
                embedding=np.asarray(vector, dtype=np.float32),
                raw_chunk=str(row.get("chunk") or ""),
                entities=[e for e in (row.get("entities") or []) if isinstance(e, dict)],
                textual_anchors=[e for e in (row.get("Ti") or []) if isinstance(e, dict)],
                visual_anchors=[e for e in (row.get("Vi") or []) if isinstance(e, dict)],
                metadata=metadata,
                links=_normalize_links(row.get("links")),
                status=row.get("status", "ACTIVE"),
            )
            bank.memories.append(item)
        _validate_memory_ids(bank.memories)
        return bank

    def __len__(self):
        return len(self.memories)


def _checkpoint_generation_path(directory: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Checkpoint generation must be a directory name")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Checkpoint generation escapes its checkpoint directory")
    return path


def _validate_memory_ids(memories: Sequence[MemoryEpisode]) -> None:
    identifiers = [str(memory.id) for memory in memories]
    if any(not identifier.strip() for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Memory episode IDs must be nonempty and unique")


def _validate_checkpoint_boundary(bank: MemoryBank, state: dict[str, Any]) -> None:
    index = state.get("next_event_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index != len(bank):
        raise ValueError("Checkpoint event boundary does not match its memory count")
    signature = state.get("signature") or {}
    if not isinstance(signature, dict):
        raise ValueError("Checkpoint build signature must be a JSON object")
    dimension = signature.get("embedding_dim")
    if dimension is not None:
        if isinstance(dimension, bool) or int(dimension) <= 0:
            raise ValueError("Checkpoint embedding dimension must be positive")
        if any(item.embedding.shape != (int(dimension),) for item in bank.memories):
            raise ValueError("Checkpoint vectors do not match the recorded embedding dimension")


_LIST_METADATA_KEYS = {
    "source_dialogue_ids",
    "source_chunk_ids",
    "image_ids",
    "image_paths",
    "image_captions",
}


def _normalize_links(links: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce stored links (including the legacy all-null schema) to the
    current {"prev", "next", "related"} shape."""
    links = dict(links or {})
    related = links.get("related")
    if not isinstance(related, list):
        related = []
    return {
        "prev": links.get("prev"),
        "next": links.get("next"),
        "related": related,
    }


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = resolved_value(dict(metadata or {}))
    for key in _LIST_METADATA_KEYS:
        value = normalized.get(key, [])
        if not isinstance(value, (list, tuple, set)):
            value = [value] if value else []
        normalized[key] = list(dict.fromkeys(str(item) for item in value if str(item)))
    return normalized


# Chunk summarization and attribute extraction

EXECUTOR_VISUAL_INPUTS = ("image",)
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
    textual_anchors: List[dict] | None = None
    visual_anchors: List[dict] | None = None

    def to_dict(self):
        return {
            "success": self.success,
            "memory_content": self.memory_content,
            "reasoning": self.reasoning,
            "entities": self.entities,
            "Ti": self.textual_anchors,
            "Vi": self.visual_anchors,
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
        selected_images = [str(path) for path in (image_paths or []) if str(path).strip()]
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
                result.visual_anchors = []
        return raw_response, results, usage, call_stats

    @staticmethod
    def prepare_chunk_text(chunk_text: str, visual_input: str = "image") -> str:
        """Prepare dialogue text while original images supply visual evidence."""
        normalize_visual_input(visual_input)

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
    ):  # Append successful items to the memory bank.
        # Each successful extraction produces one memory episode.
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
                textual_anchors=result.textual_anchors or [],
                visual_anchors=result.visual_anchors or [],
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
                textual_anchors=normalize_attributes(payload.get("Ti"), visual=False),
                visual_anchors=normalize_attributes(payload.get("Vi"), visual=True),
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
    """Validate the original-image executor input mode."""
    return normalize_visual_input(value) == "image"
