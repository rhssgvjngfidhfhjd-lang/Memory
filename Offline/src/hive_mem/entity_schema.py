"""Closed attribute schemas and normalization for HiveMem chunk nodes."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

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
