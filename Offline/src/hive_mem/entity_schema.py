"""Shared entity/attribute ontology for MAU extraction and edge building.

Agreed design (2026-08-03): every MAU carries ``entities`` produced by the
SAME LLM call that writes the summary. Each entity is

    {"name": str, "type": <ENTITY_TYPES>, "aliases": [str, ...]?,
     "attributes": {<type-specific key>: str | [str, ...], ...}}

Attribute keys form a CLOSED, human-defined ontology (derived from the
Mem-Gallery QA patterns); the LLM chooses values, code enforces the keys.
Cross-memory judgments (edges) are never made by the LLM: shared-attribute
edges are set intersections over these fields.
"""

from __future__ import annotations

import json
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
