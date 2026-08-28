"""Offline entity extraction for built memory banks.

For every ACTIVE MAU in a dataset directory this script asks an LLM to
extract normalized entity annotations from the summary and writes them into
the MAU's top-level ``entities`` field:

    [{"name": "Alice", "type": "PERSON", "aliases": ["her friend Alice"]},
     {"name": "allergy medicine", "type": "OBJECT",
      "attribute": "user medication status", "value": "in use"}]

Design contract (agreed 2026-08-03):
- The LLM only annotates single memories (linear cost, no pairwise calls).
- Extraction-time filters: type whitelist, no pronouns/generic references,
  minimum name length.
- Corpus-level outputs are sidecar files, never stored inside MAUs:
  ``entity_stats.json`` (document frequency; recomputable cache) and
  ``entity_aliases.json`` (optional LLM alias arbitration results).
- Progress is checkpointed to ``entity_extract_progress.jsonl`` so an
  interrupted run resumes without repeating LLM calls.

Usage:
    python -m hive_mem.extract_entities DATASET_DIR [DATASET_DIR ...] \
        --model Qwen/Qwen3-VL-4B-Instruct --base-url http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import difflib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backends import BaseLLMBackend, OpenAICompatibleBackend
from .entity_schema import (
    normalize_entities,
    ontology_prompt_block,
    parse_entities_payload,
)
from .mau import MAUBank


EXTRACTION_PROMPT = """You are annotating one memory item from a user's long-term conversational memory.

Memory summary:
{summary}

Extract the concrete entities this memory mentions, each with its attributes.

{ontology}

Rules:
- "name" must be a canonical, self-contained noun phrase. Resolve pronouns and
  vague references ("it", "the place") to explicit names using the summary
  itself; if you cannot resolve one, drop it entirely.
- Do NOT include the user or the assistant as entities.
- Record surface forms that differ from the canonical name in "aliases".
- "attributes" uses ONLY the keys listed above for that entity type; fill only
  what the summary explicitly states. A value is a short string, or a list of
  short strings for multiple values (e.g. several traits).
- Prefer few precise entities over many loose ones.

Answer with a single JSON array and nothing else:
[{{"name": "...", "type": "...", "aliases": ["..."], "attributes": {{"key": "value"}}}}]"""

ALIAS_PROMPT = """Two entity names were extracted from the same user's conversation memories.

Name A: {name_a}
Name B: {name_b}

Do they refer to the same specific real-world entity (same person, same animal,
same object, same place)? Answer with a single JSON object and nothing else:
{{"same": true/false}}"""


def parse_entities(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the LLM response into a list of entity dicts (unfiltered).
    Returns None when the response is unusable."""
    return parse_entities_payload(raw)


def filter_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate against the shared ontology (see entity_schema)."""
    return normalize_entities(entities)


def compute_entity_stats(bank: MAUBank) -> Dict[str, Any]:
    """Document frequency of canonical entity names over ACTIVE memories."""
    active = [item for item in bank.memories if item.status == "ACTIVE"]
    counts: Dict[str, int] = {}
    types: Dict[str, str] = {}
    for item in active:
        names = {str(e.get("name", "")).strip() for e in item.entities if e.get("name")}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        for e in item.entities:
            name = str(e.get("name", "")).strip()
            if name and name not in types:
                types[name] = str(e.get("type", ""))
    total = max(1, len(active))
    from .entity_schema import iter_attribute_items

    attribute_counts: Dict[str, int] = {}
    entities_with_attributes = 0
    total_entities = 0
    for item in active:
        for entity in item.entities:
            total_entities += 1
            pairs = list(iter_attribute_items(entity))
            if pairs:
                entities_with_attributes += 1
            for attr_key, _ in pairs:
                attribute_counts[attr_key] = attribute_counts.get(attr_key, 0) + 1
    return {
        "total_active_memories": len(active),
        "total_entities": total_entities,
        "entities_with_attributes": entities_with_attributes,
        "attribute_key_counts": dict(
            sorted(attribute_counts.items(), key=lambda kv: -kv[1])
        ),
        "entities": {
            name: {
                "type": types.get(name, ""),
                "count": count,
                "df": round(count / total, 4),
            }
            for name, count in sorted(counts.items(), key=lambda kv: -kv[1])
        },
    }


def alias_candidate_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """String-level suspicious pairs: substring containment or high fuzzy
    ratio. This is the only place cross-memory LLM judgment is allowed, and
    the candidate set is intentionally tiny."""
    pairs = []
    lowered = [(name, name.lower()) for name in names]
    for i in range(len(lowered)):
        name_a, low_a = lowered[i]
        for j in range(i + 1, len(lowered)):
            name_b, low_b = lowered[j]
            if low_a == low_b:
                continue
            if low_a in low_b or low_b in low_a:
                pairs.append((name_a, name_b))
            elif difflib.SequenceMatcher(None, low_a, low_b).ratio() >= 0.85:
                pairs.append((name_a, name_b))
    return pairs


def merge_aliases(
    names_by_frequency: List[str],
    backend: BaseLLMBackend,
    *,
    workers: int = 4,
) -> Dict[str, str]:
    """LLM-arbitrated alias map: alias name -> canonical (more frequent) name."""
    pairs = alias_candidate_pairs(names_by_frequency)
    rank = {name: position for position, name in enumerate(names_by_frequency)}
    alias_map: Dict[str, str] = {}
    lock = threading.Lock()

    def judge(pair: Tuple[str, str]) -> None:
        name_a, name_b = pair
        raw = backend.generate(ALIAS_PROMPT.format(name_a=name_a, name_b=name_b))
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return
        try:
            same = bool(json.loads(raw[start : end + 1]).get("same"))
        except json.JSONDecodeError:
            return
        if not same:
            return
        canonical, alias = sorted(pair, key=lambda name: rank.get(name, 1 << 30))
        with lock:
            alias_map[alias] = canonical

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge, pairs))
    # Collapse chains alias -> ... -> canonical.
    for alias in list(alias_map):
        target = alias_map[alias]
        seen = {alias}
        while target in alias_map and target not in seen:
            seen.add(target)
            target = alias_map[target]
        alias_map[alias] = target
    return alias_map


def _load_progress(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    done: Dict[str, List[Dict[str, Any]]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row["entities"]
    return done


def process_dataset_dir(
    dataset_dir: Path,
    backend: BaseLLMBackend,
    *,
    workers: int = 4,
    limit: int = 0,
    force: bool = False,
    alias_merge: bool = False,
) -> Dict[str, Any]:
    bank = MAUBank.load(dataset_dir)
    progress_path = dataset_dir / "entity_extract_progress.jsonl"
    done = {} if force else _load_progress(progress_path)

    pending = [
        item
        for item in bank.memories
        if item.status == "ACTIVE"
        and item.id not in done
        and (force or not item.entities)
    ]
    if limit:
        pending = pending[:limit]

    lock = threading.Lock()
    failures = [0]

    def extract(item) -> None:
        raw = backend.generate(
            EXTRACTION_PROMPT.format(summary=item.summary, ontology=ontology_prompt_block())
        )
        parsed = parse_entities(raw)
        if parsed is None:
            with lock:
                failures[0] += 1
            entities: List[Dict[str, Any]] = []
        else:
            entities = filter_entities(parsed)
        with lock:
            done[item.id] = entities
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"id": item.id, "entities": entities}, ensure_ascii=False)
                    + "\n"
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(extract, pending))

    for item in bank.memories:
        if item.id in done:
            item.entities = done[item.id]

    stats = compute_entity_stats(bank)
    alias_map: Dict[str, str] = {}
    if alias_merge:
        names = list(stats["entities"].keys())
        alias_map = merge_aliases(names, backend, workers=workers)
        (dataset_dir / "entity_aliases.json").write_text(
            json.dumps(alias_map, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    bank.save(dataset_dir)
    (dataset_dir / "entity_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "dataset_dir": str(dataset_dir),
        "extracted_this_run": len(pending),
        "parse_failures": failures[0],
        "memories_with_entities": sum(1 for item in bank.memories if item.entities),
        "distinct_entities": len(stats["entities"]),
        "alias_merges": len(alias_map),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entity annotations into MAU 'entities' fields.")
    parser.add_argument("dataset_dirs", nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Max memories per dataset (0 = all); for smoke tests")
    parser.add_argument("--force", action="store_true", help="Re-extract even when entities already present")
    parser.add_argument("--alias-merge", action="store_true", help="Run the LLM alias-arbitration pass")
    args = parser.parse_args()

    backend = OpenAICompatibleBackend(
        model=args.model,
        api_base=args.base_url,
        api_key=args.api_key,
        temperature=0.0,
        max_new_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    for dataset_dir in args.dataset_dirs:
        summary = process_dataset_dir(
            Path(dataset_dir),
            backend,
            workers=args.workers,
            limit=args.limit,
            force=args.force,
            alias_merge=args.alias_merge,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
