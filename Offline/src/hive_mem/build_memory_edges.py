"""Build memory-graph edges over a built memory bank directory.

Operates on ``memories.jsonl`` + ``vectors/text.npy`` produced by HiveMem
builder and writes the edges back into each MAU's ``links`` field:

1. Temporal chain (deterministic, no LLM): ACTIVE memories are ordered by
   (date, session, round) from their metadata and linked via
   ``links.prev`` / ``links.next``.
2. Entity-commonality pairs (deterministic, no LLM): derived from each
   MAU's ``entities`` field (see extract_entities.py) — shared rare entity
   (df <= --df-max) or >= --min-shared shared entities, with per-memory
   degree caps. These pairs are NOT stored as edges (they are recomputable
   in milliseconds); they feed stage 3 as cross-session candidates and a
   ``reports/conflicts.json`` report (same entity + same attribute,
   different value).
3. EVENT_RELATION edges (LLM-confirmed, optional): candidate pairs are the
   union of temporally-close memories (session distance <=
   --session-window) and the entity-commonality pairs.  An LLM classifies
   each pair as CAUSES / CAUSED_BY / SUBEVENT_OF / SAME_EPISODE / NONE and
   only relations at or above --min-confidence are stored in
   ``links.related`` as {"target", "type", "confidence"}.
   Vector similarity is intentionally not used anywhere in edge building.

Usage:
    python -m hive_mem.build_memory_edges DATASET_DIR [DATASET_DIR ...]
    # temporal chain only (default), add --event-relations for stage 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .llm_client import BaseLLMClient, LLMClient
from .mau import MAUBank, MAU
from .builder import _session_key
from .output_layout import DatasetLayout


EVENT_RELATION_TYPES = ("CAUSES", "CAUSED_BY", "SUBEVENT_OF", "SAME_EPISODE")

EVENT_RELATION_PROMPT = """You are annotating an agent's long-term memory graph.
Below are two memory items summarised from the same user's conversation history.
Memory A happened before (or at the same time as) memory B.

Memory A: {summary_a}
Memory B: {summary_b}

Decide the event-level relation from A to B. Options:
- CAUSES: A is a cause of B.
- CAUSED_BY: A was caused by B.
- SUBEVENT_OF: A is a sub-event of the larger event described in B.
- SAME_EPISODE: A and B are parts of the same specific episode or activity.
- NONE: no clear event-level relation (default when unsure).

Be conservative: most pairs are NONE. Only choose a relation when the two
memories clearly describe connected events, not merely a shared topic.

Answer with a single JSON object and nothing else:
{{"relation": "<one of CAUSES|CAUSED_BY|SUBEVENT_OF|SAME_EPISODE|NONE>", "confidence": <0.0-1.0>}}"""


def _temporal_sort_key(item: MAU, position: int) -> tuple:
    metadata = item.metadata or {}
    date = str(metadata.get("date") or "")
    session = str(metadata.get("session_id") or "")
    round_id = metadata.get("round_id", 0)
    try:
        round_id = int(round_id)
    except (TypeError, ValueError):
        round_id = 0
    return (date, _session_key(session), round_id, position)


def build_temporal_chain(bank: MAUBank) -> int:
    """Link ACTIVE memories into a prev/next chain ordered by
    (date, session, round). Returns the number of chained memories."""
    for item in bank.memories:
        item.links["prev"] = None
        item.links["next"] = None
    active = [
        (position, item)
        for position, item in enumerate(bank.memories)
        if item.status == "ACTIVE"
    ]
    ordered = sorted(active, key=lambda pair: _temporal_sort_key(pair[1], pair[0]))
    for (_, earlier), (_, later) in zip(ordered, ordered[1:]):
        earlier.links["next"] = later.id
        later.links["prev"] = earlier.id
    return len(ordered)


def _active_with_order(bank: MAUBank):
    active = [
        (position, item)
        for position, item in enumerate(bank.memories)
        if item.status == "ACTIVE"
    ]
    order = {
        position: rank
        for rank, (position, _) in enumerate(
            sorted(active, key=lambda pair: _temporal_sort_key(pair[1], pair[0]))
        )
    }
    return active, order


def load_alias_map(dataset_dir: Path) -> Dict[str, str]:
    path = dataset_dir / "entity_aliases.json"
    if path.exists():
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    return {}


def _canonical_entity_names(item, alias_map: Dict[str, str]) -> set:
    names = set()
    for entity in item.entities:
        name = str(entity.get("name", "")).strip()
        if name:
            names.add(alias_map.get(name, name))
    return names


def _pairs_from_buckets(
    active,
    order,
    buckets_by_position: Dict[int, set],
    *,
    df_max: float,
    df_stop: float,
    min_shared: int,
    degree_cap: int,
) -> List[Tuple[int, int]]:
    """Generic commonality pairing: memories sharing a rare bucket key
    (df <= df_max) or >= min_shared non-stoplisted keys (df <= df_stop),
    with a per-memory degree cap preferring the rarest shared key."""
    total = len(active)
    if total < 2:
        return []
    df: Dict[object, float] = {}
    members: Dict[object, List[int]] = {}
    for position, _ in active:
        for bucket in buckets_by_position.get(position, ()):
            members.setdefault(bucket, []).append(position)
    for bucket, positions in members.items():
        df[bucket] = len(positions) / total

    pair_best_df: Dict[Tuple[int, int], float] = {}
    pair_shared: Dict[Tuple[int, int], int] = {}
    for bucket, positions in members.items():
        if df[bucket] > df_stop or len(positions) < 2:
            continue
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                a, b = positions[i], positions[j]
                key = (a, b) if order[a] <= order[b] else (b, a)
                pair_shared[key] = pair_shared.get(key, 0) + 1
                if df[bucket] < pair_best_df.get(key, 2.0):
                    pair_best_df[key] = df[bucket]

    qualified = [
        key
        for key in pair_best_df
        if pair_best_df[key] <= df_max or pair_shared[key] >= min_shared
    ]
    qualified.sort(key=lambda key: (pair_best_df[key], order[key[0]], order[key[1]]))
    degree: Dict[int, int] = {}
    kept: List[Tuple[int, int]] = []
    for a, b in qualified:
        if degree.get(a, 0) >= degree_cap or degree.get(b, 0) >= degree_cap:
            continue
        kept.append((a, b))
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    kept.sort(key=lambda pair: (order[pair[0]], order[pair[1]]))
    return kept


def derive_entity_pairs(
    bank: MAUBank,
    *,
    alias_map: Optional[Dict[str, str]] = None,
    df_max: float = 0.3,
    df_stop: float = 0.5,
    min_shared: int = 2,
    degree_cap: int = 10,
) -> List[Tuple[int, int]]:
    """Entity-commonality pairs among ACTIVE memories: shared canonical
    entity *name* (see also derive_attribute_pairs). Never materialized."""
    alias_map = alias_map or {}
    active, order = _active_with_order(bank)
    buckets = {
        position: _canonical_entity_names(item, alias_map) for position, item in active
    }
    return _pairs_from_buckets(
        active, order, buckets,
        df_max=df_max, df_stop=df_stop, min_shared=min_shared, degree_cap=degree_cap,
    )


def derive_attribute_pairs(
    bank: MAUBank,
    *,
    df_max: float = 0.3,
    df_stop: float = 0.5,
    min_shared: int = 2,
    degree_cap: int = 10,
) -> List[Tuple[int, int]]:
    """Attribute-commonality pairs among ACTIVE memories: two memories
    containing entities (possibly *different* entities) that share the same
    (entity_type, attribute_key, value) — e.g. two dogs both
    trait=intelligent. Type-scoped buckets prevent cross-domain value
    collisions; DF filtering removes generic values (species=dog).
    Never materialized as stored edges."""
    from .entity_schema import iter_attribute_items

    active, order = _active_with_order(bank)
    buckets: Dict[int, set] = {}
    for position, item in active:
        keys = set()
        for entity in item.entities:
            entity_type = str(entity.get("type", "")).upper()
            for attr_key, value in iter_attribute_items(entity):
                keys.add((entity_type, attr_key, value.lower()))
        buckets[position] = keys
    return _pairs_from_buckets(
        active, order, buckets,
        df_max=df_max, df_stop=df_stop, min_shared=min_shared, degree_cap=degree_cap,
    )


def generate_candidate_pairs(
    bank: MAUBank,
    *,
    session_window: int = 1,
    entity_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Candidate (i, j) index pairs among ACTIVE memories for the
    EVENT_RELATION stage: temporally close pairs (session distance <=
    session_window) unioned with entity-commonality pairs. i always
    precedes j in temporal order.

    Vector similarity is deliberately NOT a candidate source: it feeds the
    LLM pairs that are topically similar yet event-unrelated, which is
    exactly where hallucinated causal edges come from. Cross-session
    candidates come from shared-entity signals (derive_entity_pairs)."""
    active, order = _active_with_order(bank)
    if len(active) < 2:
        return []
    pairs = set()

    def add_pair(a: int, b: int) -> None:
        if a == b:
            return
        if order[a] <= order[b]:
            pairs.add((a, b))
        else:
            pairs.add((b, a))

    session_numbers: Dict[int, int] = {}
    for position, item in active:
        session_numbers[position] = _session_key(
            str((item.metadata or {}).get("session_id") or "")
        )[0]
    for index_a in range(len(active)):
        position_a = active[index_a][0]
        for index_b in range(index_a + 1, len(active)):
            position_b = active[index_b][0]
            if abs(session_numbers[position_a] - session_numbers[position_b]) <= session_window:
                add_pair(position_a, position_b)

    for a, b in entity_pairs or []:
        add_pair(a, b)

    return sorted(pairs, key=lambda pair: (order[pair[0]], order[pair[1]]))


def find_conflict_candidates(
    bank: MAUBank,
    *,
    alias_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    """Same canonical entity + same attribute with differing values across
    two ACTIVE memories -> UPDATE/CONFLICT candidate (phase-2 input,
    reported only, no edges written)."""
    from .entity_schema import iter_attribute_items

    alias_map = alias_map or {}
    active, order = _active_with_order(bank)
    slots: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for position, item in active:
        for entity in item.entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            canonical = alias_map.get(name, name)
            for attribute, value in iter_attribute_items(entity):
                slots.setdefault((canonical, attribute), []).append((position, value))
    conflicts = []
    for (name, attribute), observations in slots.items():
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                (pos_a, value_a), (pos_b, value_b) = observations[i], observations[j]
                if pos_a == pos_b or value_a.lower() == value_b.lower():
                    continue
                earlier, later = (
                    (pos_a, pos_b) if order[pos_a] <= order[pos_b] else (pos_b, pos_a)
                )
                conflicts.append(
                    {
                        "entity": name,
                        "attribute": attribute,
                        "earlier_memory": bank.memories[earlier].id,
                        "later_memory": bank.memories[later].id,
                        "earlier_value": value_a if earlier == pos_a else value_b,
                        "later_value": value_b if later == pos_b else value_a,
                    }
                )
    return conflicts


def classify_event_relations(
    bank: MAUBank,
    pairs: Sequence[Tuple[int, int]],
    llm_client: BaseLLMClient,
    *,
    min_confidence: float = 0.7,
    max_pairs: int = 0,
    progress_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Ask the LLM to classify candidate pairs; store confident relations
    as typed edges on the earlier memory. Returns edge-count stats."""
    if max_pairs and len(pairs) > max_pairs:
        print(f"[edges] capping candidate pairs {len(pairs)} -> {max_pairs}")
        pairs = pairs[:max_pairs]
    stats = {relation: 0 for relation in EVENT_RELATION_TYPES}
    stats.update({"NONE": 0, "parse_failures": 0, "pairs": len(pairs)})
    progress_handle = progress_path.open("a", encoding="utf-8") if progress_path else None
    try:
        for index_a, index_b in pairs:
            earlier, later = bank.memories[index_a], bank.memories[index_b]
            prompt = EVENT_RELATION_PROMPT.format(
                summary_a=earlier.summary, summary_b=later.summary
            )
            raw = llm_client.generate(prompt)
            relation, confidence = _parse_relation(raw)
            if relation is None:
                stats["parse_failures"] += 1
                continue
            if relation != "NONE" and confidence >= min_confidence:
                _add_related_edge(earlier, later.id, relation, confidence)
                stats[relation] += 1
            else:
                stats["NONE"] += 1
            if progress_handle:
                progress_handle.write(
                    json.dumps(
                        {
                            "source": earlier.id,
                            "target": later.id,
                            "relation": relation,
                            "confidence": confidence,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if progress_handle:
            progress_handle.close()
    return stats


def _add_related_edge(item: MAU, target_id: str, relation: str, confidence: float) -> None:
    related = item.links.setdefault("related", [])
    for edge in related:
        if edge.get("target") == target_id and edge.get("type") == relation:
            return
    related.append({"target": target_id, "type": relation, "confidence": round(confidence, 3)})


def _parse_relation(raw: str) -> Tuple[Optional[str], float]:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, 0.0
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            data = json.loads(repair_json(payload))
        except Exception:
            return None, 0.0
    if not isinstance(data, dict):
        return None, 0.0
    relation = str(data.get("relation", "")).strip().upper()
    if relation not in EVENT_RELATION_TYPES and relation != "NONE":
        return None, 0.0
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return relation, confidence


def process_dataset_dir(
    dataset_dir: Path,
    *,
    llm_client: Optional[BaseLLMClient],
    event_relations: bool,
    session_window: int,
    min_confidence: float,
    max_pairs: int,
    df_max: float = 0.3,
    df_stop: float = 0.5,
    min_shared: int = 2,
    degree_cap: int = 10,
) -> Dict[str, object]:
    layout = DatasetLayout(dataset_dir)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    layout.traces_dir.mkdir(parents=True, exist_ok=True)
    bank = MAUBank.load(dataset_dir)
    chained = build_temporal_chain(bank)
    alias_map = load_alias_map(dataset_dir)
    entity_pairs = derive_entity_pairs(
        bank,
        alias_map=alias_map,
        df_max=df_max,
        df_stop=df_stop,
        min_shared=min_shared,
        degree_cap=degree_cap,
    )
    attribute_pairs = derive_attribute_pairs(
        bank,
        df_max=df_max,
        df_stop=df_stop,
        min_shared=min_shared,
        degree_cap=degree_cap,
    )
    entity_pairs = sorted(set(entity_pairs) | set(attribute_pairs))
    conflicts = find_conflict_candidates(bank, alias_map=alias_map)
    layout.conflict_candidates.write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary: Dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "memories": len(bank),
        "temporal_chained": chained,
        "memories_with_entities": sum(
            1 for item in bank.memories if item.status == "ACTIVE" and item.entities
        ),
        "entity_pairs_derived": len(entity_pairs),
        "attribute_pairs_derived": len(attribute_pairs),
        "conflict_candidates": len(conflicts),
    }
    if event_relations:
        pairs = generate_candidate_pairs(
            bank, session_window=session_window, entity_pairs=entity_pairs
        )
        summary["event_relation_candidates"] = len(pairs)
        summary["event_relation_stats"] = classify_event_relations(
            bank,
            pairs,
            llm_client,
            min_confidence=min_confidence,
            max_pairs=max_pairs,
            progress_path=layout.edge_progress,
        )
    bank.save(dataset_dir)
    layout.edges_manifest.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memory-graph edges (temporal chain + event relations).")
    parser.add_argument("dataset_dirs", nargs="+", help="Dataset dirs containing memories.jsonl + vectors/text.npy")
    parser.add_argument("--event-relations", action="store_true", help="Also run the LLM EVENT_RELATION stage")
    parser.add_argument("--session-window", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--max-pairs", type=int, default=0, help="Cap on LLM-judged pairs per dataset (0 = no cap)")
    parser.add_argument("--df-max", type=float, default=0.3, help="Max document frequency for a 'rare' shared entity")
    parser.add_argument("--df-stop", type=float, default=0.5, help="Entities above this df are stop-listed entirely")
    parser.add_argument("--min-shared", type=int, default=2, help="Shared-entity count that qualifies a pair without a rare entity")
    parser.add_argument("--degree-cap", type=int, default=10, help="Max entity-derived pairs per memory")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    llm_client = None
    if args.event_relations:
        if not args.model or not args.base_url:
            parser.error("--event-relations requires --model and --base-url")
        llm_client = LLMClient(
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
            llm_client=llm_client,
            event_relations=args.event_relations,
            session_window=args.session_window,
            min_confidence=args.min_confidence,
            max_pairs=args.max_pairs,
            df_max=args.df_max,
            df_stop=args.df_stop,
            min_shared=args.min_shared,
            degree_cap=args.degree_cap,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
