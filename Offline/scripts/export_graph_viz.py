"""Export a memory-graph subgraph for visualization / Neo4j.

Reads a built dataset dir (memories.jsonl + vectors/text.npy), selects the given
sessions, and writes:
- <out>.json    {nodes, edges} for the self-contained HTML viewer
- <out>.cypher  CREATE statements loadable in any Neo4j browser

Edges: TEMPORAL_NEXT (links.prev/next), stored typed edges (links.related),
and SHARES_ENTITY pairs derived on the fly from the entities field.

Usage:
  PYTHONPATH=. python scripts/export_graph_viz.py \
      outputs/.../datasets/Academic_Animal_Pet_Research_Life \
      --sessions D1,D2,D3 --out outputs/graph_sample
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hive_mem.build_memory_edges import derive_entity_pairs
from hive_mem.mau import MAUBank


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--sessions", default="", help="Comma-separated session ids (empty = all)")
    parser.add_argument("--out", required=True, help="Output path prefix (writes .json and .cypher)")
    args = parser.parse_args()

    bank = MAUBank.load(args.dataset_dir)
    keep = {s.strip() for s in args.sessions.split(",") if s.strip()}
    sel = [
        i for i, m in enumerate(bank.memories)
        if m.status == "ACTIVE" and (not keep or m.metadata.get("session_id") in keep)
    ]
    selset = set(sel)
    id2idx = {m.id: i for i, m in enumerate(bank.memories)}

    edges = []
    for i in sel:
        m = bank.memories[i]
        nxt = id2idx.get(m.links.get("next"))
        if nxt in selset:
            edges.append((i, nxt, "TEMPORAL_NEXT", None))
        for e in m.links.get("related") or []:
            t = id2idx.get(e.get("target"))
            if t in selset:
                edges.append((i, t, e["type"], e.get("confidence")))
    for a, b in derive_entity_pairs(bank):
        if a in selset and b in selset:
            ea = {x["name"] for x in bank.memories[a].entities}
            eb = {x["name"] for x in bank.memories[b].entities}
            edges.append((a, b, "SHARES_ENTITY", ", ".join(sorted(ea & eb)[:3])))

    viz = {
        "nodes": [
            {
                "id": bank.memories[i].id[-8:],
                "session": bank.memories[i].metadata.get("session_id"),
                "dialogue": bank.memories[i].metadata.get("dialogue_id"),
                "summary": bank.memories[i].summary,
                "entities": [e["name"] for e in bank.memories[i].entities],
            }
            for i in sel
        ],
        "edges": [
            {
                "s": bank.memories[a].id[-8:],
                "t": bank.memories[b].id[-8:],
                "type": t,
                "label": prop if isinstance(prop, str) else (f"{prop:.2f}" if prop else ""),
            }
            for a, b, t, prop in edges
        ],
    }
    Path(f"{args.out}.json").write_text(json.dumps(viz, ensure_ascii=False), encoding="utf-8")

    lines = ["MATCH (n:MAU) DETACH DELETE n;"]
    for i in sel:
        m = bank.memories[i]
        ents = ", ".join(e["name"] for e in m.entities[:6])
        lines.append(
            f'CREATE (:MAU {{id: "{m.id}", session: "{m.metadata.get("session_id")}", '
            f'round: {m.metadata.get("round_id", 0)}, dialogue: "{m.metadata.get("dialogue_id", "")}", '
            f'summary: "{esc(m.summary[:160])}", entities: "{esc(ents)}"}});'
        )
    for a, b, t, prop in edges:
        p = ""
        if t == "SHARES_ENTITY" and prop:
            p = f' {{entities: "{esc(prop)}"}}'
        elif isinstance(prop, float):
            p = f" {{confidence: {prop}}}"
        lines.append(
            f'MATCH (a:MAU {{id:"{bank.memories[a].id}"}}), (b:MAU {{id:"{bank.memories[b].id}"}}) '
            f"CREATE (a)-[:{t}{p}]->(b);"
        )
    Path(f"{args.out}.cypher").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("nodes:", len(sel), "| edges:", dict(Counter(t for _, _, t, _ in edges)))
    print("written:", f"{args.out}.json", f"{args.out}.cypher")


if __name__ == "__main__":
    main()
