#!/usr/bin/env python
"""Build Mem-Gallery memory banks with the WMA baseline memory systems.

Stage 1 of the Mem-Gallery baseline pipeline: drive SimpleMem / Omni-SimpleMem /
A-Mem / M2A over the Mem-Gallery dialogue rounds (the same ``chunks.jsonl``
rounds the AgentMem/RAG baselines consumed) and export each backend's memory
store to the ``agentmem`` MemoryBank JSONL format::

    <out-root>/<baseline>/datasets/<dataset>/memories.jsonl   (no vectors)
    <out-root>/<baseline>/datasets/<dataset>/build_trace.jsonl
    <out-root>/<baseline>/datasets/<dataset>/build_stats.json

Stage 2 (Offline repo, ``agentmem/embed_baseline_memories.py``)
embeds these with Qwen3-VL-Embedding-2B; stage 3 reuses
``agentmem.run_memgallery_baseline`` for top-5 retrieval + QA.

Run from the WorldMemArena repo root with its .venv python::

    .venv/bin/python eval_framework/scripts/build_memgallery_baselines.py \
        --baseline amem --llm-base-url http://127.0.0.1:8000/v1

Retrieval is unified downstream (Qwen3-VL-Embedding-2B + cosine top-5), so only
each baseline's *memory construction* runs here; the baselines' own retrievers
are not used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OFFLINE_ROOT = Path("/data1/haozhen/Visual_Primitives/Offline/Offline")

BASELINES = ("simplemem", "omnisimplemem", "amem", "m2a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--chunks", default=str(_OFFLINE_ROOT / "artifacts" / "chunks.jsonl"))
    parser.add_argument("--out-root", default=str(_OFFLINE_ROOT / "artifacts" / "baseline_memories"))
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names; empty = all")
    parser.add_argument("--max-rounds", type=int, default=0, help="Per-dataset round cap (smoke)")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument(
        "--m2a-mode",
        choices=("agent", "direct"),
        default="agent",
        help="agent = upstream ChatAgent/MemoryManager decides what to store (real M2A, "
        "~2 LLM calls/round); direct = force-store every turn into SemanticStore (fast)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip datasets already completed")
    return parser.parse_args()


def setup_env(args: argparse.Namespace) -> None:
    """Point eval_framework.config resolvers at the vLLM executor BEFORE importing adapters."""
    os.environ["OPENAI_MODEL"] = args.llm_model
    os.environ["OPENAI_BASE_URL"] = args.llm_base_url
    os.environ["OPENAI_API_KEY"] = args.llm_api_key
    # A-Mem's AgenticMemorySystem(model_name=...) is its local sentence-transformer.
    os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    scratch = Path(args.out_root) / args.baseline / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SIMPLEMEM_LANCEDB_PATH", str(scratch / "lancedb"))


# ---------------------------------------------------------------------------
# input rounds
# ---------------------------------------------------------------------------

def _session_key(session_id: str) -> tuple[int, str]:
    digits = "".join(c for c in session_id if c.isdigit())
    return (int(digits) if digits else 0, session_id)


def load_rounds(chunks_path: Path, datasets: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with chunks_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ds = str((row.get("metadata") or {}).get("dataset", ""))
            if not ds or (datasets and ds not in datasets):
                continue
            grouped.setdefault(ds, []).append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda r: (
                _session_key(str(r["metadata"].get("session_id", ""))),
                int(r["metadata"].get("round_id", 0)),
            )
        )
    return grouped


def make_turns(row: dict) -> list[Any]:
    """Split one Mem-Gallery round chunk into user/assistant NormalizedTurns.

    Chunk text layout: header (profile/session/date/round) + ``user: ...`` +
    ``assistant: ...`` + optional image lines + previous_round_summary. The
    split happens at the first ``\\nassistant: `` so the store call fires
    immediately on the assistant turn (adapters buffer the user turn).
    Attachment captions stay empty: captions are already inlined in the text.
    """
    from eval_framework.datasets.schemas import Attachment, NormalizedTurn

    meta = row["metadata"]
    text = str(row.get("text", ""))
    session_id = str(meta.get("session_id", ""))
    round_id = int(meta.get("round_id", 0))
    sample_id = str(meta.get("profile_name") or meta.get("dataset", ""))
    timestamp = str(meta.get("timestamp") or meta.get("date") or "") or None

    attachments = []
    image_ids = [str(i) for i in (meta.get("image_ids") or []) if i]
    image_paths = [str(p) for p in (row.get("images") or []) if p]
    for i, iid in enumerate(image_ids):
        attachments.append(
            Attachment(
                caption="",
                image_id=iid,
                file_path=image_paths[i] if i < len(image_paths) else None,
            )
        )

    marker = "\nassistant: "
    idx = text.find(marker)
    user_text = text[:idx] if idx >= 0 else text
    assistant_text = text[idx + len(marker):] if idx >= 0 else ""
    # Drop the inner "user: " label; the adapters re-prefix the merged text.
    user_text = user_text.replace("\nuser: ", "\n", 1)

    turns = [
        NormalizedTurn(
            sample_id=sample_id,
            session_id=session_id,
            turn_index=round_id,
            role="user",
            text=user_text,
            attachments=tuple(attachments),
            timestamp=timestamp,
        )
    ]
    if assistant_text:
        turns.append(
            NormalizedTurn(
                sample_id=sample_id,
                session_id=session_id,
                turn_index=round_id,
                role="assistant",
                text=assistant_text,
                attachments=(),
                timestamp=timestamp,
            )
        )
    return turns


# ---------------------------------------------------------------------------
# adapters + exporters
# ---------------------------------------------------------------------------

def build_adapter(baseline: str, m2a_mode: str) -> Any:
    if baseline in ("simplemem", "omnisimplemem"):
        from eval_framework.memory_adapters.simplemem_adapter import SimpleMemAdapter

        return SimpleMemAdapter(mode="text" if baseline == "simplemem" else "omni")
    if baseline == "amem":
        from eval_framework.memory_adapters.amem_v2 import AMemV2Adapter

        return AMemV2Adapter()
    # m2a
    from datetime import datetime

    from eval_framework.memory_adapters.m2a_adapter import M2AAdapter

    if m2a_mode == "direct":
        return M2AAdapter()

    class M2AAgentModeAdapter(M2AAdapter):
        """Upstream ChatAgent/MemoryManager path only — the LLM decides what
        to store via the update_memory tool; no force-store of raw turns."""

        def ingest_turn(self, turn: Any) -> None:
            self._session_id = turn.session_id
            text = self._render_turn(turn)
            if not text:
                return
            try:
                ts = datetime.fromisoformat(turn.timestamp) if turn.timestamp else datetime.now()
            except (ValueError, TypeError):
                ts = datetime.now()
            speaker = f"{turn.role}_{turn.sample_id}"
            image_path = self._turn_image_path(turn)
            try:
                self._m2a.chat_agent.chat(
                    user_text=f"({speaker}, {ts.isoformat()}) {text}",
                    user_image_path_or_url=image_path,
                    timestamp=ts,
                    role=speaker,
                )
            except Exception as exc:
                print(f"  [M2A-agent] chat failed, raw-store fallback: {exc!r}", flush=True)
                from agent.stores.raw import RawMessage  # type: ignore

                self._m2a.raw_store.append(
                    RawMessage(msg_id=-1, timestamp=ts, role=speaker, text=text, image_path=image_path)
                )

    return M2AAgentModeAdapter()


class Exporter:
    """Enumerate the backend's current memories; diff against seen ids."""

    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.seen: set[str] = set()

    def reset(self) -> None:
        self.seen = set()

    def list_memories(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def new_memories(self) -> list[dict[str, Any]]:
        current = self.list_memories()
        fresh = [m for m in current if m["backend_id"] not in self.seen]
        self.seen.update(m["backend_id"] for m in current)
        return fresh


class SimpleMemTextExporter(Exporter):
    def list_memories(self) -> list[dict[str, Any]]:
        try:
            entries = self.adapter._mem.get_all_memories()
        except Exception:
            try:
                backend = getattr(self.adapter._mem, "_backend", None) or self.adapter._mem
                entries = backend.vector_store.get_all_entries()
            except Exception:
                return []
        out = []
        for entry in entries or []:
            content = str(getattr(entry, "lossless_restatement", "") or "").strip()
            if not content:
                continue
            extra: dict[str, Any] = {}
            for field in ("keywords", "persons", "entities", "timestamp", "location", "topic"):
                val = getattr(entry, field, None)
                if val:
                    extra[field] = list(val) if isinstance(val, (list, tuple)) else val
            out.append(
                {
                    "backend_id": str(getattr(entry, "entry_id", "") or f"idx_{len(out)}"),
                    "content": content,
                    "extra": extra,
                }
            )
        return out


class OmniSimpleMemExporter(Exporter):
    def list_memories(self) -> list[dict[str, Any]]:
        mau_store = getattr(self.adapter._mem, "mau_store", None)
        if mau_store is None:
            return []
        out = []
        for mau in mau_store.iter_all():
            content = str(getattr(mau, "summary", "") or "").strip()
            if not content:
                continue
            meta_obj = getattr(mau, "metadata", None)
            tags = [str(t) for t in (getattr(meta_obj, "tags", []) or [])]
            dialogue_ids = [t.split(":", 1)[1] for t in tags if t.startswith("dialogue_id:")]
            image_ids = [t.split(":", 1)[1] for t in tags if t.startswith("image_id:")]
            extra: dict[str, Any] = {
                "modality": str(getattr(getattr(mau, "modality_type", None), "value", "") or ""),
            }
            for field in ("persons", "entities", "keywords", "location", "topic"):
                val = getattr(meta_obj, field, None)
                if val:
                    extra[field] = val
            out.append(
                {
                    "backend_id": str(mau.id),
                    "content": content,
                    "extra": extra,
                    "tag_dialogue_ids": dialogue_ids,
                    "tag_image_ids": image_ids,
                }
            )
        return out


class AMemExporter(Exporter):
    """Export note content + the LLM-distilled metadata (context/keywords/tags)."""

    def list_memories(self) -> list[dict[str, Any]]:
        backend = getattr(self.adapter, "_backend", None)
        if backend is None:
            return []
        out = []
        for mid, note in backend.memories.items():
            content = str(getattr(note, "content", "") or "").strip()
            context = str(getattr(note, "context", "") or "")
            keywords = [str(k) for k in (getattr(note, "keywords", []) or [])]
            tags = [str(t) for t in (getattr(note, "tags", []) or [])]
            parts = [content] if content else []
            if context and context.lower() != "general":
                parts.append(f"context: {context}")
            if keywords:
                parts.append(f"keywords: {', '.join(keywords)}")
            if tags:
                parts.append(f"tags: {', '.join(tags)}")
            text = "\n".join(parts).strip()
            if not text:
                continue
            out.append(
                {
                    "backend_id": str(mid),
                    "content": text,
                    "extra": {"context": context, "keywords": keywords, "tags": tags},
                }
            )
        return out


class M2AExporter(Exporter):
    def list_memories(self) -> list[dict[str, Any]]:
        rows = self.adapter._query_memory_collection()
        out = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            caption = str(row.get("image_caption") or "").strip()
            content = f"{text} | caption: {caption}" if caption else text
            if not content:
                continue
            image_path = str(row.get("image_path") or "")
            extra: dict[str, Any] = {}
            if image_path:
                extra["m2a_image_path"] = image_path
            out.append(
                {
                    "backend_id": str(row.get("id", uuid.uuid4().hex[:12])),
                    "content": content,
                    "extra": extra,
                }
            )
        return out


def build_exporter(baseline: str, adapter: Any) -> Exporter:
    return {
        "simplemem": SimpleMemTextExporter,
        "omnisimplemem": OmniSimpleMemExporter,
        "amem": AMemExporter,
        "m2a": M2AExporter,
    }[baseline](adapter)


# ---------------------------------------------------------------------------
# attribution + bank rows
# ---------------------------------------------------------------------------

def _attribute_sources(memory: dict, pending: list[dict]) -> list[dict]:
    """Pick the source rounds for one new memory.

    Preference order: explicit dialogue_id tags (Omni-SimpleMem MAUs) →
    date-narrowed pending window (SimpleMem restatements carry an ISO
    timestamp) → the whole pending window since the last flush.
    """
    by_dialogue = {p["dialogue_id"]: p for p in pending}
    tag_ids = memory.get("tag_dialogue_ids") or []
    matched = [by_dialogue[d] for d in tag_ids if d in by_dialogue]
    if matched:
        return matched
    entry_ts = str(memory.get("extra", {}).get("timestamp", "") or "")
    if entry_ts:
        date = entry_ts[:10]
        dated = [p for p in pending if str(p.get("date", ""))[:10] == date]
        if dated:
            return dated
    return list(pending)


def make_bank_row(baseline: str, dataset: str, memory: dict, sources: list[dict]) -> dict:
    image_ids: list[str] = []
    image_paths: list[str] = []
    image_captions: list[str] = []
    # Attach image metadata only when provenance is round-precise; a memory
    # attributed to a whole window/session must not inherit every image in it
    # (that would grant text-only baselines spurious visual-retrieval ability).
    image_sources = sources if len(sources) == 1 else []
    for src in image_sources:
        image_ids.extend(src.get("image_ids", []))
        image_paths.extend(src.get("image_paths", []))
        image_captions.extend(src.get("image_captions", []))
    # Omni visual MAUs are tagged with a single image id — keep only that image.
    tag_image_ids = memory.get("tag_image_ids") or []
    if tag_image_ids:
        keep = set(tag_image_ids)
        pairs = [
            (i, image_paths[pos] if pos < len(image_paths) else "")
            for pos, i in enumerate(image_ids)
            if i in keep
        ]
        if pairs:
            image_ids = [i for i, _ in pairs]
            image_paths = [p for _, p in pairs if p]
    first = sources[0] if sources else {}
    metadata = {
        "dataset": dataset,
        "session_id": first.get("session_id", ""),
        "round_id": first.get("round_id", 0),
        "dialogue_id": first.get("dialogue_id", ""),
        "source_dialogue_ids": [s["dialogue_id"] for s in sources],
        "source_chunk_ids": [s["chunk_id"] for s in sources],
        "image_id": image_ids[0] if image_ids else "",
        "image_ids": list(dict.fromkeys(image_ids)),
        "image_paths": list(dict.fromkeys(image_paths)),
        "image_captions": list(dict.fromkeys(image_captions)),
        "source": baseline,
        "backend_id": memory["backend_id"],
    }
    metadata.update(memory.get("extra", {}))
    return {
        "memory_id": f"mem_{uuid.uuid4().hex}",
        "content": memory["content"],
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# per-dataset build
# ---------------------------------------------------------------------------

def build_dataset(
    baseline: str,
    dataset: str,
    rows: list[dict],
    adapter: Any,
    exporter: Exporter,
    out_dir: Path,
    max_rounds: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "build_trace.jsonl"
    started = time.time()
    bank_rows: list[dict] = []
    pending: list[dict] = []
    if max_rounds:
        rows = rows[:max_rounds]

    def flush_new(round_label: str, trace_handle) -> int:
        fresh = exporter.new_memories()
        for memory in fresh:
            sources = _attribute_sources(memory, pending)
            bank_rows.append(make_bank_row(baseline, dataset, memory, sources))
        if fresh:
            pending.clear()
        trace_handle.write(
            json.dumps(
                {"round": round_label, "new_memories": len(fresh), "total": len(bank_rows)},
                ensure_ascii=False,
            )
            + "\n"
        )
        trace_handle.flush()
        return len(fresh)

    current_session = ""
    with trace_path.open("w", encoding="utf-8") as trace:
        for i, row in enumerate(rows, start=1):
            meta = row["metadata"]
            session_id = str(meta.get("session_id", ""))
            if current_session and session_id != current_session:
                adapter.end_session(current_session)
                flush_new(f"end_session:{current_session}", trace)
            current_session = session_id

            for turn in make_turns(row):
                adapter.ingest_turn(turn)
            pending.append(
                {
                    "dialogue_id": str(meta.get("dialogue_id", "")),
                    "chunk_id": str(row.get("chunk_id", "")),
                    "session_id": session_id,
                    "round_id": int(meta.get("round_id", 0)),
                    "date": str(meta.get("date") or meta.get("timestamp") or ""),
                    "image_ids": [str(x) for x in (meta.get("image_ids") or []) if x],
                    "image_paths": [str(x) for x in (row.get("images") or []) if x],
                    "image_captions": [str(x) for x in (meta.get("image_captions") or []) if x],
                }
            )
            flush_new(str(meta.get("dialogue_id", "")), trace)
            if i % 10 == 0 or i == len(rows):
                elapsed = time.time() - started
                print(
                    f"  [{baseline}/{dataset}] round {i}/{len(rows)} "
                    f"memories={len(bank_rows)} elapsed={elapsed:.0f}s",
                    flush=True,
                )
        if current_session:
            adapter.end_session(current_session)
            flush_new(f"end_session:{current_session}", trace)

    tmp = out_dir / "memories.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        for bank_row in bank_rows:
            handle.write(json.dumps(bank_row, ensure_ascii=False) + "\n")
    tmp.replace(out_dir / "memories.jsonl")

    stats = {
        "completed": True,
        "baseline": baseline,
        "dataset": dataset,
        "rounds": len(rows),
        "memories": len(bank_rows),
        "elapsed_sec": round(time.time() - started, 1),
        "llm_model": os.environ.get("OPENAI_MODEL", ""),
        "llm_base_url": os.environ.get("OPENAI_BASE_URL", ""),
    }
    (out_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    setup_env(args)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    grouped = load_rounds(Path(args.chunks), datasets)
    if not grouped:
        raise SystemExit(f"No rounds found in {args.chunks} for datasets={datasets or 'ALL'}")
    print(
        f"[build] baseline={args.baseline} datasets={len(grouped)} "
        f"rounds={sum(len(r) for r in grouped.values())} llm={args.llm_model} @ {args.llm_base_url}",
        flush=True,
    )

    adapter = build_adapter(args.baseline, args.m2a_mode)
    exporter = build_exporter(args.baseline, adapter)
    out_base = Path(args.out_root) / args.baseline / "datasets"
    manifest: dict[str, Any] = {}
    for pos, (dataset, rows) in enumerate(sorted(grouped.items()), start=1):
        out_dir = out_base / dataset
        stats_path = out_dir / "build_stats.json"
        if args.resume and stats_path.exists():
            try:
                if json.loads(stats_path.read_text(encoding="utf-8")).get("completed"):
                    print(f"[skip {pos}/{len(grouped)}] {dataset} (completed)", flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        print(f"[dataset {pos}/{len(grouped)}] {dataset} rounds={len(rows)}", flush=True)
        adapter.reset()
        exporter.reset()
        stats = build_dataset(
            args.baseline, dataset, rows, adapter, exporter, out_dir, args.max_rounds
        )
        manifest[dataset] = stats
        print(f"[done] {dataset}: {stats['memories']} memories in {stats['elapsed_sec']}s", flush=True)

    manifest_path = Path(args.out_root) / args.baseline / "build_manifest.json"
    existing: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(manifest)
    manifest_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
