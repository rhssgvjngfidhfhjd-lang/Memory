#!/usr/bin/env python3
"""Stage 1 diagnostic: does scheme c's image-bearing chunks displace correct
text chunks from the top-5 for non-visual query categories?

Zero-cost, read-only: compares two existing results.json files (scheme b vs
scheme c), no GPU / re-embedding needed. Rows are paired positionally (both
runs iterate the same datasets/QAs in the same deterministic order), which is
more reliable than pairing by (dataset, session_id, question, category) since
~93 of those keys are duplicated within each run (repeated question text).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def retrieval_hit(retrieved_ids: list[str], clue_ids: list[str], k: int = 5) -> float:
    if not clue_ids:
        return 0.0
    return float(bool(set(retrieved_ids[:k]) & set(clue_ids)))


def load_results(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_has_image_lookup(metadata_path: str) -> dict[tuple[str, str], bool]:
    lookup: dict[tuple[str, str], bool] = {}
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row.get("dataset", ""), str(row.get("dialogue_id", "")))
            lookup[key] = bool(row.get("has_image"))
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 retrieval-diff forensics (b vs c).")
    parser.add_argument("--results-b", required=True)
    parser.add_argument("--results-c", required=True)
    parser.add_argument("--metadata", default="artifacts/faiss_index/metadata.jsonl")
    parser.add_argument("--categories", default="FR,KR,MR,TR")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="artifacts/results/diagnostics/stage1_retrieval_diff.json")
    args = parser.parse_args()

    categories = {c.strip().upper() for c in args.categories.split(",") if c.strip()}

    results_b = load_results(args.results_b)
    results_c = load_results(args.results_c)
    has_image = build_has_image_lookup(args.metadata)

    if len(results_b) != len(results_c):
        raise SystemExit(f"length mismatch: b={len(results_b)} c={len(results_c)}")

    mismatched_alignment = 0
    regressions = []
    for rb, rc in zip(results_b, results_c):
        if (rb.get("dataset"), rb.get("session_id"), rb.get("question"), rb.get("category")) != (
            rc.get("dataset"),
            rc.get("session_id"),
            rc.get("question"),
            rc.get("category"),
        ):
            mismatched_alignment += 1
            continue
        category = str(rb.get("category", "")).upper()
        if category not in categories:
            continue
        clue = rb.get("clue", []) or []
        hit_b = retrieval_hit(rb.get("retrieved_ids", []), clue, k=args.k)
        hit_c = retrieval_hit(rc.get("retrieved_ids", []), clue, k=args.k)
        if not (hit_b == 1.0 and hit_c == 0.0):
            continue

        dataset = rb.get("dataset", "")
        retrieved_b_top5 = rb.get("retrieved_ids", [])[: args.k]
        retrieved_c_top5 = rc.get("retrieved_ids", [])[: args.k]
        new_entrants = [rid for rid in retrieved_c_top5 if rid not in retrieved_b_top5]
        exited = [rid for rid in retrieved_b_top5 if rid not in retrieved_c_top5]

        def img(rid: str) -> bool:
            return bool(has_image.get((dataset, rid), False))

        new_entrants_info = [(rid, img(rid)) for rid in new_entrants]
        exited_info = [(rid, img(rid)) for rid in exited]
        any_image_new_entrant = any(v for _, v in new_entrants_info)
        any_image_exited = any(v for _, v in exited_info)
        image_count_b_top5 = sum(1 for rid in retrieved_b_top5 if img(rid))
        image_count_c_top5 = sum(1 for rid in retrieved_c_top5 if img(rid))

        full_retrieved_c = rc.get("retrieved_ids", [])
        clue_rank_c = None
        for rank, rid in enumerate(full_retrieved_c, start=1):
            if rid in clue:
                clue_rank_c = rank
                break

        regressions.append(
            {
                "dataset": dataset,
                "category": category,
                "question": rb.get("question", ""),
                "clue": clue,
                "retrieved_b_top5": retrieved_b_top5,
                "retrieved_c_top5": retrieved_c_top5,
                "new_entrants_in_c": new_entrants_info,
                "exited_from_b": exited_info,
                "any_new_entrant_has_image": any_image_new_entrant,
                "any_exited_had_image": any_image_exited,
                "image_count_b_top5": image_count_b_top5,
                "image_count_c_top5": image_count_c_top5,
                "clue_rank_in_c_full_list": clue_rank_c,
            }
        )

    total = len(regressions)
    with_image_new_entrant = sum(1 for r in regressions if r["any_new_entrant_has_image"])
    with_image_exited = sum(1 for r in regressions if r["any_exited_had_image"])
    image_composition_increased = sum(
        1 for r in regressions if r["image_count_c_top5"] > r["image_count_b_top5"]
    )
    clue_pushed_past_k = sum(
        1 for r in regressions if r["clue_rank_in_c_full_list"] is None or r["clue_rank_in_c_full_list"] > args.k
    )
    by_category = Counter(r["category"] for r in regressions)

    summary = {
        "positional_alignment_mismatches": mismatched_alignment,
        "categories_checked": sorted(categories),
        "total_b_correct_c_wrong_regressions": total,
        "regressions_with_image_bearing_new_entrant": with_image_new_entrant,
        "fraction_with_image_bearing_new_entrant": (with_image_new_entrant / total) if total else None,
        "regressions_where_an_image_chunk_exited_b_top5": with_image_exited,
        "regressions_where_image_composition_increased_b_to_c": image_composition_increased,
        "fraction_where_image_composition_increased": (image_composition_increased / total) if total else None,
        "regressions_where_clue_pushed_past_k": clue_pushed_past_k,
        "by_category_regression_count": dict(by_category),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "regressions": regressions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
