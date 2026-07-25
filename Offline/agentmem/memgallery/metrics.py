from __future__ import annotations

from collections import defaultdict

from memgallery_harness.runner.metrics import exact_match, f1_score


def provenance_hit(source_groups: list[list[str]], clue_ids: list[str], k: int = 5) -> float:
    if not clue_ids:
        return 0.0
    clues = set(clue_ids)
    return float(any(clues.intersection(group) for group in source_groups[:k]))


def summarize_results(results: list[dict], k: int = 5) -> dict:
    rows = []
    by_category = defaultdict(list)
    for result in results:
        row = {
            "f1": f1_score(result.get("system_answer", ""), result.get("original_answer", "")),
            "em": exact_match(result.get("system_answer", ""), result.get("original_answer", "")),
            "hit": provenance_hit(
                result.get("retrieved_source_groups", []),
                result.get("clue", []),
                k=k,
            ),
        }
        rows.append(row)
        by_category[result.get("category", "")].append(row)

    def summary(values: list[dict]) -> dict:
        count = len(values)
        return {
            "count": count,
            "f1": sum(value["f1"] for value in values) / count if count else 0.0,
            "exact_match": sum(value["em"] for value in values) / count if count else 0.0,
            f"retrieval_hitrate@{k}": sum(value["hit"] for value in values) / count if count else 0.0,
        }

    output = summary(rows)
    output["by_category"] = {category: summary(values) for category, values in sorted(by_category.items())}
    return output
