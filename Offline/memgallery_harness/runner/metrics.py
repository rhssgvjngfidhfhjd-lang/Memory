from __future__ import annotations

import re
import string
from collections import Counter, defaultdict


def normalize_answer(text: str) -> str:
    s = str(text).lower()
    s = s.replace("_", " UNDERSCORE ")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = s.replace(" UNDERSCORE ", "_")
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def retrieval_hit(retrieved_ids: list[str], clue_ids: list[str], k: int = 5) -> float:
    if not clue_ids:
        return 0.0
    return float(bool(set(retrieved_ids[:k]) & set(clue_ids)))


def summarize_results(results: list[dict], k: int = 5) -> dict:
    by_cat = defaultdict(list)
    rows = []
    for r in results:
        f1 = f1_score(r.get("system_answer", ""), r.get("original_answer", ""))
        em = exact_match(r.get("system_answer", ""), r.get("original_answer", ""))
        hit = retrieval_hit(r.get("retrieved_ids", []), r.get("clue", []), k=k)
        row = {**r, "_f1": f1, "_em": em, "_hit": hit}
        rows.append(row)
        by_cat[r.get("category", "")].append(row)

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "count": len(rows),
        "f1": avg([r["_f1"] for r in rows]),
        "exact_match": avg([r["_em"] for r in rows]),
        f"retrieval_hitrate@{k}": avg([r["_hit"] for r in rows]),
        "by_category": {},
    }
    for cat, cat_rows in sorted(by_cat.items()):
        metrics["by_category"][cat] = {
            "count": len(cat_rows),
            "f1": avg([r["_f1"] for r in cat_rows]),
            "exact_match": avg([r["_em"] for r in cat_rows]),
            f"retrieval_hitrate@{k}": avg([r["_hit"] for r in cat_rows]),
        }
    return metrics
