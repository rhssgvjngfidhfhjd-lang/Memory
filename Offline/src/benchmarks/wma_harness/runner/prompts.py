from __future__ import annotations


SYSTEM_PROMPT = """You answer questions using only the supplied retrieved memories.
Be concise and directly answer the question. If the requested fact is not supported,
answer that it is unknown instead of guessing. Respect updates, contradictions, dates,
and ordering in the memory history. For visual-search questions, return the exact image ID."""


_CONSTRAINTS = {
    "MB": "The correct answer may be unknown; do not infer an unstated fact.",
    "MC": "Identify the claim that conflicts with the supported history.",
    "TR": "Order events using their session dates and stated temporal relations.",
    "VS": "Return the exact image ID shown in the retrieved visual evidence.",
    "TTL": "Apply the demonstrated long-term pattern to the hypothetical situation.",
}


def format_question_prompt(question: str, category: str) -> str:
    constraint = _CONSTRAINTS.get(category.upper(), "")
    suffix = f"\n\nAdditional constraint: {constraint}" if constraint else ""
    return (
        "Answer the following WorldMemArena question using the retrieved memory. "
        "Return only the answer, without an 'Answer:' prefix.\n\n"
        f"Question: {question}{suffix}"
    )
