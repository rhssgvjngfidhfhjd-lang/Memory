from __future__ import annotations

import re


class AnswerFormatError(ValueError):
    """Raised when a model response violates the shared answer-tag contract."""


ANSWER_BLOCK_REGEX = r"<answer>[^<]+</answer>"
# Used only after a response exhausts the 512-token budget before closing its
# tag. Keeping the common path unbounded avoids the large performance penalty
# of bounded regex decoding; the rare retry is capped without changing Prompt.
ANSWER_BLOCK_RETRY_REGEX = r"<answer>[^<]{1,800}</answer>"


_ANSWER_BLOCK = re.compile(r"\s*<answer>(.*?)</answer>\s*", re.DOTALL)


def parse_answer_block(raw: str) -> str:
    """Extract exactly one non-empty ``<answer>...</answer>`` block."""
    text = str(raw or "")
    match = _ANSWER_BLOCK.fullmatch(text)
    if match is None:
        raise AnswerFormatError(
            "Response must contain only one <answer>...</answer> block"
        )
    answer = match.group(1).strip()
    if not answer or "<answer>" in answer or "</answer>" in answer:
        raise AnswerFormatError("The <answer> block must be unique and non-empty")
    return answer
