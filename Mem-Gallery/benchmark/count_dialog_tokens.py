#!/usr/bin/env python3
"""Count English words and tokens in Mem-Gallery dialog samples.

Examples:
    python count_dialog_tokens.py
    python count_dialog_tokens.py --scope dialog --per-file
    python count_dialog_tokens.py --encoding o200k_base
    python count_dialog_tokens.py --hf-tokenizer Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable


# Treat contractions and hyphenated compounds as one English word.
ENGLISH_WORD = re.compile(r"[A-Za-z]+(?:['’‑-][A-Za-z]+)*")


def iter_strings(value: Any) -> Iterable[str]:
    """Yield all string values recursively from a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def extract_components(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Extract semantic text fields without counting IDs, dates, paths, or clues."""
    components: dict[str, list[str]] = {
        "dialog": [],
        "captions": [],
        "profile": list(iter_strings(sample.get("character_profile", {}))),
        "qa": [],
    }

    for session in sample.get("multi_session_dialogues", []):
        for turn in session.get("dialogues", []):
            for field in ("user", "assistant"):
                text = turn.get(field)
                if isinstance(text, str) and text:
                    components["dialog"].append(text)
            captions = turn.get("image_caption", [])
            if isinstance(captions, str):
                captions = [captions]
            components["captions"].extend(
                text for text in captions if isinstance(text, str) and text
            )

    for qa in sample.get("human-annotated QAs", []):
        for field in ("question", "answer"):
            text = qa.get(field)
            if isinstance(text, str) and text:
                components["qa"].append(text)

    return components


def make_token_counter(args: argparse.Namespace) -> tuple[str, Callable[[str], int]]:
    if args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "缺少 transformers，请先运行: pip install transformers"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_tokenizer, trust_remote_code=args.trust_remote_code
        )
        return (
            f"Hugging Face: {args.hf_tokenizer}",
            lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )

    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("缺少 tiktoken，请先运行: pip install tiktoken") from exc
    encoding = tiktoken.get_encoding(args.encoding)
    return f"tiktoken: {args.encoding}", lambda text: len(encoding.encode(text))


def count_text(texts: Iterable[str], count_tokens: Callable[[str], int]) -> tuple[int, int]:
    text = "\n".join(texts)
    return len(ENGLISH_WORD.findall(text)), count_tokens(text)


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent / "data" / "dialog"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", nargs="?", type=Path, default=default_dir, help="JSON 数据目录"
    )
    parser.add_argument(
        "--scope",
        choices=("dialog", "dialog+captions", "all-content"),
        default="all-content",
        help=(
            "dialog=仅 user/assistant；dialog+captions=再加图片描述；"
            "all-content=再加人物档案和 QA（默认）"
        ),
    )
    tokenizer = parser.add_mutually_exclusive_group()
    tokenizer.add_argument(
        "--encoding", default="cl100k_base", help="tiktoken encoding（默认 cl100k_base）"
    )
    tokenizer.add_argument(
        "--hf-tokenizer", metavar="MODEL_OR_PATH", help="Hugging Face tokenizer 名称或路径"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="加载 Hugging Face tokenizer 时允许远程代码",
    )
    parser.add_argument("--per-file", action="store_true", help="输出每个 sample 的结果")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"没有找到 JSON 文件: {args.data_dir}")

    tokenizer_name, count_tokens = make_token_counter(args)
    scope_components = {
        "dialog": ("dialog",),
        "dialog+captions": ("dialog", "captions"),
        "all-content": ("dialog", "captions", "profile", "qa"),
    }[args.scope]

    total_words = 0
    total_tokens = 0
    rows: list[tuple[str, int, int]] = []
    for path in files:
        with path.open(encoding="utf-8") as file:
            sample = json.load(file)
        components = extract_components(sample)
        texts = (
            text
            for component in scope_components
            for text in components[component]
        )
        words, tokens = count_text(texts, count_tokens)
        total_words += words
        total_tokens += tokens
        rows.append((path.name, words, tokens))

    print(f"数据目录: {args.data_dir.resolve()}")
    print(f"Sample 数: {len(files)}")
    print(f"统计范围: {args.scope}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"英文单词: {total_words:,}")
    print(f"Tokens: {total_tokens:,}")

    if args.per_file:
        print("\n每个 sample:")
        width = max(len(name) for name, _, _ in rows)
        for name, words, tokens in rows:
            print(f"{name:<{width}}  words={words:>7,}  tokens={tokens:>7,}")


if __name__ == "__main__":
    main()
