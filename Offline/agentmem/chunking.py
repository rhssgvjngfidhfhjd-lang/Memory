from typing import Dict, List, Sequence, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


_TOKENIZER_NAME = "cl100k_base"


class _WhitespaceTokenizer:
    def encode(self, text: str):
        return str(text or "").split()

    def decode(self, tokens):
        return " ".join(tokens)


def _get_tokenizer():
    if tiktoken is None:
        return _WhitespaceTokenizer()
    try:
        return tiktoken.get_encoding(_TOKENIZER_NAME)
    except Exception:
        return _WhitespaceTokenizer()


_VALID_INPUT_TYPES = {"text", "documents", "dialogue"}
_DIALOGUE_MODES = {"turn", "turn-pair", "full-session", "fixed-length"}
_TEXT_MODES = {"fixed-length", "full-session"}
_DOCUMENT_MODES = {"fixed-length", "full-session"}


def normalize_chunks(
    input_data,
    input_type: str,
    chunk_mode: str = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Tuple[str, List[str]]:
    input_type = str(input_type or "").strip().lower()
    if input_type not in _VALID_INPUT_TYPES:
        raise ValueError(f"Unsupported input_type: {input_type}")

    resolved_mode = chunk_mode or _default_chunk_mode(input_type)
    resolved_mode = str(resolved_mode).strip().lower()

    tokenizer = _get_tokenizer()

    if input_type == "text":
        if resolved_mode not in _TEXT_MODES:
            raise ValueError(f"text input only supports chunk_mode values: {sorted(_TEXT_MODES)}")
        if not isinstance(input_data, str):
            raise ValueError("text input_type expects a single string.")
        chunks = _chunk_text_input(
            input_data,
            chunk_mode=resolved_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        )
        return resolved_mode, chunks

    if input_type == "documents":
        if resolved_mode not in _DOCUMENT_MODES:
            raise ValueError(
                f"documents input only supports chunk_mode values: {sorted(_DOCUMENT_MODES)}"
            )
        if not isinstance(input_data, Sequence) or isinstance(input_data, (str, bytes)):
            raise ValueError("documents input_type expects a list of strings.")
        chunks = _chunk_documents_input(
            input_data,
            chunk_mode=resolved_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        )
        return resolved_mode, chunks

    if resolved_mode not in _DIALOGUE_MODES:
        raise ValueError(f"dialogue input only supports chunk_mode values: {sorted(_DIALOGUE_MODES)}")
    if not isinstance(input_data, Sequence) or isinstance(input_data, (str, bytes)):
        raise ValueError("dialogue input_type expects a list of turn dictionaries.")
    chunks = _chunk_dialogue_input(
        input_data,
        chunk_mode=resolved_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
    )
    return resolved_mode, chunks


def _default_chunk_mode(input_type: str) -> str:
    if input_type == "dialogue":
        return "turn-pair"
    return "fixed-length"


def _chunk_text_input(
    text: str,
    chunk_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    if chunk_mode == "full-session":
        return [cleaned]
    return _split_fixed_length(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer)


def _chunk_documents_input(
    documents: Sequence[str],
    chunk_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> List[str]:
    chunks: List[str] = []
    for doc_index, document in enumerate(documents, start=1):
        if not isinstance(document, str):
            raise ValueError("documents input_type expects every item to be a string.")
        cleaned = document.strip()
        if not cleaned:
            continue
        prefix = f"DOCUMENT {doc_index}:"
        if chunk_mode == "full-session":
            chunks.append(f"{prefix}\n{cleaned}")
            continue
        for piece in _split_fixed_length(
            cleaned,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
        ):
            chunks.append(f"{prefix}\n{piece}")
    return chunks


def _chunk_dialogue_input(
    turns: Sequence[Dict],
    chunk_mode: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> List[str]:
    formatted_turns = []
    for turn in turns:
        formatted = _format_turn(turn)
        if formatted:
            formatted_turns.append(formatted)
    if not formatted_turns:
        return []

    if chunk_mode == "turn":
        return formatted_turns

    if chunk_mode == "turn-pair":
        chunks = []
        for index in range(0, len(formatted_turns), 2):
            chunks.append("\n".join(formatted_turns[index:index + 2]).strip())
        return chunks

    if chunk_mode == "full-session":
        return ["\n".join(formatted_turns).strip()]

    full_text = "\n".join(formatted_turns).strip()
    return _split_fixed_length(
        full_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer,
    )


def _split_fixed_length(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer,
) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")

    tokens = tokenizer.encode(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [text.strip()]

    overlap = min(chunk_overlap, max(chunk_size - 1, 0))
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def _format_turn(turn: Dict) -> str:
    if not isinstance(turn, dict):
        raise ValueError("dialogue turns must be dictionaries.")

    if "speaker" in turn and "text" in turn:
        speaker = str(turn["speaker"]).strip()
        text = str(turn["text"]).strip()
        if speaker and text:
            return f"{speaker}: {text}"

    if "role" in turn and "content" in turn:
        role = str(turn["role"]).strip()
        content = str(turn["content"]).strip()
        if role and content:
            return f"{role}: {content}"

    raise ValueError(
        "dialogue turns must use either {'speaker', 'text'} or {'role', 'content'}."
    )
