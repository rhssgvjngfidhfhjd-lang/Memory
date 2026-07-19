import threading
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


class HFTextRetriever:
    def __init__(
        self,
        retriever: str = "contriever",
        device: str = None,
        batch_size: int = 32,
    ):
        retriever = str(retriever or "").strip().lower()
        if retriever not in {"contriever", "dpr", "dragon"}:
            raise ValueError("retriever must be one of: contriever, dpr, dragon")
        self.retriever = retriever
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def embed_texts(self, texts: Union[str, List[str]], mode: str = "context") -> np.ndarray:
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)

        tokenizer, model = self._get_components(mode)
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(text_list), self.batch_size):
                batch = text_list[start:start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                outputs = model(**encoded)
                if self.retriever == "dpr":
                    batch_embeddings = outputs.pooler_output
                else:
                    last_hidden_state = getattr(outputs, "last_hidden_state", outputs[0])
                    batch_embeddings = _mean_pooling(last_hidden_state, encoded["attention_mask"])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))

        merged = np.vstack(embeddings).astype(np.float32)
        if single_input:
            return merged[0]
        return merged

    def _get_components(self, mode: str) -> Tuple[object, object]:
        mode = str(mode or "").strip().lower()
        if mode not in {"context", "query"}:
            raise ValueError("mode must be 'context' or 'query'.")

        cache_key = (self.retriever, mode, self.device)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        tokenizer, model = _load_retriever_model(self.retriever, mode, self.device)
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model


def _load_retriever_model(retriever: str, mode: str, device: str):
    if retriever == "dpr":
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizer,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizer,
        )

        if mode == "context":
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            )
            model = DPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            )
        else:
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            )
            model = DPRQuestionEncoder.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            )
    else:
        from transformers import AutoModel, AutoTokenizer

        if retriever == "contriever":
            model_name = "facebook/contriever"
        elif mode == "context":
            model_name = "facebook/dragon-plus-context-encoder"
        else:
            model_name = "facebook/dragon-plus-query-encoder"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp(min=1)
