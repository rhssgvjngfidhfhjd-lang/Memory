from typing import Any, Dict, List

import numpy as np

from .backends import build_llm_backend
from .chunking import normalize_chunks
from .executor import MemoryExecutor
from .memory_bank import MemoryBank
from .operations import get_default_operations
from .retriever import HFTextRetriever


class MemoryExtractionPipeline:
    def __init__(
        self,
        model: str,
        api: bool = True,
        api_base: str = "",
        api_key=None,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        retriever: str = "contriever",
        device: str = None,
        llm_backend=None,
        embedder=None,
        operations=None,
    ):
        self.model = model
        self.api = api
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retriever = retriever
        self.device = device

        self.llm_backend = llm_backend or build_llm_backend(
            api=api,
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self.embedder = embedder or HFTextRetriever(
            retriever=retriever,
            device=device,
        )
        self.operations = list(operations) if operations else get_default_operations()
        self.executor = MemoryExecutor(self.llm_backend, self.embedder)

    def run(
        self,
        input_data,
        input_type: str,
        chunk_mode: str = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        resolved_chunk_mode, chunks = normalize_chunks(
            input_data=input_data,
            input_type=input_type,
            chunk_mode=chunk_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        memory_bank = MemoryBank()
        trace: List[Dict[str, Any]] = []

        if chunks:
            query_embeddings = self.embedder.embed_texts(chunks, mode="query")
            query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
        else:
            query_embeddings = np.zeros((0, 0), dtype=np.float32)

        for chunk_index, chunk_text in enumerate(chunks):
            query_embedding = query_embeddings[chunk_index]
            retrieved_memories, retrieved_indices = memory_bank.retrieve(query_embedding, top_k=top_k)
            raw_response, actions = self.executor.execute(
                operations=self.operations,
                chunk_text=chunk_text,
                retrieved_memories=retrieved_memories,
            )
            self.executor.apply_to_memory_bank(actions, memory_bank, retrieved_indices)
            memory_bank.step()

            trace.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "retrieved_memories": retrieved_memories,
                    "raw_response": raw_response,
                    "actions": [action.to_dict() for action in actions],
                    "memory_count_after": len(memory_bank),
                }
            )

        return {
            "input_type": input_type,
            "chunk_mode": resolved_chunk_mode,
            "chunks": chunks,
            "trace": trace,
            "final_memories": memory_bank.get_contents(),
        }
