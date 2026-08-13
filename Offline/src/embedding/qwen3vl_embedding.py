from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class Qwen3VLEmbeddingService:
    """Qwen3-VL embedding wrapper for text-only and image-text chunks.

    The implementation intentionally lazy-imports torch/transformers so the
    rest of the pipeline can run in lightweight environments.
    """

    supports_images = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
        expected_dim: int = 2048,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.expected_dim = int(expected_dim)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch_dtype = self._resolve_dtype(torch)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        # The Qwen3-VL-Embedding checkpoints store weights with the
        # generation-wrapper prefix ("model.language_model.*"). Loading them
        # into the bare Qwen3VLModel silently re-initializes everything on
        # transformers >= 4.57 (prefix stripping changed), so load the
        # wrapper class and unwrap its base model instead.
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if getattr(config, "model_type", "") == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            full = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
            self._model = full.model.to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            ).to(self.device)
        self._model.eval()

    def embed_chunk(self, text: str, images: list[str] | None = None) -> list[float]:
        return self._embed(text, images or [])

    def embed_query(self, query: str, images: list[str] | None = None) -> list[float]:
        return self._embed(query, images or [])

    def _embed(self, text: str, images: list[str]) -> list[float]:
        self.load()
        import torch

        messages = self._build_messages(text, images)
        inputs = self._build_inputs(messages)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self._model, "encode"):
                vec = self._try_model_encode(inputs, text, images)
            else:
                outputs = self._model(**inputs, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else outputs.last_hidden_state
                vec = self._last_token_pool(hidden, inputs.get("attention_mask"))

        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.expected_dim}, got {arr.shape[0]}"
            )
        arr = arr / (np.linalg.norm(arr) + 1e-8)
        return arr.astype(np.float32).tolist()

    def _try_model_encode(self, inputs: dict[str, Any], text: str, images: list[str]):
        # Some embedding model implementations expose an encode method. Keep this
        # broad so the wrapper survives upstream model API differences.
        try:
            out = self._model.encode(**inputs)
        except TypeError:
            try:
                out = self._model.encode(text=text, images=images)
            except TypeError:
                out = self._model.encode([text])
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "detach"):
            out = out.detach().cpu().float().numpy()
        return np.asarray(out).reshape(-1)

    def _build_messages(self, text: str, images: list[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image_path in images:
            if image_path and Path(image_path).exists():
                item: dict[str, Any] = {"type": "image", "image": image_path}
                if self.max_pixels is not None:
                    item["max_pixels"] = self.max_pixels
                if self.min_pixels is not None:
                    item["min_pixels"] = self.min_pixels
                content.append(item)
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _build_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        image_inputs = None
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            image_inputs = self._fallback_pil_images(messages)

        kwargs = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        return self._processor(**kwargs)

    def _fallback_pil_images(self, messages: list[dict[str, Any]]):
        from PIL import Image

        images = []
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "image":
                    path = item.get("image")
                    if path and os.path.exists(path):
                        img = Image.open(path).convert("RGB")
                        max_pixels = item.get("max_pixels", self.max_pixels)
                        if max_pixels and img.width * img.height > max_pixels:
                            scale = (max_pixels / (img.width * img.height)) ** 0.5
                            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                            img = img.resize(new_size, Image.BICUBIC)
                        images.append(img)
        return images

    @staticmethod
    def _last_token_pool(hidden, attention_mask):
        if attention_mask is None:
            pooled = hidden[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden[range(hidden.shape[0]), lengths]
        return pooled[0].detach().cpu().float().numpy()

    def _resolve_dtype(self, torch):
        if self.dtype == "auto":
            return "auto"
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32
        return "auto"


class QwenMemoryEmbedder:
    """Expose the simple pipeline embedder API using Qwen3-VL Embedding."""

    supports_images = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda:0",
        expected_dim: int = 2048,
        dtype: str = "auto",
        local_files_only: bool = False,
    ):
        self.expected_dim = int(expected_dim)
        self.service = Qwen3VLEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    def embed_texts(self, texts: str | Sequence[str], mode: str = "context") -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        vectors = []
        for text in values:
            if mode == "query":
                vector = self.service.embed_query(str(text), [])
            else:
                vector = self.service.embed_chunk(str(text), [])
            vectors.append(vector)
        result = np.asarray(vectors, dtype=np.float32)
        if not vectors:
            result = np.zeros((0, self.expected_dim), dtype=np.float32)
        return result[0] if single else result

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        vectors = [self.service.embed_chunk("Represent this memory image.", [path]) for path in image_paths]
        if not vectors:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)
