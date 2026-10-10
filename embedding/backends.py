"""Local and API embedding backends, with an optional HTTP service."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import tempfile
import threading
import urllib.request
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.utils import api_key_for, image_data_uri, load_runtime_config, validate_embedding_settings


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
        revision: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.expected_dim = int(expected_dim)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.revision = revision
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
            revision=self.revision,
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
            revision=self.revision,
        )
        if getattr(config, "model_type", "") == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration

            full = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                revision=self.revision,
            )
            self._model = full.model.to(self.device)
        else:
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                revision=self.revision,
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
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Embedding image not found: {path}")
            item: dict[str, Any] = {"type": "image", "image": str(path)}
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
        revision: str | None = None,
    ):
        self.expected_dim = int(expected_dim)
        self.service = Qwen3VLEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            revision=revision,
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


QWEN3_TEXT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_TEXT_EMBEDDING_DIM = 1024
DEFAULT_QUERY_INSTRUCTION = (
    "Given a memory-related question, retrieve relevant memory passages that answer the question"
)


def is_qwen3_text_embedding_model(model_name: str) -> bool:
    return str(model_name).rstrip("/").casefold() == QWEN3_TEXT_EMBEDDING_MODEL.casefold()


class Qwen3TextEmbeddingService:
    """Text-only Qwen3 embedding backend with official last-token pooling."""

    supports_images = False

    def __init__(
        self,
        model_name: str = QWEN3_TEXT_EMBEDDING_MODEL,
        device: str | None = None,
        expected_dim: int = QWEN3_TEXT_EMBEDDING_DIM,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        max_length: int = 8192,
        batch_size: int = 16,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        revision: str | None = None,
    ):
        if not is_qwen3_text_embedding_model(model_name):
            raise ValueError(f"Unsupported text embedding model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.expected_dim = int(expected_dim)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.query_instruction = str(query_instruction).strip()
        self.revision = revision
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            revision=self.revision,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype=self._resolve_dtype(torch),
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            revision=self.revision,
        ).to(self.device)
        self._model.eval()

    def embed_chunk(self, text: str, images: list[str] | None = None) -> list[float]:
        self._reject_images(images)
        return self.embed_chunks([text])[0].tolist()

    def embed_query(self, query: str, images: list[str] | None = None) -> list[float]:
        self._reject_images(images)
        return self.embed_queries([query])[0].tolist()

    def embed_chunks(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_texts([str(text) for text in texts])

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        values = [self._format_query(str(query)) for query in queries]
        return self._embed_texts(values)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        self.load()
        import torch
        import torch.nn.functional as functional

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, return_dict=True)
                vectors = self._last_token_pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                vectors = functional.normalize(vectors.float(), p=2, dim=1)
            batches.append(vectors.detach().cpu().float().numpy())
        result = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(texts), self.expected_dim):
            raise ValueError(
                f"Embedding shape mismatch for {self.model_name}: "
                f"expected ({len(texts)}, {self.expected_dim}), got {result.shape}"
            )
        return result

    def _format_query(self, query: str) -> str:
        if not self.query_instruction:
            return query
        return f"Instruct: {self.query_instruction}\nQuery: {query}"

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if bool(left_padding):
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _resolve_dtype(self, torch):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype, "auto")

    def _reject_images(self, images: Sequence[str] | None) -> None:
        if images:
            raise ValueError(
                f"{self.model_name} is text-only; pass an image caption as text instead"
            )


class Qwen3TextMemoryEmbedder:
    """Expose the memory-pipeline API for Qwen3 text embeddings."""

    supports_images = False

    def __init__(
        self,
        model_name: str = QWEN3_TEXT_EMBEDDING_MODEL,
        device: str = "cuda:0",
        expected_dim: int = QWEN3_TEXT_EMBEDDING_DIM,
        dtype: str = "auto",
        local_files_only: bool = False,
        batch_size: int = 16,
        revision: str | None = None,
    ):
        self.expected_dim = int(expected_dim)
        self.service = Qwen3TextEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            batch_size=batch_size,
            revision=revision,
        )

    def embed_texts(self, texts: str | Sequence[str], mode: str = "context") -> np.ndarray:
        single = isinstance(texts, str)
        values = [str(texts)] if single else [str(text) for text in texts]
        if mode == "query":
            result = self.service.embed_queries(values)
        else:
            result = self.service.embed_chunks(values)
        return result[0] if single else result

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        raise ValueError(
            f"{self.service.model_name} is text-only and cannot create image vectors"
        )


def create_embedding_service(
    *,
    model_name: str,
    device: str | None,
    expected_dim: int,
    dtype: str = "auto",
    local_files_only: bool = False,
    batch_size: int = 16,
    revision: str | None = None,
):
    if is_qwen3_text_embedding_model(model_name):
        return Qwen3TextEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            batch_size=batch_size,
            revision=revision,
        )
    return Qwen3VLEmbeddingService(
        model_name=model_name,
        device=device,
        expected_dim=expected_dim,
        dtype=dtype,
        local_files_only=local_files_only,
        revision=revision,
    )


def create_memory_embedder(
    *,
    model_name: str,
    device: str,
    expected_dim: int,
    dtype: str = "auto",
    local_files_only: bool = False,
    revision: str | None = None,
):
    if is_qwen3_text_embedding_model(model_name):
        return Qwen3TextMemoryEmbedder(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            revision=revision,
        )
    return QwenMemoryEmbedder(
        model_name=model_name,
        device=device,
        expected_dim=expected_dim,
        dtype=dtype,
        local_files_only=local_files_only,
        revision=revision,
    )


class OpenAIMemoryEmbedder:
    """Memory-pipeline embedder backed by an OpenAI-compatible endpoint."""

    supports_images = True
    text_batch_size = 128

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        expected_dim: int,
        api_key: str = "EMPTY",
        timeout: float = 180,
    ) -> None:
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/embeddings"):
            self.endpoint += "/embeddings"
        self.model_name = model_name
        self.expected_dim = int(expected_dim)
        self.api_key = api_key_for("embedding", api_key)
        self.timeout = float(timeout)

    def embed_texts(
        self, texts: str | Sequence[str], mode: str = "context"
    ) -> np.ndarray:
        single = isinstance(texts, str)
        values = [str(texts)] if single else [str(text) for text in texts]
        batches = [
            self._request(
                {"input": values[start : start + self.text_batch_size], "mode": mode}
            )
            for start in range(0, len(values), self.text_batch_size)
        ]
        vectors = (
            np.concatenate(batches, axis=0)
            if batches
            else np.zeros((0, self.expected_dim), dtype=np.float32)
        )
        return vectors[0] if single else vectors

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        vectors = []
        for path in image_paths:
            vectors.append(
                self.embed_multimodal(
                    "Represent this memory image.", [path], mode="context"
                )
            )
        if not vectors:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def embed_multimodal(
        self,
        text: str,
        image_paths: Sequence[str] = (),
        *,
        mode: str = "context",
    ) -> np.ndarray:
        """Embed one text/image item through the shared Qwen3-VL service."""
        content: list[dict[str, Any]] = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"embedding image does not exist: {path}")
            content.append(
                {"type": "image_url", "image_url": {"url": image_data_uri(path)}}
            )
        content.append({"type": "text", "text": str(text or " ")})
        vectors = self._request(
            {
                "mode": mode,
                "messages": [{"role": "user", "content": content}],
            }
        )
        if len(vectors) != 1:
            raise ValueError(
                f"multimodal embedding expected one vector, received {len(vectors)}"
            )
        return vectors[0]

    def _request(self, payload: dict[str, Any]) -> np.ndarray:
        body = json.dumps(
            {"model": self.model_name, **payload}, ensure_ascii=True
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rows = sorted(result.get("data") or [], key=lambda row: int(row.get("index", 0)))
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise ValueError(
                f"embedding shape mismatch: expected (*, {self.expected_dim}), got {vectors.shape}"
            )
        return vectors


class EmbeddingApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name = args.model
        self.dim = args.dim
        self.embedder = create_embedding_service(
            model_name=args.model,
            device=args.device,
            expected_dim=args.dim,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            revision=getattr(args, "revision", "") or None,
        )
        self.lock = threading.Lock()

    @staticmethod
    def _message_input(
        messages: list[dict[str, Any]],
        stack: ExitStack,
    ) -> tuple[str, list[str]]:
        texts: list[str] = []
        images: list[str] = []
        temporary_dir: Path | None = None
        for message in messages:
            content = message.get("content") or []
            if isinstance(content, str):
                texts.append(content)
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    texts.append(str(item.get("text") or ""))
                    continue
                if item.get("type") not in {"image", "image_url"}:
                    continue
                raw = item.get("image") or item.get("image_url") or ""
                if isinstance(raw, dict):
                    raw = raw.get("url") or ""
                value = str(raw)
                if value.startswith("data:"):
                    header, encoded = value.split(",", 1)
                    mime = header[5:].split(";", 1)[0]
                    suffix = mimetypes.guess_extension(mime) or ".png"
                    if temporary_dir is None:
                        temporary_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                    path = temporary_dir / f"image_{len(images)}{suffix}"
                    path.write_bytes(base64.b64decode(encoded))
                    images.append(str(path))
                elif value:
                    images.append(value)
        return "\n".join(text for text in texts if text).strip() or " ", images

    def embed(self, payload: dict[str, Any]) -> list[list[float]]:
        with ExitStack() as stack:
            if payload.get("messages"):
                items = [self._message_input(payload["messages"], stack)]
            else:
                raw = payload.get("input", [])
                values = raw if isinstance(raw, list) else [raw]
                items = [(str(value), []) for value in values]
            mode = str(payload.get("mode") or "query")
            embed = self.embedder.embed_chunk if mode == "context" else self.embedder.embed_query
            with self.lock:
                return [
                    embed(text, images)
                    for text, images in items
                ]


class Handler(BaseHTTPRequestHandler):
    app: EmbeddingApplication

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[embedding] {self.address_string()} {format % args}", flush=True)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/health", "/v1/models"}:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": self.app.model_name, "object": "model"}]})
            else:
                self._send(200, {"status": "ok", "model": self.app.model_name, "dim": self.app.dim})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requested = str(payload.get("model") or "")
            if requested and requested != self.app.model_name:
                raise ValueError(f"unsupported model {requested!r}")
            vectors = self.app.embed(payload)
            self._send(
                200,
                {
                    "object": "list",
                    "model": self.app.model_name,
                    "data": [
                        {"object": "embedding", "index": index, "embedding": vector}
                        for index, vector in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )
        except Exception as exc:
            self._send(500, {"error": {"type": type(exc).__name__, "message": str(exc)}})


def serve(args: argparse.Namespace) -> None:
    Handler.app = EmbeddingApplication(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.model} on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    runtime = load_runtime_config()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser(
        "serve", help="Start the local embedding HTTP service",
        description="Serve a local embedding model through an OpenAI-compatible HTTP API.",
    )
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8001)
    serve_parser.add_argument("--model", default=runtime.get("embedding_model", ""))
    serve_parser.add_argument("--revision", default=runtime.get("embedding_revision", ""),
                              help="Hugging Face model commit/tag; use the same revision when preparing query vectors.")
    serve_parser.add_argument("--dim", type=int, default=runtime.get("embedding_dim"))
    serve_parser.add_argument("--device", default="cuda:0")
    serve_parser.add_argument("--dtype", default="bfloat16")
    serve_parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.model, args.dim = validate_embedding_settings(
        serve_parser, args.model, args.dim, model_flag="--model", dimension_flag="--dim",
    )
    serve(args)


if __name__ == "__main__":
    main()
