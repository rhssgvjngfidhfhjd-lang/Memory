"""Small OpenAI-compatible server for the shared local Qwen3-VL embedder."""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embedding.qwen3_text_embedding import create_embedding_service  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    Handler.app = EmbeddingApplication(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.model} on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
