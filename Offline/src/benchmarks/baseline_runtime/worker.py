from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.protocol import RetrievalRequest
from benchmarks.baseline_runtime.registry import create_local_adapter
from embedding.chunk_builder import Chunk


def _dispatch(adapter: Any, operation: str, request: dict[str, Any]) -> Any:
    if operation == "reset":
        adapter.reset(str(request["sample_id"]), Path(request["state_dir"]))
        return None
    if operation == "ingest":
        adapter.ingest(Chunk.from_dict(request["chunk"]))
        return None
    if operation == "end_session":
        adapter.end_session(str(request["session_id"]))
        return None
    if operation == "retrieve":
        return adapter.retrieve(RetrievalRequest.from_dict(request["request"])).to_dict()
    if operation == "snapshot":
        return [row.to_dict() for row in adapter.snapshot()]
    if operation == "capabilities":
        return adapter.capabilities()
    if operation == "close":
        adapter.close()
        return None
    raise KeyError(f"unknown worker operation: {operation}")


def main() -> None:
    adapter = None
    for line in sys.stdin:
        if not line.strip():
            continue
        message: dict[str, Any] = {}
        should_close = False
        try:
            message = json.loads(line)
            operation = str(message.get("op") or "")
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "init":
                    adapter = create_local_adapter(
                        str(message["baseline"]),
                        config=dict(message.get("config") or {}),
                        entry=dict(message.get("entry") or {}),
                    )
                    result = adapter.capabilities()
                else:
                    if adapter is None:
                        raise RuntimeError("worker has not been initialized")
                    result = _dispatch(adapter, operation, message)
                    should_close = operation == "close"
            response = {"id": message.get("id"), "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - serialized across process boundary
            response = {
                "id": message.get("id"),
                "ok": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()
        if should_close:
            break


if __name__ == "__main__":
    main()
