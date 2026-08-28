from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.config import OFFLINE_ROOT
from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
)
from embedding.chunk_builder import Chunk


class BaselineProcess(BaselineAdapter):
    def __init__(
        self,
        baseline: str,
        *,
        entry: dict[str, Any],
        config: dict[str, Any],
        python_executable: str,
    ) -> None:
        self.baseline = baseline
        self.entry = dict(entry)
        self.config = dict(config)
        self.timeout = float(config.get("baseline_worker_timeout") or 180)
        env = os.environ.copy()
        source_path = str(OFFLINE_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_path, env.get("PYTHONPATH", "")) if value
        )
        self._process = subprocess.Popen(
            [python_executable, "-m", "benchmarks.baseline_runtime.worker"],
            cwd=str(OFFLINE_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._lock = threading.Lock()
        self._next_id = 0
        self._request(
            "init",
            baseline=baseline,
            entry=self.entry,
            config=self.config,
        )

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._responses.put(line)
        self._responses.put(None)

    def _request(self, operation: str, **payload: Any) -> Any:
        with self._lock:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.baseline} worker exited with code {self._process.returncode}"
                )
            self._next_id += 1
            request_id = self._next_id
            message = {"id": request_id, "op": operation, **payload}
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
            self._process.stdin.flush()
            try:
                line = self._responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                self._terminate()
                raise TimeoutError(
                    f"{self.baseline} worker timed out during {operation}"
                ) from exc
            if line is None:
                raise RuntimeError(f"{self.baseline} worker closed stdout during {operation}")
            response = json.loads(line)
            if int(response.get("id", -1)) != request_id:
                raise RuntimeError(f"{self.baseline} worker response id mismatch")
            if not response.get("ok"):
                error = response.get("error") or {}
                raise RuntimeError(
                    f"{self.baseline} {operation} failed: "
                    f"{error.get('type', 'Error')}: {error.get('message', '')}\n"
                    f"{error.get('traceback', '')}"
                )
            return response.get("result")

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self._request("reset", sample_id=sample_id, state_dir=str(Path(state_dir).resolve()))

    def ingest(self, chunk: Chunk) -> None:
        self._request("ingest", chunk=chunk.to_dict())

    def end_session(self, session_id: str) -> None:
        self._request("end_session", session_id=session_id)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult.from_dict(
            self._request("retrieve", request=request.to_dict()) or {}
        )

    def snapshot(self) -> list[MemoryRecord]:
        return [MemoryRecord.from_dict(row) for row in self._request("snapshot") or []]

    def capabilities(self) -> dict[str, Any]:
        return dict(self._request("capabilities") or {})

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            self._request("close")
        finally:
            self._terminate()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> "BaselineProcess":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
