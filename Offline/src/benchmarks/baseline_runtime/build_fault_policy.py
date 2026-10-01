from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.baseline_runtime.call_trace import CallRecorder


_NON_SKIPPABLE_MARKERS = (
    "status 401",
    "status code: 401",
    "status 403",
    "status code: 403",
    "unauthorized",
    "authentication",
    "invalid api key",
    "insufficient_quota",
    "insufficient quota",
    "insufficient credit",
    "insufficient balance",
    "worker exited with code",
    "worker closed stdout",
    "worker response id mismatch",
    "worker timed out during",
    "no space left on device",
    "database or disk is full",
    # ORM/database lifecycle errors may happen after a tool has already
    # written part of a request.  Skipping them could therefore duplicate
    # writes after resume, so they must remain fail-closed.
    "detachedinstanceerror",
    "objectdeletederror",
    "staledataerror",
    "pendingrollbackerror",
    "resourceclosederror",
    "unboundexecutionerror",
    "dbapierror",
    "databaseerror",
    "integrityerror",
    "operationalerror",
    "internalerror",
    "programmingerror",
    "interfaceerror",
    "statementerror",
    "partial commit",
    "duplicate write",
)


def is_non_skippable_build_failure(exc: Exception) -> bool:
    """Return whether one failure represents a global/system-level problem."""
    text = str(exc).casefold()
    return any(marker in text for marker in _NON_SKIPPABLE_MARKERS)


@dataclass
class ConsecutiveBuildFaultPolicy:
    """Audit isolated build failures and stop at a consecutive-failure limit."""

    baseline: str
    benchmark: str
    enabled: bool
    maximum: int = 10
    recorder: CallRecorder | None = None
    # M2A formal runs may deliberately prefer completing a benchmark with an
    # audited missing point over aborting the entire sample.  In that mode,
    # even failures normally classified as global/system-level are skippable.
    fail_open: bool = False
    consecutive: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.maximum = max(1, int(self.maximum))

    def success(self) -> None:
        self.consecutive = 0

    def handle(
        self,
        exc: Exception,
        *,
        chunk: Any | None,
        point_kind: str,
        session_id: str,
    ) -> None:
        if not self.enabled or (
            is_non_skippable_build_failure(exc) and not self.fail_open
        ):
            raise exc
        self.consecutive += 1
        metadata = dict(getattr(chunk, "metadata", None) or {})
        row = {
            "phase": "build_fault",
            "service": "runtime",
            "event": "skipped_build_point",
            "failed": True,
            "success": False,
            "point_kind": point_kind,
            "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
            "dialogue_id": str(metadata.get("dialogue_id") or ""),
            "session_id": session_id,
            "consecutive_failed_build_points": self.consecutive,
            "max_consecutive_failed_build_points": self.maximum,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
        }
        self.failures.append(row)
        if self.recorder is not None:
            self.recorder.append(row)
        if self.consecutive >= self.maximum:
            raise RuntimeError(
                f"{self.benchmark} stopped after {self.consecutive} consecutive "
                f"failed {self.baseline} build points"
            ) from exc
