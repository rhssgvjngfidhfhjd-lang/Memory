from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, TypeVar

from benchmarks.io_utils import write_json_atomic


Item = TypeVar("Item")
Value = TypeVar("Value")


def sample_artifact_path(root: Path, sample_id: str) -> Path:
    """Return a collision-safe, portable checkpoint filename for one sample."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:12]
    return root / f"{slug[:96]}-{digest}.json"


def load_sample_artifact(
    root: Path,
    sample_id: str,
    *,
    signature: str,
) -> dict[str, Any] | None:
    path = sample_artifact_path(root, sample_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("sample_id") != sample_id:
        return None
    signature_matches = payload.get("signature") == signature
    allow_stale = os.getenv("BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT", "0") == "1"
    if not signature_matches and not allow_stale:
        return None
    artifact = payload.get("artifact")
    return artifact if isinstance(artifact, dict) else None


def save_sample_artifact(
    root: Path,
    sample_id: str,
    *,
    signature: str,
    artifact: dict[str, Any],
) -> Path:
    path = sample_artifact_path(root, sample_id)
    write_json_atomic(
        path,
        {
            "version": 1,
            "sample_id": sample_id,
            "signature": signature,
            "artifact": artifact,
        },
    )
    return path


def signature_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validated_paired_resume_signatures(
    checkpoints: Iterable[tuple[str, Path, Path]],
) -> tuple[str, ...]:
    """Return historical signatures only when state and QA checkpoints agree.

    A long-running native baseline may outlive harmless harness-source changes
    that alter the freshly computed run signature.  Resuming such a run is safe
    only when the durable database checkpoint and its native QA checkpoint bind
    the same sample to the same historical signature.  A malformed or
    contradictory pair is treated as checkpoint corruption instead of being
    silently accepted.
    """
    signatures: list[str] = []
    for sample_id, state_path, qa_path in checkpoints:
        if not (state_path.is_file() and qa_path.is_file()):
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"damaged resume checkpoint pair for {sample_id}: {exc}"
            ) from exc
        state_signature = str(state.get("signature") or "")
        qa_signature = str(qa.get("signature") or "")
        if not state_signature or not qa_signature:
            raise RuntimeError(
                f"resume checkpoint pair for {sample_id} is missing a signature"
            )
        if state_signature != qa_signature:
            raise RuntimeError(
                f"resume checkpoint signature mismatch for {sample_id}: "
                f"state={state_signature}, qa={qa_signature}"
            )
        signatures.append(state_signature)
    return tuple(dict.fromkeys(signatures))


def validated_qa_only_resume_signatures(
    checkpoints: Iterable[tuple[str, Path, Path]],
) -> tuple[str, ...]:
    """Return native-QA signatures bound to an audited, read-only MMA bank.

    QA-only replay deliberately does not resume MMA ingestion, so the copied
    ``.offline_mma_resume.json`` belongs to the historical build configuration
    and must not be paired with the new QA run signature.  The reuse provenance
    marker is the durable binding in this mode.  We still reject malformed QA
    checkpoints instead of silently replaying already paid questions.
    """
    signatures: list[str] = []
    for sample_id, provenance_path, qa_path in checkpoints:
        if not qa_path.is_file():
            continue
        if not provenance_path.is_file():
            raise RuntimeError(
                f"QA-only resume for {sample_id} lacks audited reuse provenance: "
                f"{provenance_path}"
            )
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"damaged QA-only resume checkpoint for {sample_id}: {exc}"
            ) from exc
        if provenance.get("mode") != "qa_only_isolated_memory_reuse":
            raise RuntimeError(
                f"invalid QA-only reuse provenance mode for {sample_id}: "
                f"{provenance.get('mode')!r}"
            )
        if qa.get("version") != 1 or qa.get("sample_id") != sample_id:
            raise RuntimeError(
                f"native QA checkpoint identity mismatch for {sample_id}"
            )
        qa_signature = str(qa.get("signature") or "")
        if not qa_signature:
            raise RuntimeError(
                f"native QA checkpoint for {sample_id} is missing a signature"
            )
        signatures.append(qa_signature)
    return tuple(dict.fromkeys(signatures))


def parallel_map_ordered(
    items: Iterable[Item],
    worker: Callable[[Item], Value],
    *,
    max_workers: int,
    item_key: Callable[[Item], str] = str,
    on_complete: Callable[[str, Value], None] | None = None,
    on_error: Callable[[str, Exception], None] | None = None,
) -> list[Value]:
    """Run all independent samples and return input-ordered values.

    Failures are collected until every submitted sample has had a chance to
    finish.  This matters for benchmark runners because each successful sample
    writes its own checkpoint; aborting iteration on the first exception hides
    later failures and makes recovery appear less complete than it is.
    """
    ordered = list(items)
    if not ordered:
        return []
    failures: list[tuple[str, Exception]] = []
    if max_workers <= 1:
        values = []
        for item in ordered:
            key = item_key(item)
            try:
                value = worker(item)
            except Exception as exc:
                failures.append((key, exc))
                _report_sample_failure(key, exc)
                if on_error is not None:
                    on_error(key, exc)
                continue
            if on_complete is not None:
                on_complete(key, value)
            values.append(value)
        if failures:
            _raise_parallel_failures(failures)
        return values

    completed: dict[str, Value] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future[Value], str] = {
            pool.submit(worker, item): item_key(item) for item in ordered
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                failures.append((key, exc))
                _report_sample_failure(key, exc)
                if on_error is not None:
                    on_error(key, exc)
                continue
            completed[key] = value
            if on_complete is not None:
                on_complete(key, value)
    if failures:
        _raise_parallel_failures(failures)
    return [completed[item_key(item)] for item in ordered]


def _report_sample_failure(key: str, exc: Exception) -> None:
    """Expose deferred sample failures immediately while peers checkpoint."""
    print(
        "[sample-error] "
        + json.dumps(
            {
                "sample_id": str(key),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "action": "continue_other_samples_then_fail_job",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _raise_parallel_failures(failures: list[tuple[str, Exception]]) -> None:
    details = "; ".join(
        f"{key}: {type(exc).__name__}: {exc}" for key, exc in failures
    )
    raise RuntimeError(
        f"{len(failures)} parallel sample(s) failed after other samples were allowed "
        f"to finish: {details}"
    ) from failures[0][1]
