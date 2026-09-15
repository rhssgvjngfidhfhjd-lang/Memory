#!/usr/bin/env python3
"""Validate one M2A conformance run against the 0912 acceptance contract.

The validator intentionally does not import the benchmark harness.  It inspects
the persisted artifacts so a run cannot pass merely because its in-memory
objects looked correct before serialization.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_UPSTREAM_URL = "https://github.com/Little-Fridge/M2A"
DEFAULT_UPSTREAM_COMMIT = "edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    run_dir: str
    sample: str
    passed: bool = True
    checks: list[Check] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name=name, passed=passed, detail=detail))
        if not passed:
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_checks"] = [
            check.name for check in self.checks if not check.passed
        ]
        return payload


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _normal_url(value: Any) -> str:
    url = str(value or "").strip().rstrip("/")
    return url[:-4] if url.lower().endswith(".git") else url


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _validate_hash_value(value: Any) -> tuple[bool, str]:
    if isinstance(value, str):
        return _valid_hash(value), value
    if isinstance(value, dict):
        expected = value.get("expected") or value.get("upstream")
        actual = value.get("actual") or value.get("runtime")
        valid = _valid_hash(expected) and _valid_hash(actual) and expected == actual
        return valid, f"expected={expected}, actual={actual}"
    return False, repr(value)


def _candidate_state_dirs(run_dir: Path, manifest: dict[str, Any], sample: str) -> list[Path]:
    candidates: list[Path] = []
    configured = manifest.get("baseline_state_dir")
    if configured:
        configured_path = Path(str(configured)).expanduser()
        candidates.extend([configured_path / sample, configured_path])
    candidates.extend(
        [
            run_dir / "memory" / "datasets" / sample,
            run_dir / "memory" / sample,
            run_dir / sample,
            run_dir,
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _discover_state_dir(run_dir: Path, manifest: dict[str, Any], sample: str | None) -> tuple[Path, str]:
    if sample:
        for candidate in _candidate_state_dirs(run_dir, manifest, sample):
            if (candidate / "raw.db").is_file() and (candidate / "semantic.db").exists():
                return candidate, sample
        raise ValueError(f"cannot find raw.db and semantic.db for sample {sample!r}")

    roots = [run_dir / "memory" / "datasets", run_dir / "memory", run_dir]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "raw.db").is_file() and (root / "semantic.db").exists():
            found.append(root)
        found.extend(
            path.parent
            for path in root.glob("*/raw.db")
            if (path.parent / "semantic.db").exists()
        )
    unique = {str(path.resolve()): path for path in found}
    if len(unique) != 1:
        raise ValueError(
            "--expected-sample is required when the run does not contain exactly one state directory"
        )
    state_dir = next(iter(unique.values()))
    return state_dir, state_dir.name


def _raw_schema_and_rows(path: Path) -> tuple[list[dict[str, Any]], list[sqlite3.Row]]:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        schema = [dict(row) for row in connection.execute("PRAGMA table_info(messages)")]
        rows = list(
            connection.execute(
                "SELECT msg_id, timestamp, role, text, image_path "
                "FROM messages ORDER BY msg_id"
            )
        )
    except sqlite3.Error as exc:
        raise ValueError(f"cannot inspect raw database {path}: {exc}") from exc
    finally:
        connection.close()
    return schema, rows


def _schema_field_names(schema: dict[str, Any]) -> set[str]:
    return {
        str(field.get("name") or field.get("field_name") or "")
        for field in (schema.get("fields") or [])
        if isinstance(field, dict)
    }


def _sqlite_semantic(path: Path) -> tuple[dict[str, set[str]], dict[str, list[dict[str, Any]]]]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    schemas: dict[str, set[str]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        table_names = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for collection in ("memory", "text_embs", "image_embs"):
            if collection not in table_names:
                continue
            schemas[collection] = {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{collection}")')
            }
            rows[collection] = [
                dict(row) for row in connection.execute(f'SELECT * FROM "{collection}"')
            ]
    finally:
        connection.close()
    return schemas, rows


def _milvus_semantic(path: Path) -> tuple[dict[str, set[str]], dict[str, list[dict[str, Any]]]]:
    schemas: dict[str, set[str]] = {}
    collections_dir = path / "collections"
    for collection in ("memory", "text_embs", "image_embs"):
        schema_path = collections_dir / collection / "schema.json"
        if schema_path.is_file():
            schema = _read_json(schema_path)
            schemas[collection] = _schema_field_names(schema)

    try:
        from pymilvus import MilvusClient  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "pymilvus is required to inspect the persisted semantic rows; "
            "run this script with the configured M2A/pipeline Python environment"
        ) from exc

    client = MilvusClient(str(path))
    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        collection_names = set(client.list_collections())
        for collection in ("memory", "text_embs", "image_embs"):
            if collection not in collection_names:
                continue
            client.load_collection(collection_name=collection)
            rows[collection] = list(
                client.query(
                    collection_name=collection,
                    filter="id >= 0",
                    output_fields=["*"],
                    limit=16_384,
                )
                or []
            )
            if collection not in schemas:
                description = client.describe_collection(collection_name=collection)
                schemas[collection] = _schema_field_names(description)
    except Exception as exc:
        raise ValueError(f"cannot inspect Milvus database {path}: {exc}") from exc
    finally:
        client.close()
    return schemas, rows


def _semantic_schema_and_rows(path: Path) -> tuple[dict[str, set[str]], dict[str, list[dict[str, Any]]]]:
    if path.is_file():
        try:
            with path.open("rb") as handle:
                header = handle.read(16)
        except OSError as exc:
            raise ValueError(f"cannot read semantic database {path}: {exc}") from exc
        if header == b"SQLite format 3\x00":
            schemas, rows = _sqlite_semantic(path)
            if {"memory", "text_embs", "image_embs"} <= set(schemas):
                return schemas, rows
            # Milvus Lite itself is persisted in an SQLite container, but its
            # internal table layout is not the three logical collections.
            return _milvus_semantic(path)
    if path.is_dir() or path.is_file():
        return _milvus_semantic(path)
    raise ValueError(f"unsupported semantic database artifact: {path}")


def _parse_evidence(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid evidence_ids JSON: {value!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"evidence_ids is not a list: {value!r}")
    ranges: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or isinstance(item[0], bool)
            or isinstance(item[1], bool)
        ):
            raise ValueError(f"invalid evidence range: {item!r}")
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid evidence range: {item!r}") from exc
        ranges.append((start, end))
    return ranges


def _memory_id(value: Any) -> str:
    memory_id = str(value or "")
    return memory_id.split(":", 1)[-1] if memory_id.startswith("m2a:") else memory_id


def _ids_from_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        value = item.get("memory_id") or item.get("id") if isinstance(item, dict) else item
        memory_id = _memory_id(value)
        if memory_id:
            result.append(memory_id)
    return result


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("manifest_question_id") or row.get("query_id") or row.get("question_id") or "")


def _trace_payload(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    payload = row.get("m2a_trace") or nested.get("m2a_trace") or {}
    return payload if isinstance(payload, dict) else {}


def _execution_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key) or "")
        for key in ("event", "component", "action", "tool_name", "stage")
    ).casefold()


def _event_values(event: dict[str, Any], key: str) -> list[Any]:
    value = event.get(key)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def validate_run(
    run_dir: Path,
    *,
    expected_sample: str | None = None,
    expected_utterances: int | None = None,
    expected_images: int | None = None,
    expected_questions: int | None = None,
    expected_upstream_url: str = DEFAULT_UPSTREAM_URL,
    expected_upstream_commit: str = DEFAULT_UPSTREAM_COMMIT,
) -> ValidationReport:
    run_dir = run_dir.expanduser().resolve()
    report = ValidationReport(run_dir=str(run_dir), sample=expected_sample or "")

    try:
        manifest = _read_json(run_dir / "run_manifest.json")
        if not isinstance(manifest, dict):
            raise ValueError("run_manifest.json is not an object")
        report.add("run_manifest", True, "run_manifest.json is readable")
    except ValueError as exc:
        report.add("run_manifest", False, str(exc))
        return report

    conformance = manifest.get("m2a_conformance")
    if not isinstance(conformance, dict):
        sidecar = run_dir / "m2a_conformance.json"
        try:
            conformance = _read_json(sidecar)
        except ValueError:
            conformance = {}
    upstream_url = conformance.get("upstream_url") or conformance.get("upstream_repository")
    upstream_commit = str(conformance.get("upstream_commit") or "").lower()
    report.add(
        "upstream_url",
        _normal_url(upstream_url).casefold() == _normal_url(expected_upstream_url).casefold(),
        f"actual={upstream_url!r}, expected={expected_upstream_url!r}",
    )
    report.add(
        "upstream_commit",
        upstream_commit == expected_upstream_commit.lower(),
        f"actual={upstream_commit!r}, expected={expected_upstream_commit!r}",
    )

    deviations = conformance.get("deviations")
    deviation_text = _flatten_text(deviations).casefold()
    report.add(
        "deviations_declared",
        isinstance(deviations, list) and bool(deviations),
        f"declared={len(deviations) if isinstance(deviations, list) else 0}",
    )
    deviation_categories = {
        "model": bool(re.search(r"qwen|model|backbone", deviation_text)),
        "top7_answer_budget": bool(re.search(r"top.?7|seven|retrieval.?budget|memory.?budget", deviation_text)),
        "benchmark_answer_prompt": bool(re.search(r"benchmark|qa.?prompt|answer.?prompt", deviation_text)),
    }
    report.add(
        "known_deviations_complete",
        all(deviation_categories.values()),
        json.dumps(deviation_categories, sort_keys=True),
    )
    compatibility = _flatten_text((manifest.get("baseline_runtime") or {}).get("compatibility_mode")).casefold()
    report.add(
        "no_simplified_compatibility_mode",
        not any(token in compatibility for token in ("direct", "fallback", "simplified")),
        f"compatibility_mode={compatibility!r}",
    )

    internal_hashes = conformance.get("internal_prompt_sha256")
    hash_details: list[str] = []
    hashes_valid = isinstance(internal_hashes, dict) and bool(internal_hashes)
    if isinstance(internal_hashes, dict):
        for name, value in internal_hashes.items():
            valid, detail = _validate_hash_value(value)
            hashes_valid = hashes_valid and valid
            hash_details.append(f"{name}:{detail}")
        names = " ".join(str(name).casefold() for name in internal_hashes)
        hashes_valid = hashes_valid and "chat" in names and "manager" in names
    report.add("internal_prompt_hashes", hashes_valid, "; ".join(hash_details) or "missing")

    answer_hash = conformance.get("answer_prompt_sha256")
    if isinstance(answer_hash, dict):
        answer_valid, answer_detail = _validate_hash_value(answer_hash)
        answer_actual = answer_hash.get("actual") or answer_hash.get("runtime")
    else:
        answer_valid, answer_detail = _validate_hash_value(answer_hash)
        answer_actual = answer_hash
    harness_hash = manifest.get("prompt_sha256")
    answer_valid = answer_valid and answer_actual == harness_hash
    report.add(
        "benchmark_answer_prompt_hash",
        answer_valid,
        f"conformance={answer_detail}, run_manifest={harness_hash}",
    )

    try:
        state_dir, sample = _discover_state_dir(run_dir, manifest, expected_sample)
        report.sample = sample
        report.statistics["state_dir"] = str(state_dir)
        report.add("state_dir", True, str(state_dir))
        try:
            state_dir.resolve().relative_to(run_dir)
            local_state = True
        except ValueError:
            local_state = False
        report.add(
            "state_dir_belongs_to_run",
            local_state,
            f"state_dir={state_dir.resolve()}, run_dir={run_dir}",
        )
    except ValueError as exc:
        report.add("state_dir", False, str(exc))
        return report

    try:
        raw_schema, raw_rows = _raw_schema_and_rows(state_dir / "raw.db")
    except ValueError as exc:
        report.add("raw_database", False, str(exc))
        raw_schema, raw_rows = [], []
    raw_columns = {str(row.get("name") or "") for row in raw_schema}
    required_raw = {"msg_id", "timestamp", "role", "text", "image_path"}
    primary_keys = {str(row.get("name")) for row in raw_schema if int(row.get("pk") or 0) == 1}
    report.add(
        "raw_schema",
        required_raw <= raw_columns and primary_keys == {"msg_id"},
        f"columns={sorted(raw_columns)}, primary_keys={sorted(primary_keys)}",
    )
    raw_ids = [int(row["msg_id"]) for row in raw_rows]
    sequential = raw_ids == list(range(1, len(raw_ids) + 1))
    timestamp_values: list[datetime] = []
    timestamp_valid = True
    for row in raw_rows:
        try:
            timestamp_values.append(datetime.fromisoformat(str(row["timestamp"])))
        except (TypeError, ValueError):
            timestamp_valid = False
    try:
        ordered = timestamp_valid and timestamp_values == sorted(timestamp_values)
    except TypeError:
        # Mixed offset-aware and offset-naive values have no unambiguous order.
        ordered = False
    contents_valid = all(str(row["role"] or "").strip() and str(row["text"] or "").strip() for row in raw_rows)
    report.add(
        "raw_count",
        expected_utterances is None or len(raw_rows) == expected_utterances,
        f"actual={len(raw_rows)}, expected={expected_utterances}",
    )
    report.add(
        "raw_order",
        sequential and ordered and contents_valid,
        f"sequential_ids={sequential}, chronological={ordered}, nonempty_role_text={contents_valid}",
    )
    raw_image_paths = [str(row["image_path"]) for row in raw_rows if row["image_path"]]
    report.add(
        "raw_images",
        expected_images is None or len(raw_image_paths) == expected_images,
        f"actual={len(raw_image_paths)}, expected={expected_images}",
    )
    report.statistics.update(raw_messages=len(raw_rows), raw_images=len(raw_image_paths))

    try:
        semantic_schemas, semantic_rows = _semantic_schema_and_rows(state_dir / "semantic.db")
        report.add("semantic_database", True, "semantic database is readable")
    except ValueError as exc:
        report.add("semantic_database", False, str(exc))
        semantic_schemas, semantic_rows = {}, {}
    required_semantic = {
        "memory": {"id", "text", "image_caption", "image_path", "evidence_ids"},
        "text_embs": {"id", "text", "text_dense", "text_sparse"},
        "image_embs": {"id", "image_dense"},
    }
    collection_details: dict[str, Any] = {}
    collection_valid = True
    for collection, fields in required_semantic.items():
        actual_fields = semantic_schemas.get(collection, set())
        collection_details[collection] = sorted(actual_fields)
        collection_valid = collection_valid and fields <= actual_fields
    report.add("semantic_collections_schema", collection_valid, json.dumps(collection_details, sort_keys=True))

    if (state_dir / "semantic.db").is_dir():
        index_fields: dict[str, set[str]] = {}
        for collection in ("text_embs", "image_embs"):
            path = state_dir / "semantic.db" / "collections" / collection / "manifest.json"
            try:
                payload = _read_json(path)
                index_fields[collection] = set((payload.get("index_specs") or {}).keys())
            except ValueError:
                index_fields[collection] = set()
        index_valid = {"text_dense", "text_sparse"} <= index_fields.get("text_embs", set()) and {
            "image_dense"
        } <= index_fields.get("image_embs", set())
        report.add("semantic_indexes", index_valid, repr(index_fields))

    memories = semantic_rows.get("memory", [])
    text_ids = {_memory_id(row.get("id")) for row in semantic_rows.get("text_embs", [])}
    image_ids = {_memory_id(row.get("id")) for row in semantic_rows.get("image_embs", [])}
    memory_ids = {_memory_id(row.get("id")) for row in memories}
    evidence_errors: list[str] = []
    semantic_image_paths: dict[str, str] = {}
    text_expected: set[str] = set()
    for memory in memories:
        memory_id = _memory_id(memory.get("id"))
        if memory.get("text") or memory.get("image_caption"):
            text_expected.add(memory_id)
        if memory.get("image_path"):
            semantic_image_paths[memory_id] = str(memory["image_path"])
        try:
            ranges = _parse_evidence(memory.get("evidence_ids"))
        except ValueError as exc:
            evidence_errors.append(f"memory {memory_id}: {exc}")
            continue
        if not ranges:
            evidence_errors.append(f"memory {memory_id}: empty evidence_ids")
        for start, end in ranges:
            if start < 1 or end < start or end > len(raw_rows):
                evidence_errors.append(
                    f"memory {memory_id}: evidence [{start}, {end}] outside [1, {len(raw_rows)}]"
                )
    report.add(
        "semantic_evidence_ranges",
        bool(memories) and not evidence_errors,
        "; ".join(evidence_errors[:10]) if evidence_errors else f"validated={len(memories)}",
    )
    report.add(
        "text_memory_index_mapping",
        text_expected == text_ids,
        f"expected={len(text_expected)}, indexed={len(text_ids)}, missing={sorted(text_expected - text_ids)[:10]}",
    )
    raw_image_set = set(raw_image_paths)
    image_mapping_valid = set(semantic_image_paths) == image_ids and all(
        path in raw_image_set for path in semantic_image_paths.values()
    )
    report.add(
        "image_memory_index_mapping",
        image_mapping_valid,
        f"raw_images={len(raw_image_paths)}, semantic_images={len(semantic_image_paths)}, indexed={len(image_ids)}",
    )
    report.statistics.update(
        semantic_memories=len(memories),
        text_embeddings=len(text_ids),
        image_embeddings=len(image_ids),
    )

    try:
        results = _read_json(run_dir / "results.json")
        if not isinstance(results, list):
            raise ValueError("results.json is not a list")
    except ValueError as exc:
        report.add("results", False, str(exc))
        results = []
    question_target = expected_questions
    if question_target is None and manifest.get("questions") is not None:
        try:
            question_target = int(manifest["questions"])
        except (TypeError, ValueError):
            pass
    result_samples = {
        str(row.get("dataset") or row.get("sample_id") or "")
        for row in results
        if isinstance(row, dict)
    }
    result_errors = [row for row in results if not isinstance(row, dict) or row.get("error")]
    results_valid = (
        (question_target is None or len(results) == question_target)
        and not result_errors
        and (not sample or result_samples == {sample})
    )
    report.add(
        "results_and_question_count",
        results_valid,
        f"actual={len(results)}, expected={question_target}, samples={sorted(result_samples)}, errors={len(result_errors)}",
    )
    report.statistics["questions"] = len(results)

    try:
        retrieval_rows = _read_jsonl(run_dir / "retrieval_trace.jsonl")
    except ValueError as exc:
        report.add("retrieval_trace", False, str(exc))
        retrieval_rows = []
    results_keys = [_row_key(row) for row in results if isinstance(row, dict)]
    retrieval_keys = [_row_key(row) for row in retrieval_rows]
    report.add(
        "retrieval_trace_coverage",
        len(retrieval_rows) == len(results) and retrieval_keys == results_keys,
        f"retrieval={len(retrieval_rows)}, results={len(results)}, ordered_ids_match={retrieval_keys == results_keys}",
    )

    retrieval_errors: list[str] = []
    exactly_seven = 0
    for index, row in enumerate(retrieval_rows):
        label = _row_key(row) or f"row-{index + 1}"
        payload = _trace_payload(row)
        final_ids = _unique(_ids_from_items(payload.get("final_memory_ids")))
        if not final_ids:
            final_ids = _unique(_ids_from_items(row.get("top_k")))
        declared_final_count = payload.get("final_memory_count")
        if declared_final_count is not None:
            try:
                if int(declared_final_count) != len(final_ids):
                    retrieval_errors.append(f"{label}: final_memory_count mismatch")
            except (TypeError, ValueError):
                retrieval_errors.append(f"{label}: invalid final_memory_count")
        if len(final_ids) > 7:
            retrieval_errors.append(f"{label}: final unique memories={len(final_ids)} > 7")
        if not final_ids:
            retrieval_errors.append(f"{label}: no final memory IDs")
        external_ids = _unique(_ids_from_items(row.get("top_k")))
        if external_ids != final_ids:
            retrieval_errors.append(f"{label}: top_k differs from m2a_trace.final_memory_ids")
        searches = payload.get("semantic_searches")
        if not isinstance(searches, list) or not searches:
            retrieval_errors.append(f"{label}: no semantic_searches trace")
            searches = []
        searched_ids: list[str] = []
        candidate_count = 0
        for search in searches:
            if not isinstance(search, dict):
                retrieval_errors.append(f"{label}: malformed semantic_searches item")
                continue
            returned = _ids_from_items(search.get("returned_memory_ids") or search.get("memory_ids"))
            searched_ids.extend(returned)
            for key in ("candidate_count", "available_candidate_count"):
                try:
                    candidate_count = max(candidate_count, int(search.get(key) or 0))
                except (TypeError, ValueError):
                    retrieval_errors.append(f"{label}: invalid {key}")
        searched_unique = set(searched_ids)
        candidate_count = max(candidate_count, len(searched_unique))
        if candidate_count >= 7 and len(final_ids) != 7:
            retrieval_errors.append(
                f"{label}: {candidate_count} candidates observed but final count is {len(final_ids)}, expected 7"
            )
        if len(final_ids) == 7:
            exactly_seven += 1
        if searched_unique and not set(final_ids) <= searched_unique:
            retrieval_errors.append(f"{label}: final IDs are not a subset of searched IDs")
        for counter in ("chat_agent_query_calls", "memory_manager_query_calls"):
            try:
                if int(payload.get(counter) or 0) < 1:
                    retrieval_errors.append(f"{label}: {counter} < 1")
            except (TypeError, ValueError):
                retrieval_errors.append(f"{label}: invalid {counter}")
        unknown = set(final_ids) - memory_ids
        if memory_ids and unknown:
            retrieval_errors.append(f"{label}: unknown semantic memory IDs {sorted(unknown)}")
    report.add(
        "m2a_retrieval_budget_and_trace",
        bool(retrieval_rows) and not retrieval_errors,
        "; ".join(retrieval_errors[:20]) if retrieval_errors else f"validated={len(retrieval_rows)}, exactly_seven={exactly_seven}",
    )
    report.statistics["questions_with_exactly_seven_memories"] = exactly_seven

    execution_path = state_dir / "m2a_execution_trace.jsonl"
    if not execution_path.is_file():
        execution_path = run_dir / "m2a_execution_trace.jsonl"
    try:
        execution_rows = _read_jsonl(execution_path)
    except ValueError as exc:
        report.add("m2a_execution_trace", False, str(exc))
        execution_rows = []
    event_texts = [_execution_text(row) for row in execution_rows]
    chat_positions = [i for i, text in enumerate(event_texts) if "chat_agent" in text or "chatagent" in text]
    manager_positions = [i for i, text in enumerate(event_texts) if "memory_manager" in text or "memorymanager" in text]
    forbidden_positions = [
        i
        for i, text in enumerate(event_texts)
        if any(token in text for token in ("fallback", "semantic_store_direct", "direct_insert"))
    ]
    chain_valid = bool(chat_positions and manager_positions) and any(
        chat < manager for chat in chat_positions for manager in manager_positions
    )
    report.add(
        "original_agent_manager_chain",
        bool(execution_rows) and chain_valid and not forbidden_positions,
        f"events={len(execution_rows)}, chat_agent_events={len(chat_positions)}, memory_manager_events={len(manager_positions)}, forbidden_events={forbidden_positions[:10]}",
    )
    ingest_events = [
        row
        for row, text in zip(execution_rows, event_texts)
        if ("chat_agent" in text or "chatagent" in text)
        and str(row.get("action") or "").casefold() == "ingest"
    ]
    traced_raw_ids: list[int] = []
    invalid_raw_trace = False
    turn_ids: list[str] = []
    for event in ingest_events:
        event_raw_ids = _event_values(event, "raw_ids")
        if len(event_raw_ids) != 1:
            invalid_raw_trace = True
        for raw_id in event_raw_ids:
            try:
                traced_raw_ids.append(int(raw_id))
            except (TypeError, ValueError):
                invalid_raw_trace = True
        turn_id = str(event.get("turn_id") or "")
        if not turn_id:
            invalid_raw_trace = True
        turn_ids.append(turn_id)
    ingest_coverage_valid = (
        not invalid_raw_trace
        and len(ingest_events) == len(raw_rows)
        and sorted(traced_raw_ids) == raw_ids
        and len(set(turn_ids)) == len(turn_ids)
    )
    report.add(
        "execution_ingest_raw_coverage",
        ingest_coverage_valid,
        f"ingest_events={len(ingest_events)}, raw_rows={len(raw_rows)}, traced_raw_ids={len(traced_raw_ids)}, unique_turn_ids={len(set(turn_ids))}",
    )

    delta_events = [
        row
        for row, text in zip(execution_rows, event_texts)
        if ("memory_manager" in text or "memorymanager" in text)
        and "semantic_delta" in text.replace(" ", "_")
    ]
    traced_semantic_ids = {
        _memory_id(value)
        for event in delta_events
        for value in _event_values(event, "memory_ids")
        if _memory_id(value)
    }
    report.add(
        "execution_semantic_delta_coverage",
        bool(delta_events) and memory_ids <= traced_semantic_ids,
        f"delta_events={len(delta_events)}, current_memories={len(memory_ids)}, traced_memories={len(traced_semantic_ids)}, missing={sorted(memory_ids - traced_semantic_ids)[:10]}",
    )

    event_actions = [
        str(row.get("action") or "").strip().casefold().replace(" ", "_")
        for row in execution_rows
    ]
    manager_query_events = [
        action
        for action, text in zip(event_actions, event_texts)
        if ("memory_manager" in text or "memorymanager" in text)
        and (action == "query" or action.startswith("query_"))
    ]
    semantic_search_events = [
        action
        for action, text in zip(event_actions, event_texts)
        if ("memory_manager" in text or "memorymanager" in text)
        and ("semantic_search" in action or "search_semantic" in action)
    ]
    handoff_events = [action for action in event_actions if "handoff" in action]
    required_handoffs = len(results)
    report.add(
        "execution_retrieval_handoff_coverage",
        len(manager_query_events) >= required_handoffs
        and len(semantic_search_events) >= required_handoffs
        and len(handoff_events) == required_handoffs,
        f"questions={required_handoffs}, manager_queries={len(manager_query_events)}, semantic_searches={len(semantic_search_events)}, handoffs={len(handoff_events)}",
    )
    report.statistics["execution_trace_events"] = len(execution_rows)
    return report


def _positive_or_zero(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-sample", "--sample", dest="expected_sample")
    parser.add_argument("--expected-utterances", type=_positive_or_zero)
    parser.add_argument("--expected-images", type=_positive_or_zero)
    parser.add_argument("--expected-questions", type=_positive_or_zero)
    parser.add_argument("--expected-upstream-url", default=DEFAULT_UPSTREAM_URL)
    parser.add_argument("--expected-upstream-commit", default=DEFAULT_UPSTREAM_COMMIT)
    parser.add_argument(
        "--report",
        type=Path,
        help="report path (default: RUN_DIR/m2a_conformance_report.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_run(
        args.run_dir,
        expected_sample=args.expected_sample,
        expected_utterances=args.expected_utterances,
        expected_images=args.expected_images,
        expected_questions=args.expected_questions,
        expected_upstream_url=args.expected_upstream_url,
        expected_upstream_commit=args.expected_upstream_commit,
    )
    payload = report.to_dict()
    report_path = args.report or args.run_dir / "m2a_conformance_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
