"""Data preparation and dependency checks without initializing models."""
from __future__ import annotations

from collections import Counter
import argparse
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

from packaging.requirements import Requirement

from src.utils import (
    benchmark_path, dataset_root, load_config_document, output_root,
    portable_value, project_path, raw_data_root, sha256_file as sha256, write_json_atomic,
)


# Data download, preparation, and audit

def write_report(path: Path, report: dict) -> None:
    write_json_atomic(path, report, trailing_newline=True)


def audit(manifest_path: Path, sources: list[str] | None = None, *, check_sources: bool = True) -> dict:
    from evidence_policy.evidence import SplitManifestIndex, iter_source_questions

    index = SplitManifestIndex(manifest_path)
    selected = sources or list(index.data_sources)
    counts = Counter()
    conversations = Counter()
    for source in selected:
        for split in ("train", "val", "test"):
            for conversation in index.conversations(split, data_source=source):
                counts[f"{source}/{split}"] += len(conversation.question_ids)
                conversations[f"{source}/{split}"] += 1
    files = {}
    actual = Counter()
    if check_sources:
        # Dataset locations come from explicit roots or the common raw-data directory.
        for row in iter_source_questions(index, None, data_sources=selected):
            actual[f"{row.data_source}/{row.split}"] += 1
            source = Path(row.source_path)
            if source not in files:
                files[source] = sha256(source)
        if dict(actual) != dict(counts):
            raise ValueError(f"Source question counts differ from the frozen split: {actual} != {counts}")
    return {
        "schema_version": 1,
        "split_manifest": portable_value(str(index.path)),
        "split_manifest_sha256": index.file_sha256,
        "split_unit": "conversation",
        "selection": "exact frozen conversation and question IDs; no resampling",
        "source_validation": "passed" if check_sources else "not_run",
        "data_sources": sorted(selected),
        "question_counts": dict(sorted(counts.items())),
        "conversation_counts": dict(sorted(conversations.items())),
        "source_files": [{"path": portable_value(str(p)), "sha256": digest} for p, digest in sorted(files.items())],
    }


def audit_difference(report: dict, reference: dict, sources: list[str] | None = None) -> dict:
    """Compare source snapshots within the same selected dataset scope."""
    if reference.get("schema_version") != 1 or reference.get("source_validation") != "passed":
        raise ValueError("Audit reference must be a schema-version-1 report with source validation passed")
    scope_roots = {
        "mem_gallery": dataset_root("memgallery"),
        "h2hmem_dyadic": dataset_root("h2hmem") / "dyadic",
        "h2hmem_multiparty": dataset_root("h2hmem") / "multi-party",
        "worldmemarena_lifelong": dataset_root("wma"),
    }

    def hashes(snapshot: dict) -> dict[str, str]:
        result = {}
        for row in snapshot["source_files"]:
            path = project_path(row["path"])
            if sources and not any(path.is_relative_to(scope_roots[source]) for source in sources):
                continue
            result[portable_value(str(path))] = row["sha256"]
        return result

    def counts(snapshot: dict, key: str) -> dict:
        return {name: count for name, count in snapshot[key].items()
                if not sources or name.split("/", 1)[0] in sources}

    actual, expected = hashes(report), hashes(reference)
    return {
        "manifest_changed": report["split_manifest_sha256"] != reference["split_manifest_sha256"],
        "question_counts_changed": counts(report, "question_counts") != counts(reference, "question_counts"),
        "conversation_counts_changed": counts(report, "conversation_counts") != counts(reference, "conversation_counts"),
        "modified": sorted(path for path in actual.keys() & expected.keys() if actual[path] != expected[path]),
        "added": sorted(actual.keys() - expected.keys()),
        "missing": sorted(expected.keys() - actual.keys()),
    }


def download(source: str, revision: str | None) -> dict:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit('Install download dependencies: pip install -e ".[data]"') from error
    registry = load_config_document("defaults.json")["dataset_sources"]
    entry = registry[source]
    commit = HfApi().dataset_info(entry["repo_id"], revision=revision).sha
    destination = raw_data_root() / source
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(entry["repo_id"], repo_type="dataset", revision=commit, local_dir=str(destination))
    report = {"source": source, "repo_id": entry["repo_id"], "revision": commit,
              "requested_revision": revision, "files": []}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and ".cache" not in path.relative_to(destination).parts and path.name != "download.manifest.json":
            report["files"].append({"path": path.relative_to(destination).as_posix(),
                                    "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_report(destination / "download.manifest.json", report)
    return report


def data_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download sources and audit the frozen experiment split without loading models.")
    parser.add_argument("--data-root", help="Raw dataset root, containing memgallery/, h2hmem/, wma/, etc.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download", help="Download upstream files and record the exact resolved revision and checksums.")
    p.add_argument("source", choices=("memgallery", "h2hmem", "wma"))
    p.add_argument("--revision", help="Pin an upstream revision; if omitted, resolve HEAD and record its commit.")
    p = commands.add_parser("audit", help="Check conversation disjointness, exact question selection, and source hashes.")
    p.add_argument("--manifest", type=Path, default=benchmark_path("multimodal_split_manifest.json"))
    p.add_argument("--source", action="append", choices=("mem_gallery", "h2hmem_dyadic", "h2hmem_multiparty", "worldmemarena_lifelong"))
    p.add_argument("--manifest-only", action="store_true", help="Validate split structure without requiring raw sources.")
    p.add_argument("--output", type=Path, help="Write the full audit; defaults to <output root>/audit/current.json.")
    p.add_argument("--reference", type=Path, help="Also require the selected source files to match a previous audit's SHA-256 values.")
    p = commands.add_parser("chunks", help="Preprocess raw benchmark dialogue into portable chunks.", add_help=False)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    p = commands.add_parser("queries", help="Prepare query embeddings for training and evaluation.", add_help=False)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args, extra = parser.parse_known_args(argv)
    if args.data_root:
        os.environ["HIVE_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
    if args.command in ("chunks", "queries"):
        if args.command == "chunks":
            from embedding.chunks import main as prepare
            prepare([*extra, *args.arguments])
        else:
            from embedding.build_embeddings import main as prepare
            prepare(["queries", *extra, *args.arguments])
        return
    if extra:
        parser.error(f"Unrecognized arguments: {' '.join(extra)}")
    if args.command == "audit":
        args.output = args.output or output_root() / "audit" / "current.json"
        if args.reference:
            if args.manifest_only:
                parser.error("--reference requires source validation; remove --manifest-only")
            if args.output.resolve() == args.reference.resolve():
                parser.error("--output must differ from --reference so the reference report is preserved")
        report = audit(args.manifest, args.source, check_sources=not args.manifest_only)
        difference = {}
        if args.reference:
            reference = json.loads(args.reference.read_text(encoding="utf-8"))
            difference = audit_difference(report, reference, args.source)
            report["reference_validation"] = "failed" if any(difference.values()) else "passed"
            report["reference_difference"] = difference
        write_report(args.output, report)
        report = {key: value for key, value in report.items() if key != "source_files"} | {"hashed_source_files": len(report["source_files"])}
        report["output"] = str(args.output.resolve())
    elif args.command == "download":
        report = download(args.source, args.revision)
        report = {key: value for key, value in report.items() if key != "files"} | {"downloaded_files": len(report["files"])}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.command == "audit" and any(difference.values()):
        raise ValueError(f"Frozen source snapshot differs; see {args.output}")


# Dependency checks

def dependency_groups() -> tuple[dict[str, list[str]], str]:
    """Read the dependency definitions from this checkout or its distribution."""
    module_path = Path(__file__).resolve()
    project_file = module_path.parents[1] / "pyproject.toml"
    if project_file.is_file() and project_file.parent / "embedding/cli.py" == module_path:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError as error:
                raise SystemExit("Python 3.10 source checks require tomli; install it before running this check.") from error
        with project_file.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        groups = {"core": project.get("dependencies", [])}
        groups.update(project.get("optional-dependencies", {}))
        return groups, str(project_file)

    try:
        distribution = metadata.distribution("hive-mem")
    except metadata.PackageNotFoundError as error:
        raise SystemExit("Dependency definitions were not found; use the source checkout or install hive-mem.") from error
    requirements = distribution.requires or []
    extras = distribution.metadata.get_all("Provides-Extra") or []
    # Each installed requirement carries its own extra/Python/platform marker.
    return {group: requirements for group in ("core", *extras)}, "installed hive-mem metadata"


def check_main(argv: list[str] | None = None) -> None:
    groups, source = dependency_groups()
    parser = argparse.ArgumentParser(description="Check installed dependencies without initializing a model or contacting a service.")
    parser.add_argument("--group", action="append", choices=tuple(groups), default=[])
    parser.add_argument("--output", type=Path, help="Save the actual Python and package versions.")
    args = parser.parse_args(argv)
    selected = {"core", *args.group}
    packages, errors = {}, []
    applicable = set()
    for group in selected:
        for text in groups[group]:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": "" if group == "core" else group}):
                applicable.add(text)
    for text in sorted(applicable):
        requirement = Requirement(text)
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing {requirement}")
            continue
        packages[requirement.name] = version
        if version not in requirement.specifier:
            errors.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
    report = {"python": platform.python_version(), "dependency_source": source, "groups": sorted(selected),
              "packages": packages, "errors": errors}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Dispatch data preparation and dependency checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("data", add_help=False, help="Download, prepare, or audit datasets.")
    commands.add_parser("check", add_help=False, help="Check installed dependency versions.")
    args, remaining = parser.parse_known_args(argv)
    if args.command == "data":
        data_main(remaining)
    else:
        check_main(remaining)


if __name__ == "__main__":
    main()
