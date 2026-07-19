from __future__ import annotations

import argparse
import json

from .memgallery.input_loader import audit_ablation_chunks, write_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Mem-Gallery a/b/c chunk inputs.")
    parser.add_argument("--text-only", default="artifacts/chunks_text_only.jsonl")
    parser.add_argument("--text-caption", default="artifacts/chunks_text_caption.jsonl")
    parser.add_argument("--multimodal", default="artifacts/chunks.jsonl")
    parser.add_argument("--output", default="artifacts/simple_memory_pipeline/chunk_audit.json")
    args = parser.parse_args()
    report = audit_ablation_chunks(args.text_only, args.text_caption, args.multimodal)
    write_audit(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
