import argparse
import json
from pathlib import Path

from .operations import get_operations
from .pipeline import MemoryExtractionPipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal memory extraction pipeline.")
    parser.add_argument("--input-file", required=True, help="Path to the input file.")
    parser.add_argument(
        "--input-type",
        required=True,
        choices=["text", "documents", "dialogue"],
        help="Input format type.",
    )
    parser.add_argument(
        "--chunk-mode",
        default=None,
        choices=["turn", "turn-pair", "full-session", "fixed-length"],
        help="Chunking mode. Defaults depend on input type.",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--operations",
        default="insert",
        help="Comma-separated operations to enable. Default 'insert' (insert-only); "
        "full set: 'insert,update,delete,noop'.",
    )
    parser.add_argument("--retriever", default="contriever", choices=["contriever", "dpr", "dragon"])
    parser.add_argument("--model", required=True, help="LLM model name.")
    parser.add_argument("--api", action="store_true", help="Use an OpenAI-compatible API backend.")
    parser.add_argument("--api-base", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument(
        "--api-key",
        nargs="+",
        default=None,
        help="One or more API keys. Required when --api is set.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-file", default=None, help="Optional JSON output path.")
    return parser


def load_input_data(input_file: str, input_type: str):
    path = Path(input_file)
    if input_type == "text":
        return path.read_text(encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = MemoryExtractionPipeline(
        model=args.model,
        api=args.api,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        retriever=args.retriever,
        device=args.device,
        operations=get_operations(args.operations.split(",")),
    )
    result = pipeline.run(
        input_data=load_input_data(args.input_file, args.input_type),
        input_type=args.input_type,
        chunk_mode=args.chunk_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out_file:
        Path(args.out_file).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
