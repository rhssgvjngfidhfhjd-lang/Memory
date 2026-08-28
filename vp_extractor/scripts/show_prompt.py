from __future__ import annotations

import argparse
import json
from pathlib import Path

from vp_extractor.io import load_caption_map
from vp_extractor.vlm import ObjectDiscoverer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = (
    PROJECT_ROOT.parent
    / "Mem-Gallery/benchmark/data/image/Academic_Animal_Pet_Research_Life/D13_IMG_004.jpg"
)
DEFAULT_CAPTIONS = (
    PROJECT_ROOT.parent
    / "Mem-Gallery/benchmark/data/dialog/Academic_Animal_Pet_Research_Life.json"
)


class _UnusedVLM:
    def generate(self, prompt, image):
        raise RuntimeError("This script only builds prompts and never calls the VLM")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one real discovery prompt")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--caption-file", type=Path, default=DEFAULT_CAPTIONS)
    args = parser.parse_args()

    config = json.loads((PROJECT_ROOT / "configs/default.json").read_text(encoding="utf-8"))
    prompt_path = PROJECT_ROOT / config["prompts"]["discovery"]
    template = prompt_path.read_text(encoding="utf-8")
    guided_path = PROJECT_ROOT / config["prompts"]["caption_guided"]
    guided_template = guided_path.read_text(encoding="utf-8")
    max_primitives = int(config["discovery"]["max_primitives"])
    caption = load_caption_map(args.caption_file.resolve()).get(args.image.name)
    if not caption:
        raise SystemExit(f"Caption not found for {args.image.name}")

    discoverer = ObjectDiscoverer(
        _UnusedVLM(), template, "", max_primitives, guided_template
    )
    print(discoverer.build_prompt(caption))


if __name__ == "__main__":
    main()
