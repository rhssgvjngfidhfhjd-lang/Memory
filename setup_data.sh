#!/bin/sh
# Downloads the datasets that are intentionally NOT stored in this git repo
# (see .gitignore). Run once after cloning.
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

echo "[1/3] WorldMemArena dataset (~10 GB, HuggingFace: LCZZZZ/WorldMemArena)"
huggingface-cli download LCZZZZ/WorldMemArena --repo-type dataset \
  --local-dir "$ROOT/WorldMemArena/WorldMemArena"

echo "[2/3] MemEye dataset (~500 MB, HuggingFace: MemEyeBench/MemEye)"
huggingface-cli download MemEyeBench/MemEye --repo-type dataset \
  --local-dir "$ROOT/MemEye/data"

echo "[3/3] Mem-Gallery dataset (~520 MB, benchmark/data/)"
echo "  NOTE: the upstream repo (github.com/YuanchenBei/Mem-Gallery) does not"
echo "  track benchmark/data/ in git and does not document a public download"
echo "  link for it. Source it manually and place it at:"
echo "    $ROOT/Mem-Gallery/benchmark/data/"
echo "  (expected subdirs: dialog/, image/)"

echo "Done. WorldMemArena and MemEye datasets are ready; place Mem-Gallery data manually (see note above)."
