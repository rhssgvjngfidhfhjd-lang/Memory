#!/usr/bin/env python3
"""One-off helper: pick a ~40-sample, category-proportional pilot subset of
the same 400-sample quota selection used by run_worldmemarena.sh.

Not part of the official reproduction protocol -- this exists only to sanity
check config/throughput/correctness on a small slice before committing to the
full 400-sample (8-12h) run. Reuses the exact same SUBCAT_MAP as
run_worldmemarena.sh; QUOTA is scaled down by ~10x (min 1 per category) so
the category mix roughly matches the full run.

Usage: python3 scripts/_pilot_select_indices.py <out_file> <dataset_dir>
"""
import json, sys, collections, os
from pathlib import Path

out_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])

# Must stay in sync with scripts/run_worldmemarena.sh's SUBCAT_MAP.
SUBCAT_MAP = {
    "agent/gui/excel":        "agent/arena/excel",
    "agent/gui/file_mgmt":    "agent/arena/file_mgmt",
    "agent/gui/image_edit":   "agent/arena/image_edit",
    "agent/gui/web":          "agent/arena/web",
    "agent/gui/word_docs":    "agent/arena/word_docs",
    "agent/gui/css":          "agent/vab/css",
    "agent/gui/mobile":       "agent/vab/mobile",
    "agent/gui/webarena_lite":"agent/vab/webarena-lite",
    "agent/embodied/eb_alfred_base":               "agent/eb_alfred/base",
    "agent/embodied/eb_alfred_common_sense":       "agent/eb_alfred/common_sense",
    "agent/embodied/eb_alfred_complex_instruction":"agent/eb_alfred/complex_instruction",
    "agent/embodied/eb_alfred_long_horizon":       "agent/eb_alfred/long_horizon",
    "agent/embodied/eb_alfred_visual_appearance":  "agent/eb_alfred/visual_appearance",
    "agent/embodied/eb_nav_base":                  "agent/eb_nav/base",
    "agent/embodied/eb_nav_common_sense":          "agent/eb_nav/common_sense",
    "agent/embodied/eb_nav_complex_instruction":   "agent/eb_nav/complex_instruction",
    "agent/embodied/eb_nav_long_horizon":          "agent/eb_nav/long_horizon",
    "agent/embodied/eb_nav_visual_appearance":     "agent/eb_nav/visual_appearance",
    "agent/embodied/minecraft":                    "agent/vab/minecraft",
    "agent/embodied/omnigibson":                   "agent/vab/omnigibson",
    "lifelong/project/academic":  "lifelong/domain_a_v2/academic",
    "lifelong/project/education": "lifelong/domain_a_v2/education",
    "lifelong/project/finance":   "lifelong/domain_a_v2/finance",
    "lifelong/project/health":    "lifelong/domain_a_v2/health",
    "lifelong/project/software":  "lifelong/domain_a_v2/software",
    "lifelong/project/startup":   "lifelong/domain_a_v2/startup",
    "lifelong/personal":          "lifelong/domain_b_v2",
}

# Same as run_worldmemarena.sh's QUOTA (sums to 400).
FULL_QUOTA = {
    "lifelong/domain_a_v2/academic": 2,
    "lifelong/domain_a_v2/education": 2,
    "lifelong/domain_a_v2/finance":   2,
    "lifelong/domain_a_v2/health":    2,
    "lifelong/domain_a_v2/software":  2,
    "lifelong/domain_a_v2/startup":   2,
    "lifelong/domain_b_v2": 10,
    "agent/arena/excel": 21,
    "agent/arena/file_mgmt": 17,
    "agent/arena/image_edit": 28,
    "agent/arena/web": 24,
    "agent/arena/word_docs": 22,
    "agent/eb_alfred/base": 13,
    "agent/eb_alfred/common_sense": 12,
    "agent/eb_alfred/complex_instruction": 12,
    "agent/eb_alfred/long_horizon": 12,
    "agent/eb_alfred/visual_appearance": 10,
    "agent/eb_nav/base": 14,
    "agent/eb_nav/common_sense": 13,
    "agent/eb_nav/complex_instruction": 15,
    "agent/eb_nav/long_horizon": 16,
    "agent/eb_nav/visual_appearance": 11,
    "agent/vab/css": 18,
    "agent/vab/minecraft": 30,
    "agent/vab/mobile": 17,
    "agent/vab/omnigibson": 37,
    "agent/vab/webarena-lite": 36,
}
assert sum(FULL_QUOTA.values()) == 400

SCALE = 10
QUOTA = {k: max(1, round(v / SCALE)) for k, v in FULL_QUOTA.items()}
target = sum(QUOTA.values())

excluded = {"id_mapping.json"}
all_files = sorted(
    p for p in data_root.rglob("*.json")
    if p.name not in excluded and p.parent != data_root
)


def subcat_from_path(p):
    rel = str(p.parent.relative_to(data_root)).replace(os.sep, "/")
    return SUBCAT_MAP.get(rel, rel)


taken = collections.Counter()
picked = []
for i, p in enumerate(all_files, start=1):
    sub = subcat_from_path(p)
    if taken[sub] < QUOTA.get(sub, 0):
        picked.append(i)
        taken[sub] += 1

missing = {k: QUOTA[k] - taken[k] for k in QUOTA if taken[k] != QUOTA[k]}
if missing:
    print(f"[pilot_select] WARNING quota not fully satisfied: {missing}", file=sys.stderr)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(map(str, picked)) + "\n")
print(f"[pilot_select] selected {len(picked)} samples (target {target}) -> {out_path}")
