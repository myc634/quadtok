#!/usr/bin/env python3
"""
把 tree_planning/plans/ 下所有 pkl 导出为一个 JSON，供 HTML 可视化读取。
输出: tree_planning/demo/plans_data.json

用法: python tree_planning/export_plans_json.py
"""
import json
import pickle
from pathlib import Path

TREE_DIR = Path(__file__).resolve().parent
PLANS_DIR = TREE_DIR / "plans"
OUT_PATH = TREE_DIR / "demo" / "plans_data.json"


def _parent_8x8_of_16x16(idx_16):
    r, c = idx_16 // 16, idx_16 % 16
    return (r // 2) * 8 + (c // 2)


def process_plan(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        plan = pickle.load(f)
    pi = plan["patch_indices"]
    if hasattr(pi, "tolist"):
        pi = pi.tolist()
    leaves = list(pi[64:])
    parents = sorted(set(_parent_8x8_of_16x16(i) for i in leaves))
    pc = {}
    for i in leaves:
        p = _parent_8x8_of_16x16(i)
        pc.setdefault(p, []).append(i)
    return {
        "class_id": plan.get("class_id", ""),
        "class_name": plan.get("class_name", ""),
        "leaves_16": sorted(set(leaves)),
        "parents_8": parents,
        "parent_children": {str(k): v for k, v in pc.items()},
    }


def _build_image_to_plan_key(plans: dict, gen_dir: Path) -> dict:
    """Build mapping from image filename (no ext) to plan key."""
    import re
    mapping = {}
    for img_file in sorted(gen_dir.glob("*.png")):
        stem = img_file.stem
        m = re.match(r"class_(\d{3})_.*_plan_(\d+)$", stem)
        if not m:
            continue
        cid, pidx = m.group(1), m.group(2)
        for key in plans:
            if key.startswith(f"class_{cid}_") and key.endswith(f"/plan_{pidx}"):
                mapping[stem] = key
                break
    return mapping


def main():
    plans = {}
    for pkl in sorted(PLANS_DIR.rglob("*.pkl")):
        rel = pkl.relative_to(PLANS_DIR)
        key = str(rel.with_suffix(""))
        plans[key] = process_plan(pkl)

    gen_dir = TREE_DIR / "gen_out" / "generation"
    img_map = _build_image_to_plan_key(plans, gen_dir) if gen_dir.is_dir() else {}

    out = {"plans": plans, "images": img_map}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(plans)} plans, {len(img_map)} images -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
