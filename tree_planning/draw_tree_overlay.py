#!/usr/bin/env python3
"""
给生成图片画 quadtree overlay，自动加载对应 plan。

用法:
  python tree_planning/draw_tree_overlay.py \
    --image tree_planning/gen_out/generation/class_001_goldfish__Carassius_auratus_plan_0.png

  # 批量 (selected.txt):
  python tree_planning/draw_tree_overlay.py \
    --list tree_planning/selected.txt \
    -o tree_planning/overlay_out
"""
import re
import pickle
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

TREE_DIR = Path(__file__).resolve().parent
PLANS_DIR = TREE_DIR / "plans"

OVERLAY_RGB = np.array([255, 140, 0], dtype=np.float32) / 255.0
LOD4_OVERLAY_ALPHA = 0.3#0.22
LOD4_LINE_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
LOD4_LINE_ALPHA = 0.02
LINE_THICKNESS = 2


def resolve_plan(image_path: str) -> str:
    stem = Path(image_path).stem
    m = re.match(r"class_(\d{3})_.*_plan_(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse class_id/plan_idx from: {stem}")
    cid, pidx = m.group(1), m.group(2)
    matches = sorted(PLANS_DIR.glob(f"class_{cid}_*/plan_{pidx}.pkl"))
    if not matches:
        raise FileNotFoundError(f"No plan for class_{cid}_*/plan_{pidx}.pkl")
    return str(matches[0])


def load_plan(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_activated_sets(plan: dict):
    pi = plan["patch_indices"]
    if hasattr(pi, "tolist"):
        pi = pi.tolist()
    lod4_patches = set(pi[64:])
    lod3_cells = set()
    for p in lod4_patches:
        r, c = p // 16, p % 16
        lod3_cells.add((r // 2, c // 2))
    return lod3_cells, lod4_patches


def blend(img, x0, x1, y0, y1, color, alpha):
    """Blend color onto img[x0:x1, y0:y1] in-place. img is (H,W,3) float32."""
    region = img[x0:x1, y0:y1]
    img[x0:x1, y0:y1] = region * (1 - alpha) + color * alpha


def draw_overlay(image_path: str, plan: dict) -> Image.Image:
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    lod3_cell = W // 8
    lod4_patch = W // 16

    lod3_cells, lod4_patches = get_activated_sets(plan)

    # LOD4 patch overlay
    for p in lod4_patches:
        r, c = p // 16, p % 16
        x0, y0 = r * lod4_patch, c * lod4_patch
        blend(img, x0, x0 + lod4_patch, y0, y0 + lod4_patch, OVERLAY_RGB, LOD4_OVERLAY_ALPHA)

    # LOD4 grid lines within activated LOD3 cells
    for cr, cc in lod3_cells:
        x0, y0 = cr * lod3_cell, cc * lod3_cell
        xm, ym = x0 + lod4_patch, y0 + lod4_patch
        inset = LINE_THICKNESS
        blend(img, xm, min(H, xm + LINE_THICKNESS), y0 + inset, y0 + lod3_cell,
              LOD4_LINE_COLOR, LOD4_LINE_ALPHA)
        blend(img, x0 + inset, x0 + lod3_cell, ym, min(W, ym + LINE_THICKNESS),
              LOD4_LINE_COLOR, LOD4_LINE_ALPHA)

    return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Draw quadtree overlay on generated images")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--list", type=str, default=None,
                        help="Text file with image filenames (one per line, relative to gen_out/generation/)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (file for --image, dir for --list)")
    args = parser.parse_args()

    if args.image:
        plan_path = resolve_plan(args.image)
        plan = load_plan(plan_path)
        result = draw_overlay(args.image, plan)
        out = args.output or str(Path(args.image).with_stem(Path(args.image).stem + "_overlay"))
        result.save(out)
        print(f"Saved: {out}")

    elif args.list:
        gen_dir = TREE_DIR / "gen_out" / "generation"
        out_dir = Path(args.output) if args.output else TREE_DIR / "overlay_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = Path(args.list).read_text().strip().splitlines()
        for name in lines:
            name = name.strip()
            if not name:
                continue
            img_path = gen_dir / name
            if not img_path.exists():
                print(f"SKIP (not found): {img_path}")
                continue
            plan_path = resolve_plan(str(img_path))
            plan = load_plan(plan_path)
            result = draw_overlay(str(img_path), plan)
            result.save(out_dir / name)
            print(f"Saved: {out_dir / name}")
    else:
        parser.error("Provide --image or --list")


if __name__ == "__main__":
    main()
