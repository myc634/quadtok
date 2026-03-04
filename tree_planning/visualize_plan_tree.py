#!/usr/bin/env python3
"""
可视化 plan 的 quadtree 结构：
  上层：8×8 grid（LOD 3），激活（有 16×16 子节点）的 cell 用红色，其余灰色
  下层：16×16 grid（LOD 4），仅显示 leaf node patch，红色块
  用线连接每个 8×8 parent cell 到其 16×16 children

用法:
  python tree_planning/visualize_plan_tree.py --plan tree_planning/plans/class_001_goldfish/plan_0.pkl -o viz.png
  python tree_planning/visualize_plan_tree.py --plans-dir tree_planning/plans -o tree_planning/plan_viz/ --limit 20
"""
import pickle
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np


PATCH_ORDER_64 = [
    0, 1, 8, 9, 2, 3, 10, 11, 16, 17, 24, 25, 18, 19, 26, 27,
    4, 5, 12, 13, 6, 7, 14, 15, 20, 21, 28, 29, 22, 23, 30, 31,
    32, 33, 40, 41, 34, 35, 42, 43, 48, 49, 56, 57, 50, 51, 58, 59,
    36, 37, 44, 45, 38, 39, 46, 47, 52, 53, 60, 61, 54, 55, 62, 63,
]


def _16x16_idx_to_rc(idx):
    return idx // 16, idx % 16


def _8x8_idx_to_rc(idx):
    return idx // 8, idx % 8


def _parent_8x8_of_16x16(idx_16):
    """16×16 的 raster index -> 对应的 8×8 parent raster index。"""
    r16, c16 = _16x16_idx_to_rc(idx_16)
    return (r16 // 2) * 8 + (c16 // 2)


def visualize_plan(plan: dict, save_path: str | Path | None = None, title: str = ""):
    """
    画两层 quadtree 结构图：
      上层 8×8 grid：parent level
      下层 16×16 grid：leaf level（只画激活的 cell）
      线：从 parent cell 中心连到 child cell 中心
    """
    patch_indices = plan["patch_indices"]
    if hasattr(patch_indices, "tolist"):
        patch_indices = patch_indices.tolist()

    leaves_16 = set(patch_indices[64:])
    if not leaves_16:
        return None

    parents_8 = set(_parent_8x8_of_16x16(idx) for idx in leaves_16)

    # 布局：上层 8×8，下层 16×16，中间留空画连线
    # 坐标系：x 向右，y 向上
    # 上层 y 范围 [gap + 16, gap + 16 + 8]，映射到 8×8 grid
    # 下层 y 范围 [0, 16]，映射到 16×16 grid
    gap = 6  # 两层之间的间距

    fig, ax = plt.subplots(figsize=(8, 14))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 16 + gap + 8 + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    upper_y0 = 16 + gap  # 上层底边 y

    # ---------- Draw 8×8 grid (upper) ----------
    for idx_8 in range(64):
        r, c = _8x8_idx_to_rc(idx_8)
        # 8×8 cell 映射到 x=[2c, 2c+2], y=[upper_y0 + (7-r)*1, upper_y0 + (7-r+1)*1]
        # 每个 8×8 cell 占 2 units 宽（与 16×16 对齐）
        x0 = c * 2
        y0 = upper_y0 + (7 - r) * 1
        w, h = 2, 1
        color = "#e74c3c" if idx_8 in parents_8 else "#444444"
        alpha = 1.0 if idx_8 in parents_8 else 0.25
        rect = mpatches.FancyBboxPatch(
            (x0 + 0.05, y0 + 0.05), w - 0.1, h - 0.1,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#222", linewidth=0.5, alpha=alpha,
        )
        ax.add_patch(rect)

    ax.text(8, upper_y0 + 8.5, "LOD 3  (8×8)", ha="center", va="center", fontsize=10, color="#aaa")

    # ---------- Draw 16×16 grid (lower) ----------
    for idx_16 in range(256):
        r, c = _16x16_idx_to_rc(idx_16)
        x0 = c
        y0 = 15 - r  # y 向上
        if idx_16 in leaves_16:
            color = "#e74c3c"
            alpha = 1.0
        else:
            color = "#444444"
            alpha = 0.15
        rect = mpatches.FancyBboxPatch(
            (x0 + 0.05, y0 + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#333", linewidth=0.3, alpha=alpha,
        )
        ax.add_patch(rect)

    ax.text(8, -0.7, "LOD 4  (16×16 leaves)", ha="center", va="center", fontsize=10, color="#aaa")

    # ---------- Draw connecting lines ----------
    drawn_parents = set()
    for parent_8 in parents_8:
        pr, pc = _8x8_idx_to_rc(parent_8)
        # parent center in upper layer
        px = pc * 2 + 1
        py = upper_y0 + (7 - pr) * 1 + 0.5

        children = [
            idx for idx in leaves_16
            if _parent_8x8_of_16x16(idx) == parent_8
        ]
        for child_16 in children:
            cr, cc = _16x16_idx_to_rc(child_16)
            cx = cc + 0.5
            cy = 15 - cr + 0.5
            ax.plot(
                [px, cx], [py, cy],
                color="#e74c3c", alpha=0.35, linewidth=0.8,
                zorder=0,
            )

    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Visualize plan quadtree structure")
    parser.add_argument("--plan", type=str, default=None, help="Single plan pkl path")
    parser.add_argument("--plans-dir", type=str, default=None, help="Batch: root dir of plans")
    parser.add_argument("-o", "--output", type=str, default="plan_tree_viz.png", help="Output path (file or dir)")
    parser.add_argument("--limit", type=int, default=0, help="Batch: max plans to visualize (0=all)")
    args = parser.parse_args()

    if args.plan:
        with open(args.plan, "rb") as f:
            plan = pickle.load(f)
        class_name = plan.get("class_name", "")
        class_id = plan.get("class_id", "?")
        title = f"class {class_id}: {class_name}" if class_name else f"class {class_id}"
        out = visualize_plan(plan, save_path=args.output, title=title)
        if out:
            print(f"Saved: {out}")
        else:
            print("No 16×16 leaves in this plan, nothing to draw.")
        return

    if args.plans_dir:
        plans_dir = Path(args.plans_dir)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        pkls = sorted(plans_dir.glob("**/plan_*.pkl"))
        if args.limit > 0:
            pkls = pkls[: args.limit]
        count = 0
        for pkl_path in pkls:
            with open(pkl_path, "rb") as f:
                plan = pickle.load(f)
            if len(plan["patch_indices"]) <= 64:
                continue
            class_name = plan.get("class_name", "")
            class_id = plan.get("class_id", "?")
            safe_name = pkl_path.parent.name + "_" + pkl_path.stem
            title = f"class {class_id}: {class_name}" if class_name else f"class {class_id}"
            out = visualize_plan(plan, save_path=output_dir / f"{safe_name}.png", title=title)
            if out:
                count += 1
        print(f"Saved {count} visualizations to {output_dir}")
        return

    parser.error("One of --plan or --plans-dir is required")


if __name__ == "__main__":
    main()
