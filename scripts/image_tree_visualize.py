import os
import sys
from pathlib import Path
import numpy as np
import argparse

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# sys.path.append(parent_dir)

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, datasets
from tqdm import tqdm
from omegaconf import OmegaConf
import lpips
import pickle
from copy import deepcopy
from modeling.modules.perceptual_loss import PerceptualLoss

from eval.utils.evaluator import VQGANEvaluator
from data import SimpleImageDataset
import random
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, QuadTreeNode

def overlay_mask(img3, mask_hw, color=(1.0, 0.0, 0.0), alpha0=0.0, alpha1=0.5):
    """
    img3: (3,H,W) float in [0,1]
    mask_hw: (H,W) values 0/1 (or float in [0,1])
    color: RGB color of overlay (in [0,1])
    alpha0/alpha1: opacity for mask==0 and mask==1
    returns: (3,H,W) float in [0,1]
    """
    H, W = mask_hw.shape
    mask = mask_hw.to(img3.device).float().clamp(0, 1)  # (H,W)
    alpha = alpha0 + (alpha1 - alpha0) * mask           # (H,W)
    alpha = alpha.unsqueeze(0)                          # (1,H,W)

    color_t = torch.tensor(color, device=img3.device, dtype=img3.dtype).view(3,1,1)  # (3,1,1)
    out = img3 * (1 - alpha) + color_t * alpha
    return out.clamp(0, 1)

def coarse_split_permutation(coarse_hw=(4,4), split_hw=(2,2)):
    Hc, Wc = coarse_hw
    Hs, Ws = split_hw
    Hf, Wf = Hc*Hs, Wc*Ws
    perm = []
    for cy in range(Hc):
        for cx in range(Wc):
            for sy in range(Hs):
                for sx in range(Ws):
                    fy = cy*Hs + sy
                    fx = cx*Ws + sx
                    perm.append(fy*Wf + fx)
    return perm

def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

@torch.no_grad()
def _normalize_overlay_rgb(overlay_rgb):
    overlay_rgb = [float(v) for v in overlay_rgb]
    if max(overlay_rgb) > 1.0:
        overlay_rgb = [v / 255.0 for v in overlay_rgb]
    return [min(1.0, max(0.0, v)) for v in overlay_rgb]

def _load_single_image_tensor(image_path, args, device):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(args.resize_shorter_edge, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor


@torch.no_grad()
def visualize_single_image(model, image_tensor, image_path, overlay_rgb, token_count, output_dir):
    
    # manually generate a search tree which has exactly `token_count` tokens. Note that by, default we have 64 tokens from LOD3, and each LOD4 expansion adds 4 tokens, so the number of LOD4 expansions needed is (token_count - 64) / 4
    target_token_count = max(64, min(320, token_count))
    tokens_from_lod4 = target_token_count - 64
    assert tokens_from_lod4 % 4 == 0, "token_count must be 64 + 4k for some integer k"
    lod4_expansions_needed = tokens_from_lod4 // 4
    
    # create a search tree with the required number of tokens
    patches_per_side = sorted(list(model.num_patch_side_list))
    assert len(patches_per_side) >= 5, "model must support at least LOD0..LOD4"

    # Build a complete tree up to LOD3 (always 64 tokens at LOD3 for 8x8).
    search_tree = build_quadtree(patches_per_side[:4])
    lod3_nodes = [n for n in model._get_ordered_nodes(search_tree) if n.lod_level == 3]
    lod3_node_map = {int(n.patch_index): n for n in lod3_nodes}

    # Deterministic expansion order over the 8x8 LOD3 grid.
    # This scans 2x2 splits inside each 4x4 coarse block.
    lod3_expand_order = coarse_split_permutation(coarse_hw=(4, 4), split_hw=(2, 2))
    if len(lod3_expand_order) != len(lod3_nodes):
        lod3_expand_order = sorted(lod3_node_map.keys())

    lod3_side = patches_per_side[3]
    lod4_side = patches_per_side[4]
    selected_lod3 = lod3_expand_order[:lod4_expansions_needed]

    for lod3_patch_idx in selected_lod3:
        parent = lod3_node_map[int(lod3_patch_idx)]
        parent_row = int(lod3_patch_idx) // lod3_side
        parent_col = int(lod3_patch_idx) % lod3_side

        child_start_row = parent_row * 2
        child_start_col = parent_col * 2
        top_left_idx = child_start_row * lod4_side + child_start_col
        child_indices = [
            top_left_idx,
            top_left_idx + 1,
            top_left_idx + lod4_side,
            top_left_idx + lod4_side + 1,
        ]

        parent.children = [QuadTreeNode(lod_level=4, patch_index=idx) for idx in child_indices]
    
    ordered_nodes = model._get_ordered_nodes(search_tree)
    ordered_nodes = [n for n in ordered_nodes if n.lod_level >= 3]

    image_latent = model.encode(image_tensor)
    node_list = [deepcopy(ordered_nodes)]
    z_batch = model.selector._forward_optimize(image_latent, node_list)
    _, result_dict = model.quantize(z_batch)
    token_indices = result_dict["min_encoding_indices"]
    embeds = model.quantize.get_codebook_entry(token_indices.squeeze().long().flatten()).reshape(1, -1, 8)
    reconstructed_image = model.decoder._forward_optimize(embeds, node_list)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)[0]

    tree_vis = reconstructed_image.clone()
    H, W = tree_vis.shape[1], tree_vis.shape[2]
    lod3_cell = W // 8
    lod4_patch = W // 16
    line_thickness = 2
    lod3_line_thickness = line_thickness
    lod4_line_thickness = line_thickness

    lod3_line_col = torch.tensor([0.0, 0.0, 0.0], device=tree_vis.device).view(3, 1, 1)
    lod4_line_col = torch.tensor([1.0, 1.0, 1.0], device=tree_vis.device).view(3, 1, 1)
    lod3_line_a = 1.0
    lod4_line_a = 0.35
    lod4_overlay_col = torch.tensor(overlay_rgb, device=tree_vis.device).view(3, 1, 1)
    lod4_overlay_a = 0.22

    def blend_region(x0, x1, y0, y1, col, a):
        tree_vis[:, x0:x1, y0:y1] = tree_vis[:, x0:x1, y0:y1] * (1 - a) + col * a

    def blend_hline(x, y0=0, y1=None, thickness=1, col=None, alpha=None):
        if y1 is None:
            y1 = W
        if col is None:
            col = lod4_line_col
        if alpha is None:
            alpha = lod4_line_a
        if 0 <= x < H:
            blend_region(x, min(H, x + thickness), y0, y1, col, alpha)

    def blend_vline(y, x0=0, x1=None, thickness=1, col=None, alpha=None):
        if x1 is None:
            x1 = H
        if col is None:
            col = lod4_line_col
        if alpha is None:
            alpha = lod4_line_a
        if 0 <= y < W:
            blend_region(x0, x1, y, min(W, y + thickness), col, alpha)

    activated_lod4_patches = set(int(n.patch_index) for n in ordered_nodes if n.lod_level == 4)

    activated_lod3_cells = set()
    for p in activated_lod4_patches:
        px, py = p // 16, p % 16
        activated_lod3_cells.add((px // 2, py // 2))

    for cx, cy in activated_lod3_cells:
        x0, y0 = cx * lod3_cell, cy * lod3_cell
        xm, ym = x0 + lod4_patch, y0 + lod4_patch
        inset = lod3_line_thickness
        y0_in, y1_in = y0 + inset, y0 + lod3_cell
        x0_in, x1_in = x0 + inset, x0 + lod3_cell
        if y0_in < y1_in and x0_in < x1_in:
            blend_hline(xm, y0_in, y1_in, thickness=lod4_line_thickness, col=lod4_line_col, alpha=lod4_line_a)
            blend_vline(ym, x0_in, x1_in, thickness=lod4_line_thickness, col=lod4_line_col, alpha=lod4_line_a)

    for p in activated_lod4_patches:
        px, py = p // 16, p % 16
        x0, y0 = px * lod4_patch, py * lod4_patch
        blend_region(x0, x0 + lod4_patch, y0, y0 + lod4_patch, lod4_overlay_col, lod4_overlay_a)

    # Draw LOD3 boundaries last so they remain visually on top.
    for k in range(1, 8):
        x = k * lod3_cell
        blend_hline(x, 0, W, thickness=lod3_line_thickness, col=lod3_line_col, alpha=lod3_line_a)
        blend_vline(x, 0, H, thickness=lod3_line_thickness, col=lod3_line_col, alpha=lod3_line_a)

    

    image_output_dir = Path(output_dir) / Path(image_path).stem
    os.makedirs(image_output_dir, exist_ok=True)
    token_suffix = f"_t{len(ordered_nodes)}"
    torchvision.utils.save_image(image_tensor[0], os.path.join(image_output_dir, f"gt{token_suffix}.png"))
    torchvision.utils.save_image(reconstructed_image, os.path.join(image_output_dir, f"recon{token_suffix}.png"))
    torchvision.utils.save_image(tree_vis, os.path.join(image_output_dir, f"tree_vis{token_suffix}.png"))
    with open(os.path.join(image_output_dir, f"tree{token_suffix}.pkl"), "wb") as f:
        pickle.dump(
            {
                "lvl_idx": [int(n.lod_level) for n in ordered_nodes],
                "patch_idx": [int(n.patch_index) for n in ordered_nodes],
            },
            f,
        )

    return {
        "actual_token_count": len(ordered_nodes),
        "lod4_expansions": len(activated_lod4_patches),
        "output_dir": str(image_output_dir),
    }

def main(args):
    print("Loading model model...")
    config = OmegaConf.load(args.config)

    # config.model.vq_model.finetune_decoder = True
    # config.model.vq_model.strict_length_assertion = False

    if args.checkpoint is None:
        # search for the latest checkpoint
        # format: Path(config.experiment.output_dir) / "checkpoint-%d/ema_model/pytorch_model.bin"
        checkpoints = list(Path(config.experiment.output_dir).glob("checkpoint-*"))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint = [x / "ema_model" / "pytorch_model.bin" for x in checkpoints][-1]
        print(f"Using the latest checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    elif isinstance(args.checkpoint, str) and args.checkpoint.isdigit():
        # search for the checkpoint with the given number
        checkpoint = Path(config.experiment.output_dir) / f"checkpoint-{args.checkpoint}" / "ema_model" / "pytorch_model.bin"
        print(f"Using the checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()

    model_weight = torch.load(args.checkpoint, map_location="cpu")
    model = QuadTok(config)
    model.load_state_dict(model_weight, strict=True)
    # model_weight = torch.load("checkpoints/quadtok_sl256_vq_ts8-16kcodebook-2lods-causal-selector-wope/save-checkpoint-300000/ema_model/pytorch_model.bin", map_location="cpu")
    # model.load_state_dict(model_weight, strict=True)
    model.eval()
    model.requires_grad_(False)
    device = "cuda"
    model = model.to(device)

    overlay_rgb = _normalize_overlay_rgb(args.overlay_rgb)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_paths = args.image_path if isinstance(args.image_path, list) else [args.image_path]
    token_counts = args.token_count if isinstance(args.token_count, list) else [args.token_count]
    for image_path in image_paths:
        image_tensor = _load_single_image_tensor(image_path, args, device)
        for token_count in token_counts:
            vis_stats = visualize_single_image(
                model=model,
                image_tensor=image_tensor,
                image_path=image_path,
                overlay_rgb=overlay_rgb,
                token_count=token_count,
                output_dir=args.output_dir,
            )
            print(f"image_path={image_path} requested_token_count={token_count}", vis_stats)
    print("finished visualization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="checkpoints/tokenizer_v3_selectormask_wope_expand075_300kiter/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tokenizer_v3_selectormask_wope_expand075_300kiter/pytorch_model.bin")
    parser.add_argument("--output_dir", type=str, default="generated_recon/default/")
    parser.add_argument("--image_path", type=str, nargs="+", required=True, help="One or more image paths, e.g. --image_path a.png b.png")
    parser.add_argument("--overlay_rgb", type=float, nargs=3, default=[255, 140, 0])
    parser.add_argument("--token_count", type=int, nargs="+", default=[128], help="One or more token counts, e.g. --token_count 64 128 192")
    parser.add_argument("--resize_shorter_edge", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)

    args = parser.parse_args()
    main(args)

