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

from modeling.utils import (
    build_quadtree, 
    get_ordered_nodes, 
    build_probabilistic_quadtree,
    tree_to_decision_nodes_dict,
    _get_nodes_at_level,
    QuadTreeNode,
    _create_and_assign_children
)

def _build_tree_from_node_mask(base_tree, target_lod, max_lod, patches_per_side_list, target_nodes, node_mask):
    """
    Build a tree by expanding nodes at target_lod where node_mask is True.
    
    Args:
        base_tree: Base tree that stops at target_lod
        target_lod: LOD level of nodes to potentially expand
        max_lod: Maximum LOD to expand to
        patches_per_side_list: List of patches per side for each LOD
        target_nodes: List of nodes at target_lod
        node_mask: Boolean mask of shape [num_target_nodes] indicating which nodes to expand
    
    Returns:
        Modified tree with nodes expanded according to mask
    """
    # Create a set of patch indices to expand for faster lookup
    expand_patch_indices = {
        target_nodes[i].patch_index 
        for i in range(len(target_nodes)) 
        if node_mask[target_nodes[i].patch_index]
    }
    
    def _copy_and_expand(node):
        """Recursively copy tree and expand nodes where mask is True"""
        new_node = QuadTreeNode(node.lod_level, node.patch_index)
        
        # If this is a target node and should be expanded
        if node.lod_level == target_lod and node.patch_index in expand_patch_indices:
            # Expand this node
            if node.lod_level < max_lod:
                _create_and_assign_children(new_node, patches_per_side_list)
        elif node.children:
            # Otherwise, copy children recursively
            for child in node.children:
                new_child = _copy_and_expand(child)
                new_node.children.append(new_child)
        
        return new_node
    
    # Copy the tree recursively starting from root
    return _copy_and_expand(base_tree)

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
    lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(image_tensor.device)
    # Search process start #
    base_tree = build_quadtree(model.num_patch_side_list[:-1])
    target_nodes = _get_nodes_at_level(base_tree, 3)
    target_permutation = coarse_split_permutation()
    target_inverse_permutation = inverse_permutation(target_permutation)
    new_target_nodes = [target_nodes[i] for i in target_inverse_permutation]
    
    random_patch_mask = torch.zeros(64, dtype=torch.bool)
    # fill in 32 True values randomly
    random_patch_mask[random.sample(range(64), 32)] = True

    masked_tree = _build_tree_from_node_mask(
        base_tree, 
        target_lod=3, 
        max_lod=4, 
        patches_per_side_list=model.num_patch_side_list,
        target_nodes=new_target_nodes,
        node_mask=random_patch_mask
    )

    masked_tree_reversed = _build_tree_from_node_mask(
        base_tree, 
        target_lod=3, 
        max_lod=4, 
        patches_per_side_list=model.num_patch_side_list,
        target_nodes=new_target_nodes,
        node_mask=(~random_patch_mask)
    )


    ordered_nodes = model._get_ordered_nodes(masked_tree)
    ordered_nodes = [n for n in ordered_nodes if n.lod_level >= 3]
    
    '''fig = np.zeros((256, 256, 3), dtype=np.uint8)
    for node in ordered_nodes:
        if node.lod_level == 4:
            patch_index = node.patch_index
            patch_x, patch_y = patch_index // 16, patch_index % 16
            patch_x = patch_x * 256 // 16
            patch_y = patch_y * 256 // 16
            patch_x_start = patch_x
            patch_x_end = patch_x + 256 // 16
            patch_y_start = patch_y
            patch_y_end = patch_y + 256 // 16
            fig[patch_x_start:patch_x_end, patch_y_start:patch_y_end, 0] = 255
            fig[patch_x_start:patch_x_end, patch_y_start:patch_y_end, 1] = 0
            fig[patch_x_start:patch_x_end, patch_y_start:patch_y_end, 2] = 0

    fig_torch = torch.from_numpy(fig).permute(2, 0, 1).to(images.device)
    fig_torch = fig_torch / 255.0
    concat_image = torch.cat([images[0], fig_torch], dim=2)
    concat_image = (concat_image * 255).cpu().numpy()
    concat_image = Image.fromarray(concat_image.transpose(1, 2, 0).astype(np.uint8))
    concat_image.save(f"concatenated_image.png")'''

    ordered_nodes_reversed = model._get_ordered_nodes(masked_tree_reversed)
    ordered_nodes_reversed = [n for n in ordered_nodes_reversed if n.lod_level >= 3]


    image_latent = model.encode(image_tensor)
    node_list = [deepcopy(ordered_nodes) for _ in range(image_tensor.shape[0])]
    node_list.extend([deepcopy(ordered_nodes_reversed) for _ in range(image_tensor.shape[0])])
    z_batch = model.selector._forward_optimize(image_latent.repeat(2, 1, 1), node_list)
    _, result_dict = model.quantize(z_batch)
    token_indices = result_dict['min_encoding_indices']
    embeds = model.quantize.get_codebook_entry(token_indices.squeeze().long().flatten()).reshape(image_tensor.shape[0] * 2, -1, 8)
    reconstructed_images = model.decoder._forward_optimize(embeds, node_list)
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)

    # mse_loss = lpips_fn(images,  reconstructed_images[:images.shape[0]], normalize=True).sum(1)
    mse_loss = F.mse_loss(image_tensor,  reconstructed_images[:image_tensor.shape[0]], reduction='none').sum(1) * 10 + lpips_fn(image_tensor,  reconstructed_images[:image_tensor.shape[0]], normalize=True).sum(1) # [bs, 256, 256]
    # mse_loss_reversed = lpips_fn(images,  reconstructed_images[images.shape[0]:], normalize=True).sum(1)
    mse_loss_reversed = F.mse_loss(image_tensor, reconstructed_images[image_tensor.shape[0]:], reduction='none').sum(1) * 10 + lpips_fn(image_tensor,  reconstructed_images[image_tensor.shape[0]:], normalize=True).sum(1)
    # get the patch mean mse loss, patch size is 32
    patch_size = 32
    pooling_layer = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

    patch_bin_list = []
    k = random.randint(32, 64)
    for batch_idx in range(image_tensor.shape[0]):
        # patch_means = (mse_loss[batch_idx].view(images.shape[2] // patch_size, images.shape[3] // patch_size, patch_size, patch_size).mean(dim=(2, 3)))
        # patch_means_reversed = (mse_loss_reversed[batch_idx].view(images.shape[2] // patch_size, images.shape[3] // patch_size, patch_size, patch_size).mean(dim=(2, 3)))
        # patch_means = pooling_layer(mse_loss[batch_idx].unsqueeze(0).unsqueeze(0))[0,0]
        # patch_means_reversed = pooling_layer(mse_loss_reversed[batch_idx].unsqueeze(0).unsqueeze(0))[0,0]
        original_loss_diff = (mse_loss_reversed[batch_idx] - mse_loss[batch_idx])
        patch_loss_diff = pooling_layer(original_loss_diff.unsqueeze(0).unsqueeze(0))[0,0]
        random_patch_mask_int = random_patch_mask.int()
        random_patch_mask_int[~random_patch_mask] = -1
        diff_patch_map = (patch_loss_diff) * random_patch_mask_int.reshape(8, 8).to(patch_loss_diff.device)
        # print((diff_patch_map > 0).sum().item())
        # for diff_patch_map, we define patch_bin that select the largest 75% patches as 1, and the rest as 0
        # Flatten the diff_patch_map and sort the values
        flat_diff = diff_patch_map.flatten()
        # k = 32
        sorted_vals, sorted_idx = torch.sort(flat_diff, descending=True)
        threshold = sorted_vals[k-1]
        # print(f"threshold: {threshold}")
        threshold = 0.05
        # breakpoint()
        patch_bin = (diff_patch_map >= threshold).to(diff_patch_map.dtype)
        patch_bin_list.append(patch_bin)
        # diff_map_256 = patch_bin.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
        # all_diff_patch_maps.append(diff_map_256)
    
    all_search_nodes = []
    for patch_bin_mask in patch_bin_list:
        search_tree = _build_tree_from_node_mask(
            base_tree, 
            target_lod=3, 
            max_lod=4, 
            patches_per_side_list=model.num_patch_side_list,
            target_nodes=new_target_nodes,
            node_mask=patch_bin_mask.flatten().bool()
        )
        search_nodes = model._get_ordered_nodes(search_tree)
        search_nodes = [n for n in search_nodes if n.lod_level >= 3]
        
        # total_token_number += len(search_nodes)
        all_search_nodes.append(search_nodes)
    # Search process end #
    ordered_nodes_full = all_search_nodes[0] # sience we are using batch size=1
    
    # Handle token_count: support both absolute count and percentage
    total_tokens = len(ordered_nodes_full)
    if isinstance(token_count, str):
        if token_count.endswith('%'):
            # Percentage mode: e.g., "50%" -> 0.5
            percentage = float(token_count.rstrip('%')) / 100.0
            target_token_count = max(1, int(total_tokens * percentage))
        else:
            # Try to parse as float to check if it's a decimal percentage
            try:
                token_count_float = float(token_count)
                if 0 < token_count_float < 1:
                    # Percentage as decimal: e.g., "0.5" -> 50%
                    target_token_count = max(1, int(total_tokens * token_count_float))
                else:
                    # Absolute count as string: e.g., "128"
                    target_token_count = int(token_count_float)
            except ValueError:
                # Fallback: try as integer
                target_token_count = int(token_count)
    elif isinstance(token_count, (int, float)):
        if 0 < token_count < 1:
            # Percentage as decimal: e.g., 0.5 -> 50%
            target_token_count = max(1, int(total_tokens * token_count))
        else:
            # Absolute count
            target_token_count = int(token_count)
    else:
        # Fallback: convert to int
        target_token_count = int(token_count)
    
    # Truncate ordered_nodes to target_token_count
    ordered_nodes = ordered_nodes_full[:target_token_count]

    image_latent = model.encode(image_tensor)
    node_list = [deepcopy(ordered_nodes)]
    z_batch = model.selector._forward_optimize(image_latent, node_list)
    _, result_dict = model.quantize(z_batch)
    token_indices = result_dict["min_encoding_indices"]
    embeds = model.quantize.get_codebook_entry(token_indices.squeeze().long().flatten()).reshape(1, -1, 8)
    reconstructed_image = model.decoder._forward_optimize(embeds, node_list)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)[0]
    
    # Calculate PSNR and LPIPS
    # PSNR calculation
    mse = F.mse_loss(reconstructed_image, image_tensor[0])
    if mse.item() > 0:
        psnr = 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10 * torch.log10(mse)
        psnr_value = psnr.item()
    else:
        psnr_value = float('inf')
    
    # LPIPS calculation
    lpips_value = lpips_fn(image_tensor, reconstructed_image.unsqueeze(0), normalize=True).mean().item()

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
        "total_token_count": total_tokens,
        "lod4_expansions": len(activated_lod4_patches),
        "psnr": psnr_value,
        "lpips": lpips_value,
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
            print(f"image_path={image_path} requested_token_count={token_count} "
                  f"actual_token_count={vis_stats['actual_token_count']} "
                  f"total_token_count={vis_stats['total_token_count']} "
                  f"PSNR={vis_stats['psnr']:.4f} LPIPS={vis_stats['lpips']:.4f}")
    print("finished visualization")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="checkpoints/tokenizer_v3_selectormask_wope_expand075_300kiter/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tokenizer_v3_selectormask_wope_expand075_300kiter/pytorch_model.bin")
    parser.add_argument("--output_dir", type=str, default="generated_recon/default/")
    parser.add_argument("--image_path", type=str, nargs="+", required=True, help="One or more image paths, e.g. --image_path a.png b.png")
    parser.add_argument("--overlay_rgb", type=float, nargs=3, default=[255, 140, 0])
    parser.add_argument("--token_count", type=str, nargs="+", default=["128"], 
                        help="One or more token counts (absolute numbers) or percentages (e.g., '50%%' or '0.5'), e.g. --token_count 64 128 '50%%' '0.75'")
    parser.add_argument("--resize_shorter_edge", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=256)

    args = parser.parse_args()
    main(args)

