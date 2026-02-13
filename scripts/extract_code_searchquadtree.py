import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from omegaconf import OmegaConf
import numpy as np
import webdataset as wds
import subprocess
import json
from tqdm import tqdm
from accelerate import Accelerator
from utils.logger import setup_logger
from utils.train_utils import create_model
from data.webdataset_reader import QuadtreeImageDataset
from modeling.utils import (
    build_quadtree, 
    get_ordered_nodes, 
    build_probabilistic_quadtree,
    tree_to_decision_nodes_dict,
    _get_nodes_at_level,
    QuadTreeNode,
    _create_and_assign_children
)
from PIL import Image
import copy
import torch.nn.functional as F
import torch.nn as nn
import math
from copy import deepcopy


def complexity_proxy(
    img: torch.Tensor,
    patch_size: int,
    alpha: float = 0.7,
    blur_sigma: float | None = 0.8,
    grad_pool: str = "mean",   # "mean" or "q90"
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Improved Option-A proxy:
      score = alpha * norm(grad_energy) + (1-alpha) * norm(lap_energy)
    Fixes border artifacts (reflect pad) + optional quantile pooling.

    Returns: (B, H//p, W//p) or (H//p, W//p)
    """
    assert img.dim() in (3, 4)
    single = (img.dim() == 3)
    if single:
        img = img.unsqueeze(0)

    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0

    x = img.float()

    # luminance
    if C == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        x = x.mean(dim=1, keepdim=True)

    # optional blur (reflect padded conv)
    if blur_sigma is not None and blur_sigma > 0:
        radius = int(max(1, round(3 * blur_sigma)))
        ksize = 2 * radius + 1
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        g1 = torch.exp(-0.5 * (t / blur_sigma) ** 2)
        g1 = g1 / (g1.sum() + eps)
        g2 = (g1[:, None] * g1[None, :]).view(1, 1, ksize, ksize)
        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        x = F.conv2d(x_pad, g2, padding=0)

    # filters
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    lap = torch.tensor([[ 0.,  1.,  0.],
                        [ 1., -4.,  1.],
                        [ 0.,  1.,  0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

    # reflect pad for derivatives (IMPORTANT)
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(x_pad, sobel_x, padding=0)
    gy = F.conv2d(x_pad, sobel_y, padding=0)
    grad_mag = torch.sqrt(gx * gx + gy * gy + eps)  # (B,1,H,W)

    lap_resp = F.conv2d(x_pad, lap, padding=0)      # (B,1,H,W)

    # patch pooling
    if grad_pool == "mean":
        g_patch = F.avg_pool2d(grad_mag, patch_size, patch_size)  # (B,1,hp,wp)
    elif grad_pool == "q90":
        # quantile pooling per patch: keeps thin edges from being averaged away
        hp, wp = H // patch_size, W // patch_size
        patches = grad_mag.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, 1, hp, wp, patch_size * patch_size)
        g_patch = torch.quantile(patches, 0.9, dim=-1)  # (B,1,hp,wp)
    else:
        raise ValueError("grad_pool must be 'mean' or 'q90'")

    # Laplacian "energy": mean(abs(lap)) is often more stable than variance for this use
    lap_abs = lap_resp.abs()
    lap_patch = F.avg_pool2d(lap_abs, patch_size, patch_size)  # (B,1,hp,wp)

    # normalize components per image (robust-ish)
    def norm_per_image(t):
        # t: (B,1,hp,wp)
        m = t.flatten(2).median(dim=-1).values.view(B, 1, 1, 1)
        mad = (t - m).abs().flatten(2).median(dim=-1).values.view(B, 1, 1, 1)
        return (t - m) / (mad + eps)

    g_n = norm_per_image(g_patch)
    l_n = norm_per_image(lap_patch)

    score = alpha * g_n + (1 - alpha) * l_n
    score = score.squeeze(1)  # (B,hp,wp)

    return score[0] if single else score


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
        if node_mask[i]
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


def process_tree_for_saving(tree):
    """Convert tree dict to serializable format."""
    processed_tree = defaultdict(list)
    for lod_idx, nodes in sorted(tree.items()):
        for node in nodes:
            processed_tree[lod_idx].append({
                'patch_index': node.patch_index,
                'lod_level': node.lod_level
            })
    return dict(processed_tree)


def build_child_to_parent_map(full_tree_root, num_lod):
    """Build a mapping from child nodes to their parent nodes."""
    child_to_parent_map = {}
    queue = [(full_tree_root, None)]
    
    while queue:
        node, parent = queue.pop(0)
        if node.lod_level < num_lod:
            if parent is not None:
                child_key = (node.lod_level, node.patch_index)
                child_to_parent_map[child_key] = parent
            for child in node.children:
                queue.append((child, node))
    
    return child_to_parent_map


def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    config = OmegaConf.load(args.config_dir)

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        split_batches=False,
    )

    logger = setup_logger(
        name="ExtractCodes", 
        log_level="INFO", 
        output_file=f"{output_dir}/log{accelerator.process_index}.txt"
    )
    

    
    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Create and load model
    model, ema_model = create_model(
        config, logger, accelerator, model_type=config.model.type
    )

    logger.info("Loading Tokenizer Weight From: %s", args.tokenizer_weight)
    model_weight = torch.load(args.tokenizer_weight, map_location="cpu")
    model.load_state_dict(model_weight, strict=True)
    
    model.eval()
    model.requires_grad_(False)

    # Create the custom dataloader
    config.dataset.params.num_workers_per_gpu = args.num_workers
    shards_path = f"pipe:rclone cat hoss:jianglihan/data/imagenet/imagenet-train-{args.shards_index:06d}.tar"
    quadtree_dataset = QuadtreeImageDataset(
        shards_path=shards_path,
        resize_shorter_edge=config.dataset.preprocessing.resize_shorter_edge,
        crop_size=config.dataset.preprocessing.crop_size,
        crop_range=args.crop_range,
        random_crop=config.dataset.preprocessing.random_crop,
        random_flip=config.dataset.preprocessing.random_flip,
        num_workers_per_gpu=config.dataset.params.num_workers_per_gpu,
    )
    custom_dataloader = quadtree_dataset.dataloader

    # Prepare everything with accelerator
    logger.info("Preparing model and dataloaders")
    model, custom_dataloader = accelerator.prepare(model, custom_dataloader)

    # Random quadtree parameters
    guaranteed_depth = args.guaranteed_depth
    expansion_probs = args.expansion_probs 
    
    logger.info(f"Starting code extraction with random quadtrees")
    logger.info(f"Guaranteed depth: {guaranteed_depth}, Expansion probs: {expansion_probs}")

    token_num = 0
    num_samples = 0
    
    logger.info("Starting to process samples...")
    all_sample_list = []
    # with wds.TarWriter(process.stdin) as tar_writer:
    for batch_idx, batch in tqdm(enumerate(custom_dataloader), desc="Processing samples"):

        image_key = batch['__key__']
        class_id = batch['class_id'].to(accelerator.device)
        images = batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True) # flatten from ten crop
        
        # for ten crop, flatten the batch
        num_aug = images.shape[1]
        batch_size = images.shape[0]

        images = images.flatten(0, 1)  
        with torch.no_grad():
            image_latent = model.encode(images)

        complexity_scores = complexity_proxy(images, 32)
        patch_mask = (complexity_scores > torch.quantile(complexity_scores.flatten(1,2), 0.5, dim=1)[:, None, None]).flatten(1,2)
        base_tree = build_quadtree(model.num_patch_side_list[:-1])
        target_nodes = _get_nodes_at_level(base_tree, 3)

        all_masked_ordered_nodes = []   
        final_tree_list = []
        for aug_idx in range(num_aug):
            image_node_mask = patch_mask[aug_idx].cpu().numpy()
            base_ordered_nodes = model._get_ordered_nodes(base_tree)
            masked_tree = _build_tree_from_node_mask(
                base_tree, 
                target_lod=3, 
                max_lod=4, 
                patches_per_side_list=model.num_patch_side_list,
                target_nodes=target_nodes,
                node_mask=image_node_mask
            )
            masked_ordered_nodes = model._get_ordered_nodes(masked_tree)
            masked_ordered_nodes = [n for n in masked_ordered_nodes if n.lod_level >= 3]
            all_masked_ordered_nodes.append(deepcopy(masked_ordered_nodes))
            final_tree_list.append(tree_to_decision_nodes_dict(masked_tree, model.num_lod))

        z_batch = model.selector._forward_optimize(image_latent, all_masked_ordered_nodes)
        _, result_dict = model.quantize(z_batch)
        code_incides = result_dict['min_encoding_indices']


        '''# Validate the reconstructed images
        embeds = model.quantize.get_codebook_entry(code_incides.squeeze().long().flatten()).reshape(images.shape[0], -1, 8)
        reconstructed_images = model.decoder._forward_optimize(embeds, all_masked_ordered_nodes)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        for aug_idx in range(num_aug):
            cur_reconstructed_image = reconstructed_images[aug_idx].cpu().numpy()
            cur_reconstructed_image = Image.fromarray((cur_reconstructed_image * 255).astype(np.uint8).transpose(1, 2, 0))
            cur_reconstructed_image.save(f"./reconstructed_image_{aug_idx}.png")'''

        if batch_size > 1:
            class_id_data = class_id[i].item()
        else:
            class_id_data = class_id.item() if hasattr(class_id, 'item') else int(class_id)
            
        for aug_idx in range(num_aug):
            base_filename = image_key[0] + f"_{aug_idx}"
            cur_code_incides = code_incides[aug_idx].cpu().numpy()

            final_tree = final_tree_list[aug_idx]
            lod_indices, patch_incides = [], []

            for lod_idx, nodes in final_tree.items():
                if lod_idx >= 3:
                    for node in nodes:
                        lod_indices.append(node.lod_level)
                        patch_incides.append(node.patch_index)

            sample = {
                "__key__": base_filename,
                "code_indices.npy": cur_code_incides.squeeze(),
                "lod_indices.npy": np.array(lod_indices),
                "patch_indices.npy": np.array(patch_incides),
                "cls": str(class_id_data)
            }
            all_sample_list.append(sample)

            num_samples += 1

    random.shuffle(all_sample_list)
    # Use TarWriter to create tar file
    if args.crop_range == 1.1:
        save_idx = args.shards_index
    elif args.crop_range == 1.05:
        save_idx = args.shards_index + 71
    else:
        raise ValueError(f"Invalid crop range: {args.crop_range}")
    output_tar_path = f"{args.output_tar_path}/imagenet-train-{save_idx:06d}.tar"
    tar_writer = wds.TarWriter(output_tar_path)

    with wds.TarWriter(output_tar_path) as tar_writer:
        for s in tqdm(all_sample_list, desc="Saving to Tar"):
            tar_writer.write(s)
    
    all_sample_list.clear()
    logger.info(f"Avg Token Number: {token_num / num_samples}")
    tar_writer.close()
    logger.info(f"Total samples processed and saved: {num_samples}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True, help="Config Path directory")
    parser.add_argument("--tokenizer_weight", type=str, required=True, help="Tokenizer weight path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--shards_index", type=int, required=True)
    parser.add_argument("--output_tar_path", type=str, required=True, help="Output .tar file path for pretokenized data")
    parser.add_argument("--guaranteed_depth", type=int, default=3, help="Guaranteed depth for random quadtree")
    parser.add_argument("--expansion_probs", type=float, nargs='+', default=[0.3, 0.2], 
                        help="Expansion probabilities for random quadtree")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers per GPU")
    parser.add_argument("--crop_range", type=float, default=1.1, help="Crop range for random crop")
    args = parser.parse_args()
    main(args)

