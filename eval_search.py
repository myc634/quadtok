import os
import sys
from pathlib import Path
import numpy as np
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, datasets
from tqdm import tqdm
from omegaconf import OmegaConf

from modeling.modules.perceptual_loss import PerceptualLoss

from evaluator import VQGANEvaluator
from data import SimpleImageDataset
import random
from modeling.quadtok import QuadTok
from modeling.utils import (
    build_quadtree, 
    get_ordered_nodes, 
    build_probabilistic_quadtree,
    tree_to_decision_nodes_dict,
    _get_nodes_at_level,
    QuadTreeNode,
    _create_and_assign_children
)

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

    
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
font_size=12
try:
    font = ImageFont.truetype(font_path, font_size)
except:
    font = ImageFont.load_default()

from copy import deepcopy

def complexity_proxy(
    img: torch.Tensor,
    patch_size: int,
    alpha: float = 0.7,
    blur_sigma: float | None = 0.8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Option-A proxy = alpha * mean(|Sobel|) + (1-alpha) * Var(Laplacian),
    computed per non-overlapping patch.

    Args:
        img: torch.Tensor, shape (C,H,W) or (B,C,H,W), float. (range doesn't matter)
        patch_size: int, non-overlapping patch size in pixels
        alpha: mixing weight (default 0.7)
        blur_sigma: if not None, applies a small Gaussian blur before computing edges/textures
        eps: numerical stability

    Returns:
        score: torch.Tensor, shape (B, H//patch_size, W//patch_size) (or (H//p, W//p) if input was (C,H,W))
    """
    assert img.dim() in (3, 4), f"Expected (C,H,W) or (B,C,H,W), got {tuple(img.shape)}"
    single = (img.dim() == 3)
    if single:
        img = img.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"H,W must be divisible by patch_size. Got H={H}, W={W}, patch_size={patch_size}"

    x = img.float()

    # --- luminance ---
    if C == 3:
        # Rec.601-ish; good enough for proxy
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    elif C == 1:
        x = x
    else:
        # If more channels, just average them
        x = x.mean(dim=1, keepdim=True)

    # --- optional blur to reduce ringing/noise over-splitting ---
    if blur_sigma is not None and blur_sigma > 0:
        # Simple separable-ish Gaussian via 2D kernel
        # Kernel radius ~ 3*sigma
        radius = int(max(1, round(3 * blur_sigma)))
        ksize = 2 * radius + 1
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        gauss_1d = torch.exp(-0.5 * (t / blur_sigma) ** 2)
        gauss_1d = gauss_1d / (gauss_1d.sum() + eps)
        gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, ksize, ksize)
        x = F.conv2d(x, gauss_2d, padding=radius)

    # --- Sobel & Laplacian filters ---
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    lap = torch.tensor([[ 0.,  1.,  0.],
                        [ 1., -4.,  1.],
                        [ 0.,  1.,  0.]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + eps)  # (B,1,H,W)

    lap_resp = F.conv2d(x, lap, padding=1)  # (B,1,H,W)

    # --- per-patch statistics using pooling over non-overlapping patches ---
    # mean(|Sobel|)
    g_patch = F.avg_pool2d(grad_mag, kernel_size=patch_size, stride=patch_size)  # (B,1,hp,wp)

    # Var(Laplacian) = E[x^2] - (E[x])^2
    lap_mean = F.avg_pool2d(lap_resp, kernel_size=patch_size, stride=patch_size)
    lap_mean2 = F.avg_pool2d(lap_resp * lap_resp, kernel_size=patch_size, stride=patch_size)
    lap_var_patch = torch.clamp(lap_mean2 - lap_mean * lap_mean, min=0.0)

    # Combine
    score = alpha * g_patch + (1.0 - alpha) * lap_var_patch  # (B,1,hp,wp)
    score = score.squeeze(1)  # (B,hp,wp)

    return score[0] if single else score

@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    device,
    evaluator,
    pretrained_tokenizer=None,
    length=None,
    perceptual_loss_func=None,
):
    model.eval()
    evaluator.reset_metrics()
    local_model = model
    sum_token_number = 0
    iter_count = 0

    perceptual_loss_list = []

    # unique_id_sets = {}
    id2freq = {}

    guaranteed_depth = 3
    search_expansion_probs = [0.5]
    expansion_probs = [0.55]
    total_token_number = 0
    total_sample_number = 0
    for batch in tqdm(eval_loader):
        images = batch["image"].to(device, memory_format=torch.contiguous_format, non_blocking=True)
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
        
        fig = np.zeros((256, 256, 3), dtype=np.uint8)
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
        concat_image.save(f"concatenated_image.png")
        ordered_nodes_reversed = model._get_ordered_nodes(masked_tree_reversed)
        ordered_nodes_reversed = [n for n in ordered_nodes_reversed if n.lod_level >= 3]


        image_latent = model.encode(images)
        z_batch = model.selector._forward_reconstruction(image_latent, ordered_nodes)
        z_batch_reversed = model.selector._forward_reconstruction(image_latent, ordered_nodes_reversed)
        _, result_dict = model.quantize(z_batch)
        _ , result_dict_reversed = model.quantize(z_batch_reversed)
        token_indices = result_dict['min_encoding_indices']
        token_indices_reversed = result_dict_reversed['min_encoding_indices']

        embeds = model.quantize.get_codebook_entry(token_indices.squeeze().long().flatten()).reshape(images.shape[0], -1, 8)
        embeds_reversed = model.quantize.get_codebook_entry(token_indices_reversed.squeeze().long().flatten()).reshape(images.shape[0], -1, 8)
        reconstructed_images = model.decoder._forward_reconstruction(embeds, ordered_nodes)
        reconstructed_images_reversed = model.decoder._forward_reconstruction(embeds_reversed, ordered_nodes_reversed)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        reconstructed_images_reversed = torch.clamp(reconstructed_images_reversed, 0.0, 1.0)
        mse_loss = F.mse_loss(images, reconstructed_images, reduction='none').sum(1)
        mse_loss_reversed = F.mse_loss(images, reconstructed_images_reversed, reduction='none').sum(1)
        # get the patch mean mse loss, patch size is 32
        patch_size = 32
        pooling_layer = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)

        all_diff_patch_maps = []
        for batch_idx in range(images.shape[0]):
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
            k = random.randint(32, 64)
            sorted_vals, sorted_idx = torch.sort(flat_diff, descending=True)
            threshold = sorted_vals[k-1]
            patch_bin = (diff_patch_map >= threshold).to(diff_patch_map.dtype)

            diff_map_256 = patch_bin.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
            all_diff_patch_maps.append(diff_map_256)
        
        # Expand the diff_patch_map to the original image size 

        # save the concated images with corresponding ground truth images
        for i in range(images.shape[0]):
            diff_map = torch.abs(images[i] - reconstructed_images[i]).sum(0).repeat(3, 1, 1) / 4
            diff_map_reversed = torch.abs(images[i] - reconstructed_images_reversed[i]).sum(0).repeat(3, 1, 1) / 4
            diff_map_cor = torch.abs(reconstructed_images[i] - reconstructed_images_reversed[i]).sum(0).repeat(3, 1, 1) / 4
            overlay_image = overlay_mask(images[i], all_diff_patch_maps[i])
            concatenated_image = torch.cat([overlay_image, reconstructed_images[i], reconstructed_images_reversed[i], diff_map, diff_map_reversed, diff_map_cor], dim=2)
            concatenated_image = (concatenated_image * 255).cpu().numpy()
            concatenated_image = Image.fromarray(concatenated_image.transpose(1, 2, 0).astype(np.uint8))
            concatenated_image.save(f"./concatenated_image_{i}.png")
        breakpoint()
        perceptual_loss = perceptual_loss_func(images, reconstructed_images)
        perceptual_loss_list.append(perceptual_loss.mean().item())

        total_token_number += token_indices.flatten().shape[0]
        total_sample_number += images.shape[0]
        
        evaluator.update(images, reconstructed_images.squeeze(2), token_indices.squeeze().long())
        print(f"Average token number: {total_token_number / total_sample_number}")

    model.train()
    return evaluator.result()

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
    model.eval()
    model.requires_grad_(False)
    device = "cuda"
    model = model.to(device)

    perceptual_loss_func = PerceptualLoss(config.losses.perceptual_loss).eval().to(device)


    config.training.per_gpu_batch_size = 4
    base_params = dict(
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=config.training.per_gpu_batch_size,
        num_workers_per_gpu=config.dataset.params.num_workers_per_gpu,
        resize_shorter_edge=config.dataset.preprocessing.resize_shorter_edge,
        crop_size=config.dataset.preprocessing.crop_size,
        random_crop=config.dataset.preprocessing.random_crop,
        random_flip=config.dataset.preprocessing.random_flip,
    )
    dataset = SimpleImageDataset(
        train_shards_path="/mnt/ultracube/datasets/imagenet-wds/imagenet1k-train-{000000..000071}.tar",
        eval_shards_path="/mnt/ultracube/datasets/imagenet-1k-wds-val/imagenet1k-validation-{00..63}.tar",
        **base_params,
    )
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    evaluator = VQGANEvaluator(
        device=device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=True,
        enable_codebook_entropy_measure=True,
        num_codebook_entries=config.model.vq_model.codebook_size,
        enable_psnr=True,
        enable_ssim=True,
    )

    eval_scores = eval_reconstruction(
        model, 
        eval_dataloader, 
        device, 
        evaluator, 
        pretrained_tokenizer=None,
        length=None,
        perceptual_loss_func=perceptual_loss_func,
    )

    print(eval_scores)
    print("finished evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="./tokenizer_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="./tokenizer_v3_search_new.bin")
    parser.add_argument("--output_dir", type=str, default="generated_recon/default/")

    args = parser.parse_args()
    main(args)


