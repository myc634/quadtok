import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from omegaconf import OmegaConf
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from utils.logger import setup_logger
from tqdm import tqdm
from torch.optim import AdamW
from accelerate.utils import set_seed
from accelerate import Accelerator
from modeling.modules.blocks import Mlp
from modeling.modules import ReconstructionLoss_Reward
from utils.train_utils import create_model, create_dataloader, auto_resume, create_evaluator
from modeling.utils import build_quadtree, get_ordered_nodes, build_tree_from_decision_nodes, build_quadtree, build_random_quadtree, build_probabilistic_quadtree
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def image_generator(config, logger, accelerator):
    config.training.per_gpu_batch_size = 32
    config.dataset.params.eval_shards_path_or_url = "/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet/ILSVRC2012_validation/val-{000000..000049}.tar"
    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    count = 0
    for batch in eval_dataloader:
        for img_tensor, key, class_id in zip(batch['image'], batch['__key__'], batch['class_id']):
            img_tensor = img_tensor.to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True, dtype=torch.float)
            count += 1
            assert tuple(img_tensor.shape) == (3, 256, 256), img_tensor.shape
            # yield img_tensor.unsqueeze(0).to(device), Path(key + ".png")
            yield img_tensor.unsqueeze(0), Path(f"image_{count-1:05d}.png"), key, class_id.unsqueeze(0).to(accelerator.device)

class Namespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

def visualize_lod_binary_map(lod_idx, actions, num_patches_per_side, save_path):
    """
    Visualize a binary map for a specific LOD showing which nodes are activated.
    
    Args:
        lod_idx: LOD level (int)
        actions: Binary tensor (num_patches,) where 1=activated, 0=not activated
        num_patches_per_side: Number of patches per side for this LOD
        save_path: Path to save the visualization
    """
    # Convert to numpy
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(actions.shape[0]))
    if grid_size * grid_size != actions.shape[0]:
        # If not a perfect square, find the appropriate dimensions
        grid_size = num_patches_per_side
    
    binary_map = actions.reshape(grid_size, grid_size)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(binary_map, cmap='gray', interpolation='nearest')
    ax.set_title(f'LOD {lod_idx} Activation Map ({int(binary_map.sum())} nodes activated)')
    ax.set_xlabel(f'Grid Size: {grid_size}x{grid_size}')
    ax.set_ylabel(f'Grid Size: {grid_size}x{grid_size}')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for activated cells
    for i in range(grid_size):
        for j in range(grid_size):
            if binary_map[i, j] > 0.5:
                ax.text(j, i, '1', ha='center', va='center', 
                       color='white', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def visualize_lod_maps_with_image(image, reconstruction, lod_maps_dict, logit_map_dict, save_path, patch_info):
    """
    Visualize original image, reconstruction, LOD binary maps, and token heat map.
    
    Color coding for LOD maps:
    - WHITE (value=1) = Activated node (this patch is selected)
    - BLACK (value=0) = Not activated (this patch is skipped)
    
    Args:
        image: Original image tensor (1, 3, H, W)
        reconstruction: Reconstructed image tensor (1, 3, H, W)
        lod_maps_dict: Dictionary mapping lod_idx to binary map (1D tensor)
        save_path: Path to save the visualization
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image.squeeze(0).permute(1, 2, 0).numpy()
    
    img_np = np.clip(img_np, 0, 1)

    if isinstance(reconstruction, torch.Tensor):
        recon_np = reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()
    else:
        recon_np = reconstruction.squeeze(0).permute(1, 2, 0).numpy()
    
    recon_np = np.clip(recon_np, 0, 1)
    
    # Count how many LODs we have
    num_lods = len(lod_maps_dict)
    
    # Create subplots: 1 for image + 1 for reconstruction + num_lods for LOD maps + 1 for token heat map
    fig, axes = plt.subplots(1, num_lods + 2, figsize=(4 * (num_lods + 3), 4))
    
    # Plot original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstruction Image', fontsize=12)
    axes[1].axis('off')

    # Plot each LOD map
    for idx, ((lod_idx, lod_map), (lod_idx, logit_list)) in enumerate(zip(sorted(lod_maps_dict.items()), sorted(logit_map_dict.items()))):
        axes_idx = idx + 2
        
        patch_size = patch_info['patch_size_list'][lod_idx]
        lod_nodes = patch_info['lod_node_mapping'][lod_idx]
        num_patches_per_side = patch_info['num_patches_per_side'][lod_idx]

        binary_map = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.int64)
        logit_map = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.float32)
        
        # Convert to numpy if tensor
        if isinstance(lod_map, torch.Tensor):
            map_data = lod_map.cpu().numpy()
        else:
            map_data = lod_map
        for binary, logit, node in zip(lod_map, logit_list, lod_nodes):
            row, col = divmod(node.patch_index, num_patches_per_side)
            binary_map[row, col] = binary
            logit_map[row, col] = logit

        # # Reshape to 2D grid (assuming square grid)
        grid_size = int(np.sqrt(map_data.shape[0]))
        # binary_map = map_data.reshape(grid_size, grid_size)
        # breakpoint()
        # Plot binary map
        axes[axes_idx].imshow(binary_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[axes_idx].set_title(f'Binary Map LOD {lod_idx} ({int(binary_map.sum())} nodes)', fontsize=12)
        axes[axes_idx].axis('off')
        axes[axes_idx].grid(True, alpha=0.3)
    
    plt.suptitle('Quadtree Activation Maps & Token Heat Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def compute_patch_entropy_batched(images, patch_sizes=16, bins=512, pad_value=1e6):
    """
    Compute entropy maps for multiple patch sizes in a batch of images using fully vectorized operations.
    
    Args:
        images: torch.Tensor of shape (B, C, H, W) with values in range [0, 255]
        patch_size: base patch size (default: 16)
        num_scales: number of scales to compute (default: 2)
        bins: number of bins for histogram (default: 512)
        pad_value: high entropy value to pad incomplete patches with (default: 1e6)
    
    Returns:
        batch_entropy_maps: dict containing torch.Tensor entropy maps for each patch size
                           with shape (B, H_p, W_p) where H_p and W_p depend on the patch size
    """
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Convert batch of images to grayscale using vectorized operations
    if channels == 3:
        # Apply RGB to grayscale conversion using broadcasting
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * grayscale_weights).sum(dim=1)  # Sum across channel dimension
    else:
        grayscale_images = images[:, 0]  # Take first channel
    
    # Initialize output dictionary
    batch_entropy_maps = {}
    
    # Ensure patch_sizes is a list
    if not isinstance(patch_sizes, (list, tuple)):
        patch_sizes = [patch_sizes]
    
    # Process each patch size
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps  # Round up
        num_patches_w = (W + ps - 1) // ps  # Round up
        
        # Pad images to ensure they fit into patches cleanly
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Unfold the batch of images into patches
        # Shape: (B, num_patches_h, num_patches_w, ps, ps)
        patches = padded_images.unfold(1, ps, ps).unfold(2, ps, ps)
        
        # Reshape to (B, num_patches_h, num_patches_w, ps*ps)
        flat_patches = patches.reshape(batch_size, num_patches_h, num_patches_w, ps*ps)
        
        # Quantize the pixel values to integers in [0, bins-1]
        flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
        
        # Fully vectorized histogram computation using one-hot encoding
        # Reshape to [B * num_patches_h * num_patches_w, ps*ps]
        reshaped_patches = flat_patches_int.reshape(-1, ps*ps)
        
        # Create one-hot encoding for all pixel values
        # This creates a tensor of shape [B * num_patches_h * num_patches_w * ps*ps, bins]
        one_hot = torch.zeros(reshaped_patches.size(0), ps*ps, bins, device=device)
        one_hot = one_hot.scatter_(2, reshaped_patches.unsqueeze(2), 1)
        
        # Sum over the pixels dimension to get histograms
        # Result shape: [B * num_patches_h * num_patches_w, bins]
        histograms = one_hot.sum(1)
        
        # Reshape back to separate batch and spatial dimensions
        # Shape: [B, num_patches_h, num_patches_w, bins]
        histograms = histograms.reshape(batch_size, num_patches_h, num_patches_w, bins)
        
        # Normalize histograms to get probabilities
        probabilities = histograms.float() / (ps * ps)
        
        # Compute entropy: -sum(p * log2(p)), avoiding log(0)
        epsilon = 1e-10
        entropy_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=3)
        
        # Assign a high value to padded regions
        if pad_h > 0:
            entropy_map[:, -1, :] = pad_value  # High entropy at bottom row
        if pad_w > 0:
            entropy_map[:, :, -1] = pad_value  # High entropy at right column
        
        batch_entropy_maps[ps] = entropy_map
    
    return batch_entropy_maps

def select_patches_by_threshold(entropy_maps, thresholds, alpha=1.):
    """
    Vectorized version of patch selection based on entropy thresholds.
    
    Args:
        entropy_maps (dict): Contains patch sizes as keys mapping to
            torch.Tensor entropy maps of shape (B, H_p, W_p) where
            H_p and W_p depend on the patch size
        thresholds (list): List of thresholds for selecting patches at each scale,
            should have length = len(entropy_maps) - 1
        alpha (float): Ratio of the entropy based. 
    Returns:
        masks (dict): Dictionary mapping patch sizes to their 0/1 masks
    """
    patch_sizes = sorted(list(entropy_maps.keys()))
    
    # Special case: if num_scales == 1, just return ones-mask for the first scale
    if len(patch_sizes) == 1:
        masks = {}
        masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
        return masks
    
    if len(thresholds) != len(patch_sizes) - 1:
        raise ValueError(f'Number of thresholds ({len(thresholds)}) must be one less than number of patch sizes ({len(patch_sizes)})')

    masks = {}
    # Initialize mask for smallest patch size
    masks[patch_sizes[0]] = torch.ones_like(entropy_maps[patch_sizes[0]])
    
    # Process each scale from largest to smallest
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        threshold = thresholds[i-1]
        
        # Create mask for current patch size
        masks[current_size] = (entropy_maps[current_size] < threshold).float()
        
    for i in range(len(patch_sizes)-1, 0, -1):
        current_size = patch_sizes[i]
        for j in range(i):
            # Upscale mask to match smaller patch size
            smaller_size = patch_sizes[j]
            scale_factor = current_size // smaller_size 
            mask_upscaled = masks[current_size].repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
            
            # Ensure upscaled mask matches the dimensions of smaller patches
            H_small, W_small = entropy_maps[smaller_size].shape[1:]  # Assuming batch dimension
            mask_upscaled = mask_upscaled[:, :H_small, :W_small]
            
            # Update mask for smaller patches
            masks[smaller_size] = masks[smaller_size] * (1 - mask_upscaled)
    
    return masks

def visualize_patch_split_maps(image, masks, entropy_maps, patch_sizes, save_path):
    """
    Visualize split maps for each patch size using matplotlib.
    For each patch size, shows the original image and the split map side by side.
    
    Color coding:
    - WHITE (value=1) = Will split (high entropy)
    - BLACK (value=0) = Will not split (low entropy or masked by parent)
    
    Args:
        image: Original image tensor (1, 3, H, W) or (3, H, W) with values in [0, 1]
        masks: Dictionary mapping patch_size to binary mask (H_p, W_p) where 1=split, 0=no split
        entropy_maps: Dictionary mapping patch_size to entropy map (B, H_p, W_p) or (H_p, W_p)
        patch_sizes: List of patch sizes (sorted)
        save_path: Path to save the visualization
    """
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = image
    
    img_np = np.clip(img_np, 0, 1)
    
    # Sort patch sizes from largest to smallest for display
    sorted_patch_sizes = sorted(patch_sizes, reverse=True)
    num_levels = len(sorted_patch_sizes)
    
    # Create figure with subplots: for each level, show original + split map
    fig, axes = plt.subplots(num_levels, 2, figsize=(12, 6 * num_levels))
    if num_levels == 1:
        axes = axes.reshape(1, -1)
    
    for level_idx, patch_size in enumerate(sorted_patch_sizes):
        # Get mask for this patch size
        mask = masks[patch_size]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Remove batch dimension if present
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze(0)
        
        # Get entropy map for this patch size (for statistics)
        entropy_map = entropy_maps[patch_size]
        if isinstance(entropy_map, torch.Tensor):
            entropy_np = entropy_map.cpu().numpy()
        else:
            entropy_np = entropy_map
        
        if entropy_np.ndim == 3:
            entropy_np = entropy_np.squeeze(0)
        
        # Upscale mask to match image size for better visualization
        H_mask, W_mask = mask_np.shape
        H_img, W_img = img_np.shape[:2]
        
        # Upscale mask using nearest neighbor interpolation
        mask_upscaled = cv2.resize(mask_np.astype(np.float32), (W_img, H_img), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Calculate statistics
        num_patches = mask_np.size
        num_split = int(mask_np.sum())
        num_no_split = num_patches - num_split
        split_ratio = num_split / num_patches * 100 if num_patches > 0 else 0
        
        # Get entropy statistics for patches that split and don't split
        entropy_split = entropy_np[mask_np > 0.5] if mask_np.sum() > 0 else np.array([])
        entropy_no_split = entropy_np[mask_np <= 0.5] if (mask_np <= 0.5).sum() > 0 else np.array([])
        
        # Left subplot: Original image
        axes[level_idx, 0].imshow(img_np)
        axes[level_idx, 0].set_title(f'Original Image (Patch Size: {patch_size}x{patch_size})', 
                                    fontsize=14, fontweight='bold')
        axes[level_idx, 0].axis('off')
        
        # Right subplot: Split map (white=split, black=no split)
        im = axes[level_idx, 1].imshow(mask_upscaled, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        title = (f'Split Map (Patch Size: {patch_size}x{patch_size})\n'
                f'Split: {num_split}/{num_patches} ({split_ratio:.1f}%)')
        if len(entropy_split) > 0:
            title += f'\nSplit Entropy: {entropy_split.mean():.2f} ± {entropy_split.std():.2f}'
        if len(entropy_no_split) > 0:
            title += f'\nNo-Split Entropy: {entropy_no_split.mean():.2f} ± {entropy_no_split.std():.2f}'
        axes[level_idx, 1].set_title(title, fontsize=12)
        axes[level_idx, 1].axis('off')
        
        # Add grid to show patch boundaries
        patch_h = H_img // H_mask
        patch_w = W_img // W_mask
        for i in range(H_mask + 1):
            axes[level_idx, 1].axhline(y=i * patch_h, color='blue', linewidth=0.5, alpha=0.3)
        for j in range(W_mask + 1):
            axes[level_idx, 1].axvline(x=j * patch_w, color='blue', linewidth=0.5, alpha=0.3)
    
    plt.suptitle('Patch Split Visualization by Level', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def analyze_entropy_statistics(entropy_maps, patch_sizes, thresholds=None):
    """
    Analyze entropy statistics to help determine appropriate thresholds.
    
    Args:
        entropy_maps: Dictionary mapping patch_size to entropy map (B, H_p, W_p)
        patch_sizes: List of patch sizes
        thresholds: Optional list of current thresholds for comparison
        
    Returns:
        stats: Dictionary containing statistics for each patch size
    """
    sorted_patch_sizes = sorted(patch_sizes, reverse=True)
    stats = {}
    
    print("\n" + "="*80)
    print("ENTROPY STATISTICS ANALYSIS")
    print("="*80)
    
    for i, patch_size in enumerate(sorted_patch_sizes):
        entropy_map = entropy_maps[patch_size]
        if isinstance(entropy_map, torch.Tensor):
            entropy_np = entropy_map.cpu().numpy()
        else:
            entropy_np = entropy_map
        
        if entropy_np.ndim == 3:
            entropy_np = entropy_np.squeeze(0)
        
        # Filter out padding values (very high entropy like 1e6)
        valid_entropy = entropy_np[entropy_np < 1000]  # Filter padding
        
        if len(valid_entropy) == 0:
            continue
        
        mean_entropy = valid_entropy.mean()
        std_entropy = valid_entropy.std()
        median_entropy = np.median(valid_entropy)
        min_entropy = valid_entropy.min()
        max_entropy = valid_entropy.max()
        
        # Percentiles
        p25 = np.percentile(valid_entropy, 25)
        p50 = np.percentile(valid_entropy, 50)
        p75 = np.percentile(valid_entropy, 75)
        p90 = np.percentile(valid_entropy, 90)
        p95 = np.percentile(valid_entropy, 95)
        
        stats[patch_size] = {
            'mean': mean_entropy,
            'std': std_entropy,
            'median': median_entropy,
            'min': min_entropy,
            'max': max_entropy,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90,
            'p95': p95,
        }
        
        print(f"\nPatch Size: {patch_size}x{patch_size}")
        print(f"  Mean:    {mean_entropy:.3f} ± {std_entropy:.3f}")
        print(f"  Median:  {median_entropy:.3f}")
        print(f"  Range:   [{min_entropy:.3f}, {max_entropy:.3f}]")
        print(f"  Percentiles:")
        print(f"    P25: {p25:.3f}  P50: {p50:.3f}  P75: {p75:.3f}")
        print(f"    P90: {p90:.3f}  P95: {p95:.3f}")
        
        # Suggest threshold based on percentiles
        # Using median or p75 as a starting point
        suggested_threshold = p75
        print(f"  Suggested threshold (P75): {suggested_threshold:.3f}")
        
        if thresholds and i < len(thresholds):
            current_threshold = thresholds[i]
            split_ratio = (valid_entropy < current_threshold).mean() * 100
            print(f"  Current threshold: {current_threshold:.3f} -> {split_ratio:.1f}% patches would split")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("For a quadtree structure:")
    print("  - Higher threshold = more patches split = more fine-grained detail")
    print("  - Lower threshold = fewer patches split = more coarse representation")
    print("  - Typical thresholds: between P50 (median) and P90")
    print("  - Start with P75 and adjust based on desired split ratio")
    print("="*80 + "\n")
    
    return stats

def visualize_selected_patches_cv2_non_overlapping(
    image_tensor, 
    masks, 
    patch_sizes,
    color=(255, 255, 255),  # BGR in OpenCV, but white is the same in BGR or RGB
    thickness=1
):
    """
    Draw rectangles (using cv2) where masks are 1 for patches of different sizes,
    avoiding overlapping boundaries to prevent thick lines.

    Args:
        image_tensor   (torch.Tensor): Grayscale or RGB image of shape (H, W) or (C, H, W).
        masks          (List[torch.Tensor]): List of 0/1 masks for different patch sizes.
        patch_sizes    (List[int]): List of patch sizes corresponding to each mask.
        color          (tuple): BGR color for rectangle outlines (default white).
        thickness      (int): Thickness of the rectangle outline.

    Returns:
        annotated_image_pil (PIL.Image): The original image with drawn rectangles (in white).
    """

    # 1. Convert the image tensor to a NumPy array for OpenCV
    if image_tensor.ndim == 3:
        # If image_tensor is (C, H, W) with channels first
        if image_tensor.shape[0] in [1, 3]:
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            # Already (H, W, C)
            image_np = image_tensor.cpu().numpy()
    else:
        # (H, W) -> expand dimension for single-channel (H, W, 1)
        image_np = image_tensor.cpu().numpy()
        # Expand to 3 channels to draw colored rectangles
        image_np = np.stack([image_np]*3, axis=-1)

    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    # Get full image dimensions
    img_h, img_w = image_np.shape[:2]
    
    # 2. Create a set to track which edges have been drawn
    # We'll use (y1, x1, y2, x2) tuples to represent line segments
    drawn_edges = set()
    
    # 3. Create a copy of the image to draw on
    annotated_np = image_np.copy()
    
    # Process masks from largest to smallest patch size to handle hierarchy
    for patch_size_idx in range(len(patch_sizes)):
        patch_size = patch_sizes[patch_size_idx]
        mask = masks[patch_size]
        
        H, W = mask.shape
        for i in range(H):
            for j in range(W):
                if mask[i, j] == 1:
                    # Calculate patch coordinates
                    y1 = i * patch_size
                    x1 = j * patch_size
                    y2 = min(y1 + patch_size, img_h)  # Ensure we don't go beyond image bounds
                    x2 = min(x1 + patch_size, img_w)
                    
                    # Draw top edge if not already drawn
                    if (y1, x1, y1, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y1), (x2, y1), color, thickness)
                        drawn_edges.add((y1, x1, y1, x2))
                    
                    # Draw bottom edge if not already drawn
                    if (y2, x1, y2, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y2), (x2, y2), color, thickness)
                        drawn_edges.add((y2, x1, y2, x2))
                    
                    # Draw left edge if not already drawn
                    if (y1, x1, y2, x1) not in drawn_edges:
                        cv2.line(annotated_np, (x1, y1), (x1, y2), color, thickness)
                        drawn_edges.add((y1, x1, y2, x1))
                    
                    # Draw right edge if not already drawn
                    if (y1, x2, y2, x2) not in drawn_edges:
                        cv2.line(annotated_np, (x2, y1), (x2, y2), color, thickness)
                        drawn_edges.add((y1, x2, y2, x2))
    
    # 4. Convert back to PIL image
    annotated_image_pil = Image.fromarray(annotated_np)
    
    return annotated_image_pil

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    config = OmegaConf.load(args.config_dir)

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # output_dir = config.experiment.output_dir + "/eval_outputs"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        split_batches=False,
    )

    logger = setup_logger(name="Generation", log_level="INFO", output_file=f"{output_dir}/log{accelerator.process_index}.txt")
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    model, ema_model = create_model(
        config, logger, accelerator, model_type=config.model.type)

    # build loss
    loss_module = ReconstructionLoss_Reward(config=config)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model, loss_module = accelerator.prepare(model, loss_module)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # Start training.
    logger.info(f""" Start evaluation of the model. """)

    accelerator.print(f"Evaluation of the checkpoint started.")
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    model.eval()

    # mkdir
    Path(output_dir).mkdir(exist_ok=True)

    count = 0
    generator = image_generator(config, logger, accelerator)
    evaluator = create_evaluator(config, logger, accelerator)

    # tree_structure = generate_tree_structure(args.max_tree_depth)
    token_num = 0
    patch_sizes = [32, 16, 8]
    thresholds = [3.5, 6.5]
    
    # First pass: analyze entropy statistics to help determine thresholds
    logger.info("Analyzing entropy statistics for threshold selection...")
    entropy_stats_accumulated = {ps: [] for ps in patch_sizes}
    num_samples_for_stats = min(10, 100)  # Analyze first 10 samples
    
    for image, image_path, image_key, class_id in tqdm(generator):
        if count >= num_samples_for_stats:
            break
        entropy_map = compute_patch_entropy_batched(image * 255, patch_sizes)
        for ps in patch_sizes:
            entropy_np = entropy_map[ps].squeeze(0).cpu().numpy()
            valid_entropy = entropy_np[entropy_np < 1000]  # Filter padding
            entropy_stats_accumulated[ps].extend(valid_entropy.flatten().tolist())
        count += 1
    
    # Compute global statistics
    global_stats = {}
    for ps in patch_sizes:
        if len(entropy_stats_accumulated[ps]) > 0:
            arr = np.array(entropy_stats_accumulated[ps])
            global_stats[ps] = {
                'mean': arr.mean(),
                'std': arr.std(),
                'median': np.median(arr),
                'p25': np.percentile(arr, 25),
                'p50': np.percentile(arr, 50),
                'p75': np.percentile(arr, 75),
                'p90': np.percentile(arr, 90),
            }
    
    logger.info("\nGlobal Entropy Statistics (across multiple samples):")
    for ps in sorted(patch_sizes, reverse=True):
        if ps in global_stats:
            s = global_stats[ps]
            logger.info(f"Patch Size {ps}x{ps}: Mean={s['mean']:.3f}, Median={s['median']:.3f}, P75={s['p75']:.3f}, P90={s['p90']:.3f}")
    
    # Reset generator and process images with visualization
    count = 0
    generator = image_generator(config, logger, accelerator)
    
    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1
        
        # Compute entropy maps
        entropy_map = compute_patch_entropy_batched(image * 255, patch_sizes)
        
        # Analyze statistics for this specific image
        analyze_entropy_statistics(entropy_map, patch_sizes, thresholds=thresholds)
        
        # Select patches based on thresholds
        masks = select_patches_by_threshold(entropy_map, thresholds=thresholds)
        
        # Prepare masks for visualization (remove batch dimension)
        visualization_masks = {}
        for scale, mask_tensor in masks.items():
            visualization_masks[scale] = mask_tensor.squeeze(0)
        
        # Create visualization directory
        vis_dir = Path(output_dir) / "split_visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Save split map visualization
        # vis_path = vis_dir / f"split_map_{count:05d}_{image_key}.png"
        # visualize_patch_split_maps(
        #     image=image,
        #     masks=visualization_masks,
        #     entropy_maps=entropy_map,
        #     patch_sizes=patch_sizes,
        #     save_path=str(vis_path)
        # )
        # logger.info(f"Saved split visualization to {vis_path}")
        
        # Also save the cv2 visualization for comparison
        vis_img = visualize_selected_patches_cv2_non_overlapping(
            image_tensor=image.squeeze(0) * 255,
            masks=visualization_masks,
            patch_sizes=patch_sizes
        )
        vis_img.save(vis_dir / f"overlay_{count:05d}.png")
        
        if count >= 100:  # Process first 10 images
            break
    
    # print(token_num / count if count > 0 else 0)
    # print(evaluator.result())


    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=None, help="Config Path directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_tree_depth", type=int, default=2, help="how many lod are used")


    args = parser.parse_args()
    main(args)