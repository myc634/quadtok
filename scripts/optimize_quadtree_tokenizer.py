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

def image_generator(config, logger, accelerator):
    config.training.per_gpu_batch_size = 1
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
        # breakpoint()
        # axes[axes_idx - 1].imshow(logit_map, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
        # axes[axes_idx - 1].set_title(f'Logit Map LOD {lod_idx} ({int(binary_map.sum())} nodes)', fontsize=12)
        # axes[axes_idx - 1].axis('off')
        # axes[axes_idx - 1].grid(True, alpha=0.3)
        # plt.colorbar(im, ax=axes[axes_idx - 1], fraction=0.046, pad=0.04)
        
        # Add text annotations for activated cells
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         if binary_map[i, j] > 0.5:
        #             axes[axes_idx].text(j, i, '1', ha='center', va='center', 
        #                color='white', fontsize=6, fontweight='bold')
    
    # Calculate and plot token heat map (cumulative activation count per position)
    # token_heat_map = calculate_token_heat_map(lod_maps_dict)
    # if token_heat_map is not None:
    #     heat_map_idx = num_lods + 2
    #     im = axes[heat_map_idx].imshow(token_heat_map, cmap='coolwarm', interpolation='nearest')
    #     axes[heat_map_idx].set_title(f'Token Heat Map (max: {int(token_heat_map.max())})', fontsize=12)
    #     axes[heat_map_idx].axis('off')
        
    #     # Add colorbar
    #     plt.colorbar(im, ax=axes[heat_map_idx], fraction=0.046, pad=0.04)
    
    plt.suptitle('Quadtree Activation Maps & Token Heat Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def calculate_token_heat_map(lod_maps_dict):
    """
    Calculate cumulative token heat map showing how many tokens are activated at each position.
    
    Args:
        lod_maps_dict: Dictionary mapping lod_idx to binary map (1D tensor)
        
    Returns:
        heat_map: 2D numpy array showing cumulative token count per position
    """
    if not lod_maps_dict:
        return None
    
    # Find the finest resolution (highest LOD)
    max_lod = max(lod_maps_dict.keys())
    finest_map = lod_maps_dict[max_lod]
    
    # Convert to numpy if tensor
    if isinstance(finest_map, torch.Tensor):
        finest_data = finest_map.cpu().numpy()
    else:
        finest_data = finest_map
    
    # Get grid size for finest LOD
    grid_size = int(np.sqrt(finest_data.shape[0]))
    heat_map = np.zeros((grid_size, grid_size))
    
    # For each LOD, accumulate tokens to the finest resolution
    for lod_idx, lod_map in sorted(lod_maps_dict.items()):
        if isinstance(lod_map, torch.Tensor):
            map_data = lod_map.cpu().numpy()
        else:
            map_data = lod_map
        
        # Calculate how many tokens this LOD contributes to each finest position
        lod_grid_size = int(np.sqrt(map_data.shape[0]))
        lod_binary_map = map_data.reshape(lod_grid_size, lod_grid_size)
        
        # Calculate scaling factor (how many finest positions each LOD position covers)
        scale_factor = grid_size // lod_grid_size
        
        # Accumulate tokens to finest resolution
        for i in range(lod_grid_size):
            for j in range(lod_grid_size):
                if lod_binary_map[i, j] > 0.5:  # If this LOD position is activated
                    # Add tokens to corresponding finest positions
                    start_i = i * scale_factor
                    end_i = min((i + 1) * scale_factor, grid_size)
                    start_j = j * scale_factor
                    end_j = min((j + 1) * scale_factor, grid_size)
                    
                    heat_map[start_i:end_i, start_j:end_j] += 1
    
    return heat_map

def gumbel_sigmoid(logits, tau=1.0, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Sigmoid distribution and optionally discretize.
    Adds noise to logits, applies sigmoid, and uses STE during backward pass if hard=True.
    Returns tensor_like logits, shaped like logits.sigmoid().
    """
    # Sample Gumbel(0, 1) noise (numerically stable)
    gumbels = -torch.empty_like(logits).exponential_().log() # ~ Gumbel(0,1)
    gumbels = (logits + gumbels) / tau # ~ Gumbel(logits, tau)
    y_soft = torch.sigmoid(gumbels)

    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        # STE trick: Set gradients w.r.t. y_hard to gradients w.r.t. y_soft
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard
    else:
        return y_soft

def build_policy_result_from_actions(actions_hard_flat, model_selector, device):
    """
    Builds the policy_result dictionary from a flat tensor of hard actions.
    Input: actions_hard_flat (B, N_total-1) or (B, N_total) - Binary tensor (0 or 1)
           model_selector - Your QuadTokSelctor instance (to access tree info)
    Output: policy_result dictionary ready for model_selector._forward_policy
    """
    # Assuming actions_hard_flat corresponds to nodes 1 to N_total-1
    # We need to prepend the root node action (always 1)
    batch_size = actions_hard_flat.shape[0]
    
    # Create the full actions tensor (B, N_total) including root
    actions_full_mask = torch.cat(
        [torch.ones(batch_size, 1, device=device, dtype=torch.bool), # Root is always kept
         actions_hard_flat.bool()], # Assuming input corresponds to nodes 1..N-1
        dim=1
    )
    # If actions_hard_flat already includes root (size N_total), adjust accordingly
    # actions_full_mask = actions_hard_flat.bool()
    # actions_full_mask[:, 0] = True # Ensure root is always true

    num_total_nodes = model_selector.num_total_nodes
    parent_indices = model_selector.parent_indices # Buffer on model device
    full_tree_lods = model_selector.full_tree_lods
    full_tree_indices = model_selector.full_tree_indices

    # Calculate keep_mask based on parent relationship
    keep_mask = torch.zeros_like(actions_full_mask, device=device)
    keep_mask[:, 0] = True # Root

    for i in range(1, num_total_nodes):
        parent_idx = parent_indices[i]
        # Ensure parent_idx is valid before indexing keep_mask
        # Note: root parent (-1) handled by clamp/logic elsewhere or init
        valid_parent_idx = torch.clamp(parent_idx, min=0) 
        parent_is_kept = keep_mask[:, valid_parent_idx] 
        keep_mask[:, i] = parent_is_kept & actions_full_mask[:, i]

    pruned_lods_list = []
    pruned_indices_list = []
    
    for b in range(batch_size):
        sample_mask = keep_mask[b] # (num_total_nodes,)
        pruned_lods_list.append(full_tree_lods[sample_mask])
        pruned_indices_list.append(full_tree_indices[sample_mask])
    
    padded_pruned_lods = pad_sequence(pruned_lods_list, batch_first=True, padding_value=-1)
    padded_pruned_indices = pad_sequence(pruned_indices_list, batch_first=True, padding_value=-1)
    num_tokens = torch.sum(keep_mask, dim=1)

    # Make sure sequence length matches padding if necessary
    max_len = padded_pruned_lods.shape[1]

    return {
        "pruned_lods_padded": padded_pruned_lods,
        "pruned_indices_padded": padded_pruned_indices,
        "num_tokens": num_tokens,
    }

# --- NEW SEQUENTIAL OPTIMIZATION FUNCTION ---

def optimize_tree_for_image(
    image, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None
):
    """
    Optimizes the quadtree structure for a single image using a sequential,
    causal gradient descent process. (FIXED VERSION 4)
    
    Args:
        image_id: Optional identifier for saving visualizations
        save_dir: Directory to save visualizations
    """
    logger.info(f"Optimizing tree for one image (Sequential Causal)...")
    model.eval()
    loss_module.eval()

    device = accelerator.device
    guaranteed_depth = config.model.guaranteed_depth
    
    # Create save directory for visualizations
    if save_dir and image_id is not None:
        viz_dir = os.path.join(save_dir, f"image_{image_id:05d}_viz")
        os.makedirs(viz_dir, exist_ok=True)

    # 1. Build tree structure and mappings
    full_tree_root = build_quadtree(model.num_patch_side_list)
    ordered_full_nodes = get_ordered_nodes(full_tree_root, model.num_lod)
    num_total_nodes = len(ordered_full_nodes)
    lod_len_mapping, lod_node_mapping = defaultdict(int), defaultdict(list)
    for node in ordered_full_nodes:
        lod_len_mapping[node.lod_level] += 1
        lod_node_mapping[node.lod_level].append(node)

    node_to_idx_map = {
        (node.lod_level, node.patch_index): i 
        for i, node in enumerate(ordered_full_nodes)
    }
    
    lod_start_indices = {}
    total_nodes_count = 0
    for lod_idx in range(model.num_lod):
        lod_start_indices[lod_idx] = total_nodes_count
        total_nodes_count += lod_len_mapping[lod_idx] 

    # --- Pre-calculate mappings for ALL nodes ---
    # We need fast lookup for parent_bfs -> child_patch_indices
    # and child_patch_index -> parent_bfs_index
    parent_bfs_to_child_patch_map = defaultdict(list)
    
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            # We only care about children that exist in the tree
            if (child_node.lod_level, child_node.patch_index) in node_to_idx_map:
                parent_bfs_to_child_patch_map[parent_idx].append(child_node.patch_index)


    # 2. Init theta_I per-image
    theta_I_dict = nn.ParameterDict()
    lod_patch_idx_to_theta_idx_maps = {} 
    
    for lod_idx in range(model.num_lod - 1):
        if lod_idx >= guaranteed_depth:
            theta_I_dict[str(lod_idx)] = nn.Parameter(
                torch.randn((lod_len_mapping[lod_idx], 2048), device=device) * (1024**(-0.5))
            )
            
            lod_nodes = ordered_full_nodes[lod_start_indices[lod_idx] : lod_start_indices[lod_idx] + lod_len_mapping[lod_idx]]
            lod_patch_idx_to_theta_idx_maps[lod_idx] = {
                node.patch_index: i for i, node in enumerate(lod_nodes)
            }

    # 3. Optimization settings
    optim_lr = config.optimization.get("per_image_lr", 1e-3) 
    num_optim_steps = config.optimization.get("per_image_steps", 100)
    gumbel_tau = config.optimization.get("gumbel_tau", 1.0) 

    # 4. Get image latent (once)
    with torch.no_grad():
        image_latent = model.encode(image)

    # 5. Sequential optimization loop
    all_best_actions_hard = {}
    all_best_logit = {}
    lod_actions_vis_dict = {}
    
    current_decision_nodes = defaultdict(list)
    for node in ordered_full_nodes:
        lod_idx, index = node.lod_level, node.patch_index
        if lod_idx <= guaranteed_depth:
            current_decision_nodes[lod_idx].append(index)
            
    # NOTE: all_node_mapping is NO LONGER USED, as it's for the "full" tree
    # We will pass the sparse list of active child patch indices instead.

    # Iterate sequentially through optimizable LODs
    for lod_idx_str in theta_I_dict.keys():
        lod_idx = int(lod_idx_str)

        child_lod_idx = lod_idx + 1
        
        # logger.info(f"--- Optimizing LOD {lod_idx} ---")

        # 5a. Re-init linear layer and optimizer per-LOD
        linear = Mlp(2048, hidden_features=4096, out_features=1).to(device)
        
        lod_theta_I = theta_I_dict[lod_idx_str]
        optimizer = AdamW([
            {'params': lod_theta_I},
            {'params': linear.parameters()}
        ], lr=optim_lr)

        # 5b. Determine causally-active nodes for this LOD
        # (e.g., at LOD 4, this is the list of 140 patch indices)
        active_patch_indices_at_lod = current_decision_nodes.get(lod_idx, [])
        if not active_patch_indices_at_lod:
            logger.info(f"Skipping LOD {lod_idx}: No active parent nodes.")
            all_best_actions_hard[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
            continue
            
        current_lod_map = lod_patch_idx_to_theta_idx_maps[lod_idx]
        
        # Get the indices *within theta_I* that are active
        # (e.g., at LOD 4, a list of 140 indices into theta_I_dict['4'])
        active_theta_indices = [
            current_lod_map[patch_idx] for patch_idx in active_patch_indices_at_lod 
            if patch_idx in current_lod_map
        ]
        
        if not active_theta_indices:
            logger.info(f"Skipping LOD {lod_idx}: No matching active nodes.")
            all_best_actions_hard[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
            continue

        # (e.g., shape [140])
        active_theta_indices_tensor = torch.tensor(active_theta_indices, device=device, dtype=torch.long)
        
        # --- 5c. Find children of *only* these active nodes ---
        
        # Get the global BFS indices for the active parents
        # (e.g., [70, 72, 73, ...], len=140)
        active_parent_global_bfs_indices = [
            node_to_idx_map[(lod_idx, patch_idx)] 
            for patch_idx in active_patch_indices_at_lod
        ]
        
        active_child_patch_indices = []
        child_local_parent_indices = [] # index relative to active_theta_indices_tensor

        for i, parent_global_idx in enumerate(active_parent_global_bfs_indices):
            children_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            active_child_patch_indices.extend(children_patches)
            child_local_parent_indices.extend([i] * len(children_patches))

        # (e.g., shape [560])
        child_local_parent_indices_tensor = torch.tensor(child_local_parent_indices, device=device, dtype=torch.long)
        
        best_loss = float('inf')
        best_actions_hard_for_this_lod_sparse = None # Will be size 140
        best_logit_hard_for_this_lod_sparse = None

        for step in range(num_optim_steps):
            optimizer.zero_grad()

            # --- REVISED (5d): Fully Sparse ---
            
            # 1. Get probs ONLY for active nodes (e.g., shape 140)
            active_theta_subset = lod_theta_I.index_select(0, active_theta_indices_tensor)
            active_probs_subset = linear(active_theta_subset).sigmoid().squeeze(-1)

            # 2. STE ONLY for active nodes (e.g., shape 140)
            active_actions_hard = (active_probs_subset > 0.5).float()
            active_effective_parent = active_probs_subset + (active_actions_hard - active_probs_subset).detach()

            # 3. Gather ONLY for children of active nodes (e.g., shape 560)
            actions_effective = torch.gather(active_effective_parent, 0, child_local_parent_indices_tensor) 

            # 4. Model forward and loss
            # current_decision_nodes has the sparse tree up to LOD 4 (140 nodes)
            # actions_effective has the sparse activations for LOD 5 (560 actions)
            z = model.selector._forward_optimize(
                image_latent, 
                current_decision_nodes, 
                actions_effective 
            )
            z_quantized = model.quantize(z).sample()

            # decoder receives:
            # z: features for (base + 140 + 560) nodes
            # current_decision_nodes: sparse tree (up to 140 nodes at LOD 4)
            # [actions_effective, active_probs_subset]: sparse tensors [560], [140]
            # active_child_patch_indices: list of 560 patch indices for LOD 5

            reconstruction = model.decoder._forward_optimize(
                z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), 
                current_decision_nodes,                 
                [actions_effective, active_probs_subset], 
                active_child_patch_indices # This is the "idx list"
            )
            
            loss, loss_dict = loss_module(image, reconstruction)
            if loss.numel() == 1:
                loss = loss.mean()
            
            # Sparsity penalty only on *active* nodes
            sparsity_loss = active_probs_subset.mean() * 0.1 + (active_probs_subset * (1 - active_probs_subset)).mean()
            # sparsity_loss = (active_probs_subset * (1 - active_probs_subset)).mean()# + active_probs_subset.mean() * 0.1
            total_loss = loss + sparsity_loss
            
            # if step % 20 == 0:
            #     logger.info(f"[LOD {lod_idx} Step {step}] Prob Mean (Active): {active_probs_subset.mean().item():.4f} Total Loss: {total_loss.item():.4f}, Recon Loss: {loss_dict['reconstruction_loss']:.4f}, Perceptual Loss: {loss_dict['perceptual_loss']:.4f}")

            # 5e. Backward
            total_loss.backward()
            optimizer.step()

            current_loss_val = total_loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                # Store the *sparse* best actions (e.g., shape 140)
                best_actions_hard_for_this_lod_sparse = active_actions_hard.detach().clone()
                best_logit_hard_for_this_lod_sparse = linear(active_theta_subset).sigmoid().detach().clone().squeeze(-1)
        
        # End of optimization loop

        # 5f. Store best actions (scatter back to full size)
        if best_actions_hard_for_this_lod_sparse is None:
            # Handle case where loop didn't run or improve
            best_actions_hard_for_this_lod_sparse = torch.zeros(len(active_theta_indices_tensor), device=device, dtype=torch.float)

        best_actions_hard_for_this_lod = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
        best_actions_hard_for_this_lod.scatter_(0, active_theta_indices_tensor, best_actions_hard_for_this_lod_sparse)
        # breakpoint()
        best_logit_hard_for_this_lod = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
        best_logit_hard_for_this_lod.scatter_(0, active_theta_indices_tensor, best_logit_hard_for_this_lod_sparse)
        
        all_best_actions_hard[lod_idx] = best_actions_hard_for_this_lod
        all_best_logit[lod_idx] = best_logit_hard_for_this_lod

        # 5g. Update decision nodes for next LOD
        if child_lod_idx < model.num_lod:
            # Map patch_idx to its sparse hard action
            active_parent_patch_to_best_action = {
                patch_idx: best_actions_hard_for_this_lod_sparse[i]
                for i, patch_idx in enumerate(active_patch_indices_at_lod)
            }
            
            for parent_patch_idx, action in active_parent_patch_to_best_action.items():
                if action > 0.5:
                    parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
                    child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
                    current_decision_nodes[child_lod_idx].extend(child_patches)
        
        del optimizer
        # logger.info(f"--- Finished LOD {lod_idx}. Best Loss: {best_loss:.4f}. Activated {int(best_actions_hard_for_this_lod.sum())} nodes. ---")

    # Visualization: Save combined LOD maps with original image
    if save_dir and image_id is not None:
        # Show which LODs were actually optimized
        logger.info(f"Available LODs in all_best_actions_hard: {list(all_best_actions_hard.keys())}")
        
        # Visualize all optimized LODs (or specific ones like 4, 5)
        # Option 1: Visualize all LODs (currently active)
        lod_maps_to_visualize = all_best_actions_hard
        # Option 2: Visualize only specific LODs (commented out)
        # lod_maps_to_visualize = {}
        # for lod_idx in [4, 5]:  
        #     if lod_idx in all_best_actions_hard:
        #         lod_maps_to_visualize[lod_idx] = all_best_actions_hard[lod_idx]
        #         logger.info(f"Including LOD {lod_idx} with shape {all_best_actions_hard[lod_idx].shape}")
        if lod_maps_to_visualize:
            combined_path = os.path.join(viz_dir, "lod_maps_with_image.png")
            visualize_lod_maps_with_image(image, reconstruction.detach(), lod_maps_to_visualize, all_best_logit, combined_path, dict(patch_size_list=model.patch_size_list, lod_node_mapping=lod_node_mapping, num_patches_per_side=model.num_patch_side_list))
            logger.info(f"Saved combined LOD visualization to {combined_path}")
        else:
            logger.warning(f"No LOD maps to visualize! Available LODs: {list(all_best_actions_hard.keys())}")

    logger.info("Sequential optimization finished.")
    torch.cuda.empty_cache()
    return all_best_actions_hard, current_decision_nodes


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
    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1
        # Call the new sequential optimization function with visualization
        all_actions, final_tree = optimize_tree_for_image(
            image, model, loss_module, config, logger, accelerator, 
            image_id=count-1, save_dir=output_dir
        )
        # Debug: Print which LODs were actually optimized
        logger.info(f"After optimization - LODs available: {list(all_actions.keys())}")
        for lod_idx, actions in all_actions.items():
            logger.info(f"  LOD {lod_idx}: shape={actions.shape}, sum(activated)={actions.sum().item()}")
        
        # breakpoint()  # Uncomment to debug
        # You can inspect all_actions and final_tree here
        final_node = build_tree_from_decision_nodes(final_tree, model.num_patch_side_list)
        with torch.no_grad():
            latent_feats = model.encode(image)
            z = model.selector(latent_feats, final_node)
            token_num += z.shape[-1]
            z_quantized = model.quantize(z).sample()
            reconstructed_images = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_node)

            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
            # Quantize to uint8
            reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
            image = torch.clamp(image, 0.0, 1.0)
            torch.cuda.empty_cache()
        evaluator.update(image, reconstructed_images, None)
        if count == 100:
            break
    print(token_num / count)
    print(evaluator.result())


    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=None, help="Config Path directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_tree_depth", type=int, default=2, help="how many lod are used")


    args = parser.parse_args()
    main(args)