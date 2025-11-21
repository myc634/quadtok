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
from modeling.modules.losses import ReconstructionLoss_Reward
import matplotlib.pyplot as plt

GAIN_THRESHOLD = 0.0
GAIN_THRESHOLD_MAPPING = {2: -0.01, 3: 6.0, 4: 0.05}
LOSS_THRESHOLD_MAPPING = {2: 0.01, 3: -0.2, 4: -0.001}
ENTROPY_THRESHOLD_ESTIMATION = {
        2: 5.5, 
        3: 6.5,  
        4: 7.0 
    }
PRESERVE_PROB = {2: 0.8, 3: 0.3, 4: 0.2}
PRUNE_MAPPING = {3: 0.3, 4: 0.2}

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

# --- NEW RULE-BASED (GREEDY) SEARCH FUNCTION ---

# --- NEW RULE-BASED (GREEDY) SEARCH FUNCTION ---

def optimize_tree_rule_based(
    image, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None
):
    """
    Optimizes the quadtree structure for a single image using a 
    rule-based, greedy, per-LOD search. (Based on User Request)
    
    Algorithm:
    1. Start with the guaranteed_depth tree.
    2. For each optimizable LOD (e.g., LOD 4):
    3. Get all active decision nodes (e.g., 140 nodes at LOD 4).
    4. For each of these 140 nodes, create a *test tree* where this
       single node is "split" (i.e., its 4 children at LOD 5 are added).
    5. Run a full forward pass (encode, select, quantize, decode) with this test tree.
    6. Calculate the PSNR of the reconstruction.
    7. After testing all 140 splits, rank them by PSNR.
    8. Keep the top 50% of splits.
    9. The final tree for this LOD now includes all children from these top 50% splits.
    10. Repeat for the next LOD (e.g., LOD 5), using the newly added
        nodes as the next set of decision candidates.
    
    Args:
        image_id: Optional identifier for saving visualizations
        save_dir: Directory to save visualizations
    """
    logger.info(f"Optimizing tree for one image (Rule-Based Greedy Search)...")
    model.eval()
    # loss_module.eval()
    device = accelerator.device
    guaranteed_depth = config.model.guaranteed_depth

    # Create save directory for visualizations
    viz_dir = None
    if save_dir and image_id is not None:
        viz_dir = os.path.join(save_dir, f"image_{image_id:05d}_viz")
        os.makedirs(viz_dir, exist_ok=True)

    # 1. Build tree structure and mappings (same as original function)
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

    parent_bfs_to_child_patch_map = defaultdict(list)
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            if (child_node.lod_level, child_node.patch_index) in node_to_idx_map:
                parent_bfs_to_child_patch_map[parent_idx].append(child_node.patch_index)

    # We need this map for visualization (to scatter sparse results)
    lod_patch_idx_to_theta_idx_maps = {} 
    for lod_idx in range(model.num_lod - 1):
        if lod_idx >= guaranteed_depth:
            lod_nodes = ordered_full_nodes[lod_start_indices[lod_idx] : lod_start_indices[lod_idx] + lod_len_mapping[lod_idx]]
            lod_patch_idx_to_theta_idx_maps[lod_idx] = {
                node.patch_index: i for i, node in enumerate(lod_nodes)
            }

    # 2. Get image latent (once)
    with torch.no_grad():
        image_latent = model.encode(image)

    # 3. Initialize the tree with guaranteed nodes
    # This dict holds the list of *decision nodes* that form the tree
    current_decision_nodes = defaultdict(list)
    for node in ordered_full_nodes:
        if node.lod_level <= guaranteed_depth:
            current_decision_nodes[node.lod_level].append(node.patch_index)

    # These are for visualization, to match the original function's output
    all_best_actions_hard = {}
    all_best_logits = {} # We'll just copy actions here, as we don't have logits

    # 4. Sequential, rule-based search loop
    # We iterate up to num_lod - 2 because we are *deciding* at lod_idx
    # to add children at lod_idx + 1.
    for lod_idx in range(guaranteed_depth, model.num_lod - 1):
        
        # Get the list of potential nodes to split
        # (e.g., at lod_idx=4, this is the list of nodes from LOD 4)
        decision_candidates_at_lod = current_decision_nodes.get(lod_idx, [])
        
        if not decision_candidates_at_lod:
            logger.info(f"LOD {lod_idx}: No decision candidates. Skipping.")
            # Ensure a zero tensor is created for visualization
            if lod_idx in lod_patch_idx_to_theta_idx_maps:
                 all_best_actions_hard[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
                 all_best_logits[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
            continue

        logger.info(f"Rule-based search: Testing {len(decision_candidates_at_lod)} potential splits at LOD {lod_idx}...")
        
        psnr_scores_for_lod = [] # List to store (psnr, parent_patch_idx)
        
        # This is the "base" tree *before* testing splits at this LOD
        base_tree_nodes = copy.deepcopy(current_decision_nodes)

        # 5. Test each potential split one by one
        for parent_patch_idx in tqdm(decision_candidates_at_lod, desc=f"Testing LOD {lod_idx}"):
            
            # Create a temporary tree: base + 1 new split
            test_tree_nodes = copy.deepcopy(base_tree_nodes)
            
            # Find the children of this one parent
            parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
            child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            
            if not child_patches:
                continue

            # Add this parent's children to the test tree
            test_tree_nodes[lod_idx + 1].extend(child_patches)
            
            # Build the full tree structure from the decision node list
            test_tree_root = build_tree_from_decision_nodes(
                test_tree_nodes, model.num_patch_side_list
            )
            
            # Run the full forward pass
            with torch.no_grad():
                z = model.selector(image_latent, test_tree_root)
                z_quantized = model.quantize(z).sample()
                recon = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), test_tree_root)
                recon = torch.clamp(recon, 0.0, 1.0)
            
            # Calculate PSNR
            mse = F.mse_loss(image, recon)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_scores_for_lod.append((psnr.item(), parent_patch_idx))
            
            del test_tree_root, test_tree_nodes, z, z_quantized, recon
            torch.cuda.empty_cache()

        if not psnr_scores_for_lod:
            logger.info(f"LOD {lod_idx}: No valid splits found.")
            continue

        # 6. Rank the splits and keep top 50%
        psnr_scores_for_lod.sort(key=lambda x: x[0], reverse=True) # Higher PSNR is better
        num_to_keep = int(len(psnr_scores_for_lod) * 0.5)
        top_splits = psnr_scores_for_lod[:num_to_keep]
        top_parents = {p_idx for psnr, p_idx in top_splits}
        
        logger.info(f"LOD {lod_idx}: Kept {len(top_parents)} / {len(psnr_scores_for_lod)} splits.")
        
        # 7. Update the main tree (`current_decision_nodes`) for the next iteration
        new_children_for_next_lod = []
        for parent_patch_idx in top_parents:
            parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
            child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            new_children_for_next_lod.extend(child_patches)
            
        current_decision_nodes[lod_idx + 1] = new_children_for_next_lod

        # 8. (For Visualization) Create the binary action map for this LOD
        lod_map = lod_patch_idx_to_theta_idx_maps[lod_idx]
        lod_len = lod_len_mapping[lod_idx]
        actions_tensor = torch.zeros(lod_len, device=device, dtype=torch.float)
        
        for parent_patch_idx in top_parents:
            if parent_patch_idx in lod_map:
                theta_idx = lod_map[parent_patch_idx]
                actions_tensor[theta_idx] = 1.0
        
        all_best_actions_hard[lod_idx] = actions_tensor
        all_best_logits[lod_idx] = actions_tensor.clone() # Use actions as logits for viz

    # --- End of LOD loop ---
    
    # We need a final reconstruction for the visualization function
    # Note: This is inefficient as it's computed again in main,
    # but required to make the visualization call self-contained here.
    final_reconstruction = image # Placeholder
    try:
        final_tree_root_viz = build_tree_from_decision_nodes(current_decision_nodes, model.num_patch_side_list)
        with torch.no_grad():
            z = model.selector(image_latent, final_tree_root_viz)
            z_quantized = model.quantize(z).sample()
            final_reconstruction = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_tree_root_viz)
            final_reconstruction = torch.clamp(final_reconstruction.detach(), 0.0, 1.0)
    except Exception as e:
        logger.error(f"Error during final reconstruction for viz: {e}")

    # Visualization: Save combined LOD maps
    if viz_dir and all_best_actions_hard:
        combined_path = os.path.join(viz_dir, "lod_maps_with_image_RULE_BASED.png")
        visualize_lod_maps_with_image(
            image, 
            final_reconstruction, 
            all_best_actions_hard, 
            all_best_logits, 
            combined_path, 
            dict(patch_size_list=model.patch_size_list, lod_node_mapping=lod_node_mapping, num_patches_per_side=model.num_patch_side_list)
        )
        logger.info(f"Saved rule-based LOD visualization to {combined_path}")

    logger.info("Rule-based optimization finished.")
    torch.cuda.empty_cache()
    
    # Return the final decision tree and the action maps for logging
    return all_best_actions_hard, all_best_logits, current_decision_nodes

# --- NEW RULE-BASED (GREEDY PSNR GAIN) SEARCH FUNCTION ---

# --- NEW RULE-BASED (GREEDY PSNR GAIN) SEARCH FUNCTION ---

def get_reconstruction_for_tree(image_latent, decision_nodes, model):
    """Returns the reconstruction image for a tree defined by decision_nodes."""
    with torch.no_grad():
        tree_root = build_tree_from_decision_nodes(decision_nodes, model.num_patch_side_list)
        z = model.selector(image_latent, tree_root)
        z_quantized = model.quantize(z).sample()
        recon = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_root)
        recon = torch.clamp(recon, 0.0, 1.0)
        
        del tree_root, z, z_quantized
        torch.cuda.empty_cache()
        return recon

def calculate_patch_entropy(patch, bins=512):
    """
    Calculate entropy for a single patch.
    
    Args:
        patch: Image patch tensor (B, C, H, W) with values in [0, 1]
        bins: Number of bins for histogram
        
    Returns:
        entropy: Entropy value (scalar)
    """
    # Convert to grayscale if RGB
    if patch.shape[1] == 3:
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=patch.device).view(1, 3, 1, 1)
        gray_patch = (patch * grayscale_weights).sum(dim=1)  # (B, H, W)
    else:
        gray_patch = patch[:, 0]  # (B, H, W)
    
    # Scale to [0, 255] and quantize
    patch_int = (gray_patch * 255.0).clamp(0, 255)
    
    # Quantize to bins
    patch_quantized = (patch_int * (bins / 256.0)).long().clamp(0, bins-1)
    
    # Flatten to get all pixels
    patch_flat = patch_quantized.flatten()  # (B*H*W,)
    
    # Create histogram using one-hot encoding
    # Similar to entropy_quadtree_tokenizer.py line 230
    one_hot = torch.zeros(patch_flat.size(0), 1, bins, device=patch.device)
    one_hot = one_hot.scatter_(2, patch_flat.unsqueeze(1).unsqueeze(2), 1)
    histogram = one_hot.sum(0).squeeze(0)  # (bins,)
    
    # Normalize to probabilities
    total_pixels = patch_flat.size(0)
    probabilities = histogram.float() / total_pixels
    
    # Compute entropy: -sum(p * log2(p))
    epsilon = 1e-10
    entropy = -torch.sum(probabilities * torch.log2(probabilities + epsilon))
    
    return entropy.item()

def calculate_patch_psnr_gain(original_image, base_recon, test_recon, parent_patch_idx, lod_idx, num_patches_per_side, patch_size_list):
    """
    Calculate PSNR gain for a specific patch region.
    
    Args:
        original_image: Original image (B, C, H, W) with values in [0, 1]
        base_recon: Base reconstruction without split (B, C, H, W)
        test_recon: Test reconstruction with split (B, C, H, W)
        parent_patch_idx: The patch index at current LOD
        lod_idx: Current LOD level
        num_patches_per_side: Number of patches per side for this LOD
        patch_size_list: List of patch sizes for each LOD
        
    Returns:
        psnr_gain: The PSNR gain for this specific patch region
    """
    # Calculate patch position in the image
    patch_size = patch_size_list[lod_idx]
    row = parent_patch_idx // num_patches_per_side
    col = parent_patch_idx % num_patches_per_side
    
    # Extract patch region from all three images
    start_h = row * patch_size
    end_h = min(start_h + patch_size, original_image.shape[2])
    start_w = col * patch_size
    end_w = min(start_w + patch_size, original_image.shape[3])
    
    orig_patch = original_image[:, :, start_h:end_h, start_w:end_w]
    base_patch = base_recon[:, :, start_h:end_h, start_w:end_w]
    test_patch = test_recon[:, :, start_h:end_h, start_w:end_w]
    
    # Calculate MSE for base and test patches
    base_mse = F.mse_loss(orig_patch, base_patch)
    test_mse = F.mse_loss(orig_patch, test_patch)
    
    # Calculate PSNR
    base_psnr = 10 * torch.log10(1.0 / (base_mse + 1e-10))
    test_psnr = 10 * torch.log10(1.0 / (test_mse + 1e-10))
    
    # Calculate gain
    psnr_gain = (test_psnr - base_psnr).item()
    
    # Additionally, calculate the original patch entropy to understand the complexity
    # This can be used for logging or future enhancement
    patch_entropy = calculate_patch_entropy(orig_patch)

    return psnr_gain, patch_entropy

def optimize_tree_rule_based_gain(
    image, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None
):
    """
    Optimizes the quadtree structure for a single image using a 
    rule-based, greedy, per-LOD search based on PSNR Gain.
    
    Algorithm (v2 - PSNR Gain):
    1. Start with the guaranteed_depth tree.
    2. Calculate a `base_psnr` for this initial tree.
    3. For each optimizable LOD (e.g., LOD 4):
    4. Get all active decision nodes (e.g., 140 nodes at LOD 4).
    5. For each of these 140 nodes, create a *test tree* where this
       single node is "split" (i.e., its 4 children at LOD 5 are added).
    6. Run a full forward pass and calculate the `test_psnr`.
    7. Calculate `psnr_gain = test_psnr - base_psnr`.
    8. After testing all 140 splits, rank them by `psnr_gain`.
    9. Keep the top 50% of splits that also have `psnr_gain > 0`.
    10. The final tree for this LOD now includes all children from these top 50% splits.
    11. **Crucially: Recalculate `base_psnr` using this new, expanded tree.**
    12. Repeat for the next LOD (e.g., LOD 5).
    
    Args:
        image_id: Optional identifier for saving visualizations
        save_dir: Directory to save visualizations
    """
    # logger.info(f"Optimizing tree for one image (Rule-Based Greedy PSNR Gain Search)...")
    
    model.eval()
    # loss_module.eval()
    device = accelerator.device
    guaranteed_depth = config.model.guaranteed_depth
    # loss_fn = ReconstructionLoss_Reward(config).to(device)

    # Create save directory for visualizations
    viz_dir = None
    if save_dir and image_id is not None:
        viz_dir = os.path.join(save_dir, f"image_{image_id:05d}_viz")
        os.makedirs(viz_dir, exist_ok=True)

    # 1. Build tree structure and mappings (same as original function)
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

    parent_bfs_to_child_patch_map = defaultdict(list)
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            if (child_node.lod_level, child_node.patch_index) in node_to_idx_map:
                parent_bfs_to_child_patch_map[parent_idx].append(child_node.patch_index)

    # We need this map for visualization (to scatter sparse results)
    lod_patch_idx_to_theta_idx_maps = {} 
    for lod_idx in range(model.num_lod - 1):
        if lod_idx >= guaranteed_depth:
            lod_nodes = ordered_full_nodes[lod_start_indices[lod_idx] : lod_start_indices[lod_idx] + lod_len_mapping[lod_idx]]
            lod_patch_idx_to_theta_idx_maps[lod_idx] = {
                node.patch_index: i for i, node in enumerate(lod_nodes)
            }

    # 2. Get image latent (once)
    with torch.no_grad():
        image_latent = model.encode(image)

    # 3. Initialize the tree with guaranteed nodes
    # This dict holds the list of *decision nodes* that form the tree
    current_decision_nodes = defaultdict(list)
    for node in ordered_full_nodes:
        if node.lod_level <= guaranteed_depth:
            current_decision_nodes[node.lod_level].append(node.patch_index)

    # These are for visualization, to match the original function's output
    all_best_actions_hard = {}
    all_best_logits = {} # We'll just copy actions here, as we don't have logits

    # --- Helper function to calculate PSNR for a given tree ---
    def get_psnr_for_tree(decision_nodes):
        """Calculates PSNR for a tree defined by decision_nodes."""
        with torch.no_grad():
            tree_root = build_tree_from_decision_nodes(decision_nodes, model.num_patch_side_list)
            z = model.selector(image_latent, tree_root)
            z_quantized = model.quantize(z).sample()
            recon = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_root)
            recon = torch.clamp(recon, 0.0, 1.0)

            mse = F.mse_loss(image, recon)
            psnr = 10 * torch.log10(1.0 / mse)

            recon_loss, _ = loss_module(image, recon)

            del tree_root, z, z_quantized, recon
            torch.cuda.empty_cache()
            return psnr.item(), torch.exp(-recon_loss).item()

    # 4. Calculate initial base_psnr for the guaranteed tree
    base_psnr, base_loss = get_psnr_for_tree(current_decision_nodes)
    # logger.info(f"Initial Base PSNR (LOD <={guaranteed_depth}): {base_psnr:.4f}")

    # 5. Sequential, rule-based search loop
    # We iterate up to num_lod - 2 because we are *deciding* at lod_idx
    # to add children at lod_idx + 1.
    for lod_idx in range(guaranteed_depth, model.num_lod - 1):
        
        # Get the list of potential nodes to split
        # (e.g., at lod_idx=4, this is the list of nodes from LOD 4)
        decision_candidates_at_lod = current_decision_nodes.get(lod_idx, [])
        
        if not decision_candidates_at_lod:
            logger.info(f"LOD {lod_idx}: No decision candidates. Skipping.")
            # Ensure a zero tensor is created for visualization
            if lod_idx in lod_patch_idx_to_theta_idx_maps:
                 all_best_actions_hard[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
                 all_best_logits[lod_idx] = torch.zeros(lod_len_mapping[lod_idx], device=device, dtype=torch.float)
            continue

        # logger.info(f"Rule-based search: Testing {len(decision_candidates_at_lod)} potential splits at LOD {lod_idx} (Base PSNR: {base_psnr:.4f})...")
        
        psnr_gain_scores_for_lod = [] # List to store (psnr_gain, parent_patch_idx)
        
        # This is the "base" tree *before* testing splits at this LOD
        base_tree_nodes = copy.deepcopy(current_decision_nodes)

        # 6. Test each potential split one by one
        for parent_patch_idx in decision_candidates_at_lod:
            
            # Create a temporary tree: base + 1 new split
            test_tree_nodes = copy.deepcopy(base_tree_nodes)
            
            # Find the children of this one parent
            parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
            child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            
            if not child_patches:
                continue

            # Add this parent's children to the test tree
            test_tree_nodes[lod_idx + 1].extend(child_patches)
            
            # Get PSNR for this specific test tree
            test_psnr, test_loss = get_psnr_for_tree(test_tree_nodes)
            
            # Calculate the GAIN

            psnr_gain = test_psnr - base_psnr 
            loss_dec = test_loss - base_loss
            
            psnr_gain_scores_for_lod.append((psnr_gain, loss_dec, parent_patch_idx))
            
            del test_tree_nodes
            torch.cuda.empty_cache()

        if not psnr_gain_scores_for_lod:
            logger.info(f"LOD {lod_idx}: No valid splits found.")
            continue
        # breakpoint()
        # Filter out any splits with negative or zero gain
        psnr_gain_scores_for_lod.sort(key=lambda x: x[0], reverse=True) # Higher PSNR is better
        psnr_gain_list = [psnr_gain for psnr_gain, loss_dec, p_idx in psnr_gain_scores_for_lod]
        mean, var = np.mean(psnr_gain_list), np.var(psnr_gain_list)
        psnr_gain_advantage = (np.array(psnr_gain_list) - mean) / (var + 1e-10)
        num_to_keep = (psnr_gain_advantage > 0).sum()

        if lod_idx == 4:
            num_to_keep = (num_to_keep * 1.5).astype(np.int64)

        psnr_gain_scores_for_lod = psnr_gain_scores_for_lod[:num_to_keep]
        top_parents = {p_idx for psnr_gain, loss_dec, p_idx in psnr_gain_scores_for_lod}

        # psnr_gain_scores_for_lod.sort(key=lambda x: x[1], reverse=False) # Lower Loss is better
        # loss_dec_list = [loss_dec for psnr_gain, loss_dec, p_idx in psnr_gain_scores_for_lod]
        # mean, var = np.mean(loss_dec_list), np.var(loss_dec_list)
        # lose_dec_advantage = (np.array(loss_dec_list) - mean) / (var + 1e-10)
        # num_to_keep = (lose_dec_advantage > 0).sum() 

        # psnr_gain_scores_for_lod = psnr_gain_scores_for_lod[:num_to_keep]
        # top_parents = {p_idx for psnr_gain, loss_dec, p_idx in psnr_gain_scores_for_lod}
        
        # logger.info(f"LOD {lod_idx}: Kept {len(top_parents)} / {len(psnr_gain_scores_for_lod)} splits (Top 50% with positive gain).")
        
        # 8. Update the main tree (`current_decision_nodes`) for the next iteration
        new_children_for_next_lod = []
        if top_parents:
            for parent_patch_idx in top_parents:
                parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
                child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
                new_children_for_next_lod.extend(child_patches)
                
            current_decision_nodes[lod_idx + 1] = new_children_for_next_lod
            
            # 9. **Crucial:** Recalculate base_psnr for the next iteration
            # This is the PSNR of the tree *with* the newly added splits.
            base_psnr, base_loss = get_psnr_for_tree(current_decision_nodes)
            # logger.info(f"New Base PSNR (after LOD {lod_idx} splits): {base_psnr:.4f}")
        else:
            logger.info(f"LOD {lod_idx}: No splits provided positive gain. Base PSNR remains {base_psnr:.4f}.")
            # No new children, so base_psnr for next iter is unchanged.

        # 10. (For Visualization) Create the binary action map for this LOD
        lod_map = lod_patch_idx_to_theta_idx_maps[lod_idx]
        lod_len = lod_len_mapping[lod_idx]
        actions_tensor = torch.zeros(lod_len, device=device, dtype=torch.float)
        
        for parent_patch_idx in top_parents:
            if parent_patch_idx in lod_map:
                theta_idx = lod_map[parent_patch_idx]
                actions_tensor[theta_idx] = 1.0
        
        all_best_actions_hard[lod_idx] = actions_tensor
        all_best_logits[lod_idx] = actions_tensor.clone() # Use actions as logits for viz

    # --- End of LOD loop ---
    
    # We need a final reconstruction for the visualization function
    final_reconstruction = image # Placeholder
    try:
        # Re-use the final base_psnr calculation logic
        final_tree_root_viz = build_tree_from_decision_nodes(current_decision_nodes, model.num_patch_side_list)
        with torch.no_grad():
            z = model.selector(image_latent, final_tree_root_viz)
            z_quantized = model.quantize(z).sample()
            final_reconstruction = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_tree_root_viz)
            final_reconstruction = torch.clamp(final_reconstruction.detach(), 0.0, 1.0)
    except Exception as e:
        logger.error(f"Error during final reconstruction for viz: {e}")

    # Visualization: Save combined LOD maps
    if viz_dir and all_best_actions_hard:
        combined_path = os.path.join(viz_dir, "lod_maps_with_image_RULE_BASED_GAIN.png")
        visualize_lod_maps_with_image(
            image, 
            final_reconstruction, 
            all_best_actions_hard, 
            all_best_logits, 
            combined_path, 
            dict(patch_size_list=model.patch_size_list, lod_node_mapping=lod_node_mapping, num_patches_per_side=model.num_patch_side_list)
        )
        logger.info(f"Saved rule-based (PSNR Gain) LOD visualization to {combined_path}")

    # logger.info("Rule-based (PSNR Gain) optimization finished.")
    torch.cuda.empty_cache()
    
    # Return the final decision tree and the action maps for logging
    return all_best_actions_hard, all_best_logits, current_decision_nodes


def optimize_tree_bottom_up_pruning(
    image, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None, prune_threshold=0.25
):
    """
    Optimizes the quadtree structure using a bottom-up pruning approach.
    
    Algorithm (Bottom-Up Pruning):
    1. Start with a FULL tree (all nodes).
    2. For each optimizable LOD (e.g., LOD 5):
    3. Get all parent nodes at this LOD that have children.
    4. For each parent, test what happens if we remove ALL its children.
    5. Calculate the PSNR of the pruned tree (without children).
    6. Calculate the PSNR drop = full_psnr - pruned_psnr.
    7. If the drop is small (e.g., < prune_threshold), then remove the children.
    8. Otherwise, keep the children.
    9. Repeat for the previous LOD level (going from fine to coarse).
    
    This is the opposite of the greedy approach - we start with everything and remove
    what doesn't hurt too much.
    
    Args:
        image_id: Optional identifier for saving visualizations
        save_dir: Directory to save visualizations
        prune_threshold: Maximum allowed PSNR drop for pruning (dB)
    """
    logger.info(f"Optimizing tree for one image (Bottom-Up Pruning Approach)...")
    model.eval()
    loss_module.eval()
    device = accelerator.device
    guaranteed_depth = config.model.guaranteed_depth

    # Create save directory for visualizations
    viz_dir = None
    if save_dir and image_id is not None:
        viz_dir = os.path.join(save_dir, f"image_{image_id:05d}_viz")
        os.makedirs(viz_dir, exist_ok=True)

    # 1. Build full tree structure and mappings
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

    parent_bfs_to_child_patch_map = defaultdict(list)
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            if (child_node.lod_level, child_node.patch_index) in node_to_idx_map:
                parent_bfs_to_child_patch_map[parent_idx].append(child_node.patch_index)

    # We need this map for visualization
    lod_patch_idx_to_theta_idx_maps = {} 
    for lod_idx in range(model.num_lod - 1):
        if lod_idx >= guaranteed_depth:
            lod_nodes = ordered_full_nodes[lod_start_indices[lod_idx] : lod_start_indices[lod_idx] + lod_len_mapping[lod_idx]]
            lod_patch_idx_to_theta_idx_maps[lod_idx] = {
                node.patch_index: i for i, node in enumerate(lod_nodes)
            }

    # 2. Get image latent (once)
    with torch.no_grad():
        image_latent = model.encode(image)

    # 3. Initialize the tree with ALL nodes (full tree)
    current_decision_nodes = defaultdict(list)
    for node in ordered_full_nodes:
        current_decision_nodes[node.lod_level].append(node.patch_index)

    # Helper function to calculate PSNR for a given tree
    def get_psnr_for_tree(decision_nodes):
        """Calculates PSNR for a tree defined by decision_nodes."""
        with torch.no_grad():
            tree_root = build_tree_from_decision_nodes(decision_nodes, model.num_patch_side_list)
            z = model.selector(image_latent, tree_root)
            z_quantized = model.quantize(z).sample()
            recon = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_root)
            recon = torch.clamp(recon, 0.0, 1.0)
            
            mse = F.mse_loss(image, recon)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
            
            del tree_root, z, z_quantized, recon
            torch.cuda.empty_cache()
            return psnr.item()

    # 4. Calculate initial full tree PSNR
    full_psnr = get_psnr_for_tree(current_decision_nodes)
    logger.info(f"Initial Full Tree PSNR: {full_psnr:.4f}")

    # These are for visualization
    all_best_actions_hard = {}
    all_best_logits = {}

    # 5. Bottom-up pruning loop (from finest to coarsest LOD)
    # We start from the finest LOD and work our way up
    for lod_idx in range(model.num_lod - 2, guaranteed_depth - 1, -1):
        
        # Get all nodes at this LOD that have children
        # These are the nodes we can potentially prune
        decision_candidates_at_lod = current_decision_nodes.get(lod_idx, [])
        
        if not decision_candidates_at_lod:
            logger.info(f"LOD {lod_idx}: No decision candidates. Skipping.")
            continue

        logger.info(f"Bottom-up pruning: Testing {len(decision_candidates_at_lod)} potential prunings at LOD {lod_idx}...")
        
        psnr_drops_for_lod = []  # List to store (psnr_drop, parent_patch_idx)
        
        # Current base tree (before pruning at this level)
        current_tree_nodes = copy.deepcopy(current_decision_nodes)

        # 6. Test each potential pruning one by one
        for parent_patch_idx in tqdm(decision_candidates_at_lod, desc=f"Testing LOD {lod_idx}"):
            
            # Find the children of this parent
            parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
            child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            
            if not child_patches:
                continue

            # Create a pruned tree: current - children of this parent
            pruned_tree_nodes = copy.deepcopy(current_tree_nodes)
            # Remove all children of this parent
            if (lod_idx + 1) in pruned_tree_nodes:
                child_patches_set = set(child_patches)
                pruned_tree_nodes[lod_idx + 1] = [
                    p for p in pruned_tree_nodes[lod_idx + 1] 
                    if p not in child_patches_set
                ]
            
            # Get PSNR for this pruned tree
            pruned_psnr = get_psnr_for_tree(pruned_tree_nodes)
            
            # Calculate the PSNR DROP (how much we lose by pruning)
            psnr_drop = full_psnr - pruned_psnr
            
            psnr_drops_for_lod.append((psnr_drop, parent_patch_idx))
            
            del pruned_tree_nodes
            torch.cuda.empty_cache()

        if not psnr_drops_for_lod:
            logger.info(f"LOD {lod_idx}: No valid prunings found.")
            continue

        # 7. Decide which children to prune based on PSNR drop
        # If drop is small (< prune_threshold), we prune (remove children)
        # breakpoint()
        prune_threshold = PRUNE_MAPPING[lod_idx]
        nodes_to_prune = [p_idx for drop, p_idx in psnr_drops_for_lod if drop <= prune_threshold]
        nodes_to_keep = [p_idx for drop, p_idx in psnr_drops_for_lod if drop > prune_threshold]
        
        # Actually prune: remove children of nodes with small drops
        if nodes_to_prune:
            for parent_patch_idx in nodes_to_prune:
                parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
                child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
                
                # Remove children from current tree
                if (lod_idx + 1) in current_decision_nodes:
                    child_patches_set = set(child_patches)
                    current_decision_nodes[lod_idx + 1] = [
                        p for p in current_decision_nodes[lod_idx + 1] 
                        if p not in child_patches_set
                    ]
        
        logger.info(f"LOD {lod_idx}: Pruned {len(nodes_to_prune)} / {len(psnr_drops_for_lod)} nodes (kept {len(nodes_to_keep)} with significant drops).")
        
        # Recalculate full_psnr for the next iteration (pruned tree becomes the new baseline)
        if nodes_to_prune:
            full_psnr = get_psnr_for_tree(current_decision_nodes)
            logger.info(f"New Tree PSNR (after LOD {lod_idx} pruning): {full_psnr:.4f}")

        # 8. (For Visualization) Create the binary action map for this LOD
        lod_map = lod_patch_idx_to_theta_idx_maps[lod_idx]
        lod_len = lod_len_mapping[lod_idx]
        actions_tensor = torch.zeros(lod_len, device=device, dtype=torch.float)
        
        for parent_patch_idx in nodes_to_keep:  # Keep=1 means we did NOT prune
            if parent_patch_idx in lod_map:
                theta_idx = lod_map[parent_patch_idx]
                actions_tensor[theta_idx] = 1.0
        
        all_best_actions_hard[lod_idx] = actions_tensor
        all_best_logits[lod_idx] = actions_tensor.clone()

    # --- End of LOD loop ---
    
    # Final reconstruction for visualization
    final_reconstruction = image
    try:
        final_tree_root_viz = build_tree_from_decision_nodes(current_decision_nodes, model.num_patch_side_list)
        with torch.no_grad():
            z = model.selector(image_latent, final_tree_root_viz)
            z_quantized = model.quantize(z).sample()
            final_reconstruction = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_tree_root_viz)
            final_reconstruction = torch.clamp(final_reconstruction.detach(), 0.0, 1.0)
    except Exception as e:
        logger.error(f"Error during final reconstruction for viz: {e}")

    # Visualization
    breakpint()
    if viz_dir and all_best_actions_hard:
        combined_path = os.path.join(viz_dir, "lod_maps_with_image_BOTTOM_UP_PRUNING.png")
        visualize_lod_maps_with_image(
            image, 
            final_reconstruction, 
            all_best_actions_hard, 
            all_best_logits, 
            combined_path, 
            dict(patch_size_list=model.patch_size_list, lod_node_mapping=lod_node_mapping, num_patches_per_side=model.num_patch_side_list)
        )
        logger.info(f"Saved bottom-up pruning LOD visualization to {combined_path}")

    logger.info("Bottom-up pruning optimization finished.")
    torch.cuda.empty_cache()
    
    return all_best_actions_hard, all_best_logits, current_decision_nodes



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
    for image, image_path, image_key, class_id in tqdm(generator, desc=f"Eval Step: {count}"):
        count += 1
        # Call the new sequential optimization function with visualization
        all_actions, all_logits, final_tree = optimize_tree_rule_based_gain(
            image, model, loss_module, config, logger, accelerator, 
            image_id=count-1, save_dir=output_dir
        )
        
        # --- END OF CHOICE ---

        # Debug: Print which LODs were actually optimized
        # logger.info(f"After rule-based search - LODs available: {list(all_actions.keys())}")
        # for lod_idx, actions in all_actions.items():
        #     logger.info(f"  LOD {lod_idx}: shape={actions.shape}, sum(activated)={actions.sum().item()}")
        
        # breakpoint()  # Uncomment to debug
        # You can inspect final_tree here
        
        final_node = build_tree_from_decision_nodes(final_tree, model.num_patch_side_list)
        with torch.no_grad():
            latent_feats = model.encode(image)
            z = model.selector(latent_feats, final_node)
            token_num += z.shape[-1]
            z_quantized = model.quantize(z).sample()
            reconstructed_images = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_node)
            # breakpoint()
            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
            # Quantize to uint8
            reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
            image = torch.clamp(image, 0.0, 1.0)
            torch.cuda.empty_cache()
        # breakpoint()
        evaluator.update(image, reconstructed_images, None)
        if count == 1000:
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