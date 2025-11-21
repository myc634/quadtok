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

    return psnr_gain

def optimize_tree_rule_based_gain(
    image, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None
):
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
    
    # --- Batch helper function to calculate PSNR for multiple trees with same structure ---
    def get_psnr_for_trees_batch(decision_nodes_list):
        batch_size = len(decision_nodes_list)
        results = []
        
        with torch.no_grad():
            # Build all tree roots
            tree_roots = []
            for decision_nodes in decision_nodes_list:
                tree_root = build_tree_from_decision_nodes(decision_nodes, model.num_patch_side_list)
                tree_roots.append(tree_root)
            
            # Use batch processing: all trees have same structure, so we can batch them
            # Call _forward_optimize directly for batch processing
            z_batch = model.selector._forward_optimize(image_latent, tree_roots, [len(tree_roots) // image_latent.shape[0]] * image_latent.shape[0])
            z_quantized_batch = model.quantize(z_batch).sample()
            
            # Convert z_quantized_batch from (batch_size, C, 1, seq_len) to (batch_size, seq_len, C)
            # Each tree uses its corresponding z_quantized from selector output
            z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()  # (batch_size, seq_len, C)
            
            # Use batch decoder - each tree uses its corresponding z_quantized
            recon_batch = model.decoder._forward_optimize(z_quantized_for_decoder, tree_roots)
            recon_batch = torch.clamp(recon_batch, 0.0, 1.0)
            
            # Calculate PSNR and loss for each reconstruction

            mse = F.mse_loss(image.repeat(recon_batch.shape[0], 1, 1, 1), recon_batch, reduction="none").mean(dim=(1, 2, 3))
            psnr = 10 * torch.log10(1.0 / mse) 
            del tree_roots, z_batch, z_quantized_batch, z_quantized_for_decoder, recon_batch

        torch.cuda.empty_cache()
        return psnr.tolist()

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
        
        # This is the "base" tree *before* testing splits at this LOD
        base_tree_nodes = copy.deepcopy(current_decision_nodes)

        # 6. Batch process: Prepare all test trees first
        # All trees will have the same structure (base_tree + 4 children from each parent)
        # So they can be processed in batch
        valid_candidates = []  # List of parent_patch_idx that have valid children
        test_tree_nodes_list = []  # List of test tree structures
        
        for parent_patch_idx in decision_candidates_at_lod:
            # Find the children of this one parent
            parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
            child_patches = parent_bfs_to_child_patch_map.get(parent_global_idx, [])
            
            if not child_patches:
                continue
            
            # Create a temporary tree: base + 1 new split
            test_tree_nodes = copy.deepcopy(base_tree_nodes)
            # Add this parent's children to the test tree
            test_tree_nodes[lod_idx + 1].extend(child_patches)
            
            valid_candidates.append(parent_patch_idx)
            test_tree_nodes_list.append(test_tree_nodes)
        
        # Batch process all trees with same structure
        psnr_results = get_psnr_for_trees_batch(test_tree_nodes_list)
        
        # Calculate gains for all candidates
        psnr_gain_scores_for_lod = []

        for parent_patch_idx, test_psnr in zip(valid_candidates, psnr_results):
            psnr_gain = test_psnr - base_psnr
            psnr_gain_scores_for_lod.append((psnr_gain, parent_patch_idx))
        
        # Clean up
        del test_tree_nodes_list
        torch.cuda.empty_cache()

        if not psnr_gain_scores_for_lod:
            logger.info(f"LOD {lod_idx}: No valid splits found.")
            continue
        # breakpoint()
        # Filter out any splits with negative or zero gain
        psnr_gain_scores_for_lod.sort(key=lambda x: x[0], reverse=True) # Higher PSNR is better
        psnr_gain_list = [psnr_gain for psnr_gain, p_idx in psnr_gain_scores_for_lod]
        mean, var = np.mean(psnr_gain_list), np.var(psnr_gain_list)
        psnr_gain_advantage = (np.array(psnr_gain_list) - mean) / (var + 1e-10)
        num_to_keep = (psnr_gain_advantage > 0).sum()

        if lod_idx == 4:
            num_to_keep = (num_to_keep * 1.5).astype(np.int64)

        psnr_gain_scores_for_lod = psnr_gain_scores_for_lod[:num_to_keep]
        top_parents = {p_idx for psnr_gain, p_idx in psnr_gain_scores_for_lod}
        
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
    torch.cuda.empty_cache()
    
    # Return the final decision tree and the action maps for logging
    return current_decision_nodes

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
        final_tree = optimize_tree_rule_based_gain(
            image, model, loss_module, config, logger, accelerator, 
            image_id=count-1, save_dir=None
        )
        breakpoint()
        final_node = build_tree_from_decision_nodes(final_tree, model.num_patch_side_list)
        with torch.no_grad():
            latent_feats = model.encode(image)
            z = model.selector(latent_feats, final_node)
            token_num += z.shape[-1]
            logger.info(f"Token Num: {z.shape[-1]}")
            z_quantized = model.quantize(z).sample()
            reconstructed_images = model.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), final_node)
            reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
            # Quantize to uint8
            reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
            image = torch.clamp(image, 0.0, 1.0)
            torch.cuda.empty_cache()

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