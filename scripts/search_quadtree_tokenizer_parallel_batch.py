import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)
import concurrent.futures
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
import time

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
    config.training.per_gpu_batch_size = 32
    # config.dataset.params.eval_shards_path_or_url = "/mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet/ILSVRC2012_validation/val-{000000..000049}.tar"
    _, eval_dataloader = create_dataloader(config, logger, accelerator)

    count = 0
    for batch in eval_dataloader:
        yield batch['image'], batch['__key__'], batch['class_id'].to(accelerator.device)

class Namespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

def visualize_lod_maps_with_image(image, reconstruction, lod_maps_dict, save_path, patch_info):
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
    for idx, (lod_idx, lod_map) in enumerate(sorted(lod_maps_dict.items())):
        axes_idx = idx + 2
        
        patch_size = patch_info['patch_size_list'][lod_idx]
        lod_nodes = patch_info['lod_node_mapping'][lod_idx]
        num_patches_per_side = patch_info['num_patches_per_side'][lod_idx]

        binary_map = torch.zeros(num_patches_per_side, num_patches_per_side, dtype=torch.int64)

        for node in lod_map:
            row, col = divmod(node.patch_index, num_patches_per_side)
            binary_map[row, col] = 1

        axes[axes_idx].imshow(binary_map, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[axes_idx].set_title(f'Binary Map LOD {lod_idx} ({int(binary_map.sum())} nodes)', fontsize=12)
        axes[axes_idx].axis('off')
        axes[axes_idx].grid(True, alpha=0.3)

    plt.suptitle('Quadtree Activation Maps & Token Heat Map', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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

def create_test_tree_task(args):
    try:
        b, lod_idx, parent_patch_idx, base_tree_nodes, \
        base_children_list_at_next_lod, node_to_idx_map, \
        parent_bfs_to_child_nodes = args

        parent_global_idx = node_to_idx_map.get((lod_idx, parent_patch_idx))
        if parent_global_idx is None:
            return None

        child_nodes = parent_bfs_to_child_nodes.get(parent_global_idx, [])
        if not child_nodes:
            return None

        test_tree_nodes = base_tree_nodes.copy()
        test_tree_nodes[lod_idx + 1] = base_children_list_at_next_lod + child_nodes
        return (test_tree_nodes, b, parent_patch_idx)
    except Exception as e:
        print(f"Error in worker task: {e}")
        return None


def get_psnr_for_trees_batch(decision_nodes_list, model, image_batch, 
                             image_latent_batch, device, 
                             build_tree_from_decision_nodes, 
                             batch_expand_list=None):
    with torch.no_grad():
        tree_roots = []
        for decision_nodes in decision_nodes_list:
            tree_root = build_tree_from_decision_nodes(decision_nodes, model.num_patch_side_list)
            tree_roots.append(tree_root)
        if batch_expand_list:
            batch_indices = torch.repeat_interleave(
                torch.arange(len(batch_expand_list), device=device),
                torch.tensor(batch_expand_list, device=device, dtype=torch.long)
            )
            target_latent_batch = image_latent_batch.index_select(0, batch_indices)
            target_image_batch = image_batch.index_select(0, batch_indices)
        else:
            target_latent_batch = image_latent_batch
            target_image_batch = image_batch

        z_batch = model.selector._forward_optimize(target_latent_batch, tree_roots)
        if model.quantize_mode == "vae":
            z_quantized_batch = model.quantize(z_batch).sample()
        elif model.quantize_mode == "vq":
            z_quantized_batch, result_dict = model.quantize(z_batch)
        
        z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()
        recon_batch = model.decoder._forward_optimize(z_quantized_for_decoder, tree_roots)
        recon_batch = torch.clamp(recon_batch, 0.0, 1.0)
        
        mse = F.mse_loss(target_image_batch, recon_batch, reduction="none").mean(dim=(1, 2, 3))
        psnr = 10 * torch.log10(1.0 / mse) 
        
        del tree_roots, z_batch, z_quantized_batch, z_quantized_for_decoder, recon_batch
        del target_image_batch, target_latent_batch

    torch.cuda.empty_cache()
    return psnr 

def optimize_tree_rule_based_gain(
    image, tree_components, model, loss_module, config, logger, accelerator, image_id=None, save_dir=None
):
    # logger.info(f"Optimizing tree for one image (Rule-Based Greedy PSNR Gain Search)...")
    
    model.eval()

    device = accelerator.device
    batch_size = image.shape[0]
    guaranteed_depth = config.model.guaranteed_depth

    with torch.no_grad():
        image_latent = model.encode(image)

    current_decision_nodes = tree_components['current_decision_nodes'].copy()
    node_to_idx_map = tree_components['node_to_idx_map'].copy()
    parent_bfs_to_child_nodes = tree_components['parent_bfs_to_child_nodes'].copy()

    # 1 expand the candicate tree into batch size
    current_decision_nodes_list = [copy.deepcopy(current_decision_nodes) for _ in range(batch_size)]
    
    # --- Batch helper function to calculate PSNR for multiple trees with same structure ---
    def get_psnr_for_trees_batch(decision_nodes_list, batch_expand_list=None):

        with torch.no_grad():
            # Build all tree roots

            # Use batch processing: all trees have same structure, so we can batch them
            # Call _forward_optimize directly for batch processing
            z_batch = model.selector._forward_optimize(image_latent, decision_nodes_list, batch_expand_list)
            if model.quantize_mode == "vae":
                z_quantized_batch = model.quantize(z_batch).sample()
            elif model.quantize_mode == "vq":
                z_quantized_batch, result_dict = model.quantize(z_batch)
            
            # Convert z_quantized_batch from (batch_size, C, 1, seq_len) to (batch_size, seq_len, C)
            # Each tree uses its corresponding z_quantized from selector output
            z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()  # (batch_size, seq_len, C)
            # Use batch decoder - each tree uses its corresponding z_quantized
            recon_batch = model.decoder._forward_optimize(z_quantized_for_decoder, decision_nodes_list)
            recon_batch = torch.clamp(recon_batch, 0.0, 1.0)

            # Calculate PSNR and loss for each reconstruction
            if image.shape[0] == recon_batch.shape[0]:
                mse = F.mse_loss(image, recon_batch, reduction="none").mean(dim=(1, 2, 3))
            else:
                assert batch_expand_list is not None
                mse = F.mse_loss(torch.repeat_interleave(image, torch.tensor(batch_expand_list, device=device), dim=0), recon_batch, reduction="none").mean(dim=(1, 2, 3))
            
            psnr = 10 * torch.log10(1.0 / mse) 
            del z_batch, z_quantized_batch, z_quantized_for_decoder, recon_batch

        torch.cuda.empty_cache()
        return psnr

    # 4. Calculate initial base_psnr for the guaranteed tree
    base_psnr = get_psnr_for_trees_batch(current_decision_nodes_list)
    # logger.info(f"Initial Base PSNR (LOD <={guaranteed_depth}): {base_psnr:.4f}")

    # 5. Sequential, rule-based search loop
    # We iterate up to num_lod - 2 because we are *deciding* at lod_idx
    # to add children at lod_idx + 1.
    for lod_idx in range(guaranteed_depth, model.num_lod - 1):
        # logger.info(f"Start Processing LOD: {lod_idx}")

        valid_candidates_info = []  # Stores tuples of (batch_index, parent_patch_idx)
        mega_batch_test_tree_nodes_list = []  # Stores all corresponding tree dicts
        

        batch_size = len(current_decision_nodes_list)
        batch_expand_list, batch_indices_list, parent_patch_idx_list = [0 for _ in range(batch_size)], [], []
        tasks = []
        for b in range(batch_size):
            base_tree_nodes = current_decision_nodes_list[b] 
            decision_candidates_at_lod = base_tree_nodes.get(lod_idx, [])
            base_children_list_at_next_lod = base_tree_nodes.get(lod_idx + 1, [])
            batch_expand_list[b] = len(decision_candidates_at_lod) 

            for parent_node in decision_candidates_at_lod:
                task_args = (
                    b, lod_idx, parent_node.patch_index, base_tree_nodes,
                    base_children_list_at_next_lod, node_to_idx_map,
                    parent_bfs_to_child_nodes
                )
                tasks.append(task_args)
        batch_indices_list = []
        parent_patch_idx_list = []

        # 0.3s @ LOD 3
        # 0.5s @ LOD 5
        results = []
        for task in tasks:
            results.append(create_test_tree_task(task))

        for result in results:
            test_tree_nodes, b, parent_patch_idx = result
            
            mega_batch_test_tree_nodes_list.append(test_tree_nodes)
            batch_indices_list.append(b)
            parent_patch_idx_list.append(parent_patch_idx)
            valid_candidates_info.append((b, parent_patch_idx))

        test_tree_nodes_list = mega_batch_test_tree_nodes_list
        batch_indices_tensor = torch.tensor(batch_indices_list, device=device, dtype=torch.long)
        parent_patch_idx_tensor = torch.tensor(parent_patch_idx_list, device=device, dtype=torch.long)
        
        # Batch process all trees with same structure
        # 10s @ LOD 3
        # 36s @ LOD 4
        psnr_results = get_psnr_for_trees_batch(test_tree_nodes_list, batch_expand_list)
        all_gains_tensor = psnr_results - base_psnr[batch_indices_tensor]
        
        per_image_gains = [[] for _ in range(batch_size)]

        for idx, test_psnr in enumerate(psnr_results):
            batch_index, parent_patch_idx = valid_candidates_info[idx]
            psnr_gain = test_psnr - base_psnr[batch_index]
            per_image_gains[batch_index].append((psnr_gain, parent_patch_idx))

        # Clean up
        del test_tree_nodes_list
        torch.cuda.empty_cache()

        list_of_top_parents = []
        for b in range(batch_size):
            mask = (batch_indices_tensor == b)
            gains_for_image_b = all_gains_tensor[mask]
            patches_for_image_b = parent_patch_idx_tensor[mask]
            mean = torch.mean(gains_for_image_b)
            var = torch.var(gains_for_image_b)
            advantage = (gains_for_image_b - mean) / (var + 1e-10)

            keep_mask = (advantage > 0)
            num_to_keep = torch.sum(keep_mask).item()

            top_gains, top_indices = torch.topk(gains_for_image_b, k=min(num_to_keep, len(gains_for_image_b)))
            top_patches = patches_for_image_b[top_indices]
            list_of_top_parents.append(set(top_patches.cpu().numpy()))
            
        
        for b in range(batch_size):
            top_parents = list_of_top_parents[b]
            new_children_for_next_lod = []
            for parent_patch_idx in top_parents:
                parent_global_idx = node_to_idx_map[(lod_idx, parent_patch_idx)]
                child_patches = parent_bfs_to_child_nodes[parent_global_idx]
                new_children_for_next_lod.extend(child_patches)
                
            current_decision_nodes_list[b][lod_idx + 1] = new_children_for_next_lod
            # 9. **Crucial:** Recalculate base_psnr for the next iteration
            # This is the PSNR of the tree *with* the newly added splits.
        # Less than 0.1 s
        # st = time.time()
        base_psnr = get_psnr_for_trees_batch(current_decision_nodes_list)
        # logger.info(f"PSNR Compute post at LOD {lod_idx}: {time.time() - st}")
            # logger.info(f"New Base PSNR (after LOD {lod_idx} splits): {base_psnr:.4f}")
    torch.cuda.empty_cache()
    
    # Return the final decision tree and the action maps for logging
    return current_decision_nodes_list

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    from torch.profiler import profile, record_function, ProfilerActivity

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
    config.dataset.params.num_workers_per_gpu = 1
    train_dataloader, eval_dataloader = create_dataloader(config, None, accelerator)

    # build loss
    loss_module = ReconstructionLoss_Reward(config=config)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model, loss_module, eval_dataloader = accelerator.prepare(model, loss_module, eval_dataloader)
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

    # init all the feature need for constructure quadtree
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

    parent_bfs_to_child_nodes = defaultdict(list)
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            if (child_node.lod_level, child_node.patch_index) in node_to_idx_map:
                parent_bfs_to_child_nodes[parent_idx].append(child_node)
    # 3. Initialize the tree with guaranteed nodes
    # This dict holds the list of *decision nodes* that form the tree
    current_decision_nodes = defaultdict(list)
    for node in ordered_full_nodes:
        if node.lod_level <= config.model.guaranteed_depth:
            current_decision_nodes[node.lod_level].append(node)
    
    tree_components = dict(current_decision_nodes=current_decision_nodes, 
                           parent_bfs_to_child_nodes=parent_bfs_to_child_nodes,
                           lod_start_indices = lod_start_indices,
                           node_to_idx_map = node_to_idx_map,
                           )
    token_num = 0
    for batch_idx, (batch) in tqdm(enumerate(eval_dataloader), desc=f"Eval Step: {count}"):
        image_key, class_id = batch['__key__'], batch['class_id'].to(accelerator.device)
        images = batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        count += 1
        final_trees = optimize_tree_rule_based_gain(
            images, tree_components, model, loss_module, config, logger, accelerator, 
            image_id=count-1, save_dir=None
        )
        with torch.no_grad():
            image_latent = model.encode(images)

            z_batch = model.selector._forward_optimize(image_latent, final_trees)
            if model.quantize_mode == "vae":
                z_quantized_batch = model.quantize(z_batch).sample()
            elif model.quantize_mode == "vq":
                z_quantized_batch, result_dict = model.quantize(z_batch)
            z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()  # (batch_size, seq_len, C)
            # Use batch decoder - each tree uses its corresponding z_quantized
            recon_batch = model.decoder._forward_optimize(z_quantized_for_decoder, final_trees)
            recon_batch = torch.clamp(recon_batch, 0.0, 1.0)

        recon_batch = torch.round(recon_batch * 255.0) / 255.0

        images = torch.clamp(images, 0.0, 1.0)
        if model.quantize_mode == "vae":
            evaluator.update(images, recon_batch, None)
        elif model.quantize_mode == "vq":
            evaluator.update(images, recon_batch, result_dict["min_encoding_indices"])

    print(evaluator.result())
        # breakpoint()
        # for viz ONLY !!!!!
        # with torch.no_grad():
        #     image, tree = images[0].unsqueeze(0), [copy.deepcopy(final_trees[0])]
        #     image_latent = model.encode(image)
        #     z_batch = model.selector._forward_optimize(image_latent, tree)
        #     z_quantized_batch = model.quantize(z_batch).sample()
        #     z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()
        #     recon = model.decoder._forward_optimize(z_quantized_for_decoder, tree).clamp(0, 1)

        #     vis_tree = dict()
        #     for key, value in final_trees[0].items():
        #         if key > config.model.guaranteed_depth:
        #             vis_tree[key] = value.copy()

        #     viz_dir = os.path.join(output_dir, f"sample_{batch_idx}", "activation.png")
        #     os.makedirs(os.path.join(output_dir, f"sample_{batch_idx}"), exist_ok=True)
        #     visualize_lod_maps_with_image(image, recon, vis_tree, viz_dir, dict(patch_size_list=model.patch_size_list, lod_node_mapping=lod_node_mapping, num_patches_per_side=model.num_patch_side_list))



        




    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=None, help="Config Path directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_tree_depth", type=int, default=2, help="how many lod are used")


    args = parser.parse_args()
    main(args)