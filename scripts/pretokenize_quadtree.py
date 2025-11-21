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
import webdataset as wds
import json
from modeling.modules.blocks import Mlp
from modeling.modules import ReconstructionLoss_Reward
from utils.train_utils import create_model, create_dataloader, auto_resume, create_evaluator
from modeling.utils import build_quadtree, get_ordered_nodes, build_tree_from_decision_nodes, build_quadtree, build_random_quadtree, build_probabilistic_quadtree
from modeling.modules.losses import ReconstructionLoss_Reward
import matplotlib.pyplot as plt
import time
import subprocess
from data.webdataset_reader import QuadtreeImageDataset

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
        z_quantized_batch = model.quantize(z_batch).sample()
        
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
                z_quantized_batch, _ = model.quantize(z_batch)
            else:
                NotImplementedError
            
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

            top_gains, top_indices = torch.topk(gains_for_image_b, k=min(int(num_to_keep), len(gains_for_image_b)))
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
    
    # Generate tree status lists
    node_to_idx_map = tree_components['node_to_idx_map']
    ordered_full_nodes = tree_components['ordered_full_nodes']
    child_to_parent_map = tree_components['child_to_parent_map']
    num_total_nodes = len(ordered_full_nodes)

    all_tree_statuses = []
    for decision_nodes in current_decision_nodes_list:
        status_list = [-1] * num_total_nodes
        
        present_nodes_set = set()
        for lod, nodes in decision_nodes.items():
            for node in nodes:
                present_nodes_set.add((node.lod_level, node.patch_index))

        # Mark present nodes with 1
        for i, node in enumerate(ordered_full_nodes):
            if (node.lod_level, node.patch_index) in present_nodes_set:
                status_list[i] = 1

        # Mark potential but not chosen nodes with 0
        for i, node in enumerate(ordered_full_nodes):
            if status_list[i] == -1: # If node is not present
                node_key = (node.lod_level, node.patch_index)
                parent_node = child_to_parent_map.get(node_key)

                if parent_node:
                    parent_key = (parent_node.lod_level, parent_node.patch_index)
                    parent_bfs_idx = node_to_idx_map.get(parent_key)
                    if parent_bfs_idx is not None and status_list[parent_bfs_idx] == 1:
                        status_list[i] = 0
        
        all_tree_statuses.append(status_list)

    # Return the final decision tree and the action maps for logging
    return current_decision_nodes_list, all_tree_statuses

def process_tree_for_saving(tree):
    processed_tree = defaultdict(list)
    for lod_idx, nodes in sorted(tree.items()):
        for node in nodes:
            processed_tree[lod_idx].append({
                'patch_index': node.patch_index,
                'lod_level': node.lod_level
            })
    return dict(processed_tree)

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
    
    # Use ShardWriter to create tar file (similar to convert_imagenet_to_wds.py)
    tar_writer = wds.TarWriter(args.output_tar_path)  # Very large maxcount to avoid splitting
    
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
    
    # Create the custom dataloader
    quadtree_dataset = QuadtreeImageDataset(
        shards_path=args.shards_path,
        resize_shorter_edge=config.dataset.preprocessing.resize_shorter_edge,
        crop_size=config.dataset.preprocessing.crop_size,
        random_crop=config.dataset.preprocessing.random_crop,
        random_flip=config.dataset.preprocessing.random_flip,
        num_workers_per_gpu=config.dataset.params.num_workers_per_gpu,
    )
    custom_dataloader = quadtree_dataset.dataloader


    # build loss
    loss_module = ReconstructionLoss_Reward(config=config)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model, loss_module, custom_dataloader = accelerator.prepare(model, loss_module, custom_dataloader)
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
    child_to_parent_map = {}
    for parent_idx, parent_node in enumerate(ordered_full_nodes):
        for child_node in parent_node.children:
            child_key = (child_node.lod_level, child_node.patch_index)
            if child_key in node_to_idx_map:
                parent_bfs_to_child_nodes[parent_idx].append(child_node)
                child_to_parent_map[child_key] = parent_node
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
                           ordered_full_nodes = ordered_full_nodes,
                           child_to_parent_map=child_to_parent_map
                           )
    
    # Count samples in the input tar file using tar command
    logger.info("Counting samples in input tar file...")
    try:
        # Run tar command to list files and count them
        result = subprocess.run(
            f"tar -tvf {args.shards_path} | wc -l",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        estimated_samples = int(result.stdout.strip()) // 2
        # Each sample typically has 2 files (.jpg/.png and .cls)
        # Adjust this number based on your actual tar structure
        logger.info(f"Estimated samples in dataset: {estimated_samples} (total files: {estimated_samples})")
        print(f"\n{'='*60}")
        print(f"Estimated samples in dataset: {estimated_samples}")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.warning(f"Could not count samples: {e}")
        estimated_samples = None
    
    token_num = 0
    num_samples = 0
    logger.info("Starting to process samples...")
    for batch_idx, (batch) in tqdm(enumerate(custom_dataloader), desc=f"Processing samples", total=estimated_samples):
        image_key, class_id = batch['__key__'], batch['class_id'].to(accelerator.device)
        images = batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True)
        count += 1
        final_trees, all_tree_statuses = optimize_tree_rule_based_gain(
            images, tree_components, model, loss_module, config, logger, accelerator, 
            image_id=count-1, save_dir=None
        )

        processed_final_trees = [process_tree_for_saving(tree) for tree in final_trees]

        with torch.no_grad():
            image_latent = model.encode(images)

            z_batch = model.selector._forward_optimize(image_latent, final_trees)
            if model.quantize_mode == "vae":
                z_quantized_batch = model.quantize(z_batch).sample()
                z_quantized_for_decoder = z_quantized_batch.permute(0, 3, 2, 1).squeeze(2).contiguous()
                z_numpy = z_quantized_for_decoder[0].cpu().numpy()
            elif model.quantize_mode == "vq":
                z_quantized_batch, result_dict = model.quantize(z_batch)
                z_quantized_for_decoder = result_dict['min_encoding_indices'].squeeze(1)
                z_numpy = z_quantized_for_decoder[0].cpu().numpy()
            
            token_num += z_quantized_for_decoder.shape[1]
            # Since batch size is 1, we take the first element.
            key = image_key[0]
            base_filename = key.split('/')[-1].rsplit('.', 1)[0]
            
            
            tree_data = processed_final_trees[0]
            status_data = all_tree_statuses[0]
            class_id_data = class_id[0].item()

            # --- Build parent index array over the token sequence ---
            # The latent sequence order must match the order used in selector._forward_optimize.
            # For a given final tree (dict: lod_idx -> list[QuadTreeNode]),
            # the sequence is constructed by concatenating nodes level-by-level in lod order.
            final_tree = final_trees[0]
            ordered_nodes_for_seq = []
            for lod_idx in sorted(final_tree.keys()):
                ordered_nodes_for_seq.extend(final_tree[lod_idx])

            seq_len = len(ordered_nodes_for_seq)
            assert seq_len == z_numpy.shape[0], (
                f"Token sequence length ({z_numpy.shape[0]}) does not match number of nodes ({seq_len})."
            )

            # Map each node (lod_level, patch_index) -> token position i
            token_pos_map = {
                (node.lod_level, node.patch_index): idx
                for idx, node in enumerate(ordered_nodes_for_seq)
            }

            # child_to_parent_map is constructed once above from the full quadtree
            parent_indices = np.full(seq_len, -1, dtype=np.int32)
            for i, node in enumerate(ordered_nodes_for_seq):
                child_key = (node.lod_level, node.patch_index)
                parent_node = child_to_parent_map.get(child_key, None)
                if parent_node is None:
                    # Root node or missing parent -> keep -1
                    continue
                parent_key = (parent_node.lod_level, parent_node.patch_index)
                parent_idx = token_pos_map.get(parent_key, -1)
                parent_indices[i] = parent_idx

            metadata_to_save = {
                "tree": tree_data,
                "status": status_data
            }

            sample = {
                "__key__": base_filename,
                "npy": z_numpy,
                "parent_idx.npy": parent_indices,
                "json": json.dumps(metadata_to_save),
                "cls": str(class_id_data)
            }
            tar_writer.write(sample)
            num_samples += 1
    logger.info(f"Avg Token Number: {token_num / num_samples}")
    tar_writer.close()
    logger.info(f"Total samples processed and saved: {num_samples}")
    print(f"\n{'='*60}")
    print(f"Successfully saved {num_samples} samples to {args.output_tar_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default=None, help="Config Path directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--shards_path", type=str, default=None, help="Path to the .tar.gz shards file")
    parser.add_argument("--output_tar_path", type=str, default=None, help="Output .tar file path for pretokenized data")
    parser.add_argument("--max_tree_depth", type=int, default=2, help="how many lod are used")


    args = parser.parse_args()
    main(args)