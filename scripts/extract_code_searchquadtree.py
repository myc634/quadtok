import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict
import random
import lpips


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
# from modeling.modules.lpips import LPIPS

import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy

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

    lpips_fn = lpips.LPIPS(net='vgg', spatial=True).to(accelerator.device)

    logger.info("Loading Tokenizer Weight From: %s", args.tokenizer_weight)
    model_weight = torch.load(args.tokenizer_weight, map_location="cpu")
    model.load_state_dict(model_weight, strict=True)
    
    model.eval()
    model.requires_grad_(False)

    # Create the custom dataloader
    config.dataset.params.num_workers_per_gpu = args.num_workers
    # shards_path = f"pipe:rclone cat hoss:jianglihan/data/imagenet/imagenet-train-{args.shards_index:06d}.tar"
    shards_path = f"/mnt/ultracube/datasets/imagenet-wds/imagenet-train-{args.shards_index:06d}.tar"
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
        if batch_idx > 200:
            break
        image_key = batch['__key__']
        class_id = batch['class_id'].to(accelerator.device)
        images = batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True) # flatten from ten crop
        
        # for ten crop, flatten the batch
        num_aug = images.shape[1]
        batch_size = images.shape[0]

        images = images.flatten(0, 1)  
        # Generate random quadtree for each image in the batch

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
        ordered_nodes_reversed = model._get_ordered_nodes(masked_tree_reversed)
        ordered_nodes_reversed = [n for n in ordered_nodes_reversed if n.lod_level >= 3]
        
        image_latent = model.encode(images)
        node_list = [deepcopy(ordered_nodes) for _ in range(images.shape[0])]
        node_list.extend([deepcopy(ordered_nodes_reversed) for _ in range(images.shape[0])])
        z_batch = model.selector._forward_optimize(image_latent.repeat(2, 1, 1), node_list)
        _, result_dict = model.quantize(z_batch)
        token_indices = result_dict['min_encoding_indices']
        embeds = model.quantize.get_codebook_entry(token_indices.squeeze().long().flatten()).reshape(images.shape[0] * 2, -1, 8)
        reconstructed_images = model.decoder._forward_optimize(embeds, node_list)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        mse_loss = lpips_fn(images,  reconstructed_images[:images.shape[0]], normalize=True).sum(1)
        # mse_loss = F.mse_loss(images,  reconstructed_images[:images.shape[0]], reduction='none').sum(1) # [bs, 256, 256]
        mse_loss_reversed = lpips_fn(images,  reconstructed_images[images.shape[0]:], normalize=True).sum(1)
        # mse_loss = F.mse_loss(images,  reconstructed_images[:images.shape[0]], reduction='none').sum(1)
        # mse_loss_reversed = F.mse_loss(images, reconstructed_images[images.shape[0]:], reduction='none').sum(1)

        patch_size = 32
        pooling_layer = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        patch_bin_list = []
        k = random.randint(32, 64)
        for batch_idx in range(images.shape[0]):
            original_loss_diff = (mse_loss_reversed[batch_idx] - mse_loss[batch_idx])
            patch_loss_diff = pooling_layer(original_loss_diff.unsqueeze(0).unsqueeze(0))[0,0]
            random_patch_mask_int = random_patch_mask.int()
            random_patch_mask_int[~random_patch_mask] = -1
            diff_patch_map = (patch_loss_diff) * random_patch_mask_int.reshape(8, 8).to(patch_loss_diff.device)

            flat_diff = diff_patch_map.flatten()
            sorted_vals, sorted_idx = torch.sort(flat_diff, descending=True)
            threshold = sorted_vals[k-1]
            threshold = 0.05

            patch_bin = (diff_patch_map >= threshold).to(diff_patch_map.dtype)
            patch_bin_list.append(patch_bin)

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
            all_search_nodes.append(search_nodes)

        with torch.no_grad():
            z_batch_search = model.selector._forward_optimize(image_latent, all_search_nodes)
            _, result_dict_search = model.quantize(z_batch_search)
            code_indices = result_dict_search['min_encoding_indices'][:, 0]
            '''embeds_search = model.quantize.get_codebook_entry(code_indices.squeeze().long().flatten()).reshape(images.shape[0], -1, 8)
            reconstructed_images_search = model.decoder._forward_optimize(embeds_search, all_search_nodes)
            reconstructed_images_search = torch.clamp(reconstructed_images_search, 0.0, 1.0)'''

        for aug_idx in range(num_aug):
            base_filename = image_key[0] + f"_{aug_idx}"

            cur_code_indices = code_indices[aug_idx]
            cur_node_list = all_search_nodes[aug_idx]
            cur_length = len(cur_node_list)
            cur_code_indices = cur_code_indices[:cur_length]
            lod_indices, patch_incides = [], []
            for node in cur_node_list:
                lod_indices.append(node.lod_level)
                patch_incides.append(node.patch_index)


            if batch_size > 1:
                class_id_data = class_id[i].item()
            else:
                class_id_data = class_id.item() if hasattr(class_id, 'item') else int(class_id)

            assert cur_code_indices.shape[0] == len(lod_indices) == len(patch_incides)
            sample = {
                "__key__": base_filename,
                "code_indices.npy": cur_code_indices.cpu().numpy(),
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
