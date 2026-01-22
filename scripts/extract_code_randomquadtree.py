import os
import sys
from pathlib import Path
import argparse
from collections import defaultdict

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
    tree_to_decision_nodes_dict
)


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
    
    # Use TarWriter to create tar file
    if args.crop_range == 1.1:
        save_idx = args.shards_index
    elif args.crop_range == 1.05:
        save_idx = args.shards_index + 71
    else:
        raise ValueError(f"Invalid crop range: {args.crop_range}")
    output_tar_path = f"{args.output_tar_path}/imagenet-train-{save_idx:06d}.tar"
    tar_writer = wds.TarWriter(output_tar_path)
    
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
    # with wds.TarWriter(process.stdin) as tar_writer:
    for batch_idx, batch in tqdm(enumerate(custom_dataloader), desc="Processing samples"):

        image_key = batch['__key__']
        class_id = batch['class_id'].to(accelerator.device)
        images = batch["image"].to(accelerator.device, memory_format=torch.contiguous_format, non_blocking=True) # flatten from ten crop
        
        # for ten crop, flatten the batch
        num_aug = images.shape[1]
        batch_size = images.shape[0]

        images = images.flatten(0, 1)  
        # Generate random quadtree for each image in the batch
        root_list = []
        for _ in range(num_aug):
            tree_root = build_probabilistic_quadtree(
                model.num_patch_side_list,
                guaranteed_depth=guaranteed_depth,
                expansion_probs=expansion_probs
            )
            root_list.append(tree_root)
            # final_tree = tree_to_decision_nodes_dict(tree_root, model.num_lod)
            # tree_list.append(final_tree)

        with torch.no_grad():
            # Encode images
            image_latent = model.encode(images)

        for aug_idx in range(num_aug):
            batch_image_latent = image_latent[aug_idx].unsqueeze(0)
            batch_tree = root_list[aug_idx]
            ori_ordered_nodes = model._get_ordered_nodes(batch_tree)
            ordered_nodes = []
            for node in ori_ordered_nodes:
                if node.lod_level >= 3:
                    ordered_nodes.append(node)
            # Get tokens using the random quadtrees
            z_batch = model.selector._forward_reconstruction(batch_image_latent, ordered_nodes)

            _, result_dict = model.quantize(z_batch)
            code_incides = result_dict['min_encoding_indices'].squeeze()
            token_num += code_incides.shape[0]
            base_filename = image_key[0] + f"_{aug_idx}"
            
            final_tree = tree_to_decision_nodes_dict(batch_tree, model.num_lod)
            
            # Build parent index array over the token sequence
            # The latent sequence order matches the order used in selector._forward_optimize
            lod_indices, patch_incides = [], []

            for lod_idx, nodes in final_tree.items():
                if lod_idx >= 3:
                    for node in nodes:
                        lod_indices.append(node.lod_level)
                        patch_incides.append(node.patch_index)
            
            if batch_size > 1:
                class_id_data = class_id[i].item()
            else:
                class_id_data = class_id.item() if hasattr(class_id, 'item') else int(class_id)
            
            sample = {
                "__key__": base_filename,
                "code_indices.npy": code_incides.cpu().numpy(),
                "lod_indices.npy": np.array(lod_indices),
                "patch_indices.npy": np.array(patch_incides),
                "cls": str(class_id_data)
            }
            tar_writer.write(sample)
            num_samples += 1
        # print(f"Saved {batch_idx} samples")
        # if batch_idx == 100:
        #     break
    
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

