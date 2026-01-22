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
from PIL import Image
import torchvision.transforms as transforms
from accelerate import Accelerator
from utils.logger import setup_logger
from utils.train_utils import create_model
from modeling.utils import build_tree_from_decision_nodes, QuadTreeNode


def load_sample_from_tar(tar_path, sample_key=None):
    """Load a single sample from tar file."""
    dataset = wds.WebDataset(tar_path)
    
    for sample in dataset:
        key = sample['__key__']
        if sample_key is None or key == sample_key:
            code_indices = sample['code_indices.npy']
            lod_indices = sample['lod_indices.npy']
            patch_indices = sample['patch_indices.npy']
            class_id = sample['cls']
            
            return {
                'key': key,
                'code_indices': code_indices,
                'lod_indices': lod_indices,
                'patch_indices': patch_indices,
                'class_id': class_id
            }
    
    return None


def reconstruct_tree_from_indices(lod_indices, patch_indices, num_patch_side_list):
    """Reconstruct decision_nodes dict from lod_indices and patch_indices."""
    decision_nodes = defaultdict(list)
    
    for lod_idx, patch_idx in zip(lod_indices, patch_indices):
        node = QuadTreeNode(lod_level=int(lod_idx), patch_index=int(patch_idx))
        decision_nodes[int(lod_idx)].append(node)
    
    return dict(decision_nodes)


def get_ordered_nodes_from_indices(lod_indices, patch_indices):
    """Create ordered nodes list matching the order of saved indices."""
    ordered_nodes = []
    for lod_idx, patch_idx in zip(lod_indices, patch_indices):
        node = QuadTreeNode(lod_level=int(lod_idx), patch_index=int(patch_idx))
        ordered_nodes.append(node)
    return ordered_nodes


def visualize_code(model, code_indices, lod_indices, patch_indices, output_path, device):
    """Visualize a single sample from extracted codes."""
    model.eval()
    
    # Convert to tensors
    code_indices_tensor = torch.from_numpy(code_indices).long().to(device)
    
    # Create ordered nodes matching the exact order of saved indices
    # This ensures the order matches the code_indices
    ordered_nodes = get_ordered_nodes_from_indices(lod_indices, patch_indices)
    
    # Get codebook entries from code_indices
    # code_indices shape: (num_tokens,)
    # Need to reshape to (1, num_tokens) for batch dimension
    code_indices_batch = code_indices_tensor.unsqueeze(0)  # (1, num_tokens)
    
    # Get quantized features from codebook
    # get_codebook_entry expects (batch, num_tokens) or (num_tokens,)
    # Returns (batch, num_tokens, token_size) or (num_tokens, token_size)
    z_quantized = model.quantize.get_codebook_entry(code_indices_batch)  # (1, num_tokens, token_size)
    
    # Decode using decoder
    with torch.no_grad():
        # decoder._forward_reconstruction expects:
        # - z_quantized: (batch, seq_len, token_size)
        # - ordered_nodes: list of QuadTreeNode matching the sequence order
        reconstructed_image = model.decoder._forward_reconstruction(
            z_quantized, 
            ordered_nodes
        )
    
    # Convert to PIL Image and save
    # reconstructed_image shape: (1, 3, H, W)
    img_tensor = reconstructed_image[0].cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor)
    img_pil.save(output_path)
    
    print(f"Saved reconstructed image to {output_path}")
    return img_pil


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
        gradient_accumulation_steps=1,
        mixed_precision=config.training.mixed_precision,
        split_batches=False,
    )

    logger = setup_logger(
        name="VisualizeCode", 
        log_level="INFO", 
        output_file=f"{output_dir}/log{accelerator.process_index}.txt"
    )

    # Create and load model
    model, ema_model = create_model(
        config, logger, accelerator, model_type=config.model.type
    )

    logger.info("Loading Tokenizer Weight From: %s", args.tokenizer_weight)
    model_weight = torch.load(args.tokenizer_weight, map_location="cpu")
    model.load_state_dict(model_weight, strict=True)
    
    model.eval()
    model.requires_grad_(False)

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    device = accelerator.device

    # Load sample from tar file
    logger.info(f"Loading sample from {args.tar_path}")
    if args.sample_key:
        sample = load_sample_from_tar(args.tar_path, sample_key=args.sample_key)
    else:
        sample = load_sample_from_tar(args.tar_path)
    
    if sample is None:
        logger.error("No sample found!")
        return
    
    logger.info(f"Loaded sample: {sample['key']}")
    logger.info(f"Code indices shape: {sample['code_indices'].shape}")
    logger.info(f"LOD indices shape: {sample['lod_indices'].shape}")
    logger.info(f"Patch indices shape: {sample['patch_indices'].shape}")
    logger.info(f"Class ID: {sample['class_id']}")
    
    # Visualize
    output_path = os.path.join(output_dir, f"{sample['key']}_reconstructed.png")
    
    visualize_code(
        accelerator.unwrap_model(model),
        sample['code_indices'],
        sample['lod_indices'],
        sample['patch_indices'],
        output_path,
        device
    )
    
    logger.info("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, required=True, help="Config Path directory")
    parser.add_argument("--tokenizer_weight", type=str, required=True, help="Tokenizer weight path")
    parser.add_argument("--tar_path", type=str, required=True, help="Path to tar file containing extracted codes")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualization")
    parser.add_argument("--sample_key", type=str, default=None, help="Specific sample key to visualize (optional, uses first sample if not specified)")
    args = parser.parse_args()
    main(args)
