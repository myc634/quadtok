# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import os
from omegaconf import OmegaConf

import sys
from utils.distributed import init_distributed_mode
from data.augmentation import center_crop_arr
from data.imagenet import build_imagenet
from PIL import Image
from tqdm import tqdm
from modeling.quadtok import QuadTok, PolicyQuadTok
import json
from modeling.utils import QuadTreeNode
import copy
from modeling.utils import build_quadtree, build_random_quadtree, build_probabilistic_quadtree
from collections import defaultdict

def encode(model, image):
    z_quantized, result_dict = model.encode(image)
    encoded_tokens = result_dict["min_encoding_indices"]
    return encoded_tokens

def tree_to_decision_nodes_dict(tree_root, num_lod):
    """
    Convert tree root to decision_nodes dictionary format.
    Returns: dict {lod_idx: [list of QuadTreeNode]}
    """
    decision_nodes = defaultdict(list)
    queue = [tree_root]
    
    while queue:
        node = queue.pop(0)
        if node.lod_level < num_lod:
            decision_nodes[node.lod_level].append(node)
        for child in node.children:
            queue.append(child)
    
    return dict(decision_nodes)


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = 'cuda'
        rank = 0
    
    # Setup a feature folder:
    if args.debug or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_quadtrees'), exist_ok=True)

    # create and load model
    # create and load model
    model_weight = torch.load(args.vq_ckpt, map_location="cpu")
    config = OmegaConf.load(args.vq_config)
    vq_model = QuadTok(config)
    vq_model.load_state_dict(model_weight)
    vq_model.to(device)
    vq_model.eval()
    vq_model.requires_grad_(False)


    # Setup data:
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range)
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            transforms.Normalize(mean=[0., 0., 0.], std=[1.0, 1.0, 1.0], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            transforms.Normalize(mean=[0., 0., 0.], std=[1.0, 1.0, 1.0], inplace=True)
        ])
    dataset = build_imagenet(args, transform=transform)

    if not args.debug:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    total = 0
    patches_per_side_list = [1, 2, 4, 8, 16, 32]
    guaranteed_depth = 3
    expansion_probs = [0.3, 0.4]
    num_lod = len(patches_per_side_list)
    for x, y in loader:
        x = x.to(device)
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
        
        tree_list = []
        for _ in range(num_aug):
            tree_root = build_probabilistic_quadtree(
                patches_per_side_list, 
                guaranteed_depth=guaranteed_depth, 
                expansion_probs=expansion_probs
            )
            final_tree = tree_to_decision_nodes_dict(tree_root, num_lod)
            tree_list.append(final_tree)
        
        y = y.to(device)
        with torch.no_grad():
            image_latent = vq_model.encode(x_all)
            z_batch = vq_model.selector._forward_optimize(image_latent, tree_list)
            _, result_dict = vq_model.quantize(z_batch)
            input_tokens = result_dict['min_encoding_indices'].squeeze(1)

            # Commented below is for debugging (do reconstruction)
            '''bs, length = input_tokens.shape
            processed_tokens = vq_model.quantize.get_codebook_entry(input_tokens.flatten(0, 1).long()).view(bs, length, -1)
            reconstructed_images = vq_model.decoder._forward_optimize(processed_tokens, tree_list)

        for i in range(reconstructed_images.shape[0]):
            concat_image = torch.cat([x_all[i], reconstructed_images[i]], dim=1)
            Image.fromarray((concat_image.clamp(0, 1) * 255.0).permute(1,2,0).cpu().numpy().astype(np.uint8)).save(f"temp_{i}.png")'''
        
        # samples = vq_model.decode_tokens(indices)
        codes = input_tokens.reshape(x.shape[0], num_aug, -1)

        x = codes.detach().cpu().numpy()    # (1, num_aug, args.image_size//16 * args.image_size//16)
        train_steps = rank + total
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy', x)

        y = y.detach().cpu().numpy()    # (1,)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy', y)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_quadtrees/{train_steps}.npy', tree_list)

        if not args.debug:
            total += dist.get_world_size()
        else:
            total += 1
        print(total)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/mnt/ultracube/datasets/imagenet/ILSVRC/Data/CLS-LOC/train")
    parser.add_argument("--code-path", type=str, default="/mnt/ultracube/zec016/quadtok_codes/quadtree_fixtree")
    parser.add_argument("--vq-config", type=str, default="/mnt/ultracube/zec016/quadtok/configs/training/generator/gpt_quadtree_fixtree.yaml")
    parser.add_argument("--vq-ckpt", type=str, default="/mnt/ultracube/zec016/quadtok/vq_ckpts/tokenizer_ckpt.bin", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--ten-crop", action='store_true', help="whether using random crop")
    parser.add_argument("--crop-range", type=float, default=1.1, help="expanding range of center crop")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
