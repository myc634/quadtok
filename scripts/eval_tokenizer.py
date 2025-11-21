"""Generate images using One-D-Piece.

# For Tokenizer Models
# Use `--length` to specify the number of tokens to generate.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --config configs/one-d-piece_s256.yaml \
        --length=128 \
        --output_dir generated/one_d_piece_s256_len128

# For TiTok Models
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --config configs/titok_b64.yaml \
        --output_dir generated/titok_b64

# For Image Formats
# Use `--png`, `--jp2`, `--jpg`, `--webp` to specify the image format.
# Also use `--save_raw` to save the raw images instead of numpy arrays.
# Use `--original` to save the original images.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --output_dir generated/jpeg_comp2 \
        --save_raw \
        --original \
        --jpg \
        --jpg_compression_rate=2 \

# For Saving Original Images
# Use `--original` to save the original images.
# Use `--labels` to save the labels of the images.
WANDB_MODE=offline accelerate launch \
    --mixed_precision=bf16 \
    --num_machines=1 \
    --num_processes=1 \
    --machine_rank=0 \
    --main_process_ip=127.0.0.1 \
    --main_process_port=12389 \
    --same_network \
    scripts/reconstruct_tokenizer.py \
        --output_dir generated/original \
        --original \
        --labels
"""

import os
import sys
from pathlib import Path
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from omegaconf import OmegaConf
import copy
import numpy as np
import torch
from types import SimpleNamespace

from utils.logger import setup_logger
from tqdm import tqdm
from accelerate.utils import set_seed
from accelerate import Accelerator

from utils.train_utils import create_model, create_dataloader, auto_resume, create_evaluator
from modeling.utils import build_quadtree, _copy_subtree_to_depth, _get_nodes_at_level, build_quadtree, build_random_quadtree, build_probabilistic_quadtree

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

@torch.no_grad()
def main(args):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    config = OmegaConf.load(args.work_dir + "/config.yaml")

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = config.experiment.output_dir + "/eval_outputs"
    os.makedirs(output_dir, exist_ok=True)
    logging_dir = os.path.join(output_dir, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
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

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    model = accelerator.prepare(model)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # Start training.
    logger.info(f""" Start evaluation of the model. """)
    config.experiment.resume = True
    _, _ = auto_resume(config, logger, accelerator, ema_model, -1, strict=True)

    accelerator.print(f"Evaluation of the checkpoint started.")
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    model.eval()

    evaluator = create_evaluator(config, logger, accelerator)

    # mkdir
    Path(output_dir).mkdir(exist_ok=True)

    count = 0
    generator = image_generator(config, logger, accelerator)

    def generate_tree_structure(max_depth_for_branch):
        pruned_tree_roots = []

        complete_tree_roots = copy.deepcopy(model.default_quadtree)

        for complete_root in complete_tree_roots:
            pruned_root = _copy_subtree_to_depth(complete_root, max_depth_for_branch)
            pruned_tree_roots.append((pruned_root, max_depth_for_branch))
            
        return pruned_tree_roots

    # tree_structure = generate_tree_structure(args.max_tree_depth)
    token_num = 0
    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1

        # step_tree_structure = copy.deepcopy(tree_structure)
        latent_feats = accelerator.unwrap_model(model).encode(image)
        tree_structure = build_probabilistic_quadtree(model.num_patch_side_list, guaranteed_depth=3, expansion_probs=[0.7, 0.4])
        # tree_structure = build_quadtree(model.num_patch_side_list)
        z = accelerator.unwrap_model(model).selector(latent_feats, tree_structure)
        token_num += z.shape[-1]
        if model.quantize_mode == "vae":
            z_quantized = accelerator.unwrap_model(model).quantize(z).sample()
        elif model.quantize_mode == "vq":
            z_quantized, model_dict = accelerator.unwrap_model(model).quantize(z)
        
        reconstructed_images = accelerator.unwrap_model(model).decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_structure)
        # reconstructed_images, model_dict = accelerator.unwrap_model(model)(image)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # Quantize to uint8
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        image = torch.clamp(image, 0.0, 1.0)
        # breakpoint()
        if model.quantize_mode == "vae":
            evaluator.update(image, reconstructed_images, None)
        else:
            evaluator.update(image, reconstructed_images, model_dict["min_encoding_indices"])
        # if count == 1000:
        #     break
    print(token_num / count)
    print(evaluator.result())

    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default=None, help="Path to the checkpoint if you want to use a specific checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max_tree_depth", type=int, default=2, help="how many lod are used")


    args = parser.parse_args()
    main(args)
