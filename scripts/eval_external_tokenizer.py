import os
import sys
from pathlib import Path
import argparse
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from PIL import Image
from modeling.one_d_piece import OneDPiece
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

from eval.utils.evaluator import VQGANEvaluator
from data import SimpleImageDataset

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)


def image_generator(config, device):

    dataset_config = config.dataset.params
    preproc_config = preproc_config = config.dataset.preprocessing
    base_params = dict(
        num_train_examples=200000, # fake number here
        per_gpu_batch_size=32,
        global_batch_size=32,
        num_workers_per_gpu=12,
        resize_shorter_edge=preproc_config.resize_shorter_edge,
        crop_size=preproc_config.crop_size,
        random_crop=False,
        random_flip=False,
        normalize_mean=preproc_config.normalize_mean,
        normalize_std=preproc_config.normalize_mean,
    )

    dataset = SimpleImageDataset(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        **base_params,
    )

    count = 0
    for batch in dataset.eval_dataloader:
        for img_tensor, key, class_id in zip(batch['image'], batch['__key__'], batch['class_id']):
            img_tensor = img_tensor.to(device, memory_format=torch.contiguous_format, non_blocking=True, dtype=torch.float)
            count += 1
            assert tuple(img_tensor.shape) == (3, 256, 256), img_tensor.shape
            # yield img_tensor.unsqueeze(0).to(device), Path(key + ".png")
            yield img_tensor.unsqueeze(0), Path(f"image_{count-1:05d}.png"), key, class_id.unsqueeze(0).to(device)

def main(args):
    print("Loading model model...")
    config = OmegaConf.load(args.config)

    # model_weight = torch.load(config.ckpt_dir, map_location="cpu")

    if config.model.type == "vae-kl16": # eval ldm's vae
        from external_models.vae import AutoencoderKL
        model = AutoencoderKL(embed_dim=config.model.vae_embed_dim, ch_mult=config.model.ch_mult, ckpt_path=config.ckpt_dir)
        model.eval()
        model.requires_grad_(False)

        device = "cuda"
        print(f"Working on device: {device}")
        model = model.to(device)
    elif config.model.type == "sdxl-vae": # eval ldm's vae
        from diffusers.models import AutoencoderKL
        print("Start Loading")
        model = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae", local_files_only=True)
        model.eval()
        device = "cuda"
        model.requires_grad_(False)
        model = model.to(device)

    evaluator = VQGANEvaluator(
        device=torch.device("cuda:0"),
        enable_rfid=True,
        enable_inception_score=True,
        enable_codebook_usage_measure=False,
        enable_codebook_entropy_measure=False,
    )


    count = 0
    generator = image_generator(config, torch.device("cuda:0"))

    for image, image_path, image_key, class_id in tqdm(generator):
        count += 1
        if config.model.type == "vae-kl16":
            z = model.encode(image)
            reconstructed_images = model.decode(z.sample())
            reconstructed_images, image = (reconstructed_images + 1) / 2, (image + 1) / 2
        elif config.model.type == "sdxl-vae":
            reconstructed_images = model(image).sample
            reconstructed_images, image = (reconstructed_images + 1) / 2, (image + 1) / 2

        # Quantize to uint8
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        image = torch.clamp(image, 0.0, 1.0)
        
        # breakpoint()
        evaluator.update(image, reconstructed_images, None)

    print(evaluator.result())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/one-d-piece_s256.yaml")

    args = parser.parse_args()
    main(args)
