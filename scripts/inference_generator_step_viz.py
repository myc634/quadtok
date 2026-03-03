import os
import sys
from pathlib import Path
import argparse
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from accelerate import Accelerator
from PIL import Image
from omegaconf import OmegaConf

from utils.logger import setup_logger
from utils.train_utils import create_generater_tokenizer, create_model


def _get_fill_token_id(tokenizer) -> int:
    """Best-effort token id to fill missing/un-generated positions."""
    for name in ("mask_token_id", "pad_token_id", "eos_token_id", "mask_id", "pad_id"):
        if hasattr(tokenizer, name):
            v = getattr(tokenizer, name)
            if isinstance(v, int):
                return v
    return 0


def _decode_from_tokens(tokenizer, tokens, tree=None):
    """Decode tokens (optionally with tree) into image tensor in [0,1]."""
    if tree is not None:
        # Quadtree/VQ path (matches demo_util.sample_fn)
        if getattr(tokenizer, "quantize_mode", None) == "vq" and tokens.dtype in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.uint8,
        ):
            bs, length = tokens.shape
            tokens = tokenizer.quantize.get_codebook_entry(tokens.flatten(0, 1).long()).view(bs, length, -1)
        return tokenizer.decoder._forward_reconstruction(tokens, tree)

    # Flat token path
    if tokens.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
        raise ValueError(f"Non-integer tokens without tree are not supported: dtype={tokens.dtype}, shape={tuple(tokens.shape)}")
    return tokenizer.decode_tokens(tokens.view(tokens.shape[0], -1))


def _to_uint8_numpy(img_t: torch.Tensor):
    """Convert BCHW float tensor in [0,1] to NHWC uint8 numpy."""
    img_t = torch.clamp(img_t, 0.0, 1.0)
    return (img_t * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def _prepare_prefix_tokens(tokens, prefix_len: int, fill_token_id: int):
    """
    Create a prefix-only token tensor (keep first prefix_len tokens, fill the rest).
    Returns (tokens_for_decode, total_token_len_for_percent).
    """
    if tokens.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
        flat = tokens.view(tokens.shape[0], -1)
        total_len = flat.shape[1]
        prefix_len = min(prefix_len, total_len)
        out = flat.clone()
        if prefix_len < total_len:
            out[:, prefix_len:] = fill_token_id
        return out, total_len

    # float embeddings: treat dim1 as "token length"
    if tokens.dim() < 2:
        raise ValueError(f"Unexpected float token shape: {tuple(tokens.shape)}")
    total_len = tokens.shape[1]
    prefix_len = min(prefix_len, total_len)
    out = tokens.clone()
    if prefix_len < total_len:
        out[:, prefix_len:, ...] = 0
    return out, total_len


def main(args):
    # Initialize accelerate
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    logger = setup_logger(name="Inference Generation", log_level="INFO")
    
    if accelerator.is_main_process:
        logger.info("Loading model model...")
    config = OmegaConf.load(args.config)

    tokenizer_checkpoint = config.tokenizer.get("tokenizer_ckpt_dir", None)
    if tokenizer_checkpoint is None:
        raise ValueError("tokenizer_checkpoint is not found in the config file.")

    
    if args.checkpoint is None:
        # search for the latest checkpoint
        # format: Path(config.experiment.output_dir) / "checkpoint-%d/ema_model/pytorch_model.bin"
        checkpoints = list(Path(config.experiment.output_dir).glob("checkpoint-*"))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint = [x / "ema_model" / "pytorch_model.bin" for x in checkpoints][-1]
        if accelerator.is_main_process:
            logger.info(f"Using the latest checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    elif args.checkpoint.isdigit():
        # search for the checkpoint with the given number
        checkpoint = Path(config.experiment.output_dir) / f"checkpoint-{args.checkpoint}" / "ema_model" / "pytorch_model.bin"
        if accelerator.is_main_process:
            logger.info(f"Using the checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()

    tokenizer = create_generater_tokenizer(config, logger, accelerator)

    generator, _ = create_model(config, logger, accelerator, model_type=config.model.generator_type)
    generator.eval()
    generator.requires_grad_(False)
    generator_weight = torch.load(args.checkpoint, map_location="cpu")
    msg =generator.load_state_dict(generator_weight, strict=True)
    print(msg)
    

    if accelerator.is_main_process:
        logger.info(f"Working on device: {device}")
    tokenizer = tokenizer.to(device)
    generator = generator.to(device)
    
    # Calculate how many samples each rank should generate
    num_samples = args.num_samples
    samples_per_rank = num_samples // world_size
    remainder = num_samples % world_size
    start_idx = rank * samples_per_rank + min(rank, remainder)
    end_idx = start_idx + samples_per_rank + (1 if rank < remainder else 0)
    local_num_samples = end_idx - start_idx
    
    # Auto-generate output directory name
    if args.output_dir is None:
        base_output_dir = Path(config.experiment.output_dir) / "inference_output"
        # Format: gs{guidance_scale}_gd{guidance_decay}_n{num_samples}
        # Sanitize guidance_decay string for filesystem
        guidance_decay_str = str(config.model.generator.guidance_decay).replace("/", "_").replace("\\", "_").replace(" ", "_")
        # Format guidance_scale to avoid too many decimal places
        gs_str = f"{config.model.generator.guidance_scale:.2f}".rstrip("0").rstrip(".")
        gs_pow_str = f"{config.model.generator.guidance_scale_pow:.2f}".rstrip("0").rstrip(".")
        output_dir_name = f"scale_{gs_str}_pow_{gs_pow_str}_decay_{guidance_decay_str}_temp_{config.model.generator.randomize_temperature:.2f}_num_{num_samples}"
        args.output_dir = base_output_dir / output_dir_name
    else:
        args.output_dir = Path(args.output_dir)
    
    # mkdir (only main process needs to create, but all processes should wait)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()  # Wait for all processes to ensure directory is created
    
    if accelerator.is_main_process:
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Total samples: {num_samples}, World size: {world_size}, Batch size: {args.batch_size}")
    logger.info(f"Rank {rank}: Generating {local_num_samples} samples (indices {start_idx} to {end_idx-1})")
    # Generate random classes for this rank (pre-generate all labels)
    
    
    # Generate samples in batches to avoid OOM
    batch_size = args.batch_size
    num_batches = (local_num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    fill_token_id = _get_fill_token_id(tokenizer)
    viz_dir = Path(args.output_dir) / "prefix_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, local_num_samples)
            current_batch_size = batch_end - batch_start
            batch_labels = torch.randint(0, config.model.generator.condition_num_classes, (current_batch_size,), device=device).long()
            # batch_labels = torch.ones_like(batch_labels) * 207
            if accelerator.is_main_process and batch_idx == 0:
                logger.info(f"Starting generation: {num_batches} batches, batch size: {batch_size}")
            logger.info(f"Rank {rank}: Generating batch {batch_idx + 1}/{num_batches} (samples {batch_start}-{batch_end-1})")
            # Generate full tokens once, then decode prefix variants.
            # Note: generate() should run without autocast to avoid kv_cache dtype mismatch
            generated = generator.generate(
                condition=batch_labels,
                guidance_scale=config.model.generator.guidance_scale,
                guidance_decay=config.model.generator.guidance_decay,
                guidance_scale_pow=config.model.generator.guidance_scale_pow,
                randomize_temperature=config.model.generator.randomize_temperature,
                softmax_temperature_annealing=args.softmax_temperature_annealing,
                num_sample_steps=args.num_sample_steps,
            )

            if isinstance(generated, tuple):
                generated_tokens, tree = generated
            else:
                generated_tokens, tree = generated, None

            # Determine token length for % milestones.
            if generated_tokens.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
                total_len = generated_tokens.view(generated_tokens.shape[0], -1).shape[1]
            else:
                total_len = generated_tokens.shape[1] if generated_tokens.dim() >= 2 else int(generated_tokens.numel())

            milestones = [
                ("t64", min(64, total_len)),
                ("p10", max(1, int(math.ceil(0.10 * total_len)))),
                ("p20", max(1, int(math.ceil(0.20 * total_len)))),
                ("p30", max(1, int(math.ceil(0.30 * total_len)))),
                ("p40", max(1, int(math.ceil(0.40 * total_len)))),
                ("p50", max(1, int(math.ceil(0.50 * total_len)))),
                ("p60", max(1, int(math.ceil(0.60 * total_len)))),
                ("p70", max(1, int(math.ceil(0.70 * total_len)))),
                ("p80", max(1, int(math.ceil(0.80 * total_len)))),
                ("p90", max(1, int(math.ceil(0.90 * total_len)))),
                ("full", total_len),
            ]

            # Decode and save (per-sample) for each milestone.
            # Use autocast only for decode operations to save memory
            # Define the order for the combined visualization: 10%, 20%, 64 token, 50%, 70%, 100%
            combined_order = ["p10", "p20", "t64", "p40", "p60", "p80", "full"]
            
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                # Store images in memory for each sample
                sample_images = {}  # {sample_idx: {tag: Image}}
                
                for tag, k in milestones:
                    tokens_k, _ = _prepare_prefix_tokens(generated_tokens, k, fill_token_id=fill_token_id)
                    img_t = _decode_from_tokens(tokenizer, tokens_k, tree=tree)
                    imgs_np = _to_uint8_numpy(img_t)
                    for i, img in enumerate(imgs_np):
                        local_idx = batch_start + i
                        # Create per-sample directory
                        sample_dir = viz_dir / f"{rank:02d}_{local_idx:04d}"
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        # Save to sample-specific folder: {rank:02d}_{local_idx:04d}/{tag}.png
                        path = sample_dir / f"{tag}.png"
                        img_pil = Image.fromarray(img)
                        img_pil.save(path.as_posix())
                        
                        # Store in memory for later concatenation
                        if local_idx not in sample_images:
                            sample_images[local_idx] = {}
                        sample_images[local_idx][tag] = img_pil

                # Also keep the original final-image saving behavior (full only) in output_dir root.
                full_tokens, _ = _prepare_prefix_tokens(generated_tokens, total_len, fill_token_id=fill_token_id)
                full_img_t = _decode_from_tokens(tokenizer, full_tokens, tree=tree)
                full_imgs_np = _to_uint8_numpy(full_img_t)
            
            # Save full images to output_dir root (outside autocast since it's just numpy arrays)
            for i, img in enumerate(full_imgs_np):
                local_idx = batch_start + i
                # Format: {rank:02d}_{local_idx:04d}.png
                path = Path(args.output_dir) / f"{rank:02d}_{local_idx:04d}.png"
                Image.fromarray(img).save(path.as_posix())
            
            # Create combined visualization for each sample
            for local_idx, img_dict in sample_images.items():
                # Get images in the specified order
                images_to_combine = []
                for tag in combined_order:
                    if tag in img_dict:
                        images_to_combine.append(img_dict[tag])
                
                if len(images_to_combine) == 0:
                    continue
                
                # Calculate dimensions for the combined image
                gap = 10  # Gap between images in pixels
                img_width = images_to_combine[0].width
                img_height = images_to_combine[0].height
                combined_width = len(images_to_combine) * img_width + (len(images_to_combine) - 1) * gap
                combined_height = img_height
                
                # Create combined image
                combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
                x_offset = 0
                for img in images_to_combine:
                    combined_img.paste(img, (x_offset, 0))
                    x_offset += img_width + gap
                
                # Save combined image
                sample_dir = viz_dir / f"{rank:02d}_{local_idx:04d}"
                combined_path = sample_dir / "combined.png"
                combined_img.save(combined_path.as_posix())
            logger.info(
                f"Rank {rank}: Saved batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples) "
                f"with prefix viz to {viz_dir}"
            )
            
            # Clear memory
            del generated, generated_tokens, tree, img_t, imgs_np, full_img_t, full_imgs_np, full_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Rank {rank} finished: saved {local_num_samples} images to {args.output_dir}")
    
    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generator/maskgit_one-d-piece_s256.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="If None, auto-generate from config and parameters")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--guidance_decay", type=str, default="constant")
    parser.add_argument("--guidance_scale_pow", type=float, default=3.0)
    parser.add_argument("--randomize_temperature", type=float, default=1.0)
    parser.add_argument("--softmax_temperature_annealing", action="store_true")
    parser.add_argument("--num_sample_steps", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation (to avoid OOM)")
    args = parser.parse_args()
    main(args)
