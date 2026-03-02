import random
import os
import sys
from pathlib import Path
import argparse
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from accelerate import Accelerator
from PIL import Image
from modeling.titok import TiTok
from modeling.one_d_piece import OneDPiece
from modeling.maskgit import ImageBert
from omegaconf import OmegaConf
import numpy as np
from demo_util import sample_fn

from utils.logger import setup_logger
from utils.train_utils import create_generater_tokenizer, create_model
from utils.viz_utils import make_viz_from_samples, make_viz_from_samples_generation


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
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in '{config.experiment.output_dir}'. "
                "Please pass --checkpoint <path> explicitly."
            )
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

    # Throughput tracking (skip first batch as warm-up)
    timed_samples = 0
    timed_seconds = 0.0
    WARMUP_BATCHES = 1

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True) and torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, local_num_samples)
            current_batch_size = batch_end - batch_start
            batch_labels = torch.randint(0, config.model.generator.condition_num_classes, (current_batch_size,), device=device).long()
            # batch_labels = torch.ones_like(batch_labels) * 207
            if accelerator.is_main_process and batch_idx == 0:
                logger.info(f"Starting generation: {num_batches} batches, batch size: {batch_size}")
            logger.info(f"Rank {rank}: Generating batch {batch_idx + 1}/{num_batches} (samples {batch_start}-{batch_end-1})")

            # --- timed forward pass ---
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            # Generate batch
            result = sample_fn(
                generator,
                tokenizer,
                labels=batch_labels,
                guidance_scale=config.model.generator.guidance_scale,
                guidance_decay=config.model.generator.guidance_decay,
                guidance_scale_pow=config.model.generator.guidance_scale_pow,
                randomize_temperature=config.model.generator.randomize_temperature,
                softmax_temperature_annealing=args.softmax_temperature_annealing,
                num_sample_steps=args.num_sample_steps,
                device=device,
                return_tensor=False
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - t0

            # Skip warm-up batch from throughput stats
            if batch_idx >= WARMUP_BATCHES:
                timed_samples += current_batch_size
                timed_seconds += batch_elapsed
            batch_throughput = current_batch_size / batch_elapsed
            logger.info(
                f"Rank {rank}: Batch {batch_idx + 1}/{num_batches} — "
                f"{batch_elapsed:.3f}s, {batch_throughput:.2f} img/s"
                + (" [warm-up, excluded from stats]" if batch_idx < WARMUP_BATCHES else "")
            )
        

            # Save each image immediately after generation
            for i, img in enumerate(result):
                local_idx = batch_start + i
                # Format: {rank:02d}_{local_idx:04d}.png
                path = Path(args.output_dir) / f"{rank:02d}_{local_idx:04d}.png"
                Image.fromarray(img).save(path.as_posix())
            # breakpoint()
            logger.info(f"Rank {rank}: Saved batch {batch_idx + 1}/{num_batches} ({len(result)} images)")
            
            # Clear memory
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Rank {rank} finished: saved {local_num_samples} images to {args.output_dir}")

        # ---- per-rank throughput summary ----
        if timed_samples > 0:
            per_rank_throughput = timed_samples / timed_seconds
            logger.info(
                f"Rank {rank} throughput: {per_rank_throughput:.2f} img/s "
                f"({timed_samples} samples in {timed_seconds:.2f}s, warm-up of {WARMUP_BATCHES} batch(es) excluded)"
            )
        else:
            per_rank_throughput = 0.0
            logger.info(f"Rank {rank}: not enough batches to measure throughput (only {num_batches} batch(es) total).")

    # Wait for all processes to finish saving
    accelerator.wait_for_everyone()

    # ---- aggregate throughput across all ranks ----
    throughput_tensor = torch.tensor([per_rank_throughput], device=device, dtype=torch.float64)
    all_throughputs = accelerator.gather(throughput_tensor)
    if accelerator.is_main_process:
        total_throughput = all_throughputs.sum().item()
        mean_per_rank = all_throughputs.mean().item()
        logger.info(
            f"=== Throughput Summary ==="
            f"\n  World size       : {world_size}"
            f"\n  Per-rank avg     : {mean_per_rank:.2f} img/s"
            f"\n  Total (all ranks): {total_throughput:.2f} img/s"
        )


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
