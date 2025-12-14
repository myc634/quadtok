#!/usr/bin/env python3
"""Evaluate FID metrics on ImageNet validation set.

This script:
1. Reads ImageNet validation data from webdataset tar files via rclone
2. Applies eval_transform to images
3. Saves processed images to a directory
4. Converts images to npz format
5. Computes FID, sFID, IS, Precision, and Recall metrics
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.webdataset_reader import ImageTransform
from utils.eval_utils import Evaluator, create_npz_from_sample_folder
import webdataset as wds


def filter_keys(key_set):
    """Filter dictionary to only include specified keys."""
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}
    return _f


def process_webdataset_to_images(
    shards_path,
    output_dir,
    resize_shorter_edge=256,
    crop_size=256,
    normalize_mean=[0., 0., 0.],
    normalize_std=[1., 1., 1.],
):
    """
    Process webdataset and save images with eval_transform applied.
    
    Args:
        shards_path: Path to webdataset shards (can include pipe: commands)
        output_dir: Directory to save processed images
        resize_shorter_edge: Size to resize shorter edge
        crop_size: Size to crop image
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ImageTransform with eval settings
    transform = ImageTransform(
        resize_shorter_edge=resize_shorter_edge,
        crop_size=crop_size,
        random_crop=False,  # Use center crop for eval
        random_flip=False,  # No flip for eval
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    
    # Create processing pipeline
    processing_pipeline = [
        wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
        wds.rename(
            image="jpg;png;jpeg;webp",
            class_id="cls",
            handler=wds.warn_and_continue,
        ),
        wds.map(filter_keys(set(["image", "class_id", "filename"]))),
        wds.map_dict(
            image=transform.eval_transform,
            class_id=lambda x: int(x),
            handler=wds.warn_and_continue,
        ),
    ]
    
    # Create webdataset pipeline
    pipeline = [
        wds.SimpleShardList(shards_path),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        *processing_pipeline,
    ]
    
    dataset = wds.DataPipeline(*pipeline)
    
    # Process and save images
    image_count = 0
    for sample in tqdm(dataset, desc="Processing images"):
        if "image" not in sample:
            continue
        
        # Get image tensor (already transformed)
        image_tensor = sample["image"]  # Shape: [C, H, W], normalized to [0, 1] or [-1, 1]
        
        # Convert tensor to numpy and denormalize
        # Normalize transform: (image - mean) / std
        # To reverse: image = image * std + mean
        image_np = image_tensor.numpy().transpose(1, 2, 0)  # [H, W, C]
        
        # Denormalize: reverse the normalization
        # After denormalize, image should be in [0, 1] range (assuming input was [0, 1])
        mean_array = np.array(normalize_mean).reshape(1, 1, 3)
        std_array = np.array(normalize_std).reshape(1, 1, 3)
        image_np = image_np * std_array + mean_array
        
        # Clip to [0, 1] and convert to [0, 255] uint8
        image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert to PIL Image and save
        image_pil = Image.fromarray(image_np)
        output_path = output_dir / f"{image_count:06d}.png"
        image_pil.save(output_path)
        
        image_count += 1
    
    print(f"Processed and saved {image_count} images to {output_dir}")
    return image_count


def compute_fid_metrics(
    ref_batch_path,
    sample_batch_path,
    output_txt_path=None,
):
    """
    Compute FID, sFID, IS, Precision, and Recall metrics.
    
    Args:
        ref_batch_path: Path to reference batch npz file
        sample_batch_path: Path to sample batch npz file
        output_txt_path: Optional path to save results text file
    """
    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))
    
    print("Warming up TensorFlow...")
    evaluator.warmup()
    
    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_batch_path)
    print("Computing/reading reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch_path, ref_acts)
    
    print("Computing sample batch activations...")
    sample_acts = evaluator.read_activations(sample_batch_path)
    print("Computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(
        sample_batch_path, sample_acts
    )
    
    print("Computing evaluations...")
    IS = evaluator.compute_inception_score(sample_acts[0])
    FID = sample_stats.frechet_distance(ref_stats)
    sFID = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Inception Score: {IS:.4f}")
    print(f"FID: {FID:.4f}")
    print(f"sFID: {sFID:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print("="*50)
    
    # Save results to text file
    if output_txt_path is None:
        output_txt_path = str(Path(sample_batch_path).with_suffix(".txt"))
    
    print(f"\nWriting results to {output_txt_path}")
    with open(output_txt_path, "w") as f:
        print("Inception Score:", IS, file=f)
        print("FID:", FID, file=f)
        print("sFID:", sFID, file=f)
        print("Precision:", prec, file=f)
        print("Recall:", recall, file=f)
    
    return FID, sFID, IS, prec, recall


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FID metrics on ImageNet validation set"
    )
    parser.add_argument(
        "--shards_path",
        type=str,
        default="pipe:rclone cat hoss:jianglihan/data/imagenet/val-{000000..000049}.tar",
        help="Path to webdataset shards (can include pipe: commands)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--ref_batch_path",
        type=str,
        default="/mnt/petrelfs/jianglihan/my_code/quadtok/pretrained_weight/VIRTUAL_imagenet256_labeled.npz",
        help="Path to reference batch npz file",
    )
    parser.add_argument(
        "--resize_shorter_edge",
        type=int,
        default=256,
        help="Size to resize shorter edge",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="Size to crop image",
    )
    parser.add_argument(
        "--normalize_mean",
        type=float,
        nargs=3,
        default=[0., 0., 0.],
        help="Normalization mean (3 values for RGB)",
    )
    parser.add_argument(
        "--normalize_std",
        type=float,
        nargs=3,
        default=[1., 1., 1.],
        help="Normalization std (3 values for RGB)",
    )
    parser.add_argument(
        "--skip_processing",
        action="store_true",
        help="Skip image processing step (use existing images in output_dir)",
    )
    parser.add_argument(
        "--skip_npz",
        action="store_true",
        help="Skip npz conversion step (use existing npz file)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Maximum number of samples to process (for npz conversion)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Process webdataset and save images
    if not args.skip_processing:
        print("Step 1: Processing webdataset and saving images...")
        process_webdataset_to_images(
            shards_path=args.shards_path,
            output_dir=output_dir,
            resize_shorter_edge=args.resize_shorter_edge,
            crop_size=args.crop_size,
            normalize_mean=args.normalize_mean,
            normalize_std=args.normalize_std,
        )
    else:
        print("Step 1: Skipping image processing (using existing images)")
    
    # Step 2: Convert images to npz
    npz_path = str(output_dir) + ".npz"
    if not args.skip_npz:
        print(f"\nStep 2: Converting images to npz format...")
        create_npz_from_sample_folder(
            sample_dir=str(output_dir),
            num=args.num_samples,
        )
    else:
        print(f"\nStep 2: Skipping npz conversion (using existing {npz_path})")
        if not Path(npz_path).exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Step 3: Compute FID metrics
    print(f"\nStep 3: Computing FID metrics...")
    compute_fid_metrics(
        ref_batch_path=args.ref_batch_path,
        sample_batch_path=npz_path,
        output_txt_path=str(output_dir / "fid_results.txt"),
    )
    
    print(f"\nAll steps completed!")
    print(f"Processed images: {output_dir}")
    print(f"NPZ file: {npz_path}")
    print(f"Results: {output_dir / 'fid_results.txt'}")


if __name__ == "__main__":
    main()

