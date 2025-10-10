# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
# obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import time
from pathlib import Path

import webdataset as wds
from PIL import Image


def convert_dataset_to_wds(
    input_dir, output_dir, max_samples_per_shard, split
):
    """
    Converts a dataset with a structure of 'split/class/image.jpeg' to WebDataset format.

    Args:
        input_dir (str): The root directory of the dataset.
        output_dir (str): The directory where the .tar files will be saved.
        max_samples_per_shard (int): The maximum number of samples per shard.
        split (str): The dataset split to process (e.g., "train" or "val").
    """
    split_dir = Path(input_dir)
    if not split_dir.is_dir():
        print(f"Directory for split '{split}' not found at: {split_dir}", file=sys.stderr)
        return

    # Define the output pattern for the shards
    opat = os.path.join(output_dir, f"{split}-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_samples_per_shard)

    print(f"Processing split: {split}")
    start_time = time.time()
    sample_count = 0

    # Find all image files (case-insensitive JPG/JPEG)
    image_paths = list(split_dir.glob("*/*.jpg")) + list(split_dir.glob("*/*.JPEG"))
    
    if not image_paths:
        print(f"No images found in {split_dir}", file=sys.stderr)
        output.close()
        return

    for i, image_path in enumerate(sorted(image_paths)):
        try:
            # The class label is the name of the parent directory
            label = int(image_path.parent.name)

            # The key is the file name without the extension
            key = image_path.stem

            # Open the image file
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()

            # Create the sample dictionary
            sample = {"__key__": key, "jpg": img_data, "cls": label}

            # Write the sample to the shard
            output.write(sample)
            sample_count += 1

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} samples...", file=sys.stderr)

        except Exception as e:
            print(f"Error processing {image_path}: {e}", file=sys.stderr)
            continue

    output.close()
    time_taken = time.time() - start_time
    print(
        f"Finished processing '{split}' split. Wrote {sample_count} samples in {time_taken:.2f} seconds."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an image dataset from a directory structure to WebDataset format."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory of the dataset (e.g., 'target_dir').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the .tar shards.",
    )
    parser.add_argument(
        "--max_train_samples_per_shard",
        type=int,
        default=5000,
        help="Max number of samples per train shard.",
    )
    parser.add_argument(
        "--max_val_samples_per_shard",
        type=int,
        default=1000,
        help="Max number of samples per validation shard.",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)


    # Process the validation split
    convert_dataset_to_wds(
        args.input_dir,
        args.output_dir,
        args.max_val_samples_per_shard,
        "val",
    )

    print("\nConversion complete.")
