# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python data/convert_imagenet_to_wds.py --output_dir /mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet --max_train_samples_per_shard 40000
   

# Adapted from https://github.com/webdataset/webdataset-imagenet/blob/main/convert-imagenet.py

import argparse
import os
import sys
import time
import tarfile
import webdataset as wds
# from datasets import load_dataset
from pathlib import Path
import io
from PIL import Image
from tqdm import tqdm

DATA_DIR = "/mnt/shared-storage-user/idc2-shared/dataset/raw/Datasets--cutedataset--imagenet-1k/data"

def custom_generator(base_dir, split="train"):
    if split == "train":
        tar_paths = sorted(list(Path(base_dir).glob("train_images_*.tar.gz")))
    else:
        tar_paths = sorted(list(Path(base_dir).glob("val_images.tar.gz")))
    # breakpoint()
    for tar_path in tar_paths:
        print(f"Processing: {tar_path.name}")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member_info in tqdm(tar):
                if member_info.isfile() and (member_info.name.lower().endswith('.jpeg') or member_info.name.lower().endswith('.jpg')):
                    # breakpoint()
                    try:
                        label = member_info.name.split('_')[-1][:9]
                    except IndexError:
                        print(f"Warning: can't read label from filename: '{member_info.name}'. Skipping")
                        continue
                    # print(label)
                    image_file = tar.extractfile(member_info)
                    try:
                        image_bytes = image_file.read()
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        # breakpoint()
                    except Exception as e:
                        print(f"Warning: can't open: {member_info.name}, Error: {e}. Skipping")
                        continue
                    
                    yield {
                        "image": pil_image,
                        "label": label
                    }


def convert_imagenet_to_wds(output_dir, max_train_samples_per_shard, max_val_samples_per_shard):
    # assert not os.path.exists(os.path.join(output_dir, "imagenet-train-000000.tar"))
    # assert not os.path.exists(os.path.join(output_dir, "imagenet-val-000000.tar"))

    # opat = os.path.join(output_dir, "imagenet-train-%06d.tar")
    # output = wds.ShardWriter(opat, maxcount=max_train_samples_per_shard)
    # dataset = custom_generator(DATA_DIR, split="train")
    # now = time.time()
    # for i, example in enumerate(dataset):
    #     if i % max_train_samples_per_shard == 0:
    #         print(i, file=sys.stderr)
    #     img, label = example["image"], example["label"]
    #     output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    # output.close()
    # time_taken = time.time() - now
    # print(f"Wrote {i+1} train examples in {time_taken // 3600} hours.")

    opat = os.path.join(output_dir, "imagenet-val-%06d.tar")
    output = wds.ShardWriter(opat, maxcount=max_val_samples_per_shard)
    dataset = custom_generator(DATA_DIR, split="val")
    now = time.time()
    for i, example in enumerate(dataset):
        if i % max_val_samples_per_shard == 0:
            print(i, file=sys.stderr)
        img, label = example["image"], example["label"]
        output.write({"__key__": "%08d" % i, "jpg": img.convert("RGB"), "cls": label})
    output.close()
    time_taken = time.time() - now
    print(f"Wrote {i+1} val examples in {time_taken // 60} min.")


if __name__ == "__main__":
    # create parase object
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--max_train_samples_per_shard", type=int, default=4000, help="Path to the output directory.")
    parser.add_argument("--max_val_samples_per_shard", type=int, default=1000, help="Path to the output directory.")
    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    convert_imagenet_to_wds(args.output_dir, args.max_train_samples_per_shard, args.max_val_samples_per_shard)