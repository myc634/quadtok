import os
import glob
import argparse
import webdataset as wds
from collections import OrderedDict
from tqdm import tqdm
from classes import IMAGENET2012_CLASSES

# python data/correct_cls.py --input_dir /mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet --output_dir /mnt/shared-storage-user/idc2-shared/dataset/preprocess/imagenet_new


def correct_labels_in_shards(input_dir, output_dir):
    """
    Reads all .tar shards from input_dir, corrects the 'cls' field,
    and writes new .tar shards to output_dir.
    """
    class_to_idx = {key: i for i, key in enumerate(IMAGENET2012_CLASSES.keys())}
    print(f"Loaded class map with {len(class_to_idx)} entries.")

    os.makedirs(output_dir, exist_ok=True)
    
    input_shards = sorted(glob.glob(os.path.join(input_dir, "imagenet-val*")))
    if not input_shards:
        print(f"Error: No .tar files found in {input_dir}")
        return
        
    print(f"Found {len(input_shards)} shards to process.")

    # <-- 1. WRAP THE OUTER LOOP WITH TQDM FOR OVERALL PROGRESS
    for input_shard_path in tqdm(input_shards, desc="Overall Progress"):
        shard_basename = os.path.basename(input_shard_path)
        output_shard_path = os.path.join(output_dir, shard_basename)

        if input_shard_path == output_shard_path:
            print(f"Error: Input and output paths are the same for {shard_basename}. Skipping.")
            continue
            
        dataset = wds.WebDataset(input_shard_path)
        
        with wds.TarWriter(output_shard_path) as writer:
            # <-- 2. THE INNER LOOP TQDM SHOWS PROGRESS PER FILE
            for sample in tqdm(dataset, desc=f"Processing {shard_basename}", leave=False):
                # breakpoint()
                label_str = sample["cls"].decode('utf-8')
                label_int = class_to_idx.get(label_str) # Use .get() for safety
                # breakpoint()
                if label_int is None:
                    print(f"Warning: Skipping sample with unknown label '{label_str}' in {shard_basename}")
                    continue

                new_sample = sample.copy()
                new_sample["cls"] = label_int
                writer.write(new_sample)
                
    print("\nProcessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correct string labels in WebDataset shards to integer labels."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing the original .tar shards."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where the corrected .tar shards will be saved."
    )
    args = parser.parse_args()
    
    correct_labels_in_shards(args.input_dir, args.output_dir)