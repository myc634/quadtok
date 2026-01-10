from PIL import Image
import numpy as np
from tqdm import tqdm
import os


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    image_names = os.listdir(sample_dir)
    for image_name in tqdm(image_names, desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{image_name}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    # assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

# create_npz_from_sample_folder("/mnt/ultracube/zec016/quadtok/inference_output/maskgit_quadtree_fixtree_inference_default_guidance", num=50000)
create_npz_from_sample_folder("/mnt/ultracube/zec016/quadtok/checkpoints/generator/maskgit_quadtree_fixtree/inference_output/scale_6.9_pow_3_decay_power-cosine_temp_2.80_steps_64_num_6250", num=50000)

