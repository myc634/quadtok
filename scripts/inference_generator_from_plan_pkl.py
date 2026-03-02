"""
从 pkl plan 做单张或批量图生成（batch_size=1）。
用 pkl 里的 lod_indices / patch_indices / class_id 替代 mar 内部的建树逻辑。
支持 --plan_pkl 单张 或 --plans-dir 遍历 tree_planning/plans 逐次生成并保存（文件名含 class_id、class_name）。
"""
import os
import re
import sys
import pickle
import argparse
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from omegaconf import OmegaConf

from utils.logger import setup_logger
from utils.train_utils import create_generater_tokenizer, create_model
from tree_planning.generate_with_plan import generate_with_plan


def _sanitize_for_filename(s: str, max_len: int = 40) -> str:
    """文件名安全：只保留字母数字下划线。"""
    s = re.sub(r"[^\w]", "_", s)
    return (s or "unknown")[:max_len]


def _collect_plan_paths(plans_dir: Path) -> list[tuple[Path, int, int, str]]:
    """
    扫描 plans_dir：目录名为 class_XXX_name，其下 plan_0.pkl, plan_1.pkl, ...
    返回 [(pkl_path, class_id, plan_idx, class_dir_name), ...]，按 class_id, plan_idx 排序。
    """
    out = []
    for class_dir in sorted(plans_dir.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue
        # class_001_goldfish -> 001
        parts = class_dir.name.split("_", 2)
        if len(parts) < 3:
            continue
        try:
            class_id = int(parts[1])
        except ValueError:
            continue
        for pkl_path in sorted(class_dir.glob("plan_*.pkl")):
            stem = pkl_path.stem  # plan_0
            try:
                plan_idx = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            out.append((pkl_path, class_id, plan_idx, class_dir.name))
    out.sort(key=lambda x: (x[1], x[2]))
    return out


def _plan_to_activation_map_image(plan: dict, cell_size: int = 16) -> np.ndarray:
    """
    从 plan 的 patch_indices 得到 16×16 activation map 的 RGB 图。
    patch_indices 前 64 为 8×8 顺序，之后为 16×16 上激活的 raster index。
    返回 (H, W, 3) uint8，红=激活，黑=未激活；总尺寸 (16*cell_size, 16*cell_size)。
    """
    patch_indices = plan["patch_indices"]
    if torch.is_tensor(patch_indices):
        patch_indices = patch_indices.cpu().tolist()
    # 16×16 激活位置 = patch_indices[64:]
    activated = set(patch_indices[64:]) if len(patch_indices) > 64 else set()
    grid = 16
    block = cell_size
    out = np.zeros((grid * block, grid * block, 3), dtype=np.uint8)
    for r in range(grid):
        for c in range(grid):
            idx = r * 16 + c
            color = (255, 0, 0) if idx in activated else (0, 0, 0)
            out[r * block : (r + 1) * block, c * block : (c + 1) * block] = color
    return out


def _concat_generation_and_map(gen_img: np.ndarray, map_img: np.ndarray) -> np.ndarray:
    """左：generation，右：activation map（resize 到与左同高）。"""
    h = gen_img.shape[0]
    map_pil = Image.fromarray(map_img)
    map_resized = map_pil.resize((h, h), Image.NEAREST)
    map_np = np.array(map_resized)
    return np.concatenate([gen_img, map_np], axis=1)


def _load_plan_and_meta(pkl_path: Path, imagenet_idx2classname: dict | None):
    """加载 plan，返回 (plan_dict, class_id, class_name)。class_name 优先用 pkl 里的，否则查表。"""
    with open(pkl_path, "rb") as f:
        plan = pickle.load(f)
    class_id = plan["class_id"]
    if isinstance(class_id, torch.Tensor):
        class_id = int(class_id.item())
    else:
        class_id = int(class_id)
    class_name = plan.get("class_name", "")
    if not class_name and imagenet_idx2classname is not None:
        class_name = imagenet_idx2classname.get(class_id, "unknown")
    return plan, class_id, class_name


def _generate_one_image(
    plan: dict,
    class_id: int,
    class_name: str,
    plan_idx: int,
    generator,
    tokenizer,
    device,
    gen_cfg,
    output_dir: Path,
    softmax_temperature_annealing: bool,
) -> Path | None:
    """对单个 plan 生成一张图并保存。返回保存路径或 None。plan 为已加载的 dict。"""
    lod_indices = plan["lod_indices"]
    patch_indices = plan["patch_indices"]
    if lod_indices.dim() == 1:
        lod_indices = lod_indices.unsqueeze(0)
        patch_indices = patch_indices.unsqueeze(0)
    tree_dict = dict(lod_indices=lod_indices, patch_indices=patch_indices)
    condition = torch.tensor([class_id], device=device, dtype=torch.long)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        generated_tokens, ordered_nodes = generate_with_plan(
            generator,
            condition,
            tree_dict,
            guidance_scale=gen_cfg.get("guidance_scale", 3.0),
            guidance_decay=gen_cfg.get("guidance_decay", "constant"),
            guidance_scale_pow=gen_cfg.get("guidance_scale_pow", 3.0),
            randomize_temperature=gen_cfg.get("randomize_temperature", 1.0),
            softmax_temperature_annealing=softmax_temperature_annealing,
        )

    if getattr(tokenizer, "quantize_mode", None) == "vq":
        bs, seq_len = generated_tokens.shape
        generated_tokens = tokenizer.quantize.get_codebook_entry(generated_tokens.flatten(0, 1).long()).view(bs, seq_len, -1)
        generated_image = tokenizer.decoder._forward_reconstruction(generated_tokens, ordered_nodes)
    else:
        generated_image = tokenizer.decode_tokens(generated_tokens.view(generated_tokens.shape[0], -1))

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    img = generated_image[0]

    name_safe = _sanitize_for_filename(class_name)
    out_name = f"class_{class_id:03d}_{name_safe}_plan_{plan_idx}.png"

    dir_generation = output_dir / "generation"
    dir_with_map = output_dir / "generation_with_map"
    dir_generation.mkdir(parents=True, exist_ok=True)
    dir_with_map.mkdir(parents=True, exist_ok=True)

    Image.fromarray(img).save((dir_generation / out_name).as_posix())
    map_img = _plan_to_activation_map_image(plan)
    combined = _concat_generation_and_map(img, map_img)
    Image.fromarray(combined).save((dir_with_map / out_name).as_posix())
    return dir_generation / out_name


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    logger = setup_logger(name="InferenceFromPlanPkl", log_level="INFO")

    if accelerator.is_main_process:
        logger.info("Loading config and model...")
    config = OmegaConf.load(args.config)

    tokenizer_checkpoint = config.tokenizer.get("tokenizer_ckpt_dir", None)
    if tokenizer_checkpoint is None:
        raise ValueError("tokenizer_checkpoint not found in config")

    if args.checkpoint is None:
        checkpoints = list(Path(config.experiment.output_dir).glob("checkpoint-*"))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        checkpoint = [x / "ema_model" / "pytorch_model.bin" for x in checkpoints][-1]
        if accelerator.is_main_process:
            logger.info(f"Using latest checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()
    elif args.checkpoint.isdigit():
        checkpoint = Path(config.experiment.output_dir) / f"checkpoint-{args.checkpoint}" / "ema_model" / "pytorch_model.bin"
        if accelerator.is_main_process:
            logger.info(f"Using checkpoint: {checkpoint}")
        args.checkpoint = checkpoint.as_posix()

    tokenizer = create_generater_tokenizer(config, logger, accelerator)
    generator, _ = create_model(config, logger, accelerator, model_type=config.model.generator_type)
    generator.eval()
    generator.requires_grad_(False)
    generator_weight = torch.load(args.checkpoint, map_location="cpu")
    generator.load_state_dict(generator_weight, strict=True)

    tokenizer = tokenizer.to(device)
    generator = generator.to(device)
    gen_cfg = config.model.generator

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.experiment.output_dir) / "inference_from_plan_pkl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plans_dir:
        # 批量：遍历 plans 目录下所有 class_*/plan_*.pkl
        plans_dir = Path(args.plans_dir)
        if not plans_dir.is_dir():
            raise SystemExit(f"Plans dir not found: {plans_dir}")
        try:
            from imagenet_classes import imagenet_idx2classname
        except ImportError:
            imagenet_idx2classname = None
        plan_list = _collect_plan_paths(plans_dir)
        if args.start_idx is not None or args.end_idx is not None:
            start = args.start_idx or 0
            end = args.end_idx if args.end_idx is not None else len(plan_list)
            plan_list = plan_list[start:end]
            if accelerator.is_main_process:
                logger.info(f"Slice plans[{start}:{end}] -> {len(plan_list)} plans")
        if accelerator.is_main_process:
            logger.info(f"Found {len(plan_list)} plans under {plans_dir}")
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x
        for pkl_path, class_id, plan_idx, class_dir_name in tqdm(plan_list, desc="generate"):
            plan, _, class_name = _load_plan_and_meta(pkl_path, imagenet_idx2classname)
            out_file = _generate_one_image(
                plan, class_id, class_name, plan_idx,
                generator, tokenizer, device, gen_cfg,
                output_dir, args.softmax_temperature_annealing,
            )
            if accelerator.is_main_process and out_file:
                logger.info(f"Saved: {out_file}")
        return

    # 单张：--plan_pkl
    with open(args.plan_pkl, "rb") as f:
        plan = pickle.load(f)
    plan, class_id, class_name = _load_plan_and_meta(Path(args.plan_pkl), None)
    lod_indices = plan["lod_indices"]
    patch_indices = plan["patch_indices"]
    if lod_indices.dim() == 1:
        lod_indices = lod_indices.unsqueeze(0)
        patch_indices = patch_indices.unsqueeze(0)
    tree_dict = dict(lod_indices=lod_indices, patch_indices=patch_indices)

    if accelerator.is_main_process:
        logger.info(f"Plan: {args.plan_pkl}, class_id: {class_id}, class_name: {class_name}")

    condition = torch.tensor([class_id], device=device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        generated_tokens, ordered_nodes = generate_with_plan(
            generator,
            condition,
            tree_dict,
            guidance_scale=gen_cfg.get("guidance_scale", 3.0),
            guidance_decay=gen_cfg.get("guidance_decay", "constant"),
            guidance_scale_pow=gen_cfg.get("guidance_scale_pow", 3.0),
            randomize_temperature=gen_cfg.get("randomize_temperature", 1.0),
            softmax_temperature_annealing=args.softmax_temperature_annealing,
        )

    if getattr(tokenizer, "quantize_mode", None) == "vq":
        bs, seq_len = generated_tokens.shape
        generated_tokens = tokenizer.quantize.get_codebook_entry(generated_tokens.flatten(0, 1).long()).view(bs, seq_len, -1)
        generated_image = tokenizer.decoder._forward_reconstruction(generated_tokens, ordered_nodes)
    else:
        generated_image = tokenizer.decode_tokens(generated_tokens.view(generated_tokens.shape[0], -1))

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    img = generated_image[0]

    out_name = args.output_name or f"class_{class_id:03d}_{_sanitize_for_filename(class_name)}_plan.png"
    dir_generation = output_dir / "generation"
    dir_with_map = output_dir / "generation_with_map"
    dir_generation.mkdir(parents=True, exist_ok=True)
    dir_with_map.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save((dir_generation / out_name).as_posix())
    map_img = _plan_to_activation_map_image(plan)
    combined = _concat_generation_and_map(img, map_img)
    Image.fromarray(combined).save((dir_with_map / out_name).as_posix())
    if accelerator.is_main_process:
        logger.info(f"Saved generation: {dir_generation / out_name}")
        logger.info(f"Saved generation+map: {dir_with_map / out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/generator/maskgit_one-d-piece_s256.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--plan_pkl", type=str, default=None, help="Single plan pkl (optional if --plans-dir is set)")
    parser.add_argument("--plans-dir", type=str, default=None, help="Batch: root dir of plans (e.g. tree_planning/plans)")
    parser.add_argument("--start-idx", type=int, default=None, help="Batch: only process plans[start_idx:end_idx] (for multi-GPU split)")
    parser.add_argument("--end-idx", type=int, default=None, help="Batch: only process plans[start_idx:end_idx]")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="", help="Single mode output filename")
    parser.add_argument("--softmax_temperature_annealing", action="store_true")
    args = parser.parse_args()
    if not args.plans_dir and not args.plan_pkl:
        parser.error("One of --plan_pkl or --plans-dir is required")
    main(args)
