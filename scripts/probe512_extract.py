"""Distributed pretokenization for the 512 2-level generator.

Runs the frozen 512 tokenizer + guided LPIPS-A/B probing at the chosen operating point (tau=0.03,
~1010 tok/img) over ImageNet **train**, and saves per sample the tree structure + token ids:
  code_indices.npy (VQ code per token), lod_indices.npy, patch_indices.npy, cls
to WebDataset tars, sharded per rank (each GPU handles tars where tar_idx % world == rank).
Fast: uses the flex-accelerated _forward_optimize; extract needs only 1 A/B recon (for the
benefit map) + 1 selector+quantize (for the codes) per image (no guided-tree decoder pass).

Launch:  accelerate launch --num_processes 8 scripts/probe512_extract.py --tau 0.03 --out <dir> --ntars_per_rank 3
"""
import io, glob, tarfile, random, argparse, os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import lpips as lpips_pkg
import webdataset as wds
from omegaconf import OmegaConf
from PIL import Image
from accelerate import Accelerator
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, _create_and_assign_children, QuadTreeNode

CFG = "configs/training/single_stage/quadtok_ss512_vq_2level.yaml"
EMA = "/sensei-fs-3/users/yuchengm/models/quadtok_512_280k/Tokenizer/512/checkpoint-280000/ema_model/pytorch_model.bin"
TRAIN_DIR = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/train"


def train_stream(rank, world, bs, ntars_per_rank=None, size=512):
    """Yield (imgs[0,1], cls[list[int]], keys[list[str]]) from train tars where tar%world==rank."""
    tars = sorted(glob.glob(f"{TRAIN_DIR}/train-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    if ntars_per_rank:
        mine = mine[:ntars_per_rank]
    imgs, cls, keys = [], [], []
    for tar in mine:
        with tarfile.open(tar) as t:
            members = {m.name: m for m in t.getmembers()}
            for name, m in members.items():
                if not name.endswith(".jpg"):
                    continue
                key = name[:-4]
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                w, h = im.size; s = size / min(w, h)
                im = im.resize((max(size, round(w * s)), max(size, round(h * s))))
                w, h = im.size; l, tp = (w - size) // 2, (h - size) // 2
                im = im.crop((l, tp, l + size, tp + size))
                cm = members.get(key + ".cls")
                c = int(t.extractfile(cm).read().decode().strip()) if cm else 0
                imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
                cls.append(c); keys.append(key)
                if len(imgs) == bs:
                    yield torch.stack(imgs), cls, keys; imgs, cls, keys = [], [], []
    if imgs:
        yield torch.stack(imgs), cls, keys


def build_split_tree(npsl, coarse_lod, mask_bool):
    base = build_quadtree(list(npsl[:-1]))
    to_expand = set(torch.nonzero(mask_bool, as_tuple=True)[0].tolist())

    def ce(nd):
        x = QuadTreeNode(nd.lod_level, nd.patch_index)
        if nd.lod_level == coarse_lod and nd.patch_index in to_expand:
            _create_and_assign_children(x, list(npsl))
        elif nd.children:
            for c in nd.children:
                x.children.append(ce(c))
        return x
    return ce(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ntars_per_rank", type=int, default=None, help="cap tars/rank (smoke)")
    args = ap.parse_args()

    acc = Accelerator(); dev = acc.device
    cfg = OmegaConf.load(CFG); ts = int(cfg.model.selector.token_size)
    model = QuadTok(cfg).eval().to(dev)
    model.load_state_dict(torch.load(EMA, map_location="cpu"), strict=False)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()

    npsl = list(model.num_patch_side_list)
    gd = model.guaranteed_depth
    ncs = npsl[gd]; n_coarse = ncs ** 2
    cell_px = cfg.dataset.preprocessing.crop_size // ncs
    pool = nn.AvgPool2d(cell_px, cell_px)

    os.makedirs(args.out, exist_ok=True)
    tar_path = os.path.join(args.out, f"pretok-rank{acc.process_index:03d}.tar")
    writer = wds.TarWriter(tar_path)
    rng = random.Random(4321 + acc.process_index)
    n_written, tok_sum = 0, 0

    def ordered(root):
        return [n for n in model._get_ordered_nodes(root) if n.lod_level >= gd]

    for imgs, clss, keys in train_stream(acc.process_index, acc.num_processes, args.bs, args.ntars_per_rank):
        imgs = imgs.to(dev); B = imgs.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(imgs)
            mA = torch.zeros(n_coarse, dtype=torch.bool); mA[rng.sample(range(n_coarse), n_coarse // 2)] = True
            tA = ordered(build_split_tree(npsl, gd, mA)); tB = ordered(build_split_tree(npsl, gd, ~mA))
            ab = [deepcopy(tA) for _ in range(B)] + [deepcopy(tB) for _ in range(B)]
            zab = model.selector._forward_optimize(latent.repeat(2, 1, 1), ab)
            _, rdab = model.quantize(zab)
            emb = model.quantize.get_codebook_entry(rdab["min_encoding_indices"].squeeze().long().flatten()).reshape(2 * B, -1, ts)
            recab = model.decoder._forward_optimize(emb, ab).clamp(0, 1).float()
            lp = lpips_fn(imgs.repeat(2, 1, 1, 1), recab, normalize=True).sum(1)
            sign = torch.where(mA, 1, -1).float().to(dev).reshape(ncs, ncs)
            benefit = pool((lp[B:] - lp[:B]).unsqueeze(1))[:, 0] * sign  # (B,ncs,ncs)

            gtrees = [ordered(build_split_tree(npsl, gd, (benefit[bi].flatten() >= args.tau).cpu())) for bi in range(B)]
            zg = model.selector._forward_optimize(latent, gtrees)
            _, rdg = model.quantize(zg)
            codes = rdg["min_encoding_indices"][:, 0]  # (B, max_len)

        for bi in range(B):
            nodes = gtrees[bi]; L = len(nodes)
            writer.write({
                "__key__": keys[bi].replace("/", "_"),
                "code_indices.npy": codes[bi][:L].cpu().numpy().astype(np.int32),
                "lod_indices.npy": np.array([n.lod_level for n in nodes], dtype=np.int16),
                "patch_indices.npy": np.array([n.patch_index for n in nodes], dtype=np.int32),
                "cls": str(clss[bi]),
            })
            n_written += 1; tok_sum += L
        if acc.is_main_process and n_written % (args.bs * 5) < args.bs:
            print(f"[rank0] wrote {n_written} (mean_tok {tok_sum/max(n_written,1):.0f})", flush=True)

    writer.close()
    print(f"[rank{acc.process_index}] DONE wrote {n_written} samples, mean_tok {tok_sum/max(n_written,1):.1f} -> {tar_path}", flush=True)


if __name__ == "__main__":
    main()
