"""End-to-end pipeline val-loss probe (user's method):
  image -> DRIVE tokenizer (encode + guided tau=0.05 tree via probe256 _get_ordered_nodes order)
        -> per-node VQ codes (_forward_optimize, SAME order) -> pack varlen -> GEN.forward_varlen -> CE loss.

Uses the CORRECT node order (_get_ordered_nodes, i.e. what generate()/original extract expect),
NOT the tensor-native active_to_padded order. If drive gen -> ~7, the pretok token ORDER was the bug.

accelerate launch --num_processes 4 pipeline_val_loss.py --gen_ckpt <..> --size base --mlp_ratio 4 [--cap N]
"""
import os, sys, argparse, io, glob, tarfile, random
REPO = "/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch, torch.nn as nn
import lpips as lpips_pkg
from omegaconf import OmegaConf
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms
from modeling.quadtok import QuadTok
from modeling.mar import QuadtreeGPT
from modeling.utils import build_quadtree, _get_nodes_at_level
from data.augmentation import center_crop_arr
from scripts.probe256_2level_eval import (
    coarse_split_permutation, inverse_permutation, _build_tree_from_node_mask,
    ordered_ge3, recon_fast, CFG_DEFAULT, CKPT_DEFAULT, VAL_DIR,
)

_to_tensor = transforms.ToTensor()


def val_images(rank, world, bs, cap):
    tars = sorted(glob.glob(f"{VAL_DIR}/val-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    imgs, cls, n = [], [], 0
    for tar in mine:
        with tarfile.open(tar) as t:
            members = {}
            for m in t.getmembers():
                if "." not in m.name:
                    continue
                base, ext = m.name.rsplit(".", 1)
                members.setdefault(base, {})[ext] = m
            for base in sorted(members):
                d = members[base]
                jk = [k for k in d if k.lower() in ("jpg", "jpeg", "png")]
                if not jk or "cls" not in d:
                    continue
                try:
                    im = Image.open(io.BytesIO(t.extractfile(d[jk[0]]).read())).convert("RGB")
                    c = int(t.extractfile(d["cls"]).read().decode().strip())
                except Exception:
                    continue
                imgs.append(_to_tensor(center_crop_arr(im, 256))); cls.append(c); n += 1
                if len(imgs) == bs:
                    yield torch.stack(imgs), torch.tensor(cls); imgs, cls = [], []
                if cap and n >= cap:
                    if imgs:
                        yield torch.stack(imgs), torch.tensor(cls)
                    return
    if imgs:
        yield torch.stack(imgs), torch.tensor(cls)


@torch.no_grad()
def extract_codes_for_tree(tok, latent_i, node_list, ts):
    """Per-image code extraction (bs=1 -> no padding). Returns codes aligned to node_list order."""
    z = tok.selector._forward_optimize(latent_i, [node_list])
    _, rd = tok.quantize(z)
    codes = rd["min_encoding_indices"].squeeze().long().flatten()  # (len(node_list),)
    return codes[:len(node_list)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--mlp_ratio", type=int, default=1)
    ap.add_argument("--gen_config", default="configs/inference/gpt_16k_base.yaml")
    ap.add_argument("--tok_cfg", default=CFG_DEFAULT)
    ap.add_argument("--tok_ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--use_ema", action="store_true", help="load meta['ema'] if present")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--cap", type=int, default=500, help="per-rank image cap")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    acc = Accelerator(); dev = acc.device
    # ---- tokenizer (DRIVE tokenizer) ----
    tcfg = OmegaConf.load(a.tok_cfg); ts = int(tcfg.model.selector.token_size)
    tok = QuadTok(tcfg).eval().to(dev); tok.requires_grad_(False)
    tok.load_state_dict(torch.load(a.tok_ckpt, map_location="cpu"), strict=False)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()
    # ---- generator ----
    gcfg = OmegaConf.load(a.gen_config); OmegaConf.set_struct(gcfg, False)
    gcfg.model.generator.model_size = a.size
    gcfg.model.generator.mlp_ratio = a.mlp_ratio
    gcfg.model.grad_checkpointing = False
    gen = QuadtreeGPT(gcfg).to(dev).eval().requires_grad_(False)
    meta = torch.load(a.gen_ckpt, map_location="cpu")
    if isinstance(meta, dict) and a.use_ema and "ema" in meta:
        sd = meta["ema"]
    elif isinstance(meta, dict) and "ema" in meta:
        sd = meta["ema"]
    elif isinstance(meta, dict) and "model" in meta:
        sd = meta["model"]
    else:
        sd = meta
    msg = gen.load_state_dict(sd, strict=False)
    if acc.is_main_process:
        print(f"[gen] {os.path.basename(a.gen_ckpt)} size={a.size} mlp={a.mlp_ratio} "
              f"missing={len(msg.missing_keys)} unexp={len(msg.unexpected_keys)} "
              f"nparam={sum(p.numel() for p in gen.parameters())/1e6:.0f}M", flush=True)

    # ---- guided-search setup (probe256 convention -> _get_ordered_nodes order) ----
    npsl = list(tok.num_patch_side_list)
    base_tree = build_quadtree(npsl[:-1])
    target_nodes = _get_nodes_at_level(base_tree, 3)
    new_target_nodes = [target_nodes[i] for i in inverse_permutation(coarse_split_permutation())]
    pool = nn.AvgPool2d(32, 32)
    rng = random.Random(a.seed + acc.process_index)

    L, A, TOK = [], [], []
    for images, labels in val_images(acc.process_index, acc.num_processes, a.bs, a.cap):
        images = images.to(dev); labels = labels.to(dev); B = images.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = tok.encode(images)
            # one shared A/B split (probe256/original semantics)
            rpm = torch.zeros(64, dtype=torch.bool); rpm[rng.sample(range(64), 32)] = True
            tA = ordered_ge3(tok, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rpm))
            tB = ordered_ge3(tok, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, ~rpm))
            recA = recon_fast(tok, latent, tA); recB = recon_fast(tok, latent, tB)
            lpA = lpips_fn(images, recA, normalize=True).sum(1)
            lpB = lpips_fn(images, recB, normalize=True).sum(1)
            sign = rpm.int().clone(); sign[~rpm] = -1; sign8 = sign.reshape(8, 8).float().to(dev)
            # per-image guided tree + per-image code extraction (aligned order)
            codes_l, lod_l, patch_l, lens, labs = [], [], [], [], []
            for bi in range(B):
                pooled = pool((lpB[bi] - lpA[bi]).unsqueeze(0).unsqueeze(0))[0, 0]
                dmap = pooled * sign8
                m = (dmap >= a.tau).flatten().detach().cpu().bool()
                node_list = ordered_ge3(tok, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, m))
                codes = extract_codes_for_tree(tok, latent[bi:bi+1], node_list, ts)
                assert codes.shape[0] == len(node_list), (codes.shape, len(node_list))
                codes_l.append(codes)
                lod_l.append(torch.tensor([n.lod_level for n in node_list], device=dev, dtype=torch.long))
                patch_l.append(torch.tensor([n.patch_index for n in node_list], device=dev, dtype=torch.long))
                lens.append(len(node_list)); labs.append(int(labels[bi]))
            # pack varlen
            code_indices = torch.cat(codes_l).long()
            lod_indices = torch.cat(lod_l).long()
            patch_indices = torch.cat(patch_l).long()
            seqlens = torch.tensor(lens, device=dev, dtype=torch.int64)
            cu = torch.zeros(len(lens) + 1, device=dev, dtype=torch.int64)
            cu[1:] = torch.cumsum(seqlens, 0)
            max_seqlen = int(seqlens.max().item())
            labs_t = torch.tensor(labs, device=dev, dtype=torch.long)
            loss, ld = gen.forward_varlen(code_indices, lod_indices, patch_indices, labs_t,
                                          cu, seqlens, max_seqlen)
        L.append(float(ld["total_loss"])); A.append(float(ld["acc"])); TOK.append(float(seqlens.float().mean()))

    acc.wait_for_everyone()
    lt = acc.reduce(torch.tensor([sum(L), len(L)], device=dev, dtype=torch.float64), "sum")
    at = acc.reduce(torch.tensor([sum(A), len(A)], device=dev, dtype=torch.float64), "sum")
    tt = acc.reduce(torch.tensor([sum(TOK), len(TOK)], device=dev, dtype=torch.float64), "sum")
    if acc.is_main_process:
        print(f">>> [PIPELINE] {os.path.basename(a.gen_ckpt)} size={a.size} mlp={a.mlp_ratio} "
              f"CE_loss={lt[0]/lt[1]:.4f}  acc={at[0]/at[1]:.4f}  mean_tok={tt[0]/tt[1]:.1f}  "
              f"(order=_get_ordered_nodes/CORRECT, {int(lt[1])} batches/rank-sum)", flush=True)


if __name__ == "__main__":
    main()
