"""Distributed ImageNet-val rFID/PSNR eval for the 2-level 256x256 QuadTok tokenizer.

Faithful port of the submission single-stage LPIPS A/B search
(scripts/extract_code_searchquadtree.py) at a fixed threshold tau=0.05, GUIDED ONLY.
The original search logic (coarse_split_permutation, _build_tree_from_node_mask,
AvgPool(32)->8x8 signed benefit, tau gate) is copied verbatim so the guided tree is
identical to the paper method; we only add: val streaming, decode of the guided tree,
and VQGANEvaluator (rFID/PSNR/IS) aggregated across ranks.

Launch:  accelerate launch --num_processes 4 scripts/probe256_2level_eval.py --tau 0.05 [--cap N]
"""
import io, os, sys, glob, tarfile, random, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import lpips as lpips_pkg
from omegaconf import OmegaConf
from PIL import Image
from accelerate import Accelerator
from modeling.quadtok import QuadTok
from modeling.utils import (
    build_quadtree, _get_nodes_at_level, _create_and_assign_children, QuadTreeNode,
)
from eval.utils.evaluator import VQGANEvaluator

CFG_DEFAULT = "configs/training/single_stage/quadtok_ss256_vq.yaml"
CKPT_DEFAULT = "/sensei-fs-3/users/yuchengm/models/quadtok_2level_drive/drive_ckpt.download"
VAL_DIR = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/val"


# ----------------------------- original helpers (verbatim) -----------------------------
def coarse_split_permutation(coarse_hw=(4, 4), split_hw=(2, 2)):
    Hc, Wc = coarse_hw
    Hs, Ws = split_hw
    Hf, Wf = Hc * Hs, Wc * Ws
    perm = []
    for cy in range(Hc):
        for cx in range(Wc):
            for sy in range(Hs):
                for sx in range(Ws):
                    fy = cy * Hs + sy
                    fx = cx * Ws + sx
                    perm.append(fy * Wf + fx)
    return perm


def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _build_tree_from_node_mask(base_tree, target_lod, max_lod, patches_per_side_list, target_nodes, node_mask):
    expand_patch_indices = {
        target_nodes[i].patch_index
        for i in range(len(target_nodes))
        if node_mask[target_nodes[i].patch_index]
    }

    def _copy_and_expand(node):
        new_node = QuadTreeNode(node.lod_level, node.patch_index)
        if node.lod_level == target_lod and node.patch_index in expand_patch_indices:
            if node.lod_level < max_lod:
                _create_and_assign_children(new_node, patches_per_side_list)
        elif node.children:
            for child in node.children:
                new_node.children.append(_copy_and_expand(child))
        return new_node

    return _copy_and_expand(base_tree)


# ----------------------------- val streaming (resize-shorter + center crop, [0,1]) ------
def val_stream(rank, world, bs, size=256, cap=None):
    tars = sorted(glob.glob(f"{VAL_DIR}/val-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    buf, n = [], 0
    for tar in mine:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not (m.name.endswith(".jpg") or m.name.endswith(".jpeg") or m.name.endswith(".png") or m.name.endswith(".JPEG")):
                    continue
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                w, h = im.size
                s = size / min(w, h)
                im = im.resize((max(size, round(w * s)), max(size, round(h * s))))
                w, h = im.size
                l, tp = (w - size) // 2, (h - size) // 2
                im = im.crop((l, tp, l + size, tp + size))
                buf.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
                n += 1
                if len(buf) == bs:
                    yield torch.stack(buf); buf = []
                if cap and n >= cap:
                    if buf:
                        yield torch.stack(buf)
                    return
    if buf:
        yield torch.stack(buf)


@torch.no_grad()
def recon_opt(model, latent, node_lists, ts):
    """selector._forward_optimize -> quantize -> decoder._forward_optimize -> [0,1] images."""
    z = model.selector._forward_optimize(latent, node_lists)
    _, rd = model.quantize(z)
    ti = rd["min_encoding_indices"].squeeze().long().flatten()
    embeds = model.quantize.get_codebook_entry(ti).reshape(len(node_lists), -1, ts)
    return model.decoder._forward_optimize(embeds, node_lists).clamp(0, 1).float()


def ordered_ge3(model, root):
    return [n for n in model._get_ordered_nodes(root) if n.lod_level >= 3]


def all_reduce_eval(ev, acc):
    ne = torch.tensor(float(ev._num_examples), device=acc.device)
    ev._num_examples = int(acc.reduce(ne, reduction="sum").item())
    for name in ["_rfid_real_sigma", "_rfid_real_total", "_rfid_fake_sigma", "_rfid_fake_total",
                 "_is_prob_total", "_is_total_kl_d", "_total_psnr"]:
        if hasattr(ev, name):
            setattr(ev, name, acc.reduce(getattr(ev, name), reduction="sum"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--cap", type=int, default=None, help="per-rank image cap (debug/subset)")
    ap.add_argument("--cfg", type=str, default=CFG_DEFAULT)
    ap.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sanity", action="store_true",
                    help="dev-only: also eval random-matched + full-tree for validation")
    args = ap.parse_args()

    acc = Accelerator()
    dev = acc.device
    cfg = OmegaConf.load(args.cfg)
    ts = int(cfg.model.selector.token_size)  # 8

    model = QuadTok(cfg).eval().to(dev)
    model.requires_grad_(False)
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if acc.is_main_process:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()

    npsl = list(model.num_patch_side_list)             # [1,2,4,8,16]
    base_tree = build_quadtree(npsl[:-1])              # stops at lod3 (8x8=64)
    target_nodes = _get_nodes_at_level(base_tree, 3)   # 64 coarse cells (tree/nested order)
    inv = inverse_permutation(coarse_split_permutation())
    new_target_nodes = [target_nodes[i] for i in inv]  # spatial<->nested mapping
    pool = nn.AvgPool2d(kernel_size=32, stride=32)

    names = ["guided"] + (["random", "full"] if args.sanity else [])
    evals = {nm: VQGANEvaluator(device=dev, enable_rfid=True, enable_inception_score=True) for nm in names}
    for e in evals.values():
        e.reset_metrics()
    extra = {nm: {"psnr": 0.0, "lpips": 0.0, "tok": 0, "lod3": 0, "lod4": 0, "k": 0} for nm in names}
    rng = random.Random(args.seed + acc.process_index)
    full_mask = torch.ones(64, dtype=torch.bool)

    def eval_trees(nm, trees):
        recs = recon_opt(model, latent, trees, ts)
        evals[nm].update(batch.float(), recs.float())
        e = extra[nm]
        e["psnr"] += (10 * torch.log10(1.0 / ((recs - batch) ** 2).mean(dim=[1, 2, 3]).clamp_min(1e-12))).sum().item()
        e["lpips"] += lpips_fn(batch, recs, normalize=True).mean(dim=[1, 2, 3]).sum().item()
        e["tok"] += sum(len(t) for t in trees)
        e["lod3"] += sum(sum(1 for n in t if n.lod_level == 3) for t in trees)
        e["lod4"] += sum(sum(1 for n in t if n.lod_level == 4) for t in trees)
        e["k"] += len(trees)

    seen = 0
    for batch in val_stream(acc.process_index, acc.num_processes, args.bs, cap=args.cap):
        batch = batch.to(dev)
        B = batch.shape[0]
        seen += B
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(batch)

            # ONE random complementary 32/64 A/B split, shared across the batch (as in original)
            rpm = torch.zeros(64, dtype=torch.bool)
            rpm[rng.sample(range(64), 32)] = True
            tA = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rpm))
            tB = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, ~rpm))

            rec_ab = recon_opt(
                model, latent.repeat(2, 1, 1),
                [deepcopy(tA) for _ in range(B)] + [deepcopy(tB) for _ in range(B)], ts,
            )
            lpA = lpips_fn(batch, rec_ab[:B], normalize=True).sum(1)   # [B,256,256]
            lpB = lpips_fn(batch, rec_ab[B:], normalize=True).sum(1)

            # per-image signed-benefit -> tau gate -> guided tree (verbatim original semantics)
            sign = rpm.int().clone()
            sign[~rpm] = -1
            sign8 = sign.reshape(8, 8).float().to(dev)                 # A=+1, B=-1
            guided_masks, guided_trees = [], []
            for bi in range(B):
                pooled = pool((lpB[bi] - lpA[bi]).unsqueeze(0).unsqueeze(0))[0, 0]  # [8,8]
                dmap = pooled * sign8
                m = (dmap >= args.tau).flatten().detach().cpu().bool()             # 64
                guided_masks.append(m)
                guided_trees.append(ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, m)))
            eval_trees("guided", guided_trees)

            if args.sanity:
                # random-matched: same #split cells per image, random locations
                rand_trees = []
                for m in guided_masks:
                    kk = int(m.sum())
                    rm = torch.zeros(64, dtype=torch.bool)
                    if kk > 0:
                        rm[rng.sample(range(64), kk)] = True
                    rand_trees.append(ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rm)))
                eval_trees("random", rand_trees)
                # full tree: all 64 cells split -> 320 tokens (ceiling)
                eval_trees("full", [ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, full_mask)) for _ in range(B)])
        if acc.is_main_process and seen % (args.bs * 20) < args.bs:
            print(f"[rank0] ~{seen} imgs/rank", flush=True)

    acc.wait_for_everyone()
    for e in evals.values():
        all_reduce_eval(e, acc)
    for nm in names:
        for key in ["psnr", "lpips", "tok", "lod3", "lod4", "k"]:
            t = torch.tensor(float(extra[nm][key]), device=acc.device)
            extra[nm][key] = acc.reduce(t, reduction="sum").item()

    if acc.is_main_process:
        n_tot = int(extra["guided"]["k"])
        print("=" * 90)
        print(f"2-level 256  tau={args.tau}   (N={n_tot} val images)")
        print(f"{'config':>8} {'mean_tok':>9} {'lod3':>6} {'lod4':>7} {'PSNR':>8} {'LPIPS':>8} {'rFID':>9} {'IS':>8}")
        for nm in names:
            e = extra[nm]
            r = evals[nm].result()
            k = max(int(e["k"]), 1)
            print(f"{nm:>8} {e['tok']/k:>9.2f} {e['lod3']/k:>6.1f} {e['lod4']/k:>7.1f} "
                  f"{e['psnr']/k:>8.3f} {e['lpips']/k:>8.4f} {float(r['rFID']):>9.4f} {float(r.get('InceptionScore', 0.0)):>8.3f}")
        print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
