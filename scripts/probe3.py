#!/usr/bin/env python
"""3-level QuadTok probing (stage-2 pipeline).

Two-stage hierarchical LPIPS A/B search on a FROZEN 3-level tokenizer:
  Stage 1 (lod3 8x8 -> lod4 16x16, threshold t1): VERBATIM the validated 2-level probe
          (complementary 32/64 random split, LPIPS(vgg,spatial) benefit, AvgPool(32)->8x8).
  Stage 2 (lod4 16x16 -> lod5 32x32, threshold t2): same A/B over each image's EXISTING lod4
          nodes (complementary 128/256 split, AvgPool(16)->16x16). Regions still at lod3 cannot
          split (only existing lod4 nodes are candidates).

Modes:
  --mode eval    : reconstruct the guided 3-level tree (+ matched-random baseline) and report
                   rFID / PSNR / LPIPS + token-count distribution. Used for the (t1,t2) sweep
                   (target mean tokens ~800-900, pick best rFID/PSNR/LPIPS).
  --mode extract : write code_indices/lod_indices/patch_indices/cls webdataset tars for
                   generator (M1) training, using the chosen (t1,t2).

Based on base/t1_search.py + scripts/extract_code_searchquadtree.py (stage-1 kept verbatim).
"""
import os, sys, argparse, random, json
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import webdataset as wds
import torchvision.transforms as T
from omegaconf import OmegaConf
import lpips as lpips_lib
from copy import deepcopy
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, _get_nodes_at_level, QuadTreeNode, _create_and_assign_children
from eval.utils.evaluator import VQGANEvaluator


# ---- helpers (verbatim from extract_code_searchquadtree.py / t1_search.py) ----
def coarse_split_permutation(coarse_hw=(4, 4), split_hw=(2, 2)):
    Hc, Wc = coarse_hw; Hs, Ws = split_hw
    Hf, Wf = Hc * Hs, Wc * Ws
    perm = []
    for cy in range(Hc):
        for cx in range(Wc):
            for sy in range(Hs):
                for sx in range(Ws):
                    perm.append((cy * Hs + sy) * Wf + (cx * Ws + sx))
    return perm

def inverse_permutation(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def _build_tree_from_node_mask(base_tree, target_lod, max_lod, patches_per_side_list, target_nodes, node_mask):
    expand = {target_nodes[i].patch_index for i in range(len(target_nodes)) if node_mask[target_nodes[i].patch_index]}
    def _copy(node):
        nn_ = QuadTreeNode(node.lod_level, node.patch_index)
        if node.lod_level == target_lod and node.patch_index in expand:
            if node.lod_level < max_lod:
                _create_and_assign_children(nn_, patches_per_side_list)
        elif node.children:
            for c in node.children:
                nn_.children.append(_copy(c))
        return nn_
    return _copy(base_tree)

def olod3(model, tree):
    return [n for n in model._get_ordered_nodes(tree) if n.lod_level >= 3]

def _codes_for(model, latent, node_lists):
    """selector+quantize -> per-image code indices (no decode)."""
    z = model.selector._forward_optimize(latent, node_lists)
    _, rd = model.quantize(z)
    return rd["min_encoding_indices"][:, 0]

def _reconstruct(model, latent, node_lists):
    z = model.selector._forward_optimize(latent, node_lists)
    _, rd = model.quantize(z)
    B = latent.shape[0]
    emb = model.quantize.get_codebook_entry(rd["min_encoding_indices"].squeeze().long().flatten()).reshape(B, -1, 8)
    return model.decoder._forward_optimize(emb, node_lists).clamp(0, 1)


def load_tokenizer_weight(model, path):
    """Load a QuadTok state_dict from: a plain .bin/.pt, a .safetensors, or an accelerate
    checkpoint dir (prefer ema_model/, else model.safetensors/pytorch_model.bin)."""
    if os.path.isdir(path):
        cands = [os.path.join(path, "ema_model", "model.safetensors"),
                 os.path.join(path, "ema_model", "pytorch_model.bin"),
                 os.path.join(path, "unwrapped_model", "model.safetensors"),
                 os.path.join(path, "model.safetensors"),
                 os.path.join(path, "pytorch_model.bin")]
        path = next((p for p in cands if os.path.exists(p)), None)
        assert path, f"no loadable weight under {path}"
    print(f"[load] using {path}", flush=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return model.load_state_dict(sd, strict=False)


@torch.no_grad()
def _ab_stage(model, lpips_fn, images, latent, base_trees, target_lod, grid_side, pool_k, threshold):
    """One LPIPS A/B split stage. base_trees: per-image trees to expand at `target_lod`.
    Returns (per-image expanded trees, per-image ordered node-lists, benefit maps [B, grid^2])."""
    B = images.shape[0]
    n_pos = grid_side * grid_side
    mask = torch.zeros(n_pos, dtype=torch.bool)
    mask[random.sample(range(n_pos), n_pos // 2)] = True
    tgt_list, masked_nl, rev_nl = [], [], []
    for b in range(B):
        tn = _get_nodes_at_level(base_trees[b], target_lod)
        tgt_list.append(tn)
        mt = _build_tree_from_node_mask(base_trees[b], target_lod, target_lod + 1, model.num_patch_side_list, tn, mask)
        mr = _build_tree_from_node_mask(base_trees[b], target_lod, target_lod + 1, model.num_patch_side_list, tn, ~mask)
        masked_nl.append(olod3(model, mt)); rev_nl.append(olod3(model, mr))
    nl = masked_nl + rev_nl
    emb_codes = _codes_for(model, latent.repeat(2, 1, 1), nl)
    emb = model.quantize.get_codebook_entry(emb_codes.long().flatten()).reshape(2 * B, -1, 8)
    rec = model.decoder._forward_optimize(emb, nl).clamp(0, 1)
    lp = lpips_fn(images, rec[:B], normalize=True).sum(1)
    lpr = lpips_fn(images, rec[B:], normalize=True).sum(1)
    pool = nn.AvgPool2d(kernel_size=pool_k, stride=pool_k)
    mi = mask.int().clone(); mi[~mask] = -1
    mi = mi.reshape(grid_side, grid_side)
    trees, olists, diffs = [], [], []
    for b in range(B):
        diff = pool((lpr[b] - lp[b]).unsqueeze(0).unsqueeze(0))[0, 0]
        dm = diff * mi.to(diff.device)
        pb = (dm >= threshold).flatten().bool()
        t = _build_tree_from_node_mask(base_trees[b], target_lod, target_lod + 1, model.num_patch_side_list, tgt_list[b], pb)
        trees.append(t); olists.append(olod3(model, t)); diffs.append(dm.flatten())
    return trees, olists, torch.stack(diffs, 0)


@torch.no_grad()
def recon_correct(model, latent, nodes):
    """Batched single-tree reconstruction via the verified training path (selector
    _forward_reconstruction + decode). `nodes` is one ordered node list shared by the batch."""
    z = model.selector._forward_reconstruction(latent, nodes)
    zq, _ = model.quantize(z)
    rec = model.decode(zq.permute(0, 3, 2, 1).squeeze(2).contiguous(), nodes)
    return rec.clamp(0, 1)


@torch.no_grad()
def _ab_stage_shared(model, lpips_fn, images, latent, base_tree, target_lod, grid_side, pool_k, threshold):
    """LPIPS A/B split where the A and B trees are SHARED across the batch (stage 1). Uses the
    batched single-tree path -> 2 recon calls for the whole batch instead of 2B per-image trees."""
    B = images.shape[0]
    n_pos = grid_side * grid_side
    mask = torch.zeros(n_pos, dtype=torch.bool)
    mask[random.sample(range(n_pos), n_pos // 2)] = True
    tn = _get_nodes_at_level(base_tree, target_lod)
    mt = _build_tree_from_node_mask(base_tree, target_lod, target_lod + 1, model.num_patch_side_list, tn, mask)
    mr = _build_tree_from_node_mask(base_tree, target_lod, target_lod + 1, model.num_patch_side_list, tn, ~mask)
    nodes_m, nodes_r = olod3(model, mt), olod3(model, mr)
    rec_m = recon_correct(model, latent, nodes_m).float()
    rec_r = recon_correct(model, latent, nodes_r).float()
    lp = lpips_fn(images, rec_m, normalize=True).sum(1)
    lpr = lpips_fn(images, rec_r, normalize=True).sum(1)
    pool = nn.AvgPool2d(kernel_size=pool_k, stride=pool_k)
    mi = mask.int().clone(); mi[~mask] = -1
    mi = mi.reshape(grid_side, grid_side)
    trees, olists, diffs = [], [], []
    for b in range(B):
        diff = pool((lpr[b] - lp[b]).unsqueeze(0).unsqueeze(0))[0, 0]
        dm = diff * mi.to(diff.device)
        pb = (dm >= threshold).flatten().bool()
        t = _build_tree_from_node_mask(base_tree, target_lod, target_lod + 1, model.num_patch_side_list, tn, pb)
        trees.append(t); olists.append(olod3(model, t)); diffs.append(dm.flatten())
    return trees, olists, torch.stack(diffs, 0)


@torch.no_grad()
def guided_search_3level(model, lpips_fn, images, t1, t2):
    latent = model.encode(images)
    B = images.shape[0]
    lod3_tree = build_quadtree(model.num_patch_side_list[:4])   # [1,2,4,8] -> full lod3 (64 nodes)
    s1_trees, _, d1 = _ab_stage_shared(model, lpips_fn, images, latent, lod3_tree, 3, 8, 32, t1)
    s2_trees, s2_ol, d2 = _ab_stage(model, lpips_fn, images, latent, s1_trees, 4, 16, 16, t2)
    return latent, s2_trees, s2_ol, d1, d2


def build_loader(shards, bs, workers, want_meta=False):
    tf = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    pipe = [wds.SimpleShardList(shards), wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["jpg", "jpeg", "png", "webp"]), handler=wds.warn_and_continue),
            wds.rename(image="jpg;jpeg;png;webp", cls="cls", handler=wds.warn_and_continue),
            wds.map_dict(image=lambda im: tf(im.convert("RGB")), handler=wds.warn_and_continue)]
    if want_meta:
        pipe += [wds.to_tuple("image", "__key__", "cls"), wds.batched(bs)]
    else:
        pipe += [wds.to_tuple("image"), wds.batched(bs)]
    return wds.WebLoader(wds.DataPipeline(*pipe), batch_size=None, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/sensei-fs-3/users/yuchengm/code/quadtok/3level/configs/training/single_stage/quadtok_ss256_vq_3level.yaml")
    ap.add_argument("--tokenizer_weight", required=True)
    ap.add_argument("--shards", default="/sensei-fs-3/users/yuchengm/data/imagenet-wds/val/val-{000000..000049}.tar")
    ap.add_argument("--mode", choices=["eval", "extract"], default="eval")
    ap.add_argument("--t1", type=float, default=0.05)
    ap.add_argument("--t2", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="")
    ap.add_argument("--output_tar", default="")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    model = QuadTok(cfg).to(device)
    msg = load_tokenizer_weight(model, args.tokenizer_weight)
    print(f"[load] {args.tokenizer_weight} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    model.eval(); model.requires_grad_(False)
    lpips_sp = lpips_lib.LPIPS(net="vgg", spatial=True).to(device).eval()

    if args.mode == "eval":
        lpips_flat = lpips_lib.LPIPS(net="vgg", spatial=False).to(device).eval()
        ev_g = VQGANEvaluator(device=device, enable_rfid=True, enable_psnr=True, enable_inception_score=True)
        loader = build_loader(args.shards, args.batch_size, args.num_workers, want_meta=False)
        print(f"[cfg] mode=eval t1={args.t1} t2={args.t2} limit={args.limit}", flush=True)
        tok, lp_g = [], torch.zeros((), dtype=torch.float64, device=device); lp_n = torch.zeros((), dtype=torch.float64, device=device)
        seen, bi = 0, 0
        for images in loader:
            if isinstance(images, (list, tuple)): images = images[0]
            images = images.to(device).float().clamp(0, 1); B = images.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent, s2_trees, s2_ol, d1, d2 = guided_search_3level(model, lpips_sp, images, args.t1, args.t2)
                rec_g = _reconstruct(model, latent, s2_ol).float()
            for b in range(B): tok.append(len(s2_ol[b]))
            ev_g.update(images, torch.round(rec_g * 255) / 255, None)
            with torch.no_grad():
                lp_g += lpips_flat(images, rec_g, normalize=True).flatten().double().sum(); lp_n += B
            if bi < 2:
                a = d1.float().cpu().numpy().flatten(); c = d2.float().cpu().numpy().flatten()
                print(f"[diag b{bi}] stage1 benefit mean/p90={a.mean():.4f}/{np.percentile(a,90):.4f} "
                      f"stage2 benefit mean/p90={c.mean():.4f}/{np.percentile(c,90):.4f} "
                      f"tokens mean={np.mean([len(x) for x in s2_ol]):.1f}", flush=True)
            seen += B; bi += 1
            if args.limit > 0 and seen >= args.limit: break
        rg = ev_g.result()
        out = {"mode": "eval", "t1": args.t1, "t2": args.t2, "num_images": seen,
               "avg_tokens": float(np.mean(tok)), "p10": float(np.percentile(tok, 10)),
               "p50": float(np.percentile(tok, 50)), "p90": float(np.percentile(tok, 90)), "max": int(np.max(tok)),
               "rFID": float(rg.get("rFID", float("nan"))), "PSNR": float(rg.get("PSNR", float("nan"))),
               "IS": float(rg.get("InceptionScore", float("nan"))), "LPIPS": float((lp_g / lp_n).item())}
        print("RESULT_JSON " + json.dumps(out), flush=True)
        if args.output: json.dump(out, open(args.output, "w"), indent=2)

    else:  # extract -> generator training data
        assert args.output_tar, "--output_tar required for extract mode"
        os.makedirs(os.path.dirname(args.output_tar), exist_ok=True)
        loader = build_loader(args.shards, args.batch_size, args.num_workers, want_meta=True)
        print(f"[cfg] mode=extract t1={args.t1} t2={args.t2} -> {args.output_tar}", flush=True)
        n = 0
        with wds.TarWriter(args.output_tar) as tw:
            for images, keys, clss in loader:
                images = images.to(device).float().clamp(0, 1); B = images.shape[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent, s2_trees, s2_ol, _, _ = guided_search_3level(model, lpips_sp, images, args.t1, args.t2)
                    codes = _codes_for(model, latent, s2_ol)
                for b in range(B):
                    nodes = s2_ol[b]; L = len(nodes)
                    ci = codes[b][:L].cpu().numpy()
                    lod_idx = np.array([nd.lod_level for nd in nodes]); pat_idx = np.array([nd.patch_index for nd in nodes])
                    assert ci.shape[0] == L == len(lod_idx)
                    tw.write({"__key__": str(keys[b]), "code_indices.npy": ci,
                              "lod_indices.npy": lod_idx, "patch_indices.npy": pat_idx,
                              "cls": str(int(clss[b]))})
                    n += 1
        print(f"EXTRACT_DONE wrote {n} samples to {args.output_tar}", flush=True)


if __name__ == "__main__":
    main()
