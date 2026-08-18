"""Tensor-native 2-level guided search (single-GPU validation).

A tree = fixed 320-slot boolean active mask (lod3 64 | lod4 256). A/B expand + benefit +
decode are fully vectorized; recon uses the flex tensor cores
(selector._select_optimize_core / decoder._decode_optimize_core). No QuadTreeNode.

Modes:
  verify : recon_from_active (tensor) vs node-based recon_opt for the SAME tree -> PSNR (cores parity)
  eval   : tensor guided search @tau -> rFID/PSNR/tokens on N val images (compare to node-based)
  bench  : throughput img/s/GPU

Run: CUDA_VISIBLE_DEVICES=0 python scripts/bench256_tensor.py --mode verify
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch, torch.nn as nn
import lpips as lpips_pkg
from omegaconf import OmegaConf
from modeling.quadtok import QuadTok
from modeling.utils import QuadTreeNode
from eval.utils.evaluator import VQGANEvaluator
from scripts.probe256_2level_eval import val_stream, recon_opt, CKPT_DEFAULT, CFG_DEFAULT


def build_slot_maps(device):
    lod3, lod4 = 64, 256
    SLOT_LOD = torch.cat([torch.full((lod3,), 3), torch.full((lod4,), 4)]).to(device).long()
    SLOT_PATCH = torch.cat([torch.arange(lod3), torch.arange(lod4)]).to(device).long()
    j = torch.arange(lod4)
    PARENT4 = ((j // 16 // 2) * 8 + (j % 16 // 2)).to(device).long()   # lod4 patch -> lod3 patch (spatial)
    return SLOT_LOD, SLOT_PATCH, PARENT4


def active_to_padded(active, SLOT_LOD, SLOT_PATCH):
    device = active.device
    seqlens = active.sum(1).long()
    max_seq = int(seqlens.max().item())
    order = torch.argsort(active.int(), dim=1, descending=True, stable=True)
    sel = order[:, :max_seq]
    lod_pad = SLOT_LOD[sel]; pat_pad = SLOT_PATCH[sel]
    posmask = torch.arange(max_seq, device=device)[None, :] < seqlens[:, None]
    lod_pad = torch.where(posmask, lod_pad, torch.full_like(lod_pad, -1))
    pat_pad = torch.where(posmask, pat_pad, torch.zeros_like(pat_pad))
    return lod_pad, pat_pad, seqlens


@torch.no_grad()
def recon_from_active(model, latent, active, maps, ts=8):
    SLOT_LOD, SLOT_PATCH, _ = maps
    lod_pad, pat_pad, seqlens = active_to_padded(active, SLOT_LOD, SLOT_PATCH)
    z = model.selector._select_optimize_core(latent, lod_pad, pat_pad, seqlens)
    _, rd = model.quantize(z)
    B = latent.shape[0]
    emb = model.quantize.get_codebook_entry(rd["min_encoding_indices"].squeeze().long().flatten()).reshape(B, -1, ts)
    return model.decoder._decode_optimize_core(emb, lod_pad, pat_pad, seqlens).clamp(0, 1)


def _ones64(B, device):
    return torch.ones(B, 64, dtype=torch.bool, device=device)


@torch.no_grad()
def guided_search_fast(model, lpips_fn, images, tau, maps, rng):
    SLOT_LOD, SLOT_PATCH, PARENT4 = maps
    device = images.device; B = images.shape[0]
    latent = model.encode(images)
    a1 = torch.zeros(64, dtype=torch.bool, device=device)
    a1[rng.sample(range(64), 32)] = True
    e3A = a1[None].expand(B, -1); e3B = (~a1)[None].expand(B, -1)
    actA = torch.cat([_ones64(B, device), e3A[:, PARENT4]], 1)   # 320: lod3 all + lod4 A-children
    actB = torch.cat([_ones64(B, device), e3B[:, PARENT4]], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rA = recon_from_active(model, latent, actA, maps).float()
        rB = recon_from_active(model, latent, actB, maps).float()
    lpA = lpips_fn(images, rA, normalize=True).sum(1)
    lpB = lpips_fn(images, rB, normalize=True).sum(1)
    mi = a1.int().clone(); mi[~a1] = -1
    d = nn.functional.avg_pool2d((lpB - lpA).unsqueeze(1), 32)[:, 0] * mi.reshape(8, 8).to(images.dtype)
    E3 = d.reshape(B, 64) >= tau
    final_active = torch.cat([_ones64(B, device), E3[:, PARENT4]], 1)
    return latent, final_active


def load_model(cfg_path, ckpt, dev):
    cfg = OmegaConf.load(cfg_path)
    model = QuadTok(cfg).eval().to(dev)
    model.requires_grad_(False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    return model, int(cfg.model.selector.token_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "eval", "bench"], default="verify")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--cfg", default=CFG_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    args = ap.parse_args()
    dev = "cuda"
    model, ts = load_model(args.cfg, args.ckpt, dev)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()
    maps = build_slot_maps(dev)
    SLOT_LOD, SLOT_PATCH, PARENT4 = maps

    if args.mode == "verify":
        def psnr(a, b): return 10 * np.log10(1.0 / max(((a - b) ** 2).mean().item(), 1e-12))
        images = next(val_stream(0, 1, 16)).to(dev)
        B = images.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(images)
            e3 = torch.rand(B, 64, device=dev) < 0.9      # random per-image lod3 split
            lod4a = e3[:, PARENT4]
            active = torch.cat([_ones64(B, dev), lod4a], 1)
            rec_t = recon_from_active(model, latent, active, maps).float()
            # node-based recon of the SAME per-image trees (variable) via _forward_optimize
            node_lists = []
            for b in range(B):
                slots = torch.nonzero(active[b], as_tuple=True)[0]
                node_lists.append([QuadTreeNode(int(SLOT_LOD[s]), int(SLOT_PATCH[s])) for s in slots])
            rec_n = recon_opt(model, latent, node_lists, ts).float()
        psnrs = [psnr(rec_t[b:b + 1], rec_n[b:b + 1]) for b in range(B)]
        toks = [int(active[b].sum()) for b in range(B)]
        print(f"[verify] tokens/img (first 6)={toks[:6]}")
        print(f"[verify] PSNR(tensor_core, node_forward_optimize): min={np.min(psnrs):.2f} "
              f"mean={np.mean(psnrs):.2f} dB   (>40 = numerically identical)")
        return

    if args.mode == "eval":
        ev = VQGANEvaluator(device=dev, enable_rfid=True, enable_inception_score=True)
        ev.reset_metrics()
        rng = random.Random(1234)
        tot_tok = 0; tot_psnr = 0.0; n = 0
        for batch in val_stream(0, 1, args.bs, cap=args.cap):
            batch = batch.to(dev); B = batch.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lpips_fn, batch, args.tau, maps, rng)
                rec = recon_from_active(model, latent, active, maps).float()
            ev.update(batch.float(), rec.float())
            tot_tok += int(active.sum().item())
            tot_psnr += (10 * torch.log10(1.0 / ((rec - batch) ** 2).mean(dim=[1, 2, 3]).clamp_min(1e-12))).sum().item()
            n += B
        r = ev.result()
        print(f"[eval tensor] N={n} tau={args.tau}: mean_tok={tot_tok/n:.2f} "
              f"PSNR={tot_psnr/n:.3f} rFID={float(r['rFID']):.4f} IS={float(r.get('InceptionScore',0)):.2f}")
        return

    if args.mode == "bench":
        rng = random.Random(0)
        it = val_stream(0, 1, args.bs, cap=args.bs * 30)
        # warmup
        for i, batch in enumerate(it):
            batch = batch.to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lpips_fn, batch, args.tau, maps, rng)
                _ = recon_from_active(model, latent, active, maps)
            if i >= 2: break
        torch.cuda.synchronize()
        it = val_stream(0, 1, args.bs, cap=args.bs * 30)
        t0 = time.perf_counter(); n = 0
        for batch in it:
            batch = batch.to(dev); n += batch.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lpips_fn, batch, args.tau, maps, rng)
                _ = recon_from_active(model, latent, active, maps)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[bench tensor] {n} imgs in {dt:.2f}s -> {n/dt:.1f} img/s/GPU (bs={args.bs})")
        return


if __name__ == "__main__":
    main()
