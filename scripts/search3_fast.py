#!/usr/bin/env python
"""Tensor-native 3-level guided search. A tree is a fixed 1344-slot boolean 'active' mask
(lod3 8x8=64 | lod4 16x16=256 | lod5 32x32=1024). All tree ops (A/B expand, kinship, decode)
are vectorized boolean/gather tensor ops -- no QuadTreeNode, no per-node python. Reuses the
model's tensor cores (_select_optimize_core / _decode_optimize_core). Modes: verify|bench|eval."""
import os, sys, argparse, random, json, time
REPO = "/sensei-fs-3/users/yuchengm/code/quadtok/3level"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
import lpips as lpips_lib
from omegaconf import OmegaConf
from modeling.quadtok import QuadTok
from probe3 import load_tokenizer_weight, build_loader


def build_slot_maps(device):
    lod3, lod4, lod5 = 64, 256, 1024
    SLOT_LOD = torch.cat([torch.full((lod3,), 3), torch.full((lod4,), 4), torch.full((lod5,), 5)]).to(device).long()
    SLOT_PATCH = torch.cat([torch.arange(lod3), torch.arange(lod4), torch.arange(lod5)]).to(device).long()
    j = torch.arange(lod4); PARENT4 = ((j // 16 // 2) * 8 + (j % 16 // 2)).to(device).long()     # lod4->lod3 patch
    k = torch.arange(lod5); PARENT5 = ((k // 32 // 2) * 16 + (k % 32 // 2)).to(device).long()    # lod5->lod4 patch
    return SLOT_LOD, SLOT_PATCH, PARENT4, PARENT5


def active_to_padded(active, SLOT_LOD, SLOT_PATCH):
    """active (B,1344) bool -> compact padded (lod_pad,pat_pad,seqlens), fully vectorized."""
    device = active.device
    seqlens = active.sum(1).long()
    max_seq = int(seqlens.max().item())
    order = torch.argsort(active.int(), dim=1, descending=True, stable=True)   # active first, slot order
    sel = order[:, :max_seq]
    lod_pad = SLOT_LOD[sel]; pat_pad = SLOT_PATCH[sel]
    posmask = torch.arange(max_seq, device=device)[None, :] < seqlens[:, None]
    lod_pad = torch.where(posmask, lod_pad, torch.full_like(lod_pad, -1))
    pat_pad = torch.where(posmask, pat_pad, torch.zeros_like(pat_pad))
    return lod_pad, pat_pad, seqlens


@torch.no_grad()
def codes_pad_from_active(model, latent, active, maps):
    SLOT_LOD, SLOT_PATCH = maps[0], maps[1]
    lod_pad, pat_pad, seqlens = active_to_padded(active, SLOT_LOD, SLOT_PATCH)
    z = model.selector._select_optimize_core(latent, lod_pad, pat_pad, seqlens)
    _, rd = model.quantize(z)
    return rd["min_encoding_indices"], lod_pad, pat_pad, seqlens


@torch.no_grad()
def recon_from_active(model, latent, active, maps):
    idx, lod_pad, pat_pad, seqlens = codes_pad_from_active(model, latent, active, maps)
    B = latent.shape[0]
    emb = model.quantize.get_codebook_entry(idx.squeeze().long().flatten()).reshape(B, -1, 8)
    return model.decoder._decode_optimize_core(emb, lod_pad, pat_pad, seqlens).clamp(0, 1)


def _ones3(B, device):
    return torch.ones(B, 64, dtype=torch.bool, device=device)


@torch.no_grad()
def guided_search_fast(model, lpips_fn, images, t1, t2, maps):
    SLOT_LOD, SLOT_PATCH, PARENT4, PARENT5 = maps
    device = images.device; B = images.shape[0]
    latent = model.encode(images)
    z16 = torch.zeros(B, 1024, dtype=torch.bool, device=device)

    # ---- Stage 1: shared A/B over 64 lod3 (8x8) ----
    a1 = torch.zeros(64, dtype=torch.bool, device=device); a1[random.sample(range(64), 32)] = True
    e3A = a1[None].expand(B, -1); e3B = (~a1)[None].expand(B, -1)
    actA = torch.cat([_ones3(B, device), e3A[:, PARENT4], z16], 1)
    actB = torch.cat([_ones3(B, device), e3B[:, PARENT4], z16], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rA = recon_from_active(model, latent, actA, maps).float()
        rB = recon_from_active(model, latent, actB, maps).float()
    lp = lpips_fn(images, rA, normalize=True).sum(1); lpr = lpips_fn(images, rB, normalize=True).sum(1)
    mi1 = a1.int().clone(); mi1[~a1] = -1
    d1 = nn.functional.avg_pool2d((lpr - lp).unsqueeze(1), 32)[:, 0] * mi1.reshape(8, 8).to(images.dtype)
    E3 = d1.reshape(B, 64) >= t1

    # ---- Stage 2: A/B over existing lod4 (16x16) ----
    lod4_active = E3[:, PARENT4]                                   # (B,256)
    a2 = torch.zeros(256, dtype=torch.bool, device=device); a2[random.sample(range(256), 128)] = True
    E4A = lod4_active & a2[None]; E4B = lod4_active & (~a2)[None]
    actA = torch.cat([_ones3(B, device), lod4_active, E4A[:, PARENT5]], 1)
    actB = torch.cat([_ones3(B, device), lod4_active, E4B[:, PARENT5]], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rA = recon_from_active(model, latent, actA, maps).float()
        rB = recon_from_active(model, latent, actB, maps).float()
    lp = lpips_fn(images, rA, normalize=True).sum(1); lpr = lpips_fn(images, rB, normalize=True).sum(1)
    mi2 = a2.int().clone(); mi2[~a2] = -1
    d2 = nn.functional.avg_pool2d((lpr - lp).unsqueeze(1), 16)[:, 0] * mi2.reshape(16, 16).to(images.dtype)
    E4 = lod4_active & (d2.reshape(B, 256) >= t2)

    final_active = torch.cat([_ones3(B, device), lod4_active, E4[:, PARENT5]], 1)
    return latent, final_active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs/training/single_stage/quadtok_ss256_vq_3level.yaml"))
    ap.add_argument("--weight", default="/mnt/localssd/quadtok_hf")
    ap.add_argument("--shards", default="/sensei-fs-3/users/yuchengm/data/imagenet-wds/val/val-{000000..000049}.tar")
    ap.add_argument("--mode", choices=["verify", "bench", "eval", "calib", "full", "extract"], default="bench")
    ap.add_argument("--t1", type=float, default=0.035); ap.add_argument("--t2", type=float, default=0.012)
    ap.add_argument("--combos", default="")  # "t1:t2,t1:t2,..." -> sweep (eval mode), one model load
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10); ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--num_workers", type=int, default=8); ap.add_argument("--output", default="")
    ap.add_argument("--output_tar", default="")
    ap.add_argument("--hflip", type=int, default=1,
                    help="REQUIRED data aug: 1=bake original+horizontal-flip (2 views/img, matches LlamaGen/DiT/MAR); 0=original only")
    args = ap.parse_args()
    dev = torch.device("cuda:0"); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    cfg = OmegaConf.load(args.config)
    model = QuadTok(cfg).to(dev); load_tokenizer_weight(model, args.weight); model.eval().requires_grad_(False)
    lpips_sp = lpips_lib.LPIPS(net="vgg", spatial=True).to(dev).eval()
    maps = build_slot_maps(dev)

    if args.mode == "verify":
        from probe3 import recon_correct
        from modeling.utils import build_quadtree, _get_nodes_at_level, QuadTreeNode, _create_and_assign_children
        images = next(iter(build_loader(args.shards, 16, 4)))
        if isinstance(images, (list, tuple)): images = images[0]
        images = images.to(dev).float().clamp(0, 1); B = images.shape[0]
        def psnr(a, b): return 10 * np.log10(1.0 / max(((a - b) ** 2).mean().item(), 1e-12))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = model.encode(images)
            # random active mask (with lod5), recon via tensor path vs node-based correct path
            SLOT_LOD, SLOT_PATCH, PARENT4, PARENT5 = maps
            e3 = torch.rand(B, 64, device=dev) < 0.9
            lod4a = e3[:, PARENT4]
            e4 = lod4a & (torch.rand(B, 256, device=dev) < 0.7)
            active = torch.cat([_ones3(B, dev), lod4a, e4[:, PARENT5]], 1)
            rec_t = recon_from_active(model, latent, active, maps).float()
            # node-based recon for the same per-image tree, via _forward_reconstruction
            psnrs = []
            for b in range(B):
                slots = torch.nonzero(active[b], as_tuple=True)[0]
                nodes = [QuadTreeNode(int(SLOT_LOD[s]), int(SLOT_PATCH[s])) for s in slots]
                rc = recon_correct(model, latent[b:b + 1], nodes).float()
                psnrs.append(psnr(rec_t[b:b + 1], rc))
        print(f"[verify] tokens/img={[int(active[b].sum()) for b in range(B)][:6]}...", flush=True)
        print(f"[verify] PSNR(tensor_recon, node_correct): min={np.min(psnrs):.2f} mean={np.mean(psnrs):.2f} dB", flush=True)
        return

    if args.mode == "full":
        from eval.utils.evaluator import VQGANEvaluator
        lp_flat = lpips_lib.LPIPS(net="vgg", spatial=False).to(dev).eval()
        ev = VQGANEvaluator(device=dev, enable_rfid=True, enable_psnr=True, enable_inception_score=True)
        ld = build_loader(args.shards, args.bs, args.num_workers)
        lpsum = torch.zeros((), dtype=torch.float64, device=dev); nlp = 0; seen = 0
        for batch in ld:
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(dev).float().clamp(0, 1)
            B = images.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.encode(images)
                active = torch.ones(B, 1344, dtype=torch.bool, device=dev)  # full lod5 (all slots)
                rec = recon_from_active(model, latent, active, maps).float()
            ev.update(images, torch.round(rec * 255) / 255, None)
            with torch.no_grad():
                lpsum += lp_flat(images, rec, normalize=True).flatten().double().sum(); nlp += B
            seen += B
            if args.limit and seen >= args.limit:
                break
        r = ev.result()
        print("RESULT_JSON " + json.dumps({"mode": "full", "num_images": seen, "tokens": 1344,
              "rFID": float(r.get("rFID", float("nan"))), "PSNR": float(r.get("PSNR", float("nan"))),
              "IS": float(r.get("InceptionScore", float("nan"))), "LPIPS": float((lpsum / nlp).item())}), flush=True)
        return

    if args.mode == "extract":
        import webdataset as wds
        assert args.output_tar, "--output_tar required"
        os.makedirs(os.path.dirname(args.output_tar), exist_ok=True)
        loader = build_loader(args.shards, args.bs, args.num_workers, want_meta=True)
        n = 0; nw = 0; _bi = 0; _t0 = None; _n0 = 0

        def _emit(tw, imgs, keys, clss, suffix):
            # Tokenize one (possibly hflipped) batch and write one sample per image.
            nonlocal nw
            B = imgs.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lpips_sp, imgs, args.t1, args.t2, maps)
                idx, lod_pad, pat_pad, seqlens = codes_pad_from_active(model, latent, active, maps)
            codes = idx.reshape(B, -1).long().cpu().numpy()
            lodp = lod_pad.cpu().numpy(); patp = pat_pad.cpu().numpy(); sl = seqlens.cpu().numpy()
            for b in range(B):
                L = int(sl[b])
                tw.write({"__key__": str(keys[b]) + suffix,
                          "code_indices.npy": codes[b, :L].astype("int64"),
                          "lod_indices.npy": lodp[b, :L].astype("int64"),
                          "patch_indices.npy": patp[b, :L].astype("int64"),
                          "cls": str(int(clss[b]))})
                nw += 1

        with wds.TarWriter(args.output_tar) as tw:
            for images, keys, clss in loader:
                images = images.to(dev).float().clamp(0, 1)
                B = images.shape[0]
                if _bi == 2:
                    torch.cuda.synchronize(); _t0 = time.time(); _n0 = n
                _bi += 1
                # REQUIRED data aug (matches LlamaGen/DiT/MAR): center-crop (in build_loader) + hflip.
                # Frozen tokenizer -> bake BOTH the original and the horizontally-flipped view into the
                # codes; the flip is RE-tokenized so its quadtree/patch indices are correct.
                _emit(tw, images, keys, clss, "")
                if args.hflip:
                    _emit(tw, torch.flip(images, dims=[-1]), keys, clss, "_flip")
                n += B
                if args.limit and n >= args.limit:
                    break
        if _t0 is not None:
            torch.cuda.synchronize(); _dt = time.time() - _t0
            print("EXTRACT_RATE %.1f src-img/s (timed %d src-imgs in %.1fs, 2 warmup batches; hflip=%d => %d tokenizations)"
                  % ((n - _n0) / _dt, n - _n0, _dt, int(args.hflip), (n - _n0) * (2 if args.hflip else 1)), flush=True)
        print("EXTRACT_DONE wrote %d samples from %d src-imgs (hflip=%d) -> %s" % (nw, n, int(args.hflip), args.output_tar), flush=True)
        return

    if args.mode == "calib":
        combos = [(float(c.split(":")[0]), float(c.split(":")[1])) for c in args.combos.split(",")] if args.combos else [(args.t1, args.t2)]
        for t1, t2 in combos:
            l3 = l4 = l5 = 0.0; n = 0
            it = iter(build_loader(args.shards, args.bs, args.num_workers))
            while n < args.limit:
                try:
                    x = next(it)
                except StopIteration:
                    break
                images = (x[0] if isinstance(x, (list, tuple)) else x).to(dev).float().clamp(0, 1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    _, active = guided_search_fast(model, lpips_sp, images, t1, t2, maps)
                l3 += active[:, :64].sum().item(); l4 += active[:, 64:320].sum().item(); l5 += active[:, 320:].sum().item()
                n += active.shape[0]
            print("CALIB t1=%.4f t2=%.4f : lod3=%.1f lod4=%.1f lod5=%.1f total=%.1f  (n=%d)"
                  % (t1, t2, l3 / n, l4 / n, l5 / n, (l3 + l4 + l5) / n, n), flush=True)
        return

    loader = build_loader(args.shards, args.bs, args.num_workers, want_meta=(args.mode == "eval"))

    if args.mode == "bench":
        it = iter(loader)
        def nxt():
            x = next(it)
            if isinstance(x, (list, tuple)): x = x[0]
            return x.to(dev).float().clamp(0, 1)
        tot = 0.0; n = 0; ntok = []
        for i in range(args.warmup + args.iters):
            images = nxt(); B = images.shape[0]
            torch.cuda.synchronize(); t0 = time.time()
            _, active = guided_search_fast(model, lpips_sp, images, args.t1, args.t2, maps)
            torch.cuda.synchronize(); dt = time.time() - t0
            if i >= args.warmup:
                tot += dt; n += B; ntok += active.sum(1).tolist()
            print(f"[iter {i}]{' (warmup)' if i < args.warmup else ''} {B} imgs {dt:.3f}s", flush=True)
        print(f"\n=== FAST BENCH bs={args.bs} t1={args.t1} t2={args.t2} ===")
        print(f"  {tot/args.iters*1000:.1f} ms/batch  ({tot/n*1000:.2f} ms/img)  THROUGHPUT {n/tot:.1f} img/s  mean_tokens={np.mean(ntok):.0f}")
        return

    # eval (single or sweep over --combos), one model load
    from eval.utils.evaluator import VQGANEvaluator
    lp_flat = lpips_lib.LPIPS(net="vgg", spatial=False).to(dev).eval()
    combos = [(args.t1, args.t2)]
    if args.combos:
        combos = [(float(c.split(":")[0]), float(c.split(":")[1])) for c in args.combos.split(",")]

    def run_eval(t1, t2):
        ev = VQGANEvaluator(device=dev, enable_rfid=True, enable_psnr=True, enable_inception_score=True)
        ld = build_loader(args.shards, args.bs, args.num_workers, want_meta=False)
        tok = []; lpsum = torch.zeros((), dtype=torch.float64, device=dev); nlp = 0; seen = 0
        for batch in ld:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(dev).float().clamp(0, 1); B = images.shape[0]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lp_sp_ref[0], images, t1, t2, maps)
                rec = recon_from_active(model, latent, active, maps).float()
            tok += active.sum(1).tolist()
            ev.update(images, torch.round(rec * 255) / 255, None)
            with torch.no_grad():
                lpsum += lp_flat(images, rec, normalize=True).flatten().double().sum(); nlp += B
            seen += B
            if args.limit and seen >= args.limit: break
        r = ev.result()
        out = {"mode": "eval", "t1": t1, "t2": t2, "num_images": seen,
               "avg_tokens": float(np.mean(tok)), "p50": float(np.percentile(tok, 50)), "p90": float(np.percentile(tok, 90)),
               "rFID": float(r.get("rFID", float("nan"))), "PSNR": float(r.get("PSNR", float("nan"))),
               "IS": float(r.get("InceptionScore", float("nan"))), "LPIPS": float((lpsum / nlp).item())}
        print("RESULT_JSON " + json.dumps(out), flush=True)
        del ev
        return out

    lp_sp_ref = [lpips_sp]
    results = [run_eval(t1, t2) for (t1, t2) in combos]
    if args.output:
        json.dump(results, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
