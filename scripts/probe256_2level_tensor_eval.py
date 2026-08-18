"""Distributed ImageNet-val rFID/PSNR eval for the 2-level 256 tokenizer,
TENSOR-NATIVE guided search at tau=0.05 (no QuadTreeNode; fixed 320-slot active mask,
flex tensor cores _select_optimize_core / _decode_optimize_core). Guided-only.

Parity: recon cores 45 dB vs node-based; N=2000 search rFID 13.67 vs node 14.07 (equivalent).

Launch:  accelerate launch --num_processes 4 scripts/probe256_2level_tensor_eval.py --tau 0.05 [--cap N]
"""
import os, sys, argparse, random, threading, queue
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
import lpips as lpips_pkg


def prefetch(gen, n=4):
    """Background-thread prefetch: JPEG decode (PIL releases the GIL) overlaps GPU compute."""
    q = queue.Queue(maxsize=n)
    SENT = object()

    def worker():
        for x in gen:
            q.put(x)
        q.put(SENT)
    threading.Thread(target=worker, daemon=True).start()
    while True:
        x = q.get()
        if x is SENT:
            break
        yield x
from omegaconf import OmegaConf
from accelerate import Accelerator
from modeling.quadtok import QuadTok
from eval.utils.evaluator import VQGANEvaluator
from scripts.probe256_2level_eval import val_stream, all_reduce_eval, CFG_DEFAULT, CKPT_DEFAULT
from scripts.bench256_tensor import build_slot_maps, recon_from_active, guided_search_fast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--cfg", default=CFG_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    acc = Accelerator()
    dev = acc.device
    cfg = OmegaConf.load(args.cfg)
    model = QuadTok(cfg).eval().to(dev)
    model.requires_grad_(False)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    if acc.is_main_process:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()
    maps = build_slot_maps(dev)

    ev = VQGANEvaluator(device=dev, enable_rfid=True, enable_inception_score=True)
    ev.reset_metrics()
    extra = {"psnr": 0.0, "lpips": 0.0, "tok": 0, "lod3": 0, "lod4": 0, "k": 0}
    rng = random.Random(args.seed + acc.process_index)

    seen = 0
    for batch in prefetch(val_stream(acc.process_index, acc.num_processes, args.bs, cap=args.cap), n=4):
        batch = batch.to(dev)
        B = batch.shape[0]
        seen += B
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent, active = guided_search_fast(model, lpips_fn, batch, args.tau, maps, rng)
            recs = recon_from_active(model, latent, active, maps).float()
            ev.update(batch.float(), recs.float())
            extra["psnr"] += (10 * torch.log10(1.0 / ((recs - batch) ** 2).mean(dim=[1, 2, 3]).clamp_min(1e-12))).sum().item()
            extra["lpips"] += lpips_fn(batch, recs, normalize=True).mean(dim=[1, 2, 3]).sum().item()
            extra["tok"] += int(active.sum().item())
            extra["lod3"] += int(active[:, :64].sum().item())
            extra["lod4"] += int(active[:, 64:].sum().item())
            extra["k"] += B
        if acc.is_main_process and seen % (args.bs * 20) < args.bs:
            print(f"[rank0] ~{seen} imgs/rank", flush=True)

    acc.wait_for_everyone()
    all_reduce_eval(ev, acc)
    for key in ["psnr", "lpips", "tok", "lod3", "lod4", "k"]:
        t = torch.tensor(float(extra[key]), device=acc.device)
        extra[key] = acc.reduce(t, reduction="sum").item()

    if acc.is_main_process:
        r = ev.result()
        k = max(int(extra["k"]), 1)
        print("=" * 78)
        print(f"2-level 256 TENSOR-NATIVE GUIDED tau={args.tau}  (N={int(extra['k'])} val images)")
        print(f"  mean_tokens : {extra['tok']/k:.2f}   (lod3 {extra['lod3']/k:.1f} + lod4 {extra['lod4']/k:.1f})")
        print(f"  PSNR        : {extra['psnr']/k:.3f} dB")
        print(f"  LPIPS(vgg)  : {extra['lpips']/k:.4f}")
        print(f"  rFID        : {float(r['rFID']):.4f}")
        print(f"  IS          : {float(r.get('InceptionScore', 0.0)):.3f}")
        print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
