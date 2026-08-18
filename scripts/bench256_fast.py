"""Parity + speed test for the FAST A/B recon path.

The A/B search step reconstructs every image in the batch with the SAME two trees
(tA, tB). So instead of the slow multi-tree `_forward_optimize` on 2B node-lists, use
the fast single-tree training forward path (`selector(latent, nodes)` + `decode`).
This verifies recon_fast == recon_opt (parity) and times old vs new.

Run: CUDA_VISIBLE_DEVICES=0 python scripts/bench256_fast.py --bs 32 --nbatch 12
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from copy import deepcopy
import torch, torch.nn as nn
import lpips as lpips_pkg
from omegaconf import OmegaConf
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, _get_nodes_at_level
from scripts.probe256_2level_eval import (
    coarse_split_permutation, inverse_permutation, _build_tree_from_node_mask,
    val_stream, recon_opt, ordered_ge3,
)

CFG = "configs/training/single_stage/quadtok_ss256_vq.yaml"
CKPT = "/sensei-fs-3/users/yuchengm/models/quadtok_2level_drive/drive_ckpt.download"


@torch.no_grad()
def recon_fast(model, latent, ordered_nodes):
    """Single shared tree, batched images -> training forward path (fast)."""
    z = model.selector(latent, ordered_nodes)
    z_q, _ = model.quantize(z)
    dec = model.decode(z_q.permute(0, 3, 2, 1).squeeze(2).contiguous(), ordered_nodes)
    return dec.clamp(0, 1).float()


def psnr(a, b):
    return (10 * torch.log10(1.0 / ((a - b) ** 2).mean().clamp_min(1e-12))).item()


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--nbatch", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.05)
    args = ap.parse_args()
    dev = "cuda"

    cfg = OmegaConf.load(CFG)
    ts = int(cfg.model.selector.token_size)
    model = QuadTok(cfg).eval().to(dev)
    model.requires_grad_(False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()

    npsl = list(model.num_patch_side_list)
    base_tree = build_quadtree(npsl[:-1])
    target_nodes = _get_nodes_at_level(base_tree, 3)
    new_target_nodes = [target_nodes[i] for i in inverse_permutation(coarse_split_permutation())]
    pool = nn.AvgPool2d(32, 32)
    rng = random.Random(0)

    # ---- parity on first batch ----
    it = val_stream(0, 1, args.bs, cap=args.bs * args.nbatch)
    first = next(it)
    batch = first.to(dev); B = batch.shape[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = model.encode(batch)
        rpm = torch.zeros(64, dtype=torch.bool); rpm[rng.sample(range(64), 32)] = True
        tA = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rpm))
        tB = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, ~rpm))
        rec_ab_opt = recon_opt(model, latent.repeat(2, 1, 1),
                               [deepcopy(tA) for _ in range(B)] + [deepcopy(tB) for _ in range(B)], ts)
        rec_A_fast = recon_fast(model, latent, tA)
        rec_B_fast = recon_fast(model, latent, tB)
    p_A = psnr(rec_ab_opt[:B], rec_A_fast)
    p_B = psnr(rec_ab_opt[B:], rec_B_fast)
    print(f"[parity] recon_fast vs recon_opt : PSNR_A={p_A:.2f} dB  PSNR_B={p_B:.2f} dB  "
          f"(>40 dB = numerically identical)")

    # ---- speed: OLD (opt A/B) vs NEW (fast A/B), full guided flow ----
    def run(mode):
        rng2 = random.Random(0)
        it2 = val_stream(0, 1, args.bs, cap=args.bs * args.nbatch)
        T = {"ab": 0.0, "guided": 0.0, "other": 0.0}
        n = 0
        for batch in it2:
            batch = batch.to(dev); B = batch.shape[0]; n += B
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                sync(); t0 = time.perf_counter()
                latent = model.encode(batch)
                rpm = torch.zeros(64, dtype=torch.bool); rpm[rng2.sample(range(64), 32)] = True
                tA = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rpm))
                tB = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, ~rpm))
                sync(); T["other"] += time.perf_counter() - t0

                sync(); t0 = time.perf_counter()
                if mode == "old":
                    rec_ab = recon_opt(model, latent.repeat(2, 1, 1),
                                       [deepcopy(tA) for _ in range(B)] + [deepcopy(tB) for _ in range(B)], ts)
                    rA, rB = rec_ab[:B], rec_ab[B:]
                else:
                    rA = recon_fast(model, latent, tA)
                    rB = recon_fast(model, latent, tB)
                sync(); T["ab"] += time.perf_counter() - t0

                sync(); t0 = time.perf_counter()
                lpA = lpips_fn(batch, rA, normalize=True).sum(1)
                lpB = lpips_fn(batch, rB, normalize=True).sum(1)
                sign = rpm.int().clone(); sign[~rpm] = -1; sign8 = sign.reshape(8, 8).float().to(dev)
                guided = []
                for b in range(B):
                    pooled = pool((lpB[b] - lpA[b]).unsqueeze(0).unsqueeze(0))[0, 0]
                    m = ((pooled * sign8) >= args.tau).flatten().detach().cpu().bool()
                    guided.append(ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, m)))
                _ = recon_opt(model, latent, guided, ts)
                sync(); T["guided"] += time.perf_counter() - t0
        tot = sum(T.values())
        return n, tot, T

    for mode in ["old", "new"]:
        n, tot, T = run(mode)
        print(f"[{mode:>3}] {n/tot:6.1f} img/s/GPU  total {1000*tot/n:6.2f} ms/img  "
              f"| ab {1000*T['ab']/n:5.2f}  guided {1000*T['guided']/n:5.2f}  other {1000*T['other']/n:5.2f}")


if __name__ == "__main__":
    main()
