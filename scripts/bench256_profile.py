"""Single-GPU profiler for the 2-level 256 probing eval: splits wall time into
CPU tree-building vs GPU (encode / A-B recon / LPIPS / guided recon) to decide the
next speedup lever (tensor-native tree-build vs flex-batched _forward_optimize).

Run: CUDA_VISIBLE_DEVICES=0 python scripts/bench256_profile.py --bs 32 --nbatch 10
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from copy import deepcopy
import torch, torch.nn as nn
import lpips as lpips_pkg
from omegaconf import OmegaConf
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, _get_nodes_at_level
# reuse the exact faithful helpers from the eval script
from scripts.probe256_2level_eval import (
    coarse_split_permutation, inverse_permutation, _build_tree_from_node_mask,
    val_stream, recon_opt, ordered_ge3,
)

CFG = "configs/training/single_stage/quadtok_ss256_vq.yaml"
CKPT = "/sensei-fs-3/users/yuchengm/models/quadtok_2level_drive/drive_ckpt.download"


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--nbatch", type=int, default=10)
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

    T = {k: 0.0 for k in ["encode", "ab_build", "ab_recon", "lpips", "guided_build", "guided_recon"]}
    nimg = 0
    it = val_stream(0, 1, args.bs, cap=args.bs * args.nbatch)
    for bi, batch in enumerate(it):
        batch = batch.to(dev); B = batch.shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            sync(); t = time.perf_counter()
            latent = model.encode(batch); sync(); T["encode"] += time.perf_counter() - t

            t = time.perf_counter()
            rpm = torch.zeros(64, dtype=torch.bool); rpm[rng.sample(range(64), 32)] = True
            tA = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, rpm))
            tB = ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, ~rpm))
            node_ab = [deepcopy(tA) for _ in range(B)] + [deepcopy(tB) for _ in range(B)]
            T["ab_build"] += time.perf_counter() - t

            sync(); t = time.perf_counter()
            rec_ab = recon_opt(model, latent.repeat(2, 1, 1), node_ab, ts); sync()
            T["ab_recon"] += time.perf_counter() - t

            sync(); t = time.perf_counter()
            lpA = lpips_fn(batch, rec_ab[:B], normalize=True).sum(1)
            lpB = lpips_fn(batch, rec_ab[B:], normalize=True).sum(1); sync()
            T["lpips"] += time.perf_counter() - t

            t = time.perf_counter()
            sign = rpm.int().clone(); sign[~rpm] = -1; sign8 = sign.reshape(8, 8).float().to(dev)
            guided = []
            for b in range(B):
                pooled = pool((lpB[b] - lpA[b]).unsqueeze(0).unsqueeze(0))[0, 0]
                m = ((pooled * sign8) >= args.tau).flatten().detach().cpu().bool()
                guided.append(ordered_ge3(model, _build_tree_from_node_mask(base_tree, 3, 4, npsl, new_target_nodes, m)))
            T["guided_build"] += time.perf_counter() - t

            sync(); t = time.perf_counter()
            _ = recon_opt(model, latent, guided, ts); sync()
            T["guided_recon"] += time.perf_counter() - t
        nimg += B

    total = sum(T.values())
    print(f"\n=== profile ({nimg} imgs, bs={args.bs}) ===")
    print(f"{'component':>14} {'ms/img':>9} {'% total':>8}")
    for k, v in T.items():
        print(f"{k:>14} {1000*v/nimg:>9.2f} {100*v/total:>7.1f}%")
    print(f"{'TOTAL':>14} {1000*total/nimg:>9.2f}   -> {nimg/total:.1f} img/s/GPU")
    cpu = T["ab_build"] + T["guided_build"]
    gpu = total - cpu
    print(f"\nCPU tree-build: {100*cpu/total:.1f}%   GPU (encode+recon+lpips): {100*gpu/total:.1f}%")
    print("=> lever: " + ("tensor-native tree-build (CPU-bound)" if cpu > gpu * 0.5 else "flex-batched _forward_optimize (GPU-bound)"))


if __name__ == "__main__":
    main()
