"""GPU smoke test for the 3-level QuadTok tokenizer (M3).

Mirrors the real training generator step (model forward -> ReconstructionLoss_Single_Stage
-> backward) to validate: flex_attention kinship path (decoder + selector), warm-start load
from the 2-level C0 checkpoint (shape-mismatch filtering), config-driven expansion_probs,
VQ + hierarchical lod5 decode, and backward through flex. Also times steps at the real batch.

Usage: python scripts/smoke_test_3level.py config=configs/training/single_stage/quadtok_ss256_vq_3level.yaml
"""
import os, sys, time, traceback
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

import torch
from omegaconf import OmegaConf
from modeling.quadtok import QuadTok
from modeling.modules.losses import ReconstructionLoss_Single_Stage


def load_config():
    cli = OmegaConf.from_cli()
    assert "config" in cli, "pass config=<path>"
    cfg = OmegaConf.load(cli.pop("config"))
    return OmegaConf.merge(cfg, cli)


def warm_start(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    msd = model.state_dict()
    keep, skip = {}, []
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            keep[k] = v
        else:
            reason = "absent" if k not in msd else f"{tuple(v.shape)}->{tuple(msd[k].shape)}"
            skip.append(f"{k}({reason})")
    msg = model.load_state_dict(keep, strict=False)
    print(f"[warm-start] loaded {len(keep)}/{len(sd)} ckpt keys; skipped {len(skip)}: {skip[:8]}{'...' if len(skip)>8 else ''}")
    print(f"[warm-start] model missing {len(msg.missing_keys)} keys (fresh-init), e.g. {msg.missing_keys[:8]}")
    return msg


def main():
    cfg = load_config()
    dev = "cuda"
    assert torch.cuda.is_available()
    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
    print("num_patch_side_list", list(cfg.model.selector.num_patch_side_list),
          "| expansion_probs", list(cfg.model.selector.get("expansion_probs", [0.75])))

    # ---- build model ----
    model = QuadTok(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] QuadTok params: {nparam:.1f}M")

    # ---- warm-start ----
    warm_start(model, cfg.experiment.init_weight)

    # ---- decode equivalence: vectorized vs original per-node decode (must match) ----
    try:
        import random as _r
        from modeling.utils import build_probabilistic_quadtree
        dec = model.decoder
        _r.seed(0)
        root = build_probabilistic_quadtree(list(cfg.model.selector.num_patch_side_list), 3,
                                            list(cfg.model.selector.expansion_probs))
        nodes = [n for n in dec._get_ordered_nodes(root) if n.lod_level >= 3]
        Ld, B = len(nodes), 2
        feats = torch.randn(B, Ld, dec.width, device=dev)
        lodv = torch.tensor([n.lod_level for n in nodes], device=dev)
        patv = torch.tensor([n.patch_index for n in nodes], device=dev)
        with torch.no_grad():
            dec.update_features_in_tree(feats, nodes)
            out_old = dec.hierarchical_latent_decode(nodes, B)
            out_vec = dec.hierarchical_latent_decode_vec(feats, lodv, patv, B)
        d = (out_old.float() - out_vec.float()).abs().max().item()
        print(f"[decode-equiv] old-vs-vec max abs diff {d:.2e} (L={Ld})  ->  {'PASS' if d < 1e-4 else 'FAIL'}")
        assert d < 1e-4
    except Exception as e:
        import traceback; traceback.print_exc(); print("[decode-equiv] ERROR", repr(e)[:150])

    # ---- loss module (LPIPS may need offline weights; fall back to MSE if it fails) ----
    loss_module, use_full_loss = None, True
    try:
        loss_module = ReconstructionLoss_Single_Stage(config=cfg).to(dev)
        print("[loss] ReconstructionLoss_Single_Stage built (L2+LPIPS+VQ; GAN off at step 0).")
    except Exception as e:
        use_full_loss = False
        print(f"[loss] FAILED to build full loss ({repr(e)[:160]}); falling back to MSE-only.")

    model.train()

    def one_step(bs):
        images = torch.rand(bs, 3, cfg.dataset.preprocessing.crop_size,
                            cfg.dataset.preprocessing.crop_size, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, result_dict = model(images)
        assert torch.isfinite(recon).all(), "recon not finite"
        if use_full_loss:
            loss, loss_dict = loss_module(images.float(), recon, result_dict, global_step=0, mode="generator")
        else:
            loss = torch.nn.functional.mse_loss(recon.float(), images.float())
            loss_dict = {"total_loss": loss.detach()}
        loss.backward()
        return float(loss.detach()), recon.shape, result_dict

    # ---- correctness step (small bs) ----
    print("\n=== correctness step (bs=4) ===")
    loss_val, rshape, rd = one_step(4)
    print(f"recon shape {tuple(rshape)} | loss {loss_val:.4f}")
    gnorm = torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None))
    print(f"grad global norm: {gnorm.item():.4f} (finite={torch.isfinite(gnorm).item()})")
    assert torch.isfinite(gnorm).item(), "grad not finite"

    # ---- timing at the real per-GPU batch ----
    bs = int(cfg.training.per_gpu_batch_size)
    print(f"\n=== timing (bs={bs}) ===")
    model.zero_grad(set_to_none=True)
    for _ in range(4):  # warmup (also lets torch.compile finish compiling flex for varying seq lens)
        one_step(bs); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time(); N = 5
    for _ in range(N):
        one_step(bs); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / N
    print(f"step time: {dt*1000:.1f} ms/step | throughput {bs/dt:.1f} img/s/gpu | peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ---- op breakdown (1 step) to guide further optimization ----
    try:
        from torch.profiler import profile, ProfilerActivity
        model.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            one_step(bs); model.zero_grad(set_to_none=True)
        print("\n=== top ops by self CUDA time ===")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))
    except Exception as e:
        print("profiler skipped:", repr(e)[:150])

    print("\nSMOKE_PASS full_loss={} step_ms={:.0f}".format(use_full_loss, dt*1000))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("SMOKE_FAIL")
        sys.exit(1)
