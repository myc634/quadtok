"""Distributed (DDP) varlen training smoke -- verifies multi-GPU trainability (proxy for 8xH200).
accelerate launch --num_processes 4 scripts/train_smoke_varlen_dist.py   (SIZE, GC, BUDGET via env)"""
import os, sys, time, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
import modeling.mar as marmod
from data.varlen_pretok_loader import build_varlen_loader

marmod.QuadtreeGPT.initialize_weights = lambda self: None
CFG = "configs/inference/gpt_16k_base.yaml"
SMOKE = "/sensei-fs-3/users/yuchengm/data/imagenet-pretok-2level/smoke/train-*.tar"


def main():
    acc = Accelerator(mixed_precision="bf16")
    dev = acc.device
    size = os.environ.get("SIZE", "small")
    budget = int(os.environ.get("BUDGET", "29056"))
    accum = int(os.environ.get("ACCUM", "1"))
    cfg = OmegaConf.load(CFG)
    cfg.model.generator.model_size = size
    cfg.model.grad_checkpointing = (os.environ.get("GC", "0") == "1")
    model = marmod.QuadtreeGPT(cfg).to(dev).train()
    n = sum(p.numel() for p in model.parameters())

    shards = sorted(glob.glob(SMOKE))
    loader, per_gpu = build_varlen_loader(shards, max_token_num_global=budget * acc.num_processes,
                                          world=acc.num_processes, rank=acc.process_index,
                                          num_workers=4, repeat=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)
    model, opt = acc.prepare(model, opt)
    if acc.is_main_process:
        print(f"[dist] size={size} params={n/1e6:.1f}M procs={acc.num_processes} "
              f"per_gpu_budget={per_gpu} accum={accum} GC={cfg.model.grad_checkpointing}", flush=True)

    it = iter(loader)
    losses, t0 = [], None
    nsteps = 12
    for s in range(nsteps):
        opt.zero_grad(set_to_none=True)
        for a in range(accum):
            b = next(it)
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            m = acc.unwrap_model(model)
            with acc.autocast():
                loss, ld = m.forward_varlen(b["code_indices"], b["lod_indices"], b["patch_indices"],
                                            b["labels"], b["cu_seqlens"], b["seqlens"], b["max_seqlen"])
            acc.backward(loss / accum)
        acc.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(ld["total_loss"].item())
        if s == 2:
            acc.wait_for_everyone(); torch.cuda.synchronize(); t0 = time.perf_counter()
    acc.wait_for_everyone(); torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / (nsteps - 3)
    if acc.is_main_process:
        gtoks = per_gpu * accum * acc.num_processes
        print(f"[dist RESULT] {size}: {dt*1000:.0f} ms/step  global {gtoks} tok/step  "
              f"{gtoks/dt:.0f} tok/s ({acc.num_processes} gpu)  loss {losses[0]:.3f}->{losses[-1]:.3f}  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB/gpu", flush=True)


if __name__ == "__main__":
    main()
