"""Single-GPU varlen training smoke: verify QuadtreeGPT.forward_varlen trains + measure
ms/step for eager vs torch.compile x grad-accum {1,2}. Uses the smoke pretok tars."""
import os, sys, time, glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
from omegaconf import OmegaConf
import modeling.mar as marmod
from data.varlen_pretok_loader import build_varlen_loader

marmod.QuadtreeGPT.initialize_weights = lambda self: None  # fast random init for the smoke

CFG = "configs/inference/gpt_16k_base.yaml"
SMOKE = "/sensei-fs-3/users/yuchengm/data/imagenet-pretok-2level/smoke/train-*.tar"


def main():
    dev = "cuda"
    size = os.environ.get("SIZE", "small")
    budget = int(os.environ.get("BUDGET", "29056"))   # ~128 samples * 227 tok (per GPU)
    cfg = OmegaConf.load(CFG)
    cfg.model.generator.model_size = size
    cfg.model.grad_checkpointing = (os.environ.get("GC", "0") == "1")
    model = marmod.QuadtreeGPT(cfg).to(dev).train()
    n = sum(p.numel() for p in model.parameters())
    print(f"model={size} params={n/1e6:.1f}M budget={budget} tok/gpu", flush=True)

    shards = sorted(glob.glob(SMOKE))
    loader, _ = build_varlen_loader(shards, max_token_num_global=budget, world=1, rank=0,
                                    num_workers=4, shuffle=1000, repeat=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.05)

    def step_fn(m, b):
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ld = m.forward_varlen(b["code_indices"], b["lod_indices"], b["patch_indices"],
                                        b["labels"], b["cu_seqlens"], b["seqlens"], b["max_seqlen"])
        return loss, ld

    def run(tag, m, accum, nsteps=12):
        it = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        losses = []
        t0 = None
        for s in range(nsteps):
            opt.zero_grad(set_to_none=True)
            for a in range(accum):
                loss, ld = step_fn(m, next(it))
                (loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(ld["total_loss"].item())
            if s == 2:
                torch.cuda.synchronize(); t0 = time.perf_counter()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / (nsteps - 3)
        toks = budget * accum
        print(f"[{tag}] accum={accum}: {dt*1000:7.0f} ms/step  {toks/dt:7.0f} tok/s  "
              f"loss {losses[0]:.3f}->{losses[-1]:.3f}  acc {ld['acc'].item():.3f}  "
              f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    run("eager", model, accum=1)
    run("eager", model, accum=2)
    try:
        cmodel = torch.compile(model, dynamic=True)
        run("compile", cmodel, accum=1)
        run("compile", cmodel, accum=2)
    except Exception as e:
        print(f"[compile] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
