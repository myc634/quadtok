"""Reference training loop for the 512 QuadtreeGPT generator (base ~344M, LlamaGen-L) with VARLEN
flash-attn + torch.compile + Liger fused-CE + grad-accumulation. Each rank pulls its own
token-budget-packed stream (VarlenPretokPacked) and calls model(packed) -> forward() ->
forward_varlen() so DDP grad-sync fires. Verifies it trains (loss moves), fits 8xH200 at the per-GPU
token budget, and reports s/step + tok/s + peak mem.

This is the canonical fast path — production adds checkpointing / wandb / resume / a real step budget,
but MUST keep the three speed lines below (compile_blocks before prepare; interval + use_liger come
from the config via QuadtreeGPT.__init__).

Launch:  accelerate launch --num_processes 8 scripts/smoke_gen512.py --steps 100 --accum 1
"""
import time, argparse
from contextlib import nullcontext
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
import modeling.mar as mar
from modeling.mar import QuadtreeGPT
from modeling.varlen_pretok import VarlenPretokPacked

CFG = "configs/training/generator/gpt_quadtree_512.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100, help="optimizer steps (each = --accum micro-batches)")
    ap.add_argument("--accum", type=int, default=1, help="grad-accumulation micro-steps per optimizer step")
    ap.add_argument("--max_tokens_per_gpu", type=int, default=0, help="0 -> global_max_tokens // world")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = OmegaConf.load(CFG)
    acc = Accelerator(mixed_precision="bf16")
    mar.QuadtreeGPT.initialize_weights = lambda self: None  # smoke: skip O(M*P) init stall (production KEEPS init)
    torch.manual_seed(cfg.training.seed + acc.process_index)

    # QuadtreeGPT.__init__ already reads model.grad_ckpt_interval and model.use_liger from the config.
    model = QuadtreeGPT(cfg)
    # SPEED (must be applied by any trainer): torch.compile the blocks on the RAW model BEFORE prepare.
    # dynamic=True + constant max_seqlen -> compiles once (~50s), then stable across varying varlen steps;
    # inductor fusion also lowers peak mem (this is what lets grad_ckpt_interval stay low). See §9 doc.
    if bool(cfg.model.get("compile", False)):
        model.compile_blocks()

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.params.learning_rate),
                            betas=(0.9, 0.95), weight_decay=float(cfg.optimizer.params.weight_decay), fused=True)
    model, opt = acc.prepare(model, opt)

    per_gpu = args.max_tokens_per_gpu or (cfg.varlen.global_max_tokens // acc.num_processes)
    max_gn = float(cfg.training.max_grad_norm) if cfg.training.max_grad_norm else 0.0
    if acc.is_main_process:
        n_params = sum(p.numel() for p in acc.unwrap_model(model).parameters()) / 1e6
        print(f"[cfg] params {n_params:.1f}M | world {acc.num_processes} | per_gpu {per_gpu} "
              f"| global {per_gpu * acc.num_processes * args.accum} | accum {args.accum} "
              f"| ckpt_interval {cfg.model.get('grad_ckpt_interval', 1)} | compile {cfg.model.get('compile', False)} "
              f"| use_liger {cfg.model.get('use_liger', False)}", flush=True)

    ds = VarlenPretokPacked(cfg.varlen.pretok_glob, per_gpu, rank=acc.process_index,
                            world=acc.num_processes, shuffle=True, seed=cfg.training.seed, loop=True)
    it = iter(ds)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    t0, tok0 = None, 0
    for step in range(args.steps):
        if step == 3:
            acc.wait_for_everyone(); torch.cuda.synchronize(); t0 = time.time(); tok0 = 0
        opt.zero_grad(set_to_none=True)
        micro = [next(it) for _ in range(args.accum)]
        for j, mb in enumerate(micro):
            is_last = (j == args.accum - 1)
            ctx = model.no_sync() if (args.accum > 1 and not is_last and hasattr(model, "no_sync")) else nullcontext()
            with ctx:
                loss, d = model(mb)                      # forward -> forward_varlen (DDP grad-sync on last micro)
                acc.backward(loss / args.accum)
        if max_gn:
            acc.clip_grad_norm_(model.parameters(), max_gn)
        opt.step()
        if t0 is not None:
            tok0 += sum(int(mb["seqlens"].sum()) for mb in micro)
        if acc.is_main_process and (step % 10 == 0 or step == args.steps - 1):
            print(f"step {step:>3}: loss {loss.item():.4f} acc {float(d['acc']):.4f} "
                  f"| n {int(micro[-1]['seqlens'].shape[0])} tok/optstep {sum(int(mb['seqlens'].sum()) for mb in micro)}", flush=True)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() // (1024 ** 2)
    if acc.is_main_process and t0 is not None:
        dt = time.time() - t0; n = args.steps - 3
        print(f"[speed] {n} opt-steps in {dt:.1f}s = {dt / n:.3f} s/optstep | {tok0 / dt:.0f} tok/s/gpu", flush=True)
    print(f"[rank{acc.process_index}] peak_mem {peak} MiB (H200=143771)", flush=True)
    if acc.is_main_process:
        print("=== GEN SMOKE DONE ===", flush=True)


if __name__ == "__main__":
    main()
