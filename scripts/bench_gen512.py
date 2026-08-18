"""Benchmark ONE speed config of the 512 QuadtreeGPT varlen generator (run once per config so each
process is clean). Measures throughput (tok/s/GPU), per-optimizer-step time, steady-state peak mem, and
mean GPU SM-utilization on VARYING-length real varlen batches (data prefetched OUTSIDE the timer so
torch.compile's dynamic-shape behavior is exposed).

    accelerate launch --num_processes 8 scripts/bench_gen512.py \
        --ckpt_interval K --compile 0|1 --use_liger 0|1 --accum N --steps 30

--ckpt_interval: 0 = no grad-ckpt (fastest/most mem), k>=1 = ckpt every k-th block (1 == full ckpt).
--compile:       1 = torch.compile each TransformerBlock (dynamic=True) — blocks only.
--use_liger:     1 = fused linear-cross-entropy head (never materializes the ~8.6GB fp32 logits).
--accum:         grad-accumulation micro-steps per optimizer step (DDP no_sync on non-final micro-steps);
                 effective global batch = accum * global_max_tokens. tok/s is invariant to accum, so it
                 is the primary cross-config metric.
"""
import time, argparse, threading, subprocess
from contextlib import nullcontext
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
import modeling.mar as mar
from modeling.mar import QuadtreeGPT
from modeling.varlen_pretok import VarlenPretokPacked

CFG = "configs/training/generator/gpt_quadtree_512.yaml"


def gpu_util(idx):
    """SM utilization % for local GPU idx. Prefer nvml (torch.cuda.utilization); fall back to nvidia-smi."""
    try:
        return float(torch.cuda.utilization(idx))
    except Exception:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", str(idx)],
                stderr=subprocess.DEVNULL, timeout=2).decode().strip().splitlines()
            return float(out[0])
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30, help="optimizer steps (each = --accum micro-batches)")
    ap.add_argument("--warmup", type=int, default=12, help="opt steps skipped before timing/peak (compile+autotune)")
    ap.add_argument("--ckpt_interval", type=int, default=1)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--use_liger", type=int, default=0, help="1 = fused linear-CE head (no full logits)")
    ap.add_argument("--accum", type=int, default=1, help="grad-accumulation micro-steps per optimizer step")
    ap.add_argument("--max_tokens_per_gpu", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = OmegaConf.load(CFG)
    acc = Accelerator(mixed_precision="bf16")
    mar.QuadtreeGPT.initialize_weights = lambda self: None
    torch.manual_seed(cfg.training.seed + acc.process_index)

    model = QuadtreeGPT(cfg)
    model.grad_ckpt_interval = int(args.ckpt_interval)
    model.grad_checkpointing = args.ckpt_interval > 0
    model.use_liger = bool(args.use_liger)  # fused linear-CE head (forward_varlen); see modeling/mar.py
    if args.compile:
        model.compile_blocks()  # same helper the production trainer calls (blocks only, dynamic=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.params.learning_rate),
                            betas=(0.9, 0.95), weight_decay=float(cfg.optimizer.params.weight_decay),
                            fused=True)
    model, opt = acc.prepare(model, opt)

    per_gpu = args.max_tokens_per_gpu or (cfg.varlen.global_max_tokens // acc.num_processes)
    n_params = sum(p.numel() for p in acc.unwrap_model(model).parameters()) / 1e6
    if acc.is_main_process:
        print(f"[cfg] params {n_params:.1f}M | world {acc.num_processes} | per_gpu {per_gpu} | accum {args.accum} "
              f"| ckpt_interval {args.ckpt_interval} | compile {args.compile} | use_liger {args.use_liger} "
              f"| grad_ckpt {args.ckpt_interval > 0}", flush=True)

    ds = VarlenPretokPacked(cfg.varlen.pretok_glob, per_gpu, rank=acc.process_index,
                            world=acc.num_processes, shuffle=True, seed=cfg.training.seed, loop=True)
    it = iter(ds)
    model.train()
    max_gn = float(cfg.training.max_grad_norm) if cfg.training.max_grad_norm else 0.0

    # background GPU-util sampler (main process' GPU), active only during the timed window
    util_samples, sampler_on = [], threading.Event()
    dev_idx = acc.local_process_index

    def _sample():
        while sampler_on.is_set():
            u = gpu_util(dev_idx)
            if u == u:  # not NaN
                util_samples.append(u)
            time.sleep(0.2)

    sampler_thread = threading.Thread(target=_sample, daemon=True)

    times, toks, loss0, lossN = [], [], None, None
    micro = [next(it) for _ in range(args.accum)]  # prefetch first opt-step's micro-batches
    for step in range(args.steps):
        if step == args.warmup:
            acc.wait_for_everyone(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            if acc.is_main_process:
                sampler_on.set(); sampler_thread.start()
        torch.cuda.synchronize(); t = time.time()
        opt.zero_grad(set_to_none=True)
        for j, mb in enumerate(micro):
            is_last = (j == args.accum - 1)
            # DDP: skip gradient allreduce on non-final micro-steps (accumulate locally, sync once)
            ctx = model.no_sync() if (args.accum > 1 and not is_last and hasattr(model, "no_sync")) else nullcontext()
            with ctx:
                loss, d = model(mb)
                acc.backward(loss / args.accum)  # average grads over the accum micro-batches
        if max_gn:
            acc.clip_grad_norm_(model.parameters(), max_gn)
        opt.step()
        torch.cuda.synchronize(); dt = time.time() - t
        nb = [next(it) for _ in range(args.accum)]  # prefetch next opt-step OUTSIDE timer
        step_tok = sum(int(mb["seqlens"].sum()) for mb in micro)  # tokens this opt-step (post-timer)
        times.append(dt); toks.append(step_tok)
        if step == 0: loss0 = float(loss)
        lossN = float(loss)
        if acc.is_main_process and (step < 6 or step % 5 == 0 or step == args.steps - 1):
            print(f"  optstep {step:>3}: {dt*1000:7.1f} ms | loss {float(loss):.3f} acc {float(d['acc']):.4f} "
                  f"| micro {args.accum} tok/optstep {step_tok}", flush=True)
        micro = nb

    torch.cuda.synchronize()
    sampler_on.clear()
    peak = torch.cuda.max_memory_allocated() // (1024 ** 2)
    tail = slice(args.warmup, None)
    t_tail = times[tail]; k_tail = toks[tail]
    s_per_optstep = sum(t_tail) / max(1, len(t_tail))
    tok_s = sum(k_tail) / max(1e-9, sum(t_tail))  # per-GPU tokens/sec (invariant to accum)
    util = sum(util_samples) / len(util_samples) if util_samples else float("nan")
    if acc.is_main_process:
        print(f"RESULT ckpt_interval={args.ckpt_interval} compile={args.compile} use_liger={args.use_liger} "
              f"accum={args.accum} s_per_optstep={s_per_optstep:.3f} s_per_micro={s_per_optstep/args.accum:.3f} "
              f"tok_s_per_gpu={tok_s:.0f} gpu_util_pct={util:.1f} peak_mem_MiB={peak} "
              f"loss0={loss0:.3f} lossN={lossN:.3f}", flush=True)


if __name__ == "__main__":
    main()
