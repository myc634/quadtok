"""Varlen QuadtreeGPT generator training (flash_attn_varlen, no padding).

Config via env:
  SIZE=small|base|xlarge   MAX_TOKEN_GLOBAL (default 232448 ~ 1024 imgs)   GC=0|1
  LR=4e-4  END_LR=2e-5  WARMUP=50000  STEPS=400000  SAVE_EVERY=2500  WD=0.05
  DATA_DIR=/mnt/localssd/pretok   CKPT_S3=s3://.../ckpt/<run>/   RUN_NAME=...
  WANDB_KEY WANDB_ENTITY WANDB_PROJECT WANDB_BASE_URL (optional)

Global batch is kept identical across sizes: per-GPU token budget = MAX_TOKEN_GLOBAL // world,
so 1-node (8 gpu) and 2-node (16 gpu) both see MAX_TOKEN_GLOBAL tokens/step, grad-accum=1.
Resumes from the latest step_N in CKPT_S3 (pulled to localssd). Every SAVE_EVERY steps saves
accelerate state + EMA + meta to localssd/step_N and async-uploads to CKPT_S3/step_N/ (keeps ALL).
"""
import os, sys, glob, math, time, copy, threading, subprocess, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
import modeling.mar as marmod
from data.varlen_pretok_loader import build_varlen_loader

CFG = os.environ.get("CFG", "configs/inference/gpt_16k_base.yaml")


def env(k, d):
    return os.environ.get(k, d)


def s5(*args):
    subprocess.run(["s5cmd", *args], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def latest_step_on_s3(ckpt_s3):
    try:
        out = subprocess.run(["s5cmd", "ls", ckpt_s3], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return -1
    steps = []
    for line in out.splitlines():
        for tok in line.split():
            if tok.startswith("step_"):
                try:
                    steps.append(int(tok.strip("/").split("_")[1]))
                except Exception:
                    pass
    return max(steps) if steps else -1


def cosine_lr(step, base, end, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return end + 0.5 * (base - end) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def ema_update(ema, model, decay):
    mp = dict(model.named_parameters())
    for n, p in ema.named_parameters():
        p.mul_(decay).add_(mp[n].detach(), alpha=1 - decay)


def main():
    acc = Accelerator(mixed_precision="bf16")
    dev = acc.device
    size = env("SIZE", "small")
    max_tok_global = int(env("MAX_TOKEN_GLOBAL", "232448"))
    lr = float(env("LR", "4e-4")); end_lr = float(env("END_LR", "2e-5"))
    warmup = int(env("WARMUP", "50000")); steps = int(env("STEPS", "400000"))
    save_every = int(env("SAVE_EVERY", "2500")); wd = float(env("WD", "0.05"))
    ema_decay = float(env("EMA_DECAY", "0.9999"))
    data_dir = env("DATA_DIR", "/mnt/localssd/pretok")
    ckpt_s3 = env("CKPT_S3", "").rstrip("/") + "/"
    ckpt_local = env("CKPT_LOCAL", "/mnt/localssd/ckpt")
    run_name = env("RUN_NAME", f"gen_varlen_{size}")
    os.makedirs(ckpt_local, exist_ok=True)

    cfg = OmegaConf.load(CFG)
    cfg.model.generator.model_size = size
    cfg.model.grad_checkpointing = (env("GC", "0") == "1")
    model = marmod.QuadtreeGPT(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    shards = sorted(glob.glob(os.path.join(data_dir, "train-*.tar")))
    loader, per_gpu = build_varlen_loader(shards, max_token_num_global=max_tok_global,
                                          world=acc.num_processes, rank=acc.process_index,
                                          num_workers=int(env("NUM_WORKERS", "6")), repeat=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    model, opt = acc.prepare(model, opt)

    # ---- wandb (optional) ----
    use_wandb = acc.is_main_process and env("WANDB_KEY", "")
    if use_wandb:
        import wandb
        os.environ["WANDB_BASE_URL"] = env("WANDB_BASE_URL", "https://api.wandb.ai")
        wandb.login(key=env("WANDB_KEY", ""))
        wandb.init(project=env("WANDB_PROJECT", "quadtok-gen"), entity=env("WANDB_ENTITY", None),
                   name=run_name, id=run_name, resume="allow",
                   config=dict(size=size, params=nparam, max_tok_global=max_tok_global,
                               per_gpu=per_gpu, lr=lr, steps=steps, gc=cfg.model.grad_checkpointing))

    # ---- resume ----
    start_step = 0
    if ckpt_s3 != "/":
        last = latest_step_on_s3(ckpt_s3)
        if last >= 0:
            d = os.path.join(ckpt_local, f"step_{last}")
            if acc.is_main_process:
                os.makedirs(d, exist_ok=True); s5("cp", f"{ckpt_s3}step_{last}/*", d + "/")
            acc.wait_for_everyone()
            acc.load_state(os.path.join(d, "state"))
            meta_path = os.path.join(d, "meta.pt")
            if os.path.exists(meta_path):
                meta = torch.load(meta_path, map_location="cpu")
                ema.load_state_dict(meta["ema"]); start_step = meta["step"]
                if acc.is_main_process:
                    print(f"[resume] from step {start_step}", flush=True)
            else:
                # meta.pt missing on S3 (async dir-upload dropped the tiny meta.pt on a
                # preempt while state/ made it up). Tolerate instead of crash-looping:
                # model/opt/RNG are already restored from state/; take the step from the
                # ckpt dir name and re-init EMA from the restored weights (EMA re-warms).
                start_step = last
                ema.load_state_dict(acc.unwrap_model(model).state_dict())
                if acc.is_main_process:
                    print(f"[resume] meta.pt MISSING for step_{last}; model/opt restored from "
                          f"state/, EMA re-init from model, start_step={start_step}", flush=True)
    ema = ema.to(dev)

    if acc.is_main_process:
        print(f"[train] size={size} params={nparam/1e6:.1f}M world={acc.num_processes} "
              f"per_gpu_tok={per_gpu} global_tok={max_tok_global} gc={cfg.model.grad_checkpointing} "
              f"start_step={start_step} steps={steps}", flush=True)

    def save_ckpt(step):
        d = os.path.join(ckpt_local, f"step_{step}")
        acc.save_state(os.path.join(d, "state"))
        if acc.is_main_process:
            torch.save({"step": step, "ema": ema.state_dict()}, os.path.join(d, "meta.pt"))
            # async upload to S3 (keep all step_N); prune only very local dirs to save localssd
            if ckpt_s3 != "/":
                threading.Thread(target=s5, args=("cp", d + "/", f"{ckpt_s3}step_{step}/"), daemon=True).start()
            for old in sorted(glob.glob(os.path.join(ckpt_local, "step_*")))[:-3]:
                shutil.rmtree(old, ignore_errors=True)   # keep last 3 local; S3 keeps all

    unwrapped = acc.unwrap_model(model)
    it = iter(loader)
    model.train()
    t0 = time.perf_counter(); tok_acc = 0
    for step in range(start_step, steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, lr, end_lr, warmup, steps)
        b = next(it)
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        with acc.autocast():
            loss, ld = unwrapped.forward_varlen(b["code_indices"], b["lod_indices"], b["patch_indices"],
                                                b["labels"], b["cu_seqlens"], b["seqlens"], b["max_seqlen"])
        opt.zero_grad(set_to_none=True)
        acc.backward(loss)
        acc.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        ema_update(ema, unwrapped, ema_decay)
        tok_acc += int(b["code_indices"].shape[0])

        if acc.is_main_process and step % 50 == 0:
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            tps = tok_acc * acc.num_processes / dt
            print(f"step {step} loss {ld['total_loss'].item():.4f} acc {ld['acc'].item():.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} {tps:.0f} tok/s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"loss": ld["total_loss"].item(), "acc": ld["acc"].item(),
                           "lr": opt.param_groups[0]["lr"], "tok_per_s": tps, "step": step})
            t0 = time.perf_counter(); tok_acc = 0
        if (step + 1) % save_every == 0 or step + 1 == steps:
            save_ckpt(step + 1)
            acc.wait_for_everyone()
            if acc.is_main_process:
                print(f"[ckpt] saved step_{step+1}", flush=True)

    if acc.is_main_process:
        print("[train] done", flush=True)


if __name__ == "__main__":
    main()
