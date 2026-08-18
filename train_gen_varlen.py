"""3-level QuadtreeGPT generator training with VARLEN packing + flash-attn varlen.

Token-based batching (VarlenPackedDataset -> cu_seqlens, no padding). The generator's
varlen forward is monkey-patched onto QuadtreeGPT.forward so DDP wraps it and grads sync.
Sequence per segment: [cls] + (code[k-1]+pos[k-1]); target[p]=code[p] (causal, coarse->fine).
"""
import os, sys, time, argparse
REPO = "/sensei-fs-3/users/yuchengm/code/quadtok/3level"
sys.path.insert(0, REPO)
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from accelerate import Accelerator
from modeling.mar import QuadtreeGPT
from data.varlen_reader import VarlenPackedDataset


def varlen_forward(self, batch):
    """self = raw QuadtreeGPT. batch dict of CPU tensors -> (loss, acc, ntok)."""
    dev = self.output.weight.device
    code = batch["code"].to(dev, non_blocking=True)      # [T] long
    lod = batch["lod"].to(dev, non_blocking=True)
    patch = batch["patch"].to(dev, non_blocking=True)
    cls = batch["cls"].to(dev, non_blocking=True)        # [n_seg]
    cu = batch["cu_seqlens"].to(dev, non_blocking=True).long()   # [n_seg+1]
    T = int(code.numel()); nseg = int(cls.numel())
    seg_lens = cu[1:] - cu[:-1]                            # [n_seg]
    max_seqlen = int(seg_lens.max().item())
    seg_id = torch.repeat_interleave(torch.arange(nseg, device=dev), seg_lens)   # [T]
    local_pos = torch.arange(T, device=dev) - cu[:-1][seg_id]                     # [T] 0..L_i-1
    is_start = torch.zeros(T, dtype=torch.bool, device=dev); is_start[cu[:-1]] = True
    # tree pos-emb: reuse the model's get_token_indices_embedding on a padded [n_seg,max_L] view
    lod_pad = torch.full((nseg, max_seqlen), -1, dtype=torch.long, device=dev)
    pat_pad = torch.zeros((nseg, max_seqlen), dtype=torch.long, device=dev)
    lod_pad[seg_id, local_pos] = lod
    pat_pad[seg_id, local_pos] = patch
    pos_emb = self.get_token_indices_embedding({"lod_indices": lod_pad, "patch_indices": pat_pad})[seg_id, local_pos]  # [T,D]
    # input = shifted (code[p-1]+pos[p-1]); cls at segment starts; target = code[p]
    tok_emb = self.tok_embeddings(torch.roll(code, 1, 0).clamp(0))
    inp = tok_emb + torch.roll(pos_emb, 1, 0)
    cls_emb = self.cls_embedding(cls, train=self.training)[:, :self.cls_token_num].squeeze(1)  # [n_seg,D]
    inp[is_start] = cls_emb.to(inp.dtype)
    x = self.tok_dropout(inp).unsqueeze(0)                # [1,T,D]
    freqs = self.freqs_cis[local_pos].unsqueeze(0)        # [1,T,hd//2,2]
    cu32 = cu.to(torch.int32)
    ms = self.seq_len            # constant upper-bound max_seqlen for flash-varlen -> compile-stable
    ckpt_every = getattr(self, "ckpt_every", 1)          # 1=all ckpt (min mem), 0=none, N=every Nth layer
    for i, blk in enumerate(self.blocks):
        if self.training and ckpt_every > 0 and (i % ckpt_every == 0):
            x = checkpoint(blk, x, freqs, None, None, cu32, ms, use_reentrant=False)
        else:
            x = blk(x, freqs, start_pos=None, mask=None, cu_seqlens=cu32, max_seqlen=ms)
    x = self.out_norm(x)[0]                               # [T,D]
    logits = self.output(x).float()                       # [T,V]
    loss = F.cross_entropy(logits, code)
    with torch.no_grad():
        acc = (logits.argmax(-1) == code).float().mean()
    return loss, acc, T, nseg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--max_tokens_global", type=int, default=262144 * 4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=1,
                    help="1=checkpoint EVERY layer (min mem); 0=NO checkpointing (max speed); N=checkpoint every Nth layer")
    ap.add_argument("--compile", action="store_true", help="torch.compile each TransformerBlock (dynamic=True)")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="accumulate N microbatches per optimizer step (keep a large global batch with a small microbatch)")
    args = ap.parse_args()

    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.grad_accum)
    cfg = OmegaConf.load(args.config)
    model = QuadtreeGPT(cfg)
    model.ckpt_every = args.ckpt_every
    model.forward = varlen_forward.__get__(model, QuadtreeGPT)   # DDP will wrap this
    if args.compile:
        # Compile ONLY blocks that are NOT checkpointed. torch.compile INSIDE
        # torch.utils.checkpoint -> CheckpointError (saved-tensor count mismatch), so
        # compiled blocks must never be wrapped in checkpoint().
        ce = args.ckpt_every
        n_comp = 0
        for i in range(len(model.blocks)):
            is_ckpt = ce > 0 and (i % ce == 0)
            if not is_ckpt:
                model.blocks[i] = torch.compile(model.blocks[i], dynamic=True)
                n_comp += 1
        print("[compile] compiled %d/%d blocks (the non-checkpointed ones)" % (n_comp, len(model.blocks)), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    model, opt = acc.prepare(model, opt)

    max_tokens_gpu = args.max_tokens_global // acc.num_processes
    if acc.is_main_process:
        print("world=%d max_tokens_global=%d per_gpu=%d model=%.1fM ckpt_every=%d compile=%s" % (
            acc.num_processes, args.max_tokens_global, max_tokens_gpu,
            sum(p.numel() for p in acc.unwrap_model(model).parameters()) / 1e6,
            args.ckpt_every, args.compile), flush=True)
    ds = VarlenPackedDataset(args.shards, max_tokens_gpu, num_workers_per_gpu=args.num_workers)

    ga = max(1, args.grad_accum)
    step = 0; t0 = time.time(); run_loss = 0.0; toks = 0
    for batch in ds.dataloader:
        with acc.accumulate(model):
            loss, a, ntok, nseg = model(batch)
            acc.backward(loss / ga)          # mean over the ga microbatches
            opt.step(); opt.zero_grad()      # gated by accumulate -> only fires on the ga-th
        run_loss += loss.item(); toks += ntok
        if acc.sync_gradients:               # True only on the microbatch that actually stepped
            torch.cuda.synchronize()
            if acc.is_main_process:
                dt = time.time() - t0; t0 = time.time()
                mem = torch.cuda.max_memory_allocated() / 1e9
                print("step %3d | loss %.4f | ga=%d toks=%d | %.2fs peakmem %.1fGB" % (
                    step, run_loss / ga, ga, toks, dt, mem), flush=True)
            step += 1; run_loss = 0.0; toks = 0
            if step >= args.steps:
                break
    acc.print("SMOKE_DONE")


if __name__ == "__main__":
    main()
