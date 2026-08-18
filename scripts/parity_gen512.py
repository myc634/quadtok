"""Parity: QuadtreeGPT.forward (padded SDPA-causal) == forward_varlen (packed flash-varlen).
Same samples, eval() (no dropout) -> the ONLY difference is the attention impl; losses must match
(within bf16 SDPA-vs-flash tolerance). Proves the varlen generator forward is correct.

Run:  python scripts/parity_gen512.py
"""
import io, glob
import numpy as np
import torch
import webdataset as wds
from omegaconf import OmegaConf
import modeling.mar as mar
from modeling.mar import QuadtreeGPT

CFG = "configs/training/generator/gpt_quadtree_512.yaml"
DEV = "cuda"


def load_samples(pretok_glob, n=4):
    out = []
    for tar in sorted(glob.glob(pretok_glob)):
        for s in wds.WebDataset([tar], shardshuffle=False, empty_check=False):
            out.append((np.load(io.BytesIO(s["code_indices.npy"])).astype(np.int64),
                        np.load(io.BytesIO(s["lod_indices.npy"])).astype(np.int64),
                        np.load(io.BytesIO(s["patch_indices.npy"])).astype(np.int64),
                        int(s["cls"])))
            if len(out) >= n:
                return out
    return out


def main():
    mar.QuadtreeGPT.initialize_weights = lambda self: None  # skip the O(M*P) init stall (random weights fine)
    cfg = OmegaConf.load(CFG)
    torch.manual_seed(0)
    model = QuadtreeGPT(cfg).to(DEV).eval()
    print("params(M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
    print("freqs_cis len:", model.freqs_cis.shape[0])

    samples = load_samples(cfg.varlen.pretok_glob, n=4)
    B = len(samples); max_L = max(len(c) for c, *_ in samples)
    code_p = torch.full((B, max_L), -1, dtype=torch.long)
    lod_p = torch.full((B, max_L), -1, dtype=torch.long)
    patch_p = torch.zeros((B, max_L), dtype=torch.long)
    for i, (c, l, p, _) in enumerate(samples):
        L = len(c); code_p[i, :L] = torch.from_numpy(c); lod_p[i, :L] = torch.from_numpy(l); patch_p[i, :L] = torch.from_numpy(p)
    code_p, lod_p, patch_p = code_p.to(DEV), lod_p.to(DEV), patch_p.to(DEV)
    labels = torch.tensor([cl for *_, cl in samples], dtype=torch.long, device=DEV)
    print("sample lens:", [len(c) for c, *_ in samples])

    packed = {
        "code": torch.from_numpy(np.concatenate([c for c, *_ in samples])).to(DEV),
        "lod": torch.from_numpy(np.concatenate([l for _, l, _, _ in samples])).to(DEV),
        "patch": torch.from_numpy(np.concatenate([p for _, _, p, _ in samples])).to(DEV),
        "cls": labels.clone(),
        "seqlens": torch.tensor([len(c) for c, *_ in samples], dtype=torch.long, device=DEV),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # (ref) padded path (SDPA causal, vanilla materialized-logits CE)
        loss_pad, _ = model.forward(code_p[:, :-1].clone(), code_p.clone(),
                                    {"lod_indices": lod_p.clone(), "patch_indices": patch_p.clone()}, labels)
        # (b) vectorized no-host-loop varlen path, vanilla CE head
        model.use_liger = False
        loss_van, dvan = model.forward_varlen(packed)
        # (c) same varlen path, liger fused linear-CE head (skipped if liger not installed)
        model.use_liger = True; model._flce = None
        try:
            loss_lig, dlig = model.forward_varlen(packed)
            liger_ok = True
        except ImportError as e:
            print("[warn] liger not installed, skipping (c):", str(e).splitlines()[0]); liger_ok = False

    d_b = abs(loss_pad.item() - loss_van.item())
    print("=" * 64)
    print(f"(ref) loss padded  (SDPA, vanilla) = {loss_pad.item():.5f}")
    print(f"(b)   loss varlen  (flash, vanilla)= {loss_van.item():.5f}  | acc {float(dvan['acc']):.4f}")
    print(f"(b)   abs diff  padded vs varlen   = {d_b:.5f}  -> PARITY {'OK' if d_b < 0.03 else 'FAIL'} (<0.03 SDPA-vs-flash)")
    if liger_ok:
        d_c = abs(loss_van.item() - loss_lig.item())
        print(f"(c)   loss varlen  (flash, liger)  = {loss_lig.item():.5f}  | acc {float(dlig['acc']):.4f}")
        print(f"(c)   abs diff  vanilla vs liger   = {d_c:.6f}  -> PARITY {'OK' if d_c < 1e-2 else 'FAIL'} (<1e-2 fused-CE)")
    print("=" * 64)


if __name__ == "__main__":
    main()
