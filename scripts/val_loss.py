"""Measure the generator's next-token CE loss/acc on pretok data via forward_varlen (current codebase).
Run for the drive ckpt (base/mlp_ratio=4) and gen_100m (small) on the SAME data to compare."""
import os, sys, argparse, glob
REPO="/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update"
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch
from omegaconf import OmegaConf
from modeling.mar import QuadtreeGPT
from data.varlen_pretok_loader import build_varlen_loader

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gen_config", default="configs/inference/gpt_16k_base.yaml")
    ap.add_argument("--size", required=True)
    ap.add_argument("--mlp_ratio", type=int, default=1)
    ap.add_argument("--gen_ckpt", required=True)
    ap.add_argument("--data_dir", default="/mnt/localssd/valdata")
    ap.add_argument("--n_batches", type=int, default=30)
    a=ap.parse_args()
    dev="cuda"
    cfg=OmegaConf.load(a.gen_config); OmegaConf.set_struct(cfg,False)
    cfg.model.generator.model_size=a.size; cfg.model.generator.mlp_ratio=a.mlp_ratio; cfg.model.grad_checkpointing=False
    model=QuadtreeGPT(cfg).to(dev).eval().requires_grad_(False)
    meta=torch.load(a.gen_ckpt, map_location="cpu"); sd=meta["ema"] if isinstance(meta,dict) and "ema" in meta else meta
    msg=model.load_state_dict(sd, strict=False)
    print(f"[load] size={a.size} mlp={a.mlp_ratio} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} nparam={sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    shards=sorted(glob.glob(os.path.join(a.data_dir,"train-*.tar")))
    loader,per=build_varlen_loader(shards, max_token_num_global=20000, world=1, rank=0, num_workers=4, repeat=True)
    it=iter(loader); L=[]; A=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for i in range(a.n_batches):
            b=next(it); b={k:(v.to(dev,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
            loss,ld=model.forward_varlen(b["code_indices"],b["lod_indices"],b["patch_indices"],b["labels"],b["cu_seqlens"],b["seqlens"],b["max_seqlen"])
            L.append(ld["total_loss"].item()); A.append(ld["acc"].item())
    print(f"[RESULT] {a.size} ckpt={os.path.basename(a.gen_ckpt)}: CE_loss={np.mean(L):.4f}  acc={np.mean(A):.4f}  (over {a.n_batches} batches)", flush=True)

if __name__=="__main__":
    main()
