"""Generate N c2i images for ONE CFG/expansion setting -> save PNGs to a dir.
QuadtreeGPT (100m/small) + QuadTok VQ decoder. Multi-GPU via accelerate.

Tree at inference = probabilistic quadtree (guaranteed_depth=3, expansion_probs=[EXP]).
We monkey-patch build_probabilistic_quadtree so generate() uses the requested EXP
(the model's generate() otherwise hardcodes 0.75) -- NO edit to mar.py.

Decode: result_tokens (B,S) code indices -> quantize.get_codebook_entry (B*S->emb)
-> (B, token_size, S) -> tok.decode(z, ordered_nodes) -> image in [0,1].
"""
import os, sys, argparse, glob, subprocess, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from accelerate import Accelerator

import modeling.mar as marmod
from modeling.mar import QuadtreeGPT
from modeling.quadtok import QuadTok


def s5(*a):
    subprocess.run(["s5cmd", *a], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_gen(gen_config, size, gen_ckpt, device, mlp_ratio=1):
    cfg = OmegaConf.load(gen_config)
    OmegaConf.set_struct(cfg, False)
    cfg.model.generator.model_size = size
    cfg.model.generator.mlp_ratio = mlp_ratio   # 4 = old drive ckpt (buggy FFN 11008); 1 = base-update
    cfg.model.grad_checkpointing = False
    model = QuadtreeGPT(cfg).to(device).eval().requires_grad_(False)
    # gen_ckpt is a local meta.pt (has {"ema": state_dict, "step": int})
    meta = torch.load(gen_ckpt, map_location="cpu")
    sd = meta["ema"] if "ema" in meta else meta
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load_gen] step={meta.get('step','?')} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    return model


def load_tok(tok_config, tok_ckpt, device):
    cfg = OmegaConf.load(tok_config)
    tok = QuadTok(cfg).to(device).eval().requires_grad_(False)
    sd = torch.load(tok_ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    msg = tok.load_state_dict(sd, strict=False)
    print(f"[load_tok] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    return tok


@torch.no_grad()
def decode_tokens(tok, result_tokens, ordered_nodes):
    B, S = result_tokens.shape
    zq = tok.quantize.get_codebook_entry(result_tokens.reshape(-1).long())   # (B*S, token_size)
    zq = zq.reshape(B, S, -1).contiguous()                                   # (B, S, token_size) -- decoder_embed wants channels-LAST
    img = tok.decode(zq, ordered_nodes)                                      # (B, 3, H, W) ~[0,1]
    return img.clamp(0, 1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_config", default="configs/inference/gpt_16k_base.yaml")
    ap.add_argument("--tok_config", default="configs/training/single_stage/quadtok_ss256_vq.yaml")
    ap.add_argument("--tok_ckpt", default="/sensei-fs-3/users/yuchengm/models/quadtok_2level_drive/drive_ckpt.download")
    ap.add_argument("--gen_ckpt", required=True, help="local meta.pt with EMA")
    ap.add_argument("--size", default="small")
    ap.add_argument("--mlp_ratio", type=int, default=1)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--cfg_decay", default="constant", choices=["constant", "linear", "power-cosine", "lod_scheduler"])
    ap.add_argument("--cfg_pow", type=float, default=1.3)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--expansion", type=float, default=0.75)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc = Accelerator()
    dev = acc.device
    rank, world = acc.process_index, acc.num_processes

    # ---- monkey-patch tree expansion (generate() otherwise hardcodes 0.75) ----
    _orig = marmod.build_probabilistic_quadtree
    exp = float(args.expansion)
    def _patched(npsl, guaranteed_depth=3, expansion_probs=None):
        return _orig(npsl, guaranteed_depth=3, expansion_probs=[exp])
    marmod.build_probabilistic_quadtree = _patched

    model = load_gen(args.gen_config, args.size, args.gen_ckpt, dev, mlp_ratio=args.mlp_ratio)
    tok = load_tok(args.tok_config, args.tok_ckpt, dev)
    num_classes = int(model.num_classes)

    os.makedirs(args.out, exist_ok=True)
    # split n across ranks
    per = args.n // world + (1 if rank < args.n % world else 0)
    g = torch.Generator(device="cpu").manual_seed(args.seed * 100003 + rank)
    saved = 0
    nb = math.ceil(per / args.bs)
    if acc.is_main_process:
        print(f"[gen] n={args.n} world={world} per_rank~{per} bs={args.bs} cfg={args.cfg_scale}/{args.cfg_decay}/pow{args.cfg_pow} exp={exp} temp={args.temp} -> {args.out}", flush=True)
    with torch.no_grad():
        for b in range(nb):
            cur = min(args.bs, per - b * args.bs)
            if cur <= 0:
                break
            labels = torch.randint(0, num_classes, (cur,), generator=g).to(dev).long()
            # generate() in fp32 (NO autocast): KV cache is alloc'd at class_embedding dtype (fp32);
            # autocast would make k/v bf16 -> "index put dtype mismatch". fp32 gen is correct.
            result_tokens, ordered_nodes = model.generate(
                labels,
                guidance_scale=args.cfg_scale,
                guidance_decay=args.cfg_decay,
                guidance_scale_pow=args.cfg_pow,
                randomize_temperature=args.temp,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                imgs = decode_tokens(tok, result_tokens, ordered_nodes)       # (cur,3,H,W)
            arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            for i in range(cur):
                Image.fromarray(arr[i]).save(os.path.join(args.out, f"{rank:02d}_{saved:06d}.png"))
                saved += 1
            if acc.is_main_process and b % 5 == 0:
                print(f"[gen] batch {b+1}/{nb} saved~{saved}", flush=True)
    acc.wait_for_everyone()
    if acc.is_main_process:
        total = len(glob.glob(os.path.join(args.out, "*.png")))
        print(f"[gen] DONE total_pngs={total} in {args.out}", flush=True)


if __name__ == "__main__":
    main()
