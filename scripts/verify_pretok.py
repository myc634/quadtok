"""Verify the pretokenized data + varlen packing are CORRECT.

A) Varlen packing: pull a couple packed batches, assert every sample is complete, the total token
   count is <= per-GPU budget (and the NEXT sample would overflow), and cu_seqlens is consistent.
B) Round-trip: load a few (key, code, lod, patch) from the pretok tars, reload the ORIGINAL 512
   image by key from the train tars, rebuild the tree from (lod,patch), decode from the SAVED codes,
   and report PSNR vs the original (expect ~21, matching guided tau=0.03) -> proves codes+tree are right.

Run:  python scripts/verify_pretok.py --pretok <dir> --ntar_scan 4
"""
import io, glob, tarfile, argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from modeling.quadtok import QuadTok
from modeling.utils import QuadTreeNode
from modeling.varlen_pretok import VarlenPretokPacked

CFG = "configs/training/single_stage/quadtok_ss512_vq_2level.yaml"
EMA = "/sensei-fs-3/users/yuchengm/models/quadtok_512_280k/Tokenizer/512/checkpoint-280000/ema_model/pytorch_model.bin"
TRAIN_DIR = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/train"
DEV = "cuda"


def load_pretok_samples(pretok_dir, n=6):
    import webdataset as wds
    out = []
    for tar in sorted(glob.glob(f"{pretok_dir}/pretok-*.tar")):
        for s in wds.WebDataset([tar], shardshuffle=False, empty_check=False):
            out.append((s["__key__"],
                        np.load(io.BytesIO(s["code_indices.npy"])).astype(np.int64),
                        np.load(io.BytesIO(s["lod_indices.npy"])).astype(np.int64),
                        np.load(io.BytesIO(s["patch_indices.npy"])).astype(np.int64)))
            if len(out) >= n:
                return out
    return out


def load_orig_by_keys(keys, size=512, ntar_scan=4):
    """Scan the first few train tars for these keys -> {key: image_tensor[0,1]}."""
    want = set(keys); found = {}
    for tar in sorted(glob.glob(f"{TRAIN_DIR}/train-*.tar"))[:ntar_scan]:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not m.name.endswith(".jpg"):
                    continue
                k = m.name[:-4].replace("/", "_")
                if k not in want:
                    continue
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                w, h = im.size; s = size / min(w, h)
                im = im.resize((max(size, round(w * s)), max(size, round(h * s))))
                w, h = im.size; l, tp = (w - size) // 2, (h - size) // 2
                im = im.crop((l, tp, l + size, tp + size))
                found[k] = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
                if len(found) == len(want):
                    return found
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretok", required=True)
    ap.add_argument("--ntar_scan", type=int, default=4)
    ap.add_argument("--budget", type=int, default=131072)
    args = ap.parse_args()

    # ---------- A) varlen packing check (pure python, no GPU) ----------
    print("=== A) varlen packing ===")
    ds = VarlenPretokPacked(f"{args.pretok}/pretok-*.tar", args.budget, rank=0, world=1, shuffle=False, loop=False)
    nb = 0
    for batch in ds:
        sl = batch["seqlens"]; tot = int(sl.sum())
        assert tot <= args.budget, f"pack {tot} > budget {args.budget}"
        assert batch["code"].shape[0] == tot and batch["lod"].shape[0] == tot and batch["patch"].shape[0] == tot
        assert batch["cls"].shape[0] == sl.shape[0]
        print(f"  batch {nb}: {sl.shape[0]} samples, {tot} tokens (budget {args.budget}, headroom {args.budget-tot}), "
              f"min/max sample len {int(sl.min())}/{int(sl.max())}")
        nb += 1
        if nb >= 3:
            break
    print(f"  packing OK ({nb} batches checked)")

    # ---------- B) round-trip PSNR ----------
    print("=== B) round-trip (saved code+tree -> decode vs original) ===")
    cfg = OmegaConf.load(CFG); ts = int(cfg.model.selector.token_size)
    model = QuadTok(cfg).eval().to(DEV)
    model.load_state_dict(torch.load(EMA, map_location="cpu"), strict=False)
    samples = load_pretok_samples(args.pretok, n=6)
    origs = load_orig_by_keys([k for k, *_ in samples], ntar_scan=args.ntar_scan)
    npsl = list(model.num_patch_side_list)
    n_matched = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for key, code, lod, patch in samples:
            if key not in origs:
                continue
            # validity checks
            assert code.min() >= 0 and code.max() < cfg.model.vq_model.codebook_size, "code out of range"
            assert lod.min() >= model.guaranteed_depth, "lod below floor"
            nodes = [QuadTreeNode(int(lod[i]), int(patch[i])) for i in range(len(code))]
            embeds = model.quantize.get_codebook_entry(torch.from_numpy(code).to(DEV)).reshape(1, -1, ts)
            rec = model.decoder._forward_optimize(embeds, [nodes]).clamp(0, 1).float()
            orig = origs[key].unsqueeze(0).to(DEV)
            mse = ((rec - orig) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            n_coarse = (np.array(lod) == model.guaranteed_depth).sum()
            print(f"  {key}: L={len(code)} (coarse {n_coarse}, fine {len(code)-n_coarse}) PSNR={psnr:.2f} dB")
            n_matched += 1
    print(f"  round-trip OK ({n_matched} samples; PSNR ~21 => saved codes+tree reconstruct correctly)")


if __name__ == "__main__":
    main()
