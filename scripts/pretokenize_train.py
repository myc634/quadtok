"""Fast parallel tensor-native pretokenization of ImageNet-train at tau=0.05.

Each rank processes the train tars with idx % world == rank; runs the tensor-native guided
search (flex Kinship cores), and writes per-sample {code_indices, lod_indices, patch_indices,
cls} to one output tar per input shard (webdataset, compatible with PretokenizedDataset).
Codes are tiny (~2 KB/sample). Background-thread prefetch overlaps JPEG decode with GPU.

accelerate launch --num_processes 8 scripts/pretokenize_train.py --out_dir <dir> [--shards A B] [--crop center|random]
"""
import os, sys, io, glob, tarfile, random, argparse, threading, queue, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import numpy as np
import torch
import lpips as lpips_pkg
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
import webdataset as wds
from modeling.quadtok import QuadTok
from data.augmentation import center_crop_arr
from scripts.bench256_tensor import build_slot_maps, active_to_padded, guided_search_fast
from scripts.probe256_2level_eval import CFG_DEFAULT, CKPT_DEFAULT

TRAIN_DIR = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/train"


def prefetch(gen, n=4):
    q = queue.Queue(maxsize=n); SENT = object()

    def worker():
        for x in gen:
            q.put(x)
        q.put(SENT)
    threading.Thread(target=worker, daemon=True).start()
    while True:
        x = q.get()
        if x is SENT:
            break
        yield x


_to_tensor = transforms.ToTensor()
_tencrop = transforms.Compose([
    transforms.Lambda(lambda im: center_crop_arr(im, int(256 * 1.1))),  # crop_range 1.1
    transforms.TenCrop(256),
    transforms.Lambda(lambda crops: torch.stack([_to_tensor(c) for c in crops])),  # [10,3,256,256]
])


def aug_crops(im, mode):
    """mode='center_hflip' (DiT/MAR/LlamaGen-default): center_crop_arr(256) -> [orig, h-flip] (2 crops).
    mode='tencrop' (LlamaGen option): center_crop_arr(281) -> TenCrop(256) (10 crops). All in [0,1]."""
    if mode == "tencrop":
        return _tencrop(im)                             # [10,3,256,256]
    t = _to_tensor(center_crop_arr(im, 256))            # [3,256,256]
    return torch.stack([t, torch.flip(t, dims=[-1])])   # [2,3,256,256] orig + h-flip


def shard_stream(tar_path, bs, aug):
    """Yield (crops[B,3,256,256], class_ids[B], keys[B]) from one train tar; each source image
    contributes 2 (center+hflip) or 10 (tencrop) samples (keys = <base>_<j>)."""
    imgs, cls, keys = [], [], []
    with tarfile.open(tar_path) as t:
        members = {}
        for m in t.getmembers():
            if "." not in m.name:
                continue
            base, ext = m.name.rsplit(".", 1)
            members.setdefault(base, {})[ext] = m
        for base in sorted(members):
            d = members[base]
            if "jpg" not in d or "cls" not in d:
                continue
            try:
                im = Image.open(io.BytesIO(t.extractfile(d["jpg"]).read())).convert("RGB")
                crops = aug_crops(im, aug)   # [2 or 10, 3, 256, 256]
                c = int(t.extractfile(d["cls"]).read().decode().strip())
            except Exception:
                continue
            for j in range(crops.shape[0]):
                imgs.append(crops[j]); cls.append(c); keys.append(f"{base}_{j}")
                if len(imgs) == bs:
                    yield torch.stack(imgs), torch.tensor(cls), keys
                    imgs, cls, keys = [], [], []
    if imgs:
        yield torch.stack(imgs), torch.tensor(cls), keys


@torch.no_grad()
def extract_codes(model, latent, active, maps, ts):
    """Final guided tree -> per-token VQ codes (no decode)."""
    lod_pad, pat_pad, seqlens = active_to_padded(active, maps[0], maps[1])
    z = model.selector._select_optimize_core(latent, lod_pad, pat_pad, seqlens)
    _, rd = model.quantize(z)
    codes = rd["min_encoding_indices"].reshape(active.shape[0], -1)  # (B, max_seq)
    return codes, lod_pad, pat_pad, seqlens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--aug", choices=["center_hflip", "tencrop"], default="center_hflip",
                    help="center_hflip=2x (DiT/MAR/LlamaGen-default); tencrop=10x (LlamaGen option)")
    ap.add_argument("--shards", type=int, nargs=2, default=None, help="[start end) input shard range; default all")
    ap.add_argument("--cfg", default=CFG_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc = Accelerator()
    dev = acc.device
    rank, world = acc.process_index, acc.num_processes
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    ts = int(cfg.model.selector.token_size)
    model = QuadTok(cfg).eval().to(dev)
    model.requires_grad_(False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
    lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True).to(dev).eval()
    maps = build_slot_maps(dev)

    all_tars = sorted(glob.glob(f"{TRAIN_DIR}/train-*.tar"))
    if args.shards is not None:
        all_tars = all_tars[args.shards[0]:args.shards[1]]
    my_tars = [t for i, t in enumerate(all_tars) if i % world == rank]
    if acc.is_main_process:
        print(f"[pretok] world={world} total_shards={len(all_tars)} tau={args.tau} bs={args.bs} aug={args.aug}", flush=True)

    rng = random.Random(args.seed + rank)
    n_done = 0
    for tar_path in my_tars:
        shard_idx = int(os.path.basename(tar_path).split("-")[1].split(".")[0])
        out_path = os.path.join(args.out_dir, f"train-{shard_idx:06d}.tar")
        tmp_path = out_path + ".tmp"
        writer = wds.TarWriter(tmp_path)
        n_shard, tok_sum = 0, 0
        t_shard = time.perf_counter()
        for images, class_ids, keys in prefetch(shard_stream(tar_path, args.bs, args.aug)):
            images = images.to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                latent, active = guided_search_fast(model, lpips_fn, images, args.tau, maps, rng)
                codes, lod_pad, pat_pad, seqlens = extract_codes(model, latent, active, maps, ts)
            codes = codes.cpu().numpy(); lodp = lod_pad.cpu().numpy(); patp = pat_pad.cpu().numpy()
            sl = seqlens.cpu().numpy()
            for b in range(len(keys)):
                L = int(sl[b])
                writer.write({
                    "__key__": keys[b],
                    "code_indices.npy": codes[b, :L].astype(np.int32),
                    "lod_indices.npy": lodp[b, :L].astype(np.int16),
                    "patch_indices.npy": patp[b, :L].astype(np.int16),
                    "cls": str(int(class_ids[b])),
                })
                n_shard += 1; tok_sum += L
        writer.close()
        os.replace(tmp_path, out_path)
        n_done += n_shard
        dt = time.perf_counter() - t_shard
        print(f"[rank{rank}] shard {shard_idx:06d}: {n_shard} samples in {dt:.1f}s "
              f"({n_shard/dt:.1f} img/s/gpu) avg_tok {tok_sum/max(n_shard,1):.1f} -> {out_path}", flush=True)

    acc.wait_for_everyone()
    tot = torch.tensor(float(n_done), device=dev)
    tot = acc.reduce(tot, reduction="sum").item()
    if acc.is_main_process:
        print(f"[pretok DONE] total samples written across ranks: {int(tot)}", flush=True)


if __name__ == "__main__":
    main()
