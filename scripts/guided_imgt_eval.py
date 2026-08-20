"""Guided search (tau=0.05) recon eval with the CORRECT ImageTransform.eval_transform (BILINEAR)
preprocessing, to reproduce the reference Guided row: ~230 tok / PSNR 20.37 / rFID 1.46.

Monkeypatches probe256_2level_eval.val_stream -> ImageTransform version, then calls its main().
Launch: accelerate launch --num_processes N guided_imgt_eval.py --tau 0.05
"""
import os, sys, io, glob, tarfile
REPO = "/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update"
sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from PIL import Image
import scripts.probe256_2level_eval as P
from data.webdataset_reader import ImageTransform

# EXACT reference eval preprocessing: transforms.Resize(256) [BILINEAR default] + CenterCrop(256) -> [0,1]
_imgt = ImageTransform(resize_shorter_edge=256, crop_size=256, random_crop=False, random_flip=False,
                       normalize_mean=[0., 0., 0.], normalize_std=[1., 1., 1.]).eval_transform


def val_stream_imgt(rank, world, bs, size=256, cap=None):
    tars = sorted(glob.glob(f"{P.VAL_DIR}/val-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    buf, n = [], 0
    for tar in mine:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not (m.name.endswith(".jpg") or m.name.endswith(".jpeg")
                        or m.name.endswith(".png") or m.name.endswith(".JPEG")):
                    continue
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                buf.append(_imgt(im))
                n += 1
                if len(buf) == bs:
                    yield torch.stack(buf); buf = []
                if cap and n >= cap:
                    if buf:
                        yield torch.stack(buf)
                    return
    if buf:
        yield torch.stack(buf)


P.val_stream = val_stream_imgt  # <- swap in BILINEAR ImageTransform preprocessing
P.main()
