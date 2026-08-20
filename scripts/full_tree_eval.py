"""Reproduce the FULL-tree recon result (all 64 cells split -> 320 tokens) on the FULL 50k
ImageNet-val (no search). rFID needs 50k to be accurate. Compare to reported Full: rFID 1.50 / PSNR 20.39."""
import os, sys, argparse
REPO="/sensei-fs-3/users/yuchengm/code/quadtok/quadtok-base-update"
sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from modeling.quadtok import QuadTok
from modeling.utils import build_quadtree, _get_nodes_at_level
from eval.utils.evaluator import VQGANEvaluator
from scripts.probe256_2level_eval import (
    val_stream, recon_opt, ordered_ge3, all_reduce_eval,
    _build_tree_from_node_mask, coarse_split_permutation, inverse_permutation,
    CFG_DEFAULT, CKPT_DEFAULT, VAL_DIR,
)
import io, glob, tarfile
import numpy as np
from PIL import Image
from torchvision import transforms
from data.augmentation import center_crop_arr
from data.webdataset_reader import ImageTransform
_to_tensor = transforms.ToTensor()
_imgt = ImageTransform(resize_shorter_edge=256, crop_size=256, random_crop=False, random_flip=False,
                       normalize_mean=[0.,0.,0.], normalize_std=[1.,1.,1.]).eval_transform  # reference eval preproc (BILINEAR)

def val_stream_imgt(rank, world, bs, cap=None):
    """val loader using the EXACT ImageTransform.eval_transform (Resize(256)=BILINEAR + CenterCrop) -> reference preproc."""
    tars = sorted(glob.glob(f"{VAL_DIR}/val-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    buf, n = [], 0
    for tar in mine:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                buf.append(_imgt(im))
                n += 1
                if len(buf) == bs:
                    yield torch.stack(buf); buf = []
                if cap and n >= cap:
                    if buf: yield torch.stack(buf)
                    return
    if buf:
        yield torch.stack(buf)

def val_stream_cca(rank, world, bs, cap=None):
    """val loader using the TRAINING preprocessing: center_crop_arr(256) (DiT BOX+BICUBIC) -> ToTensor."""
    tars = sorted(glob.glob(f"{VAL_DIR}/val-*.tar"))
    mine = [t for i, t in enumerate(tars) if i % world == rank]
    buf, n = [], 0
    for tar in mine:
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                buf.append(_to_tensor(center_crop_arr(im, 256)))   # [3,256,256] in [0,1]
                n += 1
                if len(buf) == bs:
                    yield torch.stack(buf); buf = []
                if cap and n >= cap:
                    if buf: yield torch.stack(buf)
                    return
    if buf:
        yield torch.stack(buf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None)   # None = full 50k val
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--fp32", action="store_true", help="disable bf16 autocast (match fp32 reference eval)")
    ap.add_argument("--cca", action="store_true", help="use center_crop_arr val preprocessing")
    ap.add_argument("--imgt", action="store_true", help="use EXACT ImageTransform.eval_transform (reference preproc, BILINEAR)")
    a=ap.parse_args()
    vstream = val_stream_imgt if a.imgt else (val_stream_cca if a.cca else val_stream)
    acc=Accelerator(); dev=acc.device
    cfg=OmegaConf.load(CFG_DEFAULT); ts=int(cfg.model.selector.token_size)
    model=QuadTok(cfg).eval().to(dev); model.requires_grad_(False)
    model.load_state_dict(torch.load(CKPT_DEFAULT, map_location="cpu"), strict=False)
    npsl=list(model.num_patch_side_list)
    base_tree=build_quadtree(npsl[:-1])
    target_nodes=_get_nodes_at_level(base_tree,3)
    new_target_nodes=[target_nodes[i] for i in inverse_permutation(coarse_split_permutation())]
    full_mask=torch.ones(64,dtype=torch.bool)
    mk_full=lambda: ordered_ge3(model,_build_tree_from_node_mask(base_tree,3,4,npsl,new_target_nodes,full_mask))
    n_full=len(mk_full())
    if acc.is_main_process: print(f"[full] lod>=3 nodes = {n_full}", flush=True)
    ev=VQGANEvaluator(device=dev, enable_rfid=True, enable_inception_score=True); ev.reset_metrics()
    psnr=0.0; k=0; seen=0
    for batch in vstream(acc.process_index, acc.num_processes, a.bs, cap=a.cap):
        batch=batch.to(dev); B=batch.shape[0]; seen+=B
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16, enabled=not a.fp32):
            latent=model.encode(batch)
            recs=recon_opt(model, latent, [mk_full() for _ in range(B)], ts)   # FRESH nodes per image (node.node_feature is written in-place)
        ev.update(batch.float(), recs.float())
        psnr += (10*torch.log10(1.0/((recs-batch)**2).mean(dim=[1,2,3]).clamp_min(1e-12))).sum().item()
        k+=B
        if acc.is_main_process and seen%(a.bs*40)<a.bs: print(f"[rank0] ~{seen}", flush=True)
    acc.wait_for_everyone(); all_reduce_eval(ev, acc)
    kt=acc.reduce(torch.tensor(float(psnr),device=dev),reduction="sum").item()
    kk=acc.reduce(torch.tensor(float(k),device=dev),reduction="sum").item()
    if acc.is_main_process:
        r=ev.result()
        print(f">>> FULL-TREE  N={int(kk)}  tokens={n_full}  PSNR={kt/kk:.4f}  rFID={float(r['rFID']):.4f}  IS={float(r.get('InceptionScore',0)):.3f}", flush=True)
        print("   (reported Full: PSNR 20.39  rFID 1.50)", flush=True)
    print("FULL_EVAL_DONE", flush=True)

if __name__=="__main__":
    main()
