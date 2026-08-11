import os, io, tarfile, glob, sys
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "ILSVRC/imagenet-1k"
NVAL = 14
pq_dir = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/_val_parquet"
outdir = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/val"
os.makedirs(pq_dir, exist_ok=True); os.makedirs(outdir, exist_ok=True)
tok = os.environ.get("HF_TOKEN")

files = []
for k in range(NVAL):
    fn = f"data/validation-{k:05d}-of-{NVAL:05d}.parquet"
    print("download", fn, flush=True)
    p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=pq_dir, token=tok)
    files.append(p)

def get_bytes(img):
    if isinstance(img, dict):
        if img.get("bytes") is not None: return img["bytes"]
    if isinstance(img, (bytes, bytearray)): return bytes(img)
    b = io.BytesIO(); img.save(b, format="JPEG"); return b.getvalue()

i = 0; per = 1000; shard = None; sidx = -1
for f in sorted(files):
    t = pq.read_table(f)
    imgs = t.column("image").to_pylist()
    labs = t.column("label").to_pylist()
    for img, lab in zip(imgs, labs):
        if i % per == 0:
            if shard: shard.close()
            sidx += 1; shard = tarfile.open(os.path.join(outdir, f"val-{sidx:06d}.tar"), "w")
            print("shard", sidx, flush=True)
        data = get_bytes(img); key = f"{i:08d}"
        ti = tarfile.TarInfo(f"{key}.jpg"); ti.size = len(data); shard.addfile(ti, io.BytesIO(data))
        cb = str(int(lab)).encode(); tc = tarfile.TarInfo(f"{key}.cls"); tc.size = len(cb); shard.addfile(tc, io.BytesIO(cb))
        i += 1
if shard: shard.close()
print("DONE wrote", i, "images,", sidx + 1, "shards", flush=True)
