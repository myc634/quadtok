import os, io, tarfile, json
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# Mirror of dl_convert_val.py but for the ImageNet-1k TRAIN split.
# Downloads HF parquet shards one at a time, converts to WebDataset tars
# (.jpg + .cls), then deletes the parquet to bound disk usage.
# Each parquet maps to a whole number of shards -> exact resume at parquet boundary.
REPO = "ILSVRC/imagenet-1k"
NTRAIN = 294
pq_dir = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/_train_parquet"
outdir = "/sensei-fs-3/users/yuchengm/data/imagenet-wds/train"
progress_path = os.path.join(outdir, "_progress.json")
os.makedirs(pq_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)

tok = os.environ.get("HF_TOKEN")
assert tok, "HF_TOKEN not set"

PER = 1000

def get_bytes(img):
    if isinstance(img, dict) and img.get("bytes") is not None:
        return img["bytes"]
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    b = io.BytesIO(); img.save(b, format="JPEG"); return b.getvalue()

# Resume: progress.json holds next parquet index, running image count, last shard index.
next_k, i, sidx = 0, 0, -1
if os.path.exists(progress_path):
    with open(progress_path) as f:
        st = json.load(f)
    next_k, i, sidx = st["next_k"], st["i"], st["sidx"]
    print(f"RESUME from parquet {next_k}, img {i}, last_shard {sidx}", flush=True)

for k in range(next_k, NTRAIN):
    fn = f"data/train-{k:05d}-of-{NTRAIN:05d}.parquet"
    print("download", fn, flush=True)
    p = hf_hub_download(REPO, fn, repo_type="dataset", local_dir=pq_dir, token=tok)
    t = pq.read_table(p)
    imgs = t.column("image").to_pylist()
    labs = t.column("label").to_pylist()
    shard = None
    for j, (img, lab) in enumerate(zip(imgs, labs)):
        if j % PER == 0:                       # new shard at each 1000 within this parquet
            if shard: shard.close()
            sidx += 1
            shard = tarfile.open(os.path.join(outdir, f"train-{sidx:06d}.tar"), "w")
        data = get_bytes(img); key = f"{i:08d}"
        ti = tarfile.TarInfo(f"{key}.jpg"); ti.size = len(data); shard.addfile(ti, io.BytesIO(data))
        cb = str(int(lab)).encode(); tc = tarfile.TarInfo(f"{key}.cls"); tc.size = len(cb); shard.addfile(tc, io.BytesIO(cb))
        i += 1
    if shard: shard.close()
    try:
        os.remove(p)
    except OSError:
        pass
    with open(progress_path, "w") as f:
        json.dump({"next_k": k + 1, "i": i, "sidx": sidx}, f)
    print(f"done parquet {k}  total_imgs={i}  last_shard={sidx}", flush=True)

print("DONE wrote", i, "images,", sidx + 1, "shards", flush=True)
