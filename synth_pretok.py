"""Synthetic 3-level pretokenized shards for the generator VARLEN smoke.

Writes tars byte-identical in layout to search3_fast.py --mode extract:
  <key>.code_indices.npy  int64 [T]   (random VQ ids in [0,codebook))
  <key>.lod_indices.npy   int64 [T]   (slot-order, lod-ascending: 3..3,4..4,5..5)
  <key>.patch_indices.npy int64 [T]   (patch id within that lod's side^2 grid)
  <key>.cls               utf-8 int
Trees are structurally valid: every split node's 4 children exist at the next lod
with the exact child-patch layout get_token_indices_embedding expects. Variable T
per sample (exercises varlen packing). No torch / GPU / tokenizer weights needed.
"""
import io, os, argparse, tarfile
import numpy as np

# num_patch_side_list = [1,2,4,8,16,32]; tokens live at lod>=3 (side 8/16/32).


def children_patches(patch, side_k):
    """4 child patch ids in the side_{k+1}=2*side_k grid (matches mar.py)."""
    row, col = patch // side_k, patch % side_k
    side_c = side_k * 2
    tl = (2 * row) * side_c + (2 * col)
    return [tl, tl + 1, tl + side_c, tl + side_c + 1]


def make_tree(rng):
    lod3 = list(range(64))                                  # side 8: all present
    n3 = int(rng.integers(44, 65))
    split3 = rng.choice(64, size=n3, replace=False)
    lod4 = sorted({c for p in split3 for c in children_patches(int(p), 8)})
    n4 = int(rng.integers(len(lod4) // 2, len(lod4) + 1)) if lod4 else 0
    split4 = rng.choice(lod4, size=n4, replace=False) if n4 else np.array([], int)
    lod5 = sorted({c for p in split4 for c in children_patches(int(p), 16)})
    lods = np.array([3] * 64 + [4] * len(lod4) + [5] * len(lod5), np.int64)
    pats = np.array(lod3 + lod4 + lod5, np.int64)
    return lods, pats


def npy_bytes(arr):
    b = io.BytesIO(); np.lib.format.write_array(b, arr); return b.getvalue()


def write_shard(path, n, rng, cb=16384, learnable=False):
    with tarfile.open(path, "w") as tw:
        for i in range(n):
            lod, pat = make_tree(rng)
            cls = int(rng.integers(0, 1000))
            if learnable:
                # deterministic per-class code = f(cls, lod): the model sees cls at the
                # segment start, so it CAN learn to predict every node -> loss must drop.
                code = ((cls * 131 + lod.astype(np.int64) * 7) % cb).astype(np.int64)
            else:
                code = rng.integers(0, cb, size=lod.shape[0], dtype=np.int64)
            key = "%06d" % i
            for field, data in [("code_indices.npy", npy_bytes(code)),
                                ("lod_indices.npy", npy_bytes(lod)),
                                ("patch_indices.npy", npy_bytes(pat)),
                                ("cls", str(cls).encode())]:
                info = tarfile.TarInfo("%s.%s" % (key, field))
                info.size = len(data)
                tw.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--per_shard", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learnable", action="store_true",
                    help="code=f(cls,lod) so the model can learn it -> loss decreases")
    a = ap.parse_args()
    tot = 0
    for s in range(a.shards):
        rng = np.random.default_rng(a.seed * 1000 + s)
        p = os.path.join(a.out, "eb_%d.tar" % s)
        write_shard(p, a.per_shard, rng, learnable=a.learnable)
        tot += a.per_shard
        print("wrote", p, flush=True)
    # quick length stats from one shard
    rng = np.random.default_rng(0)
    ls = [make_tree(rng)[0].shape[0] for _ in range(200)]
    print("SYNTH_DONE shards=%d total=%d len[min/mean/max]=%d/%.0f/%d"
          % (a.shards, tot, min(ls), sum(ls) / len(ls), max(ls)), flush=True)


if __name__ == "__main__":
    main()
