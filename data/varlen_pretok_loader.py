"""Dynamic varlen token-budget loader for pretokenized quadtree codes.

Each rank streams pretok tars (sharded by rank), greedily packs COMPLETE samples up to a
per-GPU token budget (= max_token_num_global // world_size), and yields varlen batches (NO
padding) for flash_attn_varlen training:

  code_indices  (T,)   int64  concat of every packed sample's VQ codes
  lod_indices   (T,)   int64
  patch_indices (T,)   int64
  labels        (N,)   int64  one class label per packed sample
  cu_seqlens    (N+1,) int32  sample boundaries (0, L0, L0+L1, ...)  [code tokens, no cls]
  seqlens       (N,)   int32
  max_seqlen    int

The generator prepends one class token per sample, so the actual attention sequence length
is L_i + 1 per sample; cu_seqlens here is over CODE tokens (the +1 is handled in the model).
"""
import glob
import numpy as np
import torch
import webdataset as wds


class VarlenPretokDataset(torch.utils.data.IterableDataset):
    def __init__(self, shards, token_budget, rank=0, world=1, shuffle=2000, seed=0, repeat=True):
        super().__init__()
        if isinstance(shards, str):
            shards = sorted(glob.glob(shards))
        shards = sorted(shards)
        # per-rank shard assignment (deterministic, disjoint)
        self.shards = [s for i, s in enumerate(shards) if i % world == rank]
        self.token_budget = int(token_budget)
        self.shuffle = shuffle
        self.seed = seed
        self.repeat = repeat
        assert len(self.shards) > 0, f"rank {rank}/{world}: no shards assigned from {len(shards)}"

    def _raw(self):
        # per-rank shards already selected in __init__ (nodesplitter=None); webdataset's default
        # split_by_worker handles DataLoader workers -> do NOT split by worker again here.
        ds = wds.WebDataset(self.shards, shardshuffle=True, nodesplitter=None,
                            handler=wds.warn_and_continue, empty_check=False)
        if self.repeat:
            ds = ds.repeat()
        ds = (ds.shuffle(self.shuffle)
                .decode()
                .to_tuple("code_indices.npy", "lod_indices.npy", "patch_indices.npy", "cls",
                          handler=wds.warn_and_continue))
        return ds

    def _pack(self, c, l, p, lab, lens):
        lens_t = torch.tensor(lens, dtype=torch.int32)
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens_t, 0)
        return {
            "code_indices": torch.cat(c),
            "lod_indices": torch.cat(l),
            "patch_indices": torch.cat(p),
            "labels": torch.tensor(lab, dtype=torch.long),
            "cu_seqlens": cu,
            "seqlens": lens_t,
            "max_seqlen": int(lens_t.max()),
            "num_samples": len(lens),
        }

    def __iter__(self):
        c, l, p, lab, lens = [], [], [], [], []
        cur = 0
        for codes, lods, patches, cls in self._raw():
            codes = torch.as_tensor(np.asarray(codes), dtype=torch.long)
            L = int(codes.shape[0])
            if L == 0 or L > self.token_budget:
                continue
            if cur + L > self.token_budget and lens:
                yield self._pack(c, l, p, lab, lens)
                c, l, p, lab, lens = [], [], [], [], []
                cur = 0
            c.append(codes)
            l.append(torch.as_tensor(np.asarray(lods), dtype=torch.long))
            p.append(torch.as_tensor(np.asarray(patches), dtype=torch.long))
            lab.append(int(cls))
            lens.append(L)
            cur += L
        if lens:
            yield self._pack(c, l, p, lab, lens)


def build_varlen_loader(shards, max_token_num_global, world, rank, num_workers=4,
                        shuffle=2000, seed=0, repeat=True):
    """DataLoader over VarlenPretokDataset. token_budget per GPU = max_token_num_global // world.
    The dataset yields fully-formed varlen batches, so batch_size=None."""
    budget = max_token_num_global // world
    ds = VarlenPretokDataset(shards, budget, rank=rank, world=world, shuffle=shuffle,
                             seed=seed, repeat=repeat)
    return torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=(4 if num_workers > 0 else None),
    ), budget
