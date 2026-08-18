"""Varlen token-budget packing dataloader for the pretokenized 512 2-level generator data.

Each pretok sample = (code_indices[L], lod_indices[L], patch_indices[L], cls). The generator trains
with varlen flash-attn (no padding): we greedily pack COMPLETE samples until adding the next would
exceed `max_tokens_per_gpu`, then emit one packed batch (concatenated tokens + per-sample seqlens).
The model builds cu_seqlens (with the per-sample cls prepended) + per-sample RoPE positions from this.

`max_tokens_per_gpu` = global_max_tokens // world_size (global default 262144*4=1,048,576 -> 131072/gpu
for 8 GPUs, ~130 samples/step at ~1010 tok/sample, matching the old 1024-sample global batch).
"""
import glob, io
import numpy as np
import torch
import webdataset as wds


def _decode(sample):
    code = np.load(io.BytesIO(sample["code_indices.npy"])).astype(np.int64)
    lod = np.load(io.BytesIO(sample["lod_indices.npy"])).astype(np.int64)
    patch = np.load(io.BytesIO(sample["patch_indices.npy"])).astype(np.int64)
    cls = int(sample["cls"]) if isinstance(sample["cls"], (str, bytes)) else int(sample["cls"])
    return {"code": code, "lod": lod, "patch": patch, "cls": cls, "L": len(code)}


def _collate(buf):
    """buf: list of per-sample dicts -> one packed batch (concat tokens + per-sample metadata)."""
    return {
        "code": torch.from_numpy(np.concatenate([b["code"] for b in buf])),   # [sum_L] long
        "lod": torch.from_numpy(np.concatenate([b["lod"] for b in buf])),     # [sum_L] long
        "patch": torch.from_numpy(np.concatenate([b["patch"] for b in buf])), # [sum_L] long
        "cls": torch.tensor([b["cls"] for b in buf], dtype=torch.long),       # [n]
        "seqlens": torch.tensor([b["L"] for b in buf], dtype=torch.long),     # [n] per-sample token counts
    }


class VarlenPretokPacked(torch.utils.data.IterableDataset):
    """Yields token-budget-packed batches of COMPLETE samples, sharded by rank."""

    def __init__(self, tar_glob, max_tokens_per_gpu, rank=0, world=1, shuffle=True, seed=0, loop=True):
        super().__init__()
        self.tar_glob = tar_glob
        self.max_tokens = max_tokens_per_gpu
        self.rank, self.world = rank, world
        self.shuffle, self.seed, self.loop = shuffle, seed, loop

    def _urls(self):
        urls = sorted(glob.glob(self.tar_glob))
        assert urls, f"no tars match {self.tar_glob}"
        # if there are >=world tars, shard tars by rank; else every rank reads all (loop) and
        # shuffles differently by seed (smoke-friendly; production has 1 tar/rank already).
        if len(urls) >= self.world:
            urls = [u for i, u in enumerate(urls) if i % self.world == self.rank]
        return urls

    def __iter__(self):
        ds = wds.WebDataset(self._urls(), shardshuffle=self.shuffle, resampled=self.loop,
                            nodesplitter=wds.split_by_node if False else None,
                            seed=self.seed + self.rank, empty_check=False)
        if self.shuffle:
            ds = ds.shuffle(1000)
        ds = ds.map(_decode)
        buf, cur = [], 0
        for s in ds:
            L = s["L"]
            if L <= 0 or L > self.max_tokens:
                continue
            if cur + L > self.max_tokens and buf:
                yield _collate(buf)
                buf, cur = [], 0
            buf.append(s); cur += L
        if buf and not self.loop:
            yield _collate(buf)


def build_cu_seqlens(seqlens, add_cls=True, device="cuda"):
    """Per-sample token counts [n] -> cu_seqlens [n+1] for flash_attn_varlen (with +1 cls/sample)."""
    lens = seqlens.to(device) + (1 if add_cls else 0)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(lens, 0)
    return cu, int(lens.max().item())


def per_sample_positions(seqlens, add_cls=True, device="cuda"):
    """RoPE positions that RESET per sample: [0..len_i-1] concatenated. [total] long."""
    lens = seqlens.to(device) + (1 if add_cls else 0)
    return torch.cat([torch.arange(int(l), device=device) for l in lens])
