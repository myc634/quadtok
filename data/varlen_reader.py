"""Varlen token-packing dataloader for the 3-level QuadtreeGPT generator.

Replaces PretokenizedDataset's fixed-count + pad_collate. Streams pretokenized samples
(code_indices/lod_indices/patch_indices/cls) and GREEDILY packs COMPLETE samples up to
`max_tokens` tokens per batch (no padding), emitting concatenated tensors + `cu_seqlens`
(int32, [n_seg+1]) for flash-attn varlen block-diagonal causal attention.

Each DDP rank builds its own packs from its shard split; max_tokens is the PER-GPU budget
(global token budget / world_size). ResampledShards => infinite stream => no rank desync.
"""
import math
import numpy as np
import torch
import webdataset as wds


def _to_sample(d):
    return {
        "code": torch.from_numpy(d["code_indices.npy"]).long(),
        "lod": torch.from_numpy(d["lod_indices.npy"]).long(),
        "patch": torch.from_numpy(d["patch_indices.npy"]).long(),
        "cls": int(d["cls"]) if not isinstance(d["cls"], (bytes, bytearray)) else int(d["cls"].decode()),
    }


def _pack(code_b, lod_b, patch_b, cls_b, cu):
    return {
        "code": torch.cat(code_b),                              # [T] long
        "lod": torch.cat(lod_b),                                # [T] long
        "patch": torch.cat(patch_b),                            # [T] long
        "cls": torch.tensor(cls_b, dtype=torch.long),           # [n_seg]
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32),      # [n_seg+1]
    }


def make_token_packer(max_tokens):
    def token_packer(src):
        code_b, lod_b, patch_b, cls_b, cu, cur = [], [], [], [], [0], 0
        for s in src:
            L = int(s["code"].numel())
            if L == 0 or L > max_tokens:
                continue
            if cur + L > max_tokens and cur > 0:
                yield _pack(code_b, lod_b, patch_b, cls_b, cu)
                code_b, lod_b, patch_b, cls_b, cu, cur = [], [], [], [], [0], 0
            code_b.append(s["code"]); lod_b.append(s["lod"]); patch_b.append(s["patch"]); cls_b.append(s["cls"])
            cur += L; cu.append(cur)
        if cur > 0:
            yield _pack(code_b, lod_b, patch_b, cls_b, cu)
    return token_packer


class VarlenPackedDataset:
    def __init__(self, shards_path, max_tokens_per_gpu, num_train_examples=1_281_167,
                 num_workers_per_gpu=8, shuffle_bufsize=20000):
        self.max_tokens = int(max_tokens_per_gpu)
        pipeline = [
            wds.ResampledShards(shards_path),
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_bufsize, initial=shuffle_bufsize),
            wds.decode(wds.autodecode.basichandlers, handler=wds.warn_and_continue),
            wds.map(_to_sample, handler=wds.warn_and_continue),
            make_token_packer(self.max_tokens),
        ]
        self._dataset = wds.DataPipeline(*pipeline)
        self._dataloader = wds.WebLoader(
            self._dataset, batch_size=None, shuffle=False,
            num_workers=num_workers_per_gpu, pin_memory=True,
            persistent_workers=num_workers_per_gpu > 0,
        )
        # rough bookkeeping (steps are variable under packing)
        self._dataloader.num_batches = max(1, num_train_examples // max(1, (self.max_tokens // 989)))

    @property
    def dataloader(self):
        return self._dataloader
