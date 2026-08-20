# QuadTok 2-level Generator — Data & Training (order-fixed)

**Status (2026-08-19):** the pretokenization token-ORDER bug is fixed (see `QUADTOK_GEN_ORDER_BUG.md`).
New correct-order data is regenerated and pushed to HuggingFace. Old (wrong-order) data was deleted
from both S3 and HF.

## 1. Data

- **HuggingFace (primary):** `datasets/yuchengm/quadtok_data` → `pretok_2level/train-000000.tar … train-001469.tar` (1470 shards).
- **S3 (training mirror):** `s3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/train-*.tar`.
- **Format** (unchanged, WebDataset tar; one sample = one image crop):
  - `code_indices.npy`  int32 `(L,)` — VQ code per node
  - `lod_indices.npy`   int16 `(L,)` — LOD level (3 or 4)
  - `patch_indices.npy` int16 `(L,)` — spatial patch index at that LOD
  - `cls`               str          — ImageNet class id
  - `L` ≈ 226 tokens/sample (guided τ=0.05 quadtree).
  - **Token order = `_get_ordered_nodes` (BFS grouped by LOD)** — the order the generator's
    `forward_varlen`/`generate` expect. (The previous data used slot/spatial order → broke AR.)
- **Recipe:** tokenizer = `models/quadtok_2level_drive/drive_ckpt.download`; guided search τ=0.05;
  aug = `center_hflip` (center_crop_arr(256) + h-flip, 2× per image); produced by
  `scripts/pretokenize_train.py` (A100 P2 job `exp-quadtok-pretok-orderfix-a100p2`).

### Pull the data
```bash
# S3 -> localssd (training)
s5cmd cp 's3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/*' /mnt/localssd/pretok/
# or HuggingFace
huggingface-cli download yuchengm/quadtok_data --repo-type dataset \
  --include 'pretok_2level/*' --local-dir /mnt/localssd/quadtok_data
```

## 2. Train the generator

Entry: `scripts/pluto_train_gen.sh` (runs `scripts/train_generator_varlen.py`, flash-attn varlen, no padding).

```bash
# on a Pluto node (8×GPU); pulls S3 pretok -> localssd, then trains
SIZE=small  RUN_NAME=gen_100m_orderfix  GC=0  WANDB_KEY=<key>  bash scripts/pluto_train_gen.sh
# SIZE = small(111M) | base(947M w/ mlp_ratio) | xlarge
```
Key knobs (env): `MAX_TOKEN_GLOBAL=232448` (global token budget/step), `LR=4e-4`, `STEPS=400000`,
`SAVE_EVERY=2500`, `CKPT_S3=s3://g3i-data/yuchengm/quadtok_gen_ckpt/<RUN_NAME>/`.

> `mlp_ratio`: config `configs/inference/gpt_16k_*.yaml`. The Google-Drive reference gen used
> `mlp_ratio=4` (FFN 11008); base-update default is `1` (fixed FFN sizing).

## 3. Sanity check (order correctness)

Use the known-good Drive generator as a pipeline probe (`scripts/pipeline_val_loss.py`):
`image → tokenizer → guided tree (_get_ordered_nodes) → codes → gen.forward_varlen → CE`.

| order | Drive gen CE | gen_100m CE |
|---|---|---|
| wrong (old slot/spatial) | 9.55 (≈random) | 7.63 |
| **correct (_get_ordered_nodes)** | **6.57–7.06** ✅ | 10.01 (OOD) |

A correctly-pretokenized dataset must yield Drive-gen CE ≈ 7 or below. If it's ~9.7 (=ln(16384)),
the token order is wrong.

## 4. Key files

- `scripts/pretokenize_train.py` — pretok (tensor-native guided search + `compute_slot_rank` reorder → `_get_ordered_nodes` order; `--s3_out` per-shard resume/upload).
- `scripts/bench256_tensor.py` — `compute_slot_rank`, `active_to_padded(SLOT_RANK)` (the order fix).
- `scripts/train_generator_varlen.py` / `data/varlen_pretok_loader.py` — generator training (unchanged; consume saved order as-is).
- `scripts/pipeline_val_loss.py` — end-to-end order/pipeline probe.
- `QUADTOK_GEN_ORDER_BUG.md` — full root-cause writeup.
