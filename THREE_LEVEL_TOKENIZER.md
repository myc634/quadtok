# QuadTok 3-Level Tokenizer

Extends the 2-level QuadTok tokenizer (8×8 → 16×16) with a third level of detail
(**32×32, patch size 8**), so the content-adaptive token budget spans roughly **64 → ~1024**
tokens per image instead of 64 → 320. Trained **from scratch** on ImageNet-1K at 256×256.

This branch adds the 3-level training + probing code on top of the `submission` branch.

---

## 1. What changed vs. the 2-level tokenizer

| Area | Change | File |
|---|---|---|
| LODs | `num_patch_side_list: [1,2,4,8,16,32]`, `patch_size_list: [16,16,16,16,16,8]` (adds lod5 = 32×32). Tokens live at lod ≥ 3 (8×8 → 16×16 → 32×32). | config |
| **Kinship mask** | Generalized to arbitrary depth. Rule: a finer node attends **all** coarser-level nodes + **same-parent siblings** at its own level (lod5 → all lod3 + all lod4 + lod5 siblings; the coarsest token level lod3 is fully connected). | `modeling/modules/kinship_mask.py` |
| **Attention** | The Kinship mask is non-standard, so the tokenizer uses **`flex_attention`** (block-sparse) instead of a dense mask. `ResidualAttentionBlock` is **dual-path**: `block_mask` → `flex_attention` (training: encoder full-attn + decoder/selector kinship); dense `attention_mask`/`key_padding_mask` → `nn.MultiheadAttention` (batched multi-tree `_forward_optimize`, used by probing). `flex_attention` is `torch.compile(dynamic=True)`. | `modeling/modules/blocks.py` |
| **Decode** | Vectorized hierarchical decode (`hierarchical_latent_decode_vec`, batched `scatter_patches`) + batched per-LOD token-index embeddings, replacing per-node Python loops. ~3.8× faster than eager. | `modeling/modules/blocks.py` |
| Training tree | `expansion_probs` is config-driven. See §2. | `modeling/quadtok.py` |
| Resume | `use_wandb_id`: persist a wandb run id to `output_dir/wandb_id.txt` and reuse it (`resume="allow"`) so a preempted/restarted job continues the **same** wandb run. | `scripts/train_tokenizer.py` |

## 2. How the training tree is expanded

Each training step builds **one random probabilistic quadtree** (shared across the batch, re-drawn
every step) via `build_probabilistic_quadtree(npsl, guaranteed_depth=3, expansion_probs=[p0,p1])`:

- **lod 0→1→2**: always expanded (`depth < guaranteed_depth`) → the full 8×8 lod3 grid (64 nodes) always exists.
- **lod3 → lod4**: each lod3 node expands (into 4 children) with probability `p0`.
- **lod4 → lod5**: each lod4 node expands with probability `p1`.
- lod5 is the max depth (leaves).

Tokens = every node at **lod ≥ 3** (internal + leaf). With `expansion_probs = [0.95, 0.7]` the mean
token count is **~987** (max 1344), calibrated on 10k random trees to sit around the ~1024 budget
while keeping the finest ps=8 level from over-expanding. Training on **random** topologies teaches the
tokenizer to reconstruct arbitrary trees; **content-adaptive** topology selection happens at inference
(probing, §5), not during training.

## 3. Config

`configs/training/single_stage/quadtok_ss256_vq_3level.yaml` — single-stage VQ:

- From scratch (no warm-start), **350k steps**, `per_gpu_batch_size: 64` (global 512 on 1×8 GPU).
- Losses: L2 + LPIPS(convnext_s) + VQ + LeCam from step 0; **GAN (PatchGAN) after `discriminator_start: 200000`**.
- `expansion_probs: [0.95, 0.7]`, `guaranteed_depth: 3`.
- codebook 16384, token_size 8, encoder small / decoder large.
- LR 1e-4, cosine, 10k warmup; EMA 0.999; bf16.
- LPIPS weights expected at `pretrained_weight/{convnext_small-0c510722.pth, vgg16-397923af.pth, vgg.pth}`.

## 4. How to launch training

**Data** — ImageNet-1K train/val as WebDataset (`.jpg` + `.cls`, 1000/shard):
```bash
# train (HF ILSVRC/imagenet-1k parquet -> wds tars), ~1470 shards
python dl_convert_train.py     # -> /sensei-fs-3/users/yuchengm/data/imagenet-wds/train/train-*.tar
```

**Checkpointing** — checkpoints are written to **local SSD** (`/mnt/localssd/quadtok_3level_out`, off the
500 GB sensei-fs quota, ephemeral) and continuously **mirrored to S3**; on (re)start the latest checkpoint
is pulled back from S3 so training survives localssd destruction + preemption. This is handled by the launch
script — no code change needed. `save_every: 2500` (~26 min).

**Launch** (Adobe AI Platform, 1 node / 8 GPU H200, preemptible) — the launch script activates the venv
(`python -m accelerate.commands.launch`, robust to a relocated venv), sets up S3 sync, and runs
`scripts/train_tokenizer.py`:
```bash
# from the machine with the aip CLI (WANDB_API_KEY = your wandb key; WANDB_BASE_URL = your wandb host)
aip job create \
  --name exp-quadtok-3l-tok-h200p2 \
  --project SceneStaging --job-type training --preemptible \
  --gpu-instance-type p5en.48xlarge --xpus-per-pod 8 --num-pods 1 \
  --image docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:12.8.1-runtime-ubuntu22.04-CUDNN-v9.11.0.98-NCCL-v2.27.3-1 \
  --main-script scripts/pluto_launch_m3.sh \
  --env WANDB_API_KEY=$WANDB_API_KEY --env WANDB_BASE_URL=$WANDB_BASE_URL \
  --start
```
`scripts/pluto_launch_m3.sh` runs, per node:
```bash
python -m accelerate.commands.launch --mixed_precision=bf16 --num_machines=1 --num_processes=8 \
  scripts/train_tokenizer.py config=configs/training/single_stage/quadtok_ss256_vq_3level.yaml
```
For preemptible jobs, add the job name to a watchdog (poll status → `aip job start` on preemption);
`resume: True` + `use_wandb_id` make the restart continue from the latest S3 checkpoint on the same wandb run.

**A single-GPU dry run** (no aip): `python scripts/smoke_test_3level.py config=configs/training/single_stage/quadtok_ss256_vq_3level.yaml`
(warm-start-loads a 2-level checkpoint if `experiment.init_weight` is set, runs fwd+bwd, times a step).

## 5. Probing (stage 2, content-adaptive trees for the generator)

`scripts/probe3.py` — two-stage hierarchical **LPIPS A/B** search on the frozen 3-level tokenizer:

- **Stage 1** (lod3 → lod4, threshold `t1`): the verbatim 2-level probe — split a complementary 32/64 half,
  measure per-patch LPIPS(vgg, spatial) benefit via `AvgPool(32)→8×8`, split patches with benefit ≥ `t1`.
- **Stage 2** (lod4 → lod5, threshold `t2`): the same A/B over each image's **existing** lod4 nodes
  (`AvgPool(16)→16×16`); only lod4 nodes can split to lod5.

Modes: `--mode eval` (reconstruct the guided tree, report rFID / PSNR / LPIPS + token-count distribution
via `eval.utils.evaluator.VQGANEvaluator`) and `--mode extract` (write `code_indices/lod_indices/patch_indices/cls`
WebDataset tars for generator training).

Submit scripts (A100 P1, one task per GPU, mirror to S3):
- `scripts/pluto_probe3_sweep.sh` — `(t1,t2)` grid × 10k val → pick mean tokens ~800–900 with best rFID/PSNR/LPIPS.
- `scripts/pluto_probe3_extract.sh` — per-shard extraction of the generator training set with the chosen `(t1,t2)`.

> The `(t1,t2)` sweep is only meaningful on a **sufficiently trained** tokenizer: early in training the fresh
> lod5 modules are not yet useful, so lod4→lod5 splits give ~0 LPIPS benefit and the guided tree stays ~2-level.

## 6. Notes

- Attention: `flex_attention` (block-sparse Kinship) is the tokenizer's fast path; the encoder (no mask) uses
  full flex attention. Requires torch ≥ 2.5 (this repo uses torch 2.7.1+cu128).
- The dead repo symlinks (`checkpoints`, `logs`, `wandb`, `pretrained_weight`) from the original clone are not
  used here; outputs go to localssd/S3 and LPIPS/FID weights are placed under `pretrained_weight/`.
