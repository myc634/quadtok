# QuadTok 512×512 (High-Res) 2-Level Tokenizer

Trains the QuadTok tokenizer at **512×512** with a **2-level** content-adaptive quadtree
(**16×16 → 32×32**, patch size 16), so the token budget spans roughly **256 → 1280** tokens per
image (mean ~1000). This is the **resolution-scaling** track — kept to two levels so it stays
isolated from the separate 3-level *depth-scaling* experiment (`THREE_LEVEL_TOKENIZER.md`).
Trained **from scratch** on ImageNet-1K at 512×512.

The design is simply the original **256 2-level tokenizer shifted one pyramid level deeper**, on a
2× larger image: the two token levels move from lod3/lod4 (8×8+16×16) to **lod4/lod5 (16×16+32×32)**,
the finest patch stays **16px** (512/32), and token/pixel density is unchanged (4× tokens for 4× pixels).

---

## 1. What changed vs. the 2-/3-level tokenizer

| Area | Change | File |
|---|---|---|
| LODs | `num_patch_side_list: [1,2,4,8,16,32]`, `patch_size_list: [16,16,16,16,16,16]`. Tokens live at **lod ≥ 4** (16×16 → 32×32). | config |
| **guaranteed_depth** | **4** (was 3). The full 16×16 grid (256 coarse tokens) is always present. | config |
| **Token floor** | The hardcoded literal `lod == 3` / `>= 3` (coarsest token level, base decode canvas, per-LOD embeddings, dtype lookups) is generalized to **`self.guaranteed_depth`** everywhere in the training path (16 sites). Rule: **token floor == guaranteed_depth**. Default stays 3, so 256/3-level behavior is unchanged. | `modeling/quadtok.py`, `modeling/modules/blocks.py` |
| **Decoder upsampler** | Rewritten **resolution-ratio based**: the side length at a lod = `patch_size_list[lod] * num_patch_side_list[lod]`; the upsampler i→i+1 is `ConvTranspose ×2` iff that side length doubles, else a same-resolution `Conv`. Reproduces the 3-level ladder exactly (lod3→4 doubles, lod4→5 same) **and** gives 512's lod4→lod5 ×2 (256→512). | `modeling/modules/blocks.py` |
| **expansion_probs** | Single value `[0.73]` (only lod4→lod5). Mean tokens = `256 + 1024·p ≈ 1003` at p=0.73. | config |
| Resolution | `crop_size: 512`, `resize_shorter_edge: 512`. | config |
| Init | **From scratch** (`init_weight:` empty) — the deeper-shifted 2-level does not warm-start cleanly from the 256 recipe. | config |
| Attention | **Unchanged.** `ResidualAttentionBlock` already computes with **`torch.compile(flex_attention, dynamic=True)`** (block-sparse Kinship for decoder/selector, full-attn for encoder). `kinship_mask.py` is depth-general (`min_lod = lod_t.min()`), so the 2-level lod4/lod5 mask works with no change. | `modeling/modules/blocks.py`, `modeling/modules/kinship_mask.py` |

> **Not wired:** `grad_checkpointing` is *not* threaded through the tokenizer's `ResidualAttentionBlock`
> (only the MAR generator uses it). See §5.

## 2. How the training tree is expanded

Each step builds one random probabilistic quadtree (shared across the batch, re-drawn every step)
via `build_probabilistic_quadtree(npsl, guaranteed_depth=4, expansion_probs=[0.73])`:

- **lod 0→1→2→3→4**: always expanded (`depth < guaranteed_depth`) → the full **16×16 = 256** lod4 grid always exists.
- **lod4 → lod5**: each lod4 node expands (into 4 children) with probability **0.73**.
- lod5 (32×32) is the max depth.

Tokens = every node at **lod ≥ 4** (the coarse 256 + the fine children). Measured over 2000 random
trees: **mean 1003.5**, p10 968 / p50 1004 / p90 1040, min 916 / max 1280. lod3 (64 internal nodes)
is **not** a token level here — that is the key difference from the 3-level tree (64+243+680≈987, three levels).

## 3. Config

`configs/training/single_stage/quadtok_ss512_vq_2level.yaml` — single-stage VQ:

- From scratch, **350k steps**, **global batch 512** = 1 node × 8 GPU × `per_gpu_batch_size: 32` × `gradient_accumulation_steps: 2`.
- Losses: L2 + LPIPS(convnext_s) + VQ + LeCam from step 0; **GAN (PatchGAN) after `discriminator_start: 200000`**.
- `num_patch_side_list: [1,2,4,8,16,32]`, `patch_size_list: [16,…,16]`, `guaranteed_depth: 4`, `expansion_probs: [0.73]`.
- codebook 16384, token_size 8, encoder small / decoder large.
- LR 1e-4, cosine, 10k warmup, end_lr 1e-5; **EMA 0.999**; bf16; TF32 on.
- Data: local ImageNet WebDataset at 512 crop (see §4). LPIPS weights under `pretrained_weight/`.

## 4. Batch / memory / throughput (from the 512 smoke)

Measured on **H200 (139.8 GB usable)**, 500-step smokes, generate off, GAN not yet active:

| per_gpu | peak mem | step time | throughput | notes |
|---|---|---|---|---|
| 16 | 43.8 GB | 0.31 s | 51.6 img/s/gpu | huge headroom |
| **32** | **82.2 GB** | **0.52 s** | 61.9 img/s/gpu | **production microbatch** |
| 64 (raw) | **OOM ~140 GB** | — | — | doesn't fit (2.4 GB/sample × 64 + fixed) |
| 64 + `expandable_segments:True` | ~fits (~90%) | ~0.96 s | 65.8 img/s/gpu | reclaims ~11 GB fragmentation; risky headroom |

**Memory scales ~2.4 GB/sample, ~5 GB fixed.** The 82 GB at bs32 is dominated by 40 transformer
layers' activations + the 512² CNN feature maps — **flex already makes attention O(N); it is not the
bottleneck**, so flex changes nothing here.

**1-node global-512, three ways:**

| option | fits? | step time | verdict |
|---|---|---|---|
| **bs32 + grad_accum 2** | ✅ 82 GB (59%) | ~1.03 s | **chosen** — same gradients as bs64, zero recompute, safe headroom for the GAN/generate spikes over 350k |
| bs64 + `expandable_segments` (no accum) | ✅ ~90% | ~0.96 s | ~7% faster but ~90% memory → OOM risk once the discriminator fires (200k) + generate spikes |
| bs64 + grad_ckpt (no accum) | ✅ (needs wiring) | ~1.3 s | ~30% slower (recompute) **and** grad_ckpt isn't wired into the tokenizer |

**grad_accum 2 > grad_ckpt**: accumulation splits the batch (identical FLOPs, no recompute) while
checkpointing recomputes the forward in backward (~+30%). For this per-sample loss they are
gradient-identical, so accumulation wins on speed and simplicity.

## 5. Data

**No new data processing.** Reuses the existing ImageNet-1K WebDataset (the same tars the 3-level run
uses); the 512 crop happens in the dataloader, not offline:

- Train: `/sensei-fs-3/users/yuchengm/data/imagenet-wds/train/train-{000000..001469}.tar` (1470 shards, 1,281,167 imgs).
- Val: `/sensei-fs-3/users/yuchengm/data/imagenet-wds/val/val-{000000..000049}.tar` (50 shards).
- Preprocessing: `resize_shorter_edge: 512`, `crop_size: 512`, random crop + flip. ImageNet short edges
  are often < 512 → upscaled (accepted; a quality ceiling, not a correctness issue). Data loading is
  **not** a bottleneck at 512 (Data(t) ≈ 0.0013 s).

If you need to (re)build the tars from scratch, use the 3-level scripts (`scripts/dl_convert_train.py`,
`scripts/dl_convert_val.py`, `HF_TOKEN` required for the gated `ILSVRC/imagenet-1k`).

## 6. How to launch training

**Production** — Adobe AI Platform, **1 node / 8 GPU H200**, preemptible (P2) + auto-requeue.
`scripts/pluto_launch_512.sh` activates the venv (`python -m accelerate.commands.launch`), sets up S3
checkpoint mirror/resume, and runs `scripts/train_tokenizer.py`:

```bash
# from the machine with the aip CLI. WANDB (public wandb.ai, entity maoyucheng0321) makes generate/
# log_images work + gives loss curves; pass the key or it falls back to offline (still fine).
aip job create \
  --name exp-quadtok-512-2level \
  --project SceneStaging --job-type training --preemptible --auto-requeue \
  --gpu-instance-type p5en.48xlarge --xpus-per-pod 8 --num-pods 1 \
  --image docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:12.8.1-runtime-ubuntu22.04-CUDNN-v9.11.0.98-NCCL-v2.27.3-1 \
  --main-script scripts/pluto_launch_512.sh \
  --env WANDB_API_KEY=$WANDB_API_KEY --env WANDB_BASE_URL=https://api.wandb.ai \
  --start
```

**Checkpointing / resume** — checkpoints go to local SSD (`/mnt/localssd/quadtok_512_out`, off the
500 GB sensei-fs quota) and are mirrored to **S3** (`s3://g3i-data/yuchengm/quadtok_512/…`) every 5 min;
on (re)start the latest checkpoint (+ `wandb_id.txt`) is pulled back so training survives localssd
destruction + preemption. `resume: True` + `use_wandb_id` continue the same run. `save_every: 2500`
(~40 min at ~1.03 s/step). For preemptible jobs add the job to a watchdog (poll status → `aip job start`).

**Smoke** (validity check, not a paper result) — `scripts/pluto_launch_512_smoke.sh` runs a short
real-training smoke (`STEPS`/`PER_GPU`/`GRAD_CKPT` env-overridable), reports peak GPU memory + throughput,
and mirrors logs to `s3://g3i-data/yuchengm/quadtok_512/smoke`.

## 7. Notes / watch items

- **flex_attention** is the tokenizer's fast path and is already `torch.compile`d; eager would be ~10× slower.
  Requires torch ≥ 2.5 (repo uses 2.7.1+cu128).
- **Discriminator memory:** the 82 GB above is *before* the PatchGAN turns on at 200k. When it fires,
  memory rises (disc fwd/bwd on 512² images) — expected to fit in the ~57 GB headroom, but watch the first
  logs after step 200k; if tight, drop to `per_gpu 16 + accum 4` or add `expandable_segments:True`.
- **Generate:** production keeps `generate_every: 500`. Image logging uses the **wandb** tracker's
  `log_images` — it works with wandb on (online or offline). Do **not** set `enable_wandb: false` for a
  run that generates (the tensorboard tracker has no `log_images` and will crash the generate step).
- **Runtime:** ~1.03 s/step × 350k ≈ ~100 GPU-hours wall on 1 node ≈ ~4 days on P2 (with preemption/resume).
- **grad_checkpointing** is not wired into `ResidualAttentionBlock`; `model.grad_checkpointing=True` is a
  no-op for the tokenizer. Only needed if a true no-accum bs64 is required — then wire it in (≈+30% slower
  than the chosen grad_accum 2).
