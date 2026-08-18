# QuadTok 512 2-Level — Pretokenization + Generator (varlen) Training Guide

End-to-end pipeline to train the **512 2-level QuadtreeGPT generator** on content-adaptive quadtree
tokens: pretokenize ImageNet with the frozen 512 tokenizer at the best operating point (guided
**τ=0.03**, ~1010 tok/img), then train the AR generator with **varlen flash-attention** (no padding).
Everything below is verified on the smoke; this doc is for launching the **full** run.

Repo: `/sensei-fs-3/users/yuchengm/code/quadtok/512` (branch `high-res-tokenizer`). Venv:
`/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv` (torch 2.7.1+cu128, **flash_attn 2.8.3.post1**,
webdataset 1.0.2). Relay for driving Pluto: `int-relay-a10080-p0-0731` (see the tokenizer doc).

---

## 0. What was built + verified

| Piece | File | Verified |
|---|---|---|
| **FFN fix** (generator param bug) | `modeling/mar.py` QuadtreeGPT `mlp_ratio 4→1` | base = **344.0M** (was 947M) |
| **Pretokenize** (guided τ=0.03) | `scripts/probe512_extract.py` | round-trip: saved code+tree → decode → PSNR ~ guided quality |
| **Varlen dataloader** (token-budget pack) | `modeling/varlen_pretok.py` | packs 126-128 complete samples/batch ≤ per-GPU budget, no padding |
| **Generator varlen forward** | `modeling/mar.py` `QuadtreeGPT.forward_varlen` + `modeling/modules/attention.py` flash_attn_varlen branch | **parity vs padded forward: loss diff 2e-5** |
| **(b) no-pad vectorized embedding** | `modeling/mar.py` `forward_varlen` (vectorized unpack, no host loop) | parity vs padded **3e-5**; removes ~128 CPU-GPU syncs/step |
| **(c) Liger fused-CE head** | `modeling/mar.py` `_head_loss` + `use_liger` config | parity vs vanilla CE **0.0**; frees **~22 GB** (no fp32 logits) |
| **Config** | `configs/training/generator/gpt_quadtree_512.yaml` | `grad_ckpt_interval 3` + `compile` + `use_liger` |
| **Smoke / reference trainer** | `scripts/smoke_gen512.py` | applies compile+liger+grad-accum; loss ↓; 8×H200: SEE §9.1 |

## 1. The FFN fix (must be present)

`QuadtreeGPT` (`modeling/mar.py`) passed `ffn_dim_multiplier = mlp_ratio = 4` to `TransformerBlock`,
whose LlamaGen `FeedForward` already computes `hidden = int(2/3 · 4·dim)` — so the 4× was
double-counted → FFN hidden ≈ 11008 → **base = 947M**. Fixed to `mlp_ratio = 1` → hidden ≈ 2816 →
LlamaGen-L (embed 1024 / depth 24 / heads 16) = **344M** (~300M target). `self.seq_len 512→1536`
(512 max = 256 lod4 + 1024 lod5 = 1280 tok/sample + cls; RoPE resets per sample under varlen).

## 2. Pretokenization (guided τ=0.03)

`scripts/probe512_extract.py` — distributed (accelerate), shards ImageNet **train** tars by rank
(`tar_idx % world == rank`), runs the frozen tokenizer + guided LPIPS-A/B search at τ=0.03, and writes
per sample `code_indices(int32) / lod_indices(int16) / patch_indices(int32) / cls` to WDS tars
`pretok-rank{r}.tar`. Codes come from `selector._forward_optimize + quantize.min_encoding_indices`,
aligned to the guided tree's node order (lod4 coarse first, then lod5). Fast (flex `_forward_optimize`).

**Full run** — all 1470 train tars over an 8-GPU (or multi-node) H200 job:
```bash
accelerate launch --num_processes 8 scripts/probe512_extract.py \
  --tau 0.03 --bs 32 --out /sensei-fs-3/users/yuchengm/data/quadtok_512_pretok  # (no --ntars_per_rank = all)
```
Throughput ≈ the tokenizer probe rate (H200 bs64 ~15 img/s/gpu for the A/B; extract skips the guided
decode). 1.28M imgs on 8×H200 ≈ a few hours; use more GPUs/nodes to shard further. Output: one
`pretok-rank{r}.tar` per rank (~4-5k samples each per assigned tar-set). mean ~1010 tok/img.

> The smoke used `--ntars_per_rank 5` (4×A100 → ~17.4k samples) into
> `/sensei-fs-3/users/yuchengm/data/quadtok_512_pretok_smoke/`.

## 3. Varlen dataloader (token-budget packing, no padding)

`modeling/varlen_pretok.py::VarlenPretokPacked` — an IterableDataset that greedily packs COMPLETE
samples until adding the next would exceed the **per-GPU token budget**, then emits one packed batch:
`code/lod/patch [ΣL]`, `cls [n]`, `seqlens [n]`. Shards tars by rank when `#tars ≥ world`, else all
ranks read all tars (loop, per-rank shuffle seed). Helpers `build_cu_seqlens` (adds the per-sample cls,
here `add_cls=False` because the AR sequence length already equals the code count) and
`per_sample_positions` (RoPE positions reset 0..L-1 per sample).

**Token budget** — global `262144 * 4 = 1,048,576` (old 256px: 256 tok/img × 1024-sample global batch
= 262144; 512-guided ~1010 tok ≈ 4× 256 tok → 4× to keep ~1024 samples/global-step). Per-GPU =
`global // world` = **131072** for 8 GPUs (~128 samples/step). Set in the config `varlen.global_max_tokens`.

## 4. Generator varlen forward

`QuadtreeGPT.forward(dict)` dispatches to `forward_varlen(packed)` (so DDP/accelerate grad-sync fires).
It reuses the padded forward's embedding path VERBATIM (cls-prepended causal AR + hierarchical
token-index posemb), then **unpads** `z` to a packed `[1, total, dim]` sequence and runs the 24 blocks
with **block-diagonal causal `flash_attn_varlen_func`** (cu_seqlens per sample) + per-sample RoPE +
grad-checkpointing. Mathematically identical to the padded path — **parity loss diff = 2e-5**.
`modeling/modules/attention.py::Attention.forward` gained a `cu_seqlens`/`max_seqlen` varlen branch
(guarded `from flash_attn import flash_attn_varlen_func`).

## 5. Model + config

`configs/training/generator/gpt_quadtree_512.yaml`: `model_size: base` (embed 1024/depth 24/heads 16,
**344M** after the FFN fix), codebook 16384, npsl `[1,2,4,8,16,32]`, guaranteed_depth 4, AdamW lr 4e-4 /
wd 0.05 / betas (0.9,0.95), cosine 50k warmup, bf16, `varlen.global_max_tokens: 1048576`,
`varlen.pretok_glob` → the pretok tars. **Speed knobs (see §9, benchmarked):**
`grad_checkpointing: True` + `grad_ckpt_interval: 2` (checkpoint every 2nd block) + `compile: True`
(→ `QuadtreeGPT.compile_blocks()`). The trainer MUST call `model.compile_blocks()` on the raw model
BEFORE `accelerate.prepare` when `cfg.model.compile`, and set
`model.grad_ckpt_interval = cfg.model.grad_ckpt_interval` (QuadtreeGPT reads it from config already).

## 6. Launch — 8×H200 (or multi-node)

Smoke launcher `scripts/pluto_gensmoke.sh` (100 steps). Production = same but with checkpointing,
wandb, resume, a real step budget, and the full pretok glob. aip submit (P2 + auto-requeue; the account
has no P1):
```bash
aip job create --name exp-quadtok-512-gen \
  --project SceneStaging --job-type training --preemptible --auto-requeue \
  --gpu-instance-type p5en.48xlarge --xpus-per-pod 8 --num-pods 1 \
  --image docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:12.8.1-runtime-ubuntu22.04-CUDNN-v9.11.0.98-NCCL-v2.27.3-1 \
  --main-script scripts/pluto_gensmoke.sh --start
```
Each rank runs `accelerate launch ... scripts/smoke_gen512.py --steps N`. For multi-node, use the
RunAI rendezvous env (RUNAI_* → `--num_machines/--num_processes/--machine_rank/--main_process_ip`) as in
the tokenizer's `scripts/pluto_launch_512.sh`, and shard the pretok tars across all ranks.

**Smoke result — MEASURED on 8×H200** (`exp-quadtok-512-gensmoke`, 100 steps, Succeeded/EXIT 0), plus
a 4×A100 relay run at the *same* per-GPU budget (131072) for cross-check:

| where | params | per-GPU tokens | global | packing | loss (100 / 30 steps) | peak mem/GPU | s/step |
|---|---|---|---|---|---|---|---|
| **8×H200** (target) | 344.0M | 131072 | **1048576** | 126–131 complete samples, ≤131072, no pad | **9.870 → 9.639** | **43.76 GB / 143.7 (30%)** | **2.23** |
| 4×A100-80G (relay) | 344.0M | 131072 | 524288 | 128–130, same | 9.869 → 9.664 | 43.70 GB | 5.21 |

- FFN fix confirmed (**344.0M**, not 947M); flash varlen active on every rank (`attention mode is flash`).
- Greedy token-budget packing: each step packs 126–131 **complete** samples, tokens 130112–131052
  (always ≤ the 131072 budget, never a partial sample), **no padding**.
- Loss decreases monotonically (9.870→9.639 over 100 steps on 8×H200); acc rises above the 1/16384
  chance (step-99 acc 0.0016 ≈ 26× chance) — healthy early training.
- **43.76 GB/GPU = only 30% of an H200** at the real global 1,048,576-token batch → large headroom.
- H200 = **2.23 s/step** (2.3× the A100's 5.21). The 4×A100 row confirms the per-GPU behavior is
  identical to each H200 (same 131072 budget); 8 GPUs only double the global batch + speed each step.

## 7. Correctness summary (all verified)

- Pretokenize: saved (code, lod, patch) reconstruct the originals (round-trip PSNR = guided quality).
- Varlen packing: complete samples only, ≤ budget, cu_seqlens consistent.
- Generator: `forward_varlen` == padded `forward` (loss diff **2e-5**), params **344M**.
- Trains end-to-end (loss decreasing), fits H200.

## 8. Env / gotchas

- `base/.venv` is **uv-built (no pip)** → install with `uv pip install --python base/.venv/bin/python <wheel>`
  or `python -m` console scripts (the `pip`/`accelerate`/`wandb` console-scripts have stale shebangs).
- flash_attn = **2.8.3.post1** (wheel `cu12torch2.7cxx11abiTRUE-cp310`), varlen verified.
- The tokenizer EMA (for optional generate-viz) = `/sensei-fs-3/users/yuchengm/models/quadtok_512_280k/…`.
- Push code to the cluster over the relay COMMAND channel in 4KB base64 chunks (stdin/data channel
  0-byte-blocks); verify md5. See the tokenizer doc.

## 9. Training speedups — MEASURED on 8×H200 (grad-ckpt interval × torch.compile)

Swept grad-ckpt granularity (`grad_ckpt_interval`) × `torch.compile` at the real 131072 tok/GPU budget
(`scripts/bench_gen512.py` per config, 30 steps / 12 warmup, timing pure fwd+bwd+opt on VARYING varlen
batches with data prefetched outside the timer so compile's dynamic-shape behavior is exposed;
launchers `scripts/pluto_genbench.sh` + `scripts/pluto_genbench2.sh`).

| grad_ckpt_interval | compile | s/step | peak mem/GPU | verdict |
|---|---|---|---|---|
| 1 (full ckpt) | ✗ | 2.23 | 43.8 GB | baseline |
| 2 | ✗ | — | OOM (needs >141 GB) | eager can't fit interval≥2 |
| 4 / 0 (no-ckpt) | ✗ | — | OOM (>139 GB) | — |
| 0 (no-ckpt) | ✓ | — | OOM (>139 GB) | 512 @131k tok too big for no-ckpt even compiled |
| 3 / 4 | ✓ | — | OOM (>141 GB) | stores 16/18 layers of activations |
| **2** | **✓** | **1.24** | **106.5 GB (74%)** | **← PRODUCTION: 1.80× baseline throughput** |

**Two findings (match the 3-level result):**
1. **compile lowers memory, not just time.** interval=2 needs >141 GB eager (OOM) but only **106.5 GB
   compiled** (~35 GB saved) — inductor fuses elementwise ops so fewer activations are materialized.
   This is the *only* reason a low-ckpt config fits.
2. **compile does NOT recompile on varying step length.** With `dynamic=True` on the packed seqlen dim +
   a **constant `max_seqlen`** (`self.seq_len`, not the per-step `max_L`), it compiles once (~50s) then
   holds a flat ~0.61 ms/token across steps whose sample-count and token-total change every step
   (verified 40+ steps). Compile the **blocks only** (`compile_blocks()`), never the whole `forward` —
   the host-side pad/scatter loop would recompile per batch-size.

**Why interval=2 is the operating point:** 512-guided packs ~2× the tokens of the 256/3-level runs, so
no-ckpt (and even interval≥3) overflow 143.7 GB. interval=2 (checkpoint every 2nd of 24 blocks) is the
fewest recomputes that still fit under compile. loss falls normally (9.867→9.653 @30 steps), so compile
+ partial-ckpt do not change training. Headroom is thin (74%); do not raise the interval.

**Trainer integration (required for the speedup):** on the raw model, before `accelerate.prepare`:
```python
model.grad_ckpt_interval = cfg.model.grad_ckpt_interval   # 2  (QuadtreeGPT also reads this in __init__)
if cfg.model.compile:
    model.compile_blocks()                                 # torch.compile(block, dynamic=True), ~50s once
```
Both are already set in `configs/training/generator/gpt_quadtree_512.yaml`
(`grad_ckpt_interval: 2`, `compile: True`). Expect the first optimizer step to take ~50 s (compilation);
set `TORCHINDUCTOR_CACHE_DIR` to a persistent path to cache it across restarts.

**Further speedups — now APPLIED (see §9.1 for measured numbers):** (b) no-pad vectorized embedding in
`forward_varlen` (removes the host-side Python scatter loop / ~128 CPU-GPU syncs), and (c) Liger fused
linear-cross-entropy head (`use_liger`) which frees ~22 GB and unlocks `grad_ckpt_interval 3`. Still open:
fused AdamW + TF32 (already in the bench); DataLoader workers/prefetch if data becomes the bottleneck (the
bench times model-only); inference — batch CFG cond/uncond into one forward + CUDA-graph the decode step.

## 9.1 no-pad embedding (b) + Liger fused-CE (c) + grad-accum — MEASURED

All three are parity-verified before benchmarking (padded `forward` == vectorized `forward_varlen` ==
liger head), so the numbers are trustworthy. Bench = `scripts/bench_gen512.py` per config on **8×H200**
at the production **131072 tok/GPU** budget; launcher `scripts/pluto_genbench3.sh` runs the parity gate
first, then the sweep. Metrics: s/opt-step, per-GPU tok/s, mean GPU SM-util%, peak mem.

**(b) No-pad vectorized varlen embedding** (`QuadtreeGPT.forward_varlen`). Attention was already no-pad
varlen; the only waste was the embedding preamble, which padded packed→`(B,max_L)` via a **host-side
`for i in range(B)` scatter** (~128 `.item()` CPU-GPU syncs/step) plus a per-sample RoPE-position loop.
Both are now fully vectorized (masked gather + reuse of the validity mask for RoPE positions) — zero
per-sample Python loops, zero `.item()` syncs. The tree parent→child embedding is byte-identical.
**Parity vs padded: loss diff 3e-5.** Effect: removes CPU stalls (util stays high); the pure-compute
delta is within noise (embedding isn't the bottleneck), exactly as expected.

**(c) Liger fused linear-cross-entropy head** (`use_liger: True`). The vanilla head materializes the full
`(≈131k, 16384)` **fp32 logits (~8.6GB)** + softmax/backward buffers; `liger_kernel`'s
`LigerFusedLinearCrossEntropyLoss` fuses `nn.Linear` + CE and never materializes them. **Parity vs
vanilla CE: 0.000000** (exact). Measured peak-mem drop at interval 2: **106.5 → 84.6 GB (~22 GB freed)**
— more than the 8.6GB estimate (the fp32 copy, softmax and stored logit-gradient all vanish too). That
headroom lets `grad_ckpt_interval` drop from 2 to 3 (fewer recomputes = faster).

**grad-accumulation** (`--accum N`). Runs N varlen micro-batches with **DDP `no_sync` on all but the
last** (one allreduce/opt-step) + `loss/N` averaging → effective global batch ×N at **no extra memory**;
per-token tok/s is invariant to N by construction. Use it to grow the global batch, not to speed a step.

### Results (8×H200, 131072 tok/GPU, 344M generator, parity-gated)

| ckpt_interval | compile | use_liger | s/opt-step | tok/s/GPU | GPU util | peak mem | verdict |
|---|---|---|---|---|---|---|---|
| 2 | ✓ | ✗ | 1.263 | 103.4k | 98.1% | 106.5 GB | §9 baseline, new (b) code — no regression vs the old 1.24 |
| 2 | ✓ | ✓ | 1.221 | 106.9k | 96.0% | **84.6 GB** | (c) liger: **−22 GB**, +3% |
| **3** | **✓** | **✓** | **1.169–1.184** | **~111k** | **~95%** | **105.6 GB (73%)** | **← PRODUCTION: fastest with a safe margin** |
| 4 | ✓ | ✓ | 1.164 | 112.2k | 92.2% | 116.0 GB (81%) | 0.4% faster (noise) but +10 GB, lower util |
| 0 (no-ckpt) | ✓ | ✓ | — | — | — | **OOM (~138 GB)** | doesn't fit even with liger |

**Recommended production config (set in the yaml): `grad_ckpt_interval: 3` + `compile: True` +
`use_liger: True`** → **1.169–1.184 s/opt-step, 105.6 GB (73% of H200), ~95% util, ~111k tok/s/GPU**.
That is the fastest interval with headroom to spare: interval 4 is only 0.4% faster (within noise) for
+10 GB and lower util, and no-ckpt OOMs. Net vs the pre-(c) `interval=2` baseline: **~4% faster and it
lifts the memory pressure that forced interval=2** (liger frees 22 GB). All configs stay compute-bound
(90–98% SM util). Note: a hard no-ckpt OOM leaves the CUDA/NCCL context dirty, so keep OOM-prone configs
LAST in a sweep (fixed in `pluto_genbench3.sh`).

## 11. Usage — train the 512 generator on another machine

The generator is **344M** (LlamaGen-L: embed 1024 / depth 24 / heads 16; ~300M target). Prereqs: the repo
on a shared path; the uv venv (torch 2.7.1+cu128, **flash_attn 2.8.3.post1**); `liger-kernel`
(`uv pip install --python <venv>/bin/python --no-deps liger-kernel` — `--no-deps` so it uses torch's
bundled triton and does not churn torch); the pretok tars.

1. **Pretokenize** ImageNet with the frozen 512 tokenizer (guided τ=0.03, ~1010 tok/img) — see §2 — and
   point `varlen.pretok_glob` at the output `pretok-*.tar`:
   ```bash
   accelerate launch --num_processes 8 scripts/probe512_extract.py --tau 0.03 --bs 32 --out <PRETOK_DIR>
   ```
2. **Speed knobs are already the defaults** in `configs/training/generator/gpt_quadtree_512.yaml`
   (`grad_ckpt_interval: 3`, `compile: True`, `use_liger: True`).
3. **Train** — varlen flash-attn + torch.compile + Liger + optional grad-accum. `smoke_gen512.py` is the
   reference loop; it applies the three speed lines (compile_blocks before `prepare`; interval/use_liger
   come from the config). Production adds checkpointing/wandb/resume but MUST keep those lines.
   ```bash
   export TORCHINDUCTOR_CACHE_DIR=<persistent>   # cache the ~50s first-step compile across restarts
   accelerate launch --num_processes 8 scripts/smoke_gen512.py --steps <N> --accum <A>
   ```
   Multi-node: RunAI rendezvous env (RUNAI_* → `--num_machines/--num_processes/--machine_rank/
   --main_process_ip`); shard the pretok tars across all ranks.
4. **8×H200 aip job** (smoke launcher `scripts/pluto_gensmoke.sh`; production adds ckpt/wandb/resume):
   ```bash
   aip job create --name <name> --project SceneStaging --job-type training --preemptible --auto-requeue \
     --gpu-instance-type p5en.48xlarge --xpus-per-pod 8 --num-pods 1 \
     --image docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:12.8.1-runtime-ubuntu22.04-CUDNN-v9.11.0.98-NCCL-v2.27.3-1 \
     --main-script scripts/pluto_gensmoke.sh --start
   ```
5. **Re-benchmark the speed knobs** on your hardware: `scripts/pluto_genbench3.sh` (parity gate + the
   full interval × compile × liger × accum sweep, one RESULT line per config).
