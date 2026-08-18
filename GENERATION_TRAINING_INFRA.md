# 3-Level QuadtreeGPT — Pretokenization + VARLEN Generator Training Infra

This documents (1) how to pretokenize ImageNet with the trained 3-level tokenizer at the
chosen operating point, and (2) the **varlen** generator-training pipeline that replaces the
original padded/slow path. A 100-step smoke on 8×H200 has been validated (see §6). Another
session can submit the real training job by following §7.

Repo: `/sensei-fs-3/users/yuchengm/code/quadtok/3level`
Operating point (tokenizer thresholds): **t1 = 0.004, t2 = 0.021** → ~989 tokens/img
(rFID 0.770 / PSNR 23.31; content-adaptive beats random-tree at equal budget).

---

## 0. TL;DR

- **Pretokenize** (real, GPU): `scripts/search3_fast.py --mode extract` at (0.004, 0.021).
  Measured **94 img/s/GPU → ~751 img/s/node** (8×H200). Full ImageNet train (1.28M) ≈
  **28 min compute / ~30–35 min wall per 8-GPU node** (no decode). Output = webdataset tars
  with `code_indices.npy / lod_indices.npy / patch_indices.npy / cls` per sample, in
  slot-order (lod-ascending → valid causal AR order).
- **Generator** = `QuadtreeGPT` (`modeling/mar.py`), **causal AR + CE over the 16384 codebook**
  (NOT diffusion; the tree is GIVEN). LlamaGen-L size = **344.0M** (embed 1024 / depth 24 /
  heads 16), after the FFN-bug fix (§2).
- **Varlen**: token-packed batches (no padding) + `flash_attn_varlen_func` (block-diagonal
  causal). This is the key speed change vs the original padded dataloader/attention.
- **New files**: `data/varlen_reader.py`, `train_gen_varlen.py`,
  `configs/gpt_quadtree_3level.yaml`, `synth_pretok.py` (smoke only),
  `run_gen_smoke_reuse.sh` / `run_gen_sanity_reuse.sh`.
- **Patched files**: `modeling/modules/attention.py` (varlen path + import),
  `modeling/mar.py` (`mlp_ratio 4→1`, `seq_len 512→1408`).

---

## 1. Pretokenization (real data)

`scripts/search3_fast.py --mode extract` runs the two-stage content-adaptive tree search on a
frozen tokenizer and writes the tree structure + VQ codes. One GPU handles one shard range;
launch 8 in parallel per node (each GPU independent → near-linear scaling).

```bash
REPO=/sensei-fs-3/users/yuchengm/code/quadtok/3level
PY=/sensei-fs-3/users/yuchengm/code/quadtok/base/.venv/bin/python   # python3.10 (training-base image)
W=/path/to/tokenizer_weight        # accelerate ckpt dir (ema_model/model.safetensors) OR .safetensors
export TORCH_HOME=/sensei-fs-3/users/yuchengm/.torch \
       HF_HOME=/sensei-fs-3/users/yuchengm/.cache/huggingface \
       TOKENIZERS_PARALLELISM=false GRPC_ENABLE_FORK_SUPPORT=1
cd "$REPO"
for g in 0 1 2 3 4 5 6 7; do
  lo=$((g*160)); hi=$((g*160+159))     # 160 source shards/GPU for the full set; tune to cover all
  SH=$(printf "/sensei-fs-3/users/yuchengm/data/imagenet-wds/train/train-{%06d..%06d}.tar" $lo $hi)
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/search3_fast.py --mode extract --weight "$W" \
    --shards "$SH" --t1 0.004 --t2 0.021 --bs 128 --num_workers 6 \
    --output_tar "/OUT/pretok-rank$g.tar" > "/tmp/pt_$g.log" 2>&1 &
done; wait
```

- **Write to persistent storage** (sensei-fs or S3), NOT pod-local `/tmp` — `/tmp` is wiped on
  preemption. For the real corpus, extract to a sharded dir on sensei-fs / upload to S3.
- **`--weight`** accepts an accelerate checkpoint dir (looks for
  `ema_model/model.safetensors`, `unwrapped_model/…`, `model.safetensors`, or
  `pytorch_model.bin`) or a direct `.safetensors`/`.bin`. The HF repo `yuchengm/quadtok` is the
  authoritative tokenizer; `snapshot_download` it to a persistent dir once and reuse.
- **Output format (per sample)**: tar members `<key>.code_indices.npy` (int64 [T]),
  `<key>.lod_indices.npy` (int64 [T]), `<key>.patch_indices.npy` (int64 [T]), `<key>.cls`
  (utf-8 int). `T` ≈ 989 at (0.004,0.021); nodes are in **slot-order** (all lod3, then lod4,
  then lod5) → lod is monotonic non-decreasing → a valid coarse→fine causal AR order.

---

## 2. Generator model — `QuadtreeGPT` (344M) + the FFN bug

`QuadtreeGPT` (in `modeling/mar.py`, class at ~L1462) is a LlamaGen-style causal transformer:
`wqkv → q/k RMSNorm → RoPE → SDPA/flash causal → wo`; SwiGLU FFN; **CE over the 16384 codebook**
via an untied `nn.Linear(embed, 16384)`. The tree (lod/patch of every node) is provided; the
model only predicts each node's VQ **code**. There is **no diffusion head** (the `diffloss_*`
config keys belong to sibling classes and are dead here).

**⚠️ FFN bug (fixed — do not reintroduce):** the SwiGLU `FeedForward` already applies the
LLaMA `2/3 × 4` expansion. Passing `mlp_ratio = 4` as its multiplier **double-applies** it →
hidden `find_multiple(4·(2/3·4·d), 256) = 11008` (10.75×d) → base ≈ **947M**, not 343M.
Fix in `mar.py`: `mlp_ratio = 1` → hidden **2816** → the LlamaGen-L target.

LlamaGen family (for reference): B 111M (768/12/12), **L 343M (1024/24/16)**, XL 775M (1280/36/20).

Verified build (`configs/gpt_quadtree_3level.yaml`, `model_size: base`):
```
QuadtreeGPT params: 344.0M
embed_dim=1024 depth=24 heads=16 seq_len=1408   FFN w1 (2816,1024)
tok_emb (16384,1024)  output (16384,1024)        # CE head, untied
token_indices_emb (1024,1024)  lod_emb (6,1024)  # 3-level: 32²=1024 patches, 6 lods
freqs_cis (2817,32,2)                             # RoPE table = 1 + seq_len*2
```

**Patches applied to `modeling/mar.py` (QuadtreeGPT `__init__`):**
- `self.seq_len = 512 → 1408` (3-level max ~1344 tokens; RoPE table sized `1+seq_len*2`).
- `mlp_ratio = 4 → 1` (the FFN fix above).

---

## 3. Varlen data loader — `data/varlen_reader.py`

Replaces `PretokenizedDataset` + `pad_collate_fn` (fixed count, `-1` padding → wasted compute).
`VarlenPackedDataset` **greedily packs complete samples** up to a **per-GPU token budget**
(`max_tokens_global / world_size`), emitting concatenated tensors + `cu_seqlens` — **no padding**.

Each yielded batch (a "pack"):
```
code       int64 [T]        concatenated node codes of all packed samples
lod        int64 [T]        concatenated node lods
patch      int64 [T]        concatenated node patch ids
cls        int64 [n_seg]    one class label per packed sample
cu_seqlens int32 [n_seg+1]  segment offsets (0, L0, L0+L1, …, T) for flash-attn varlen
```
`wds.ResampledShards` → infinite stream (no rank desync); `split_by_node/worker`. A sample with
`T==0` or `T>max_tokens` is skipped. Data-parallel: point every rank at the same shard glob;
resampling + node/worker split give each rank a different stream.

---

## 4. Varlen training forward — `train_gen_varlen.py`

The forward is **monkey-patched onto `QuadtreeGPT.forward`** before `accelerate.prepare`, so DDP
wraps it and gradients sync. It reproduces the original padded forward **bit-for-bit**, but
packed + varlen:

- Build per-token: `seg_id`, `local_pos` (0..Lᵢ-1 within each segment), `is_start`.
- **Tree position embedding**: scatter packed `(lod, patch)` into a padded `[n_seg, max_L]`
  view, call the model's existing `get_token_indices_embedding`, gather back to `[T, D]`.
- **Input**: `tok_embeddings(roll(code,1)) + roll(pos_emb,1)`, then overwrite segment-start rows
  with `cls_embedding(cls)` (LabelEmbedder, keeps CFG label-dropout). This matches the original
  `cat([cond, tok + token_indices_embedding[:, :-1]])`: cls at seq-pos 0, node k's code at
  seq-pos k+1 carrying node k's tree-position, predicting node k's code.
- **RoPE**: `freqs_cis[local_pos]` → resets per segment.
- **Attention**: each block called with `cu_seqlens/max_seqlen` → `flash_attn_varlen_func(...,
  causal=True)` (block-diagonal causal; a token attends only within its own sample). Blocks are
  gradient-checkpointed when `model.grad_checkpointing`.
- **Loss**: `F.cross_entropy(output(out_norm(x)), code)` over all T (every packed position is
  valid — no padding mask needed).

**Patches applied to `modeling/modules/attention.py`:**
- Added `from flash_attn import flash_attn_varlen_func` (import guarded).
- `Attention.forward` + `TransformerBlock.forward` take optional `cu_seqlens, max_seqlen`; when
  given, take the varlen flash path (per-token RoPE, `causal=True`); the padded SDPA path is
  untouched.

---

## 5. Config + launcher

`configs/gpt_quadtree_3level.yaml` — the exact fields `QuadtreeGPT.__init__` reads
(`model_size: base`, `num_patch_side_list [1,2,4,8,16,32]`, `patch_size_list [16×5, 8]`,
`guaranteed_depth 3`, `codebook_size 16384`, dropouts) + a `train:` block
(`max_tokens_global 1048576`, `lr 4e-4`, wd 0.05, betas (0.9,0.95), warmup 2000, steps 300000).

Launch (8-GPU, single node):
```bash
cd /sensei-fs-3/users/yuchengm/code/quadtok/3level
python -m torch.distributed.run --nproc_per_node 8 --master_port 29513 train_gen_varlen.py \
  --config configs/gpt_quadtree_3level.yaml \
  --shards "/PERSISTENT/pretok-{000000..NNNNNN}.tar" \
  --max_tokens_global 1048576 --steps 300000 --lr 4e-4 --num_workers 4
```
`train_gen_varlen.py` CLI: `--config --shards --max_tokens_global (default 262144*4) --steps
--lr --num_workers`. Per-GPU token budget = `max_tokens_global / num_processes`.

---

## 6. Smoke result (validated)

100-step smoke launched on 8×H200 at the **exact target config** `max_tokens_global = 262144*4
= 1,048,576` (**131,072 tokens/GPU**) with **learnable synthetic** data (`synth_pretok.py
--learnable`; codes = f(cls,lod) so a correct pipeline must show a *decreasing* loss). Observed:

```
world=8  per_gpu=131072  model=344.0M
step  0 : loss ≈ 9.70 (= ln 16384, correct random-init CE)   # from the 1-GPU sanity
step 26 : loss 7.4701   ntok 130784  nseg 141   2.13s  peakmem 45.8GB
step 50 : loss 7.1283   ntok 130156  nseg 138   2.14s  peakmem 45.9GB
```
- All 8 GPUs at 100% util; **ntok ≈ 131072/GPU** (target), **nseg ≈ 140** samples/pack (no padding).
- **Peak memory 45.9 GB** on 141 GB H200 — large headroom (fits 80 GB H100 too).
- **~2.14 s/step** steady.
- **Loss strictly decreasing 9.70 → 7.11** over 50 steps → forward + flash-varlen + CE + backward
  + **DDP grad-sync across 8 GPUs** + optimizer all correct and *learning*.

The borrowed idle node was reclaimed at step ~50; the run therefore did not print the step-100
line, but every metric that the smoke exists to validate (runs, no-OOM, correct CE, decreasing
loss, DDP, per-GPU token load, throughput) was captured. A 1-GPU sanity separately completed
cleanly (`SANITY_EXIT 0`, loss 9.70 at init).

---

## 7. Submitting the REAL training job (for another session)

1. **Stage the tokenizer weight** to a persistent path (once):
   `snapshot_download("yuchengm/quadtok", local_dir=/sensei-fs-3/users/yuchengm/data/quadtok_hf_weight)`.
2. **Pretokenize the full train set** to persistent sharded tars (§1), covering all
   `imagenet-wds/train` shards across GPUs/nodes. Verify a shard: lod monotonic, T≈989.
3. **Submit generator training** as an AIP job (self-submitting pattern, see
   `~/.script/submit_exp9079.sh`): `--job-type training`, image
   `docker-matrix-experiments-snapshot.ff.adobe.io/training-base-07-23-25:...` (python3.10 so
   `base/.venv` works), `--gpu-instance-type p5en.48xlarge --xpus-per-pod 8 --num-pods N`. The
   pod-side script should `cp -a` the repo to local, then run the §5 `torch.distributed.run`
   line with `--shards` = the persistent pretok tars, `--steps/--lr` for the real schedule, and
   `--max_tokens_global = 262144*4*num_pods*?` (scale the global budget with world size; keep
   ~131072/GPU). Add checkpoint saving + resume + W&B logging to `train_gen_varlen.py` for the
   real run (the smoke omits these).
4. **Multi-node**: `torch.distributed.run --nnodes N --node_rank $RANK --master_addr … ` (or the
   repo's `scripts/multinode_train.sh` pattern). Varlen packing is per-rank; the global token
   budget divides by total world size.

**Not yet done (real-run TODOs, out of smoke scope):** checkpoint save/resume, W&B, LR schedule
(warmup+cosine), and switching `generate()`'s inference node ordering to slot-order for
train/inference consistency.

---

## 8. Environment + gotchas

- **Interpreter**: `base/.venv` (python3.10.12, torch 2.7.1+cu128, flash_attn 2.8.3.post1, wds
  1.0.2, accelerate 1.14.0) works only on the **training-base (python3.10) image**. Its
  `bin/python` symlinks `/usr/bin/python3.10`; on a python3.12 image the venv is "broken".
- **Reuse the venv on any image** (used for this smoke): install a standalone cp310 once —
  `UV_PYTHON_INSTALL_DIR=/sensei-fs-3/users/yuchengm/.uvpy uv python install 3.10` (persists on
  sensei-fs) → then run with
  `PYTHONPATH=<base/.venv>/lib/python3.10/site-packages <that python3.10> …`. Reuses torch +
  flash_attn (verified CUDA True) with zero reinstall. The venv's `torchrun` console-script has
  a hard-coded shebang, so on a reused env call `python -m torch.distributed.run` instead.
- **`aip job ssh <job> -- '<one single-quoted string>'`** — the command must be ONE argument
  (multiple tokens → "unexpected extra argument"). `GRPC_ENABLE_FORK_SUPPORT=1`; filter noise
  with `grep -vE "poll_posix|fork parent|Other threads|INFO ›"`.
- **Non-login shells have no `python` on PATH** — always use an absolute interpreter path.
- **`/tmp` and `/mnt/localssd` are ephemeral** (wiped on stop/preempt). Code lives on sensei-fs
  (persistent); pretokenized data for a real run must go to persistent storage.
- **Interactive/preemptible jobs get preempted** (this smoke lost two H200 nodes mid-run). Prefer
  a proper training job (auto-recovery) for the real run.
