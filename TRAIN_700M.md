# 训练 700M Generator（QuadtreeGPT xlarge, varlen）

从头训练 QuadtreeGPT **xlarge = 775M** 的完整配方。同一套脚本也训 100M / 300M，只改 `SIZE`。
B200 环境搭建见 [`B200_MIGRATION.md`](./B200_MIGRATION.md)。

---

## 1. 是什么

- **模型**：QuadtreeGPT（LlamaGen 风格 causal AR），在 2-level 256 tokenizer 的离散 code 上做 next-token 预测。
- **训练方式**：**varlen** —— 用 `flash_attn_varlen_func`（block-diagonal causal，无 padding），按 **token 预算**动态打包不同长度的样本，`grad-accum = 1`。
- **入口**：`scripts/train_generator_varlen.py`（`accelerate launch` 起）。
- **配置**：`configs/inference/gpt_16k_base.yaml`（`CFG` 环境变量可换）。

### 模型尺寸（`SIZE`）
| SIZE | 参数量 | 对应 | LlamaGen 类比 |
|---|---|---|---|
| `small` | ~111M | 100m | B |
| `base` | ~343M | 300m | L |
| `xlarge` | **~775M** | **700m** | **XL** |

---

## 2. 数据

pretok tars（webdataset 分片），每个 tar 若干样本，字段是每张图的 **code_indices / lod_indices / patch_indices**（2-level: lod3 64 + lod4 256，center+hflip 2× 增广）。

- **HuggingFace**：`yuchengm/quadtok_data` → `pretok_2level/`（1470 个 `train-*.tar`，~25 GB）
- **原始 S3**：`s3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/`

下载见 [`B200_MIGRATION.md`](./B200_MIGRATION.md) §3。训练时 `DATA_DIR` 指向含 `train-*.tar` 的目录即可。

varlen loader（`data/varlen_pretok_loader.py`）会：per-GPU token 预算 = `MAX_TOKEN_GLOBAL // world`，贪心打包样本到该预算，产出 `code_indices/lod_indices/patch_indices (T,)`、`labels (N,)`、`cu_seqlens (N+1,)`。

---

## 3. 环境

见 [`B200_MIGRATION.md`](./B200_MIGRATION.md) §1。关键固定版本（Blackwell 必需）：
`torch 2.7.1+cu128` · `flash-attn 2.8.3.post1` · `accelerate 1.14.0`。

---

## 4. 起训

### 4.1 环境变量（全部配置项）
| 变量 | 默认 | 说明 |
|---|---|---|
| `SIZE` | small | `small`/`base`/`xlarge` |
| `MAX_TOKEN_GLOBAL` | 232448 | 全局每步 token 数（~1024 图）。per-GPU = 此值 // world |
| `GC` | 0 | grad checkpointing；80GB 卡训 700m 需 `1`，B200 用 `0` |
| `LR` / `END_LR` | 4e-4 / 2e-5 | cosine 峰值 / 结束 lr |
| `WARMUP` / `STEPS` | 50000 / 400000 | 线性 warmup 步 / 总步 |
| `SAVE_EVERY` | 2500 | 每多少步存一次 ckpt |
| `WD` | 0.05 | AdamW weight decay |
| `EMA_DECAY` | 0.9999 | EMA 衰减 |
| `NUM_WORKERS` | 6 | dataloader worker |
| `DATA_DIR` | /mnt/localssd/pretok | 含 `train-*.tar` 的目录 |
| `CKPT_S3` | — | ckpt 读（resume）+ 写（上传）位置 |
| `CKPT_LOCAL` | /mnt/localssd/ckpt | 本地 ckpt 暂存 |
| `RUN_NAME` | gen_varlen_<size> | wandb run id + S3 子目录 |
| `WANDB_KEY` `WANDB_ENTITY` `WANDB_PROJECT` | — | 可选监控；空 KEY 则关 wandb |

### 4.2 从头训练（单机 8 卡）
**从头 = `CKPT_S3` 指向一个没有 `step_N` 的新位置**（`latest_step_on_s3` 返回 -1 → `start_step=0`）。
```bash
export SIZE=xlarge GC=0 RUN_NAME=gen_700m
export MAX_TOKEN_GLOBAL=232448 LR=4e-4 END_LR=2e-5 WARMUP=50000 STEPS=400000 SAVE_EVERY=2500 WD=0.05
export DATA_DIR=/data/quadtok/pretok_2level
export CKPT_S3=s3://<你的桶>/quadtok_gen_ckpt/gen_700m/     # 新的空路径 = 从头
export CKPT_LOCAL=/data/ckpt_local
export WANDB_KEY=<...> WANDB_ENTITY=<...> WANDB_PROJECT=quadtok-gen-varlen

accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
  scripts/train_generator_varlen.py
```

### 4.3 续训（resume）
**同一个 `CKPT_S3` 里已有 `step_N`** 时自动续：脚本找最大 `step_N` → 拉到 `CKPT_LOCAL` → `accelerate.load_state` + 载入 EMA → 从该 step 继续。wandb 用 `id=RUN_NAME + resume="allow"`，续同一个 run，log 不乱。**无需改任何参数，重跑同一命令即可。**

### 4.4 多机
```bash
accelerate launch --num_processes $((8*NNODES)) --num_machines $NNODES \
  --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  --mixed_precision bf16 scripts/train_generator_varlen.py
```
per-GPU token 预算 = `MAX_TOKEN_GLOBAL // (8*NNODES)`，保证任意 GPU 数下 global batch 恒定。

---

## 5. 显存 / 吞吐 / GC

- **GC（grad checkpointing）**：80 GB 卡训 700m 必须 `GC=1`（否则 OOM）；**B200/180GB 用 `GC=0` 更快**。
- **Grad accumulation**：本配方**不用**（accum=1）；global batch 靠 token 预算堆，不靠累积。
- **吞吐参考**：单卡 700m ≈ 13.5k tok/s（H200，含 GC）；B200 无 GC 约 H100 的 2–2.5×。
- **torch.compile / grad accum**：实测无收益（flash 已融合、accum 线性），**最快 = varlen + 不 compile + 不 accum**。

---

## 6. Checkpoint

- 每 `SAVE_EVERY` 步存 `accelerate state`（模型+优化器+RNG）+ `meta.pt`（EMA + step）到 `CKPT_LOCAL/step_N`，异步 `s5cmd` 上传到 `CKPT_S3/step_N/`。
- **保留所有 `step_N`**（S3 端全留；本地只留最近 3 个省 localssd）。
- 一个 700m ckpt ≈ **11.5 GB**（模型 + Adam 双矩 fp32 + EMA）。

### 脱离 S3（B200 无 Adobe S3 时）
把 `scripts/train_generator_varlen.py` 的读/写改成纯本地，约 10 行：
```python
# resume：用本地目录代替 latest_step_on_s3
def latest_step_local(ckpt_local):
    import glob, os
    steps = [int(os.path.basename(d).split("_")[1])
             for d in glob.glob(os.path.join(ckpt_local, "step_*"))]
    return max(steps) if steps else -1
# ... resume 段：last = latest_step_local(ckpt_local); 直接 load_state(CKPT_LOCAL/step_last/state)，不做 s5 cp

# save_ckpt：删掉这行（不上传 S3），只保留本地写
# threading.Thread(target=s5, args=("cp", d + "/", f"{ckpt_s3}step_{step}/"), daemon=True).start()
```
或更省事：把 `CKPT_S3` 指向自建 S3/MinIO，代码零改动。

---

## 7. 监控 & 冒烟

- **wandb**：project `quadtok-gen-varlen`，run = `RUN_NAME`。指标 `loss / acc / lr / tok_per_s`，x 轴用 metric `step`（续训后单调续画）。
- **起训应看到**：
```
[train] size=xlarge params=775.xM world=8 per_gpu_tok=29056 global_tok=232448 gc=False start_step=0 steps=400000
step 0 loss ~9.7 acc ~0.00 lr ... <tok/s>
```
- **loss 基线**：codebook=16384 → 初始 CE ≈ ln(16384) ≈ 9.70，随训练下降即正常。

---

*代码*：GitHub `myc634/quadtok` 分支 `base-update`｜入口 `scripts/train_generator_varlen.py`｜配置 `configs/inference/gpt_16k_base.yaml`
