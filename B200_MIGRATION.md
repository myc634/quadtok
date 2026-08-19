# 700M Generator → B200 迁移指南（从头训练）

把 `gen_700m`（QuadtreeGPT xlarge, 775M, varlen 训练）搬到一台 **B200 (Blackwell / sm_100)** 机器，**从头开始训练**（不迁移 checkpoint）。

> 训练配方细节见同目录 [`TRAIN_700M.md`](./TRAIN_700M.md)。本文只讲 **B200 环境搭建 + 数据获取 + 从头起训**。

---

## TL;DR

好消息：**当前软件栈本身就支持 B200**。venv 里是 `torch 2.7.1+cu128` + `flash_attn 2.8.3.post1`，torch 的 arch list 已含 `sm_100`（B200）和 `sm_120`。所以不用换 CUDA/torch。三步：

1. B200 机器上**复现依赖**（torch 2.7.1+cu128 / flash-attn 2.8.3 / accelerate 1.14.0）。
2. 拉 **代码**（GitHub `myc634/quadtok`）+ **数据**（HuggingFace `yuchengm/quadtok_data`, 23.4 GB）。**不搬 ckpt**。
3. `accelerate launch` 从头起训（`start_step=0`）。

---

## 1. B200 环境（Blackwell / sm_100）

硬性要求：CUDA ≥ 12.8、torch ≥ 2.7（cu128）、flash-attn 带 sm_100 kernel。

> ⚠️ Adobe 内部镜像 `docker-matrix-experiments-snapshot.ff.adobe.io/...` 在 Adobe 外拉不到，用公开等价镜像自建。

```bash
# 基础镜像二选一（都含 CUDA 12.8 + cuDNN9 + torch 2.7.x）
#   pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
#   nvcr.io/nvidia/pytorch:25.03-py3   (NGC，已带 Blackwell 优化)

pip install accelerate==1.14.0 omegaconf webdataset einops numpy wandb \
            'huggingface_hub[hf_transfer]'
pip install flash-attn==2.8.3.post1 --no-build-isolation
```

**验证 Blackwell**
```bash
python - <<'PY'
import torch
print("dev:", torch.cuda.get_device_name(0))     # 期望含 B200
print("arch:", torch.cuda.get_arch_list())        # 期望含 'sm_100'
from flash_attn import flash_attn_varlen_func
print("flash_attn_varlen ok")
PY
```
若 flash-attn 在 sm_100 报 `no kernel image is available` → 针对 Blackwell 源码重编：
```bash
TORCH_CUDA_ARCH_LIST="10.0" pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir
```

---

## 2. 拿代码

```bash
git clone -b base-update https://github.com/myc634/quadtok.git
cd quadtok
```
训练入口 `scripts/train_generator_varlen.py`，配置 `configs/inference/gpt_16k_base.yaml`。

---

## 3. 拿数据（HuggingFace，23.4 GB）✅ 已上传

数据已传到 **`https://huggingface.co/datasets/yuchengm/quadtok_data`** → 路径 `pretok_2level/`
（**1470 个 `train-*.tar`，共 23.4 GB**，center+hflip 2× 增广的 2-level 256 tokenizer codes）。

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1                 # 关键：加速下载，别省（纯 http 会很慢）
export HF_TOKEN=<你的 HF read token>               # 私有 repo，下载需对该 repo 有读权限

huggingface-cli download yuchengm/quadtok_data \
  --repo-type dataset --include 'pretok_2level/*' \
  --local-dir /data/quadtok
# 数据落在 /data/quadtok/pretok_2level/train-*.tar → 训练用 DATA_DIR=/data/quadtok/pretok_2level
```
先装 hf_transfer：`pip install 'huggingface_hub[hf_transfer]'`。25 GB 级数据 1–3 分钟到位。

> 备选：若 B200 机器有该 S3 桶凭证，可直接
> `s5cmd cp 's3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/*' /data/quadtok/pretok_2level/`
> 数据是**怎么传上 HF 的**（含踩坑）见 §7 附录。

---

## 4. 从头起训

**最省事：用 `scripts/train_b200.sh`**（已固化 `GC=0` + 固定 global batch + 8 卡最快配方），只需给 `CKPT_S3` / `DATA_DIR`：
```bash
CKPT_S3=s3://<你的桶>/quadtok_gen_ckpt/gen_700m_b200/ \
DATA_DIR=/data/quadtok/pretok_2level RUN_NAME=gen_700m_b200 \
bash scripts/train_b200.sh
```

**关键：从头训练 = 让 `CKPT_S3` 指向一个空的新位置**，脚本 `latest_step_on_s3()` 找不到 `step_N` 就返回 -1，`start_step=0` 从头开始。

手动展开（等价）：
```bash
export SIZE=xlarge GC=0 RUN_NAME=gen_700m_b200        # B200 显存大，GC=0（见 §5）
export MAX_TOKEN_GLOBAL=232448                        # ~1024 图/step，保持同 global batch
export LR=4e-4 END_LR=2e-5 WARMUP=50000 STEPS=400000 SAVE_EVERY=2500 WD=0.05
export DATA_DIR=/data/quadtok/pretok_2level
export CKPT_S3=<新的空 ckpt 位置>                      # 从头训 -> 用没有 step_N 的新路径
export CKPT_LOCAL=/data/ckpt_local
# 可选 wandb
export WANDB_KEY=<...> WANDB_ENTITY=<你的 entity> WANDB_PROJECT=quadtok-gen-varlen

accelerate launch --num_processes 8 --num_machines 1 --mixed_precision bf16 \
  scripts/train_generator_varlen.py
```

> **⚠️ ckpt 读写走 S3（s5cmd）**：`train_generator_varlen.py` 的 `latest_step_on_s3` / `save_ckpt` 用 `s5cmd`。B200 若连不上 Adobe S3，两种改法：
> - 把 `CKPT_S3` 指向你自己的 S3/MinIO（零改动）；
> - 或把 resume/save 改成读写本地目录（`latest_step_on_s3` 改成 `glob` 本地 `step_*`；`save_ckpt` 去掉 S3 上传线程）。约 10 行，见 [`TRAIN_700M.md`](./TRAIN_700M.md) §「脱离 S3」。

---

## 5. B200 专属注意点

**① grad 相关（回答「是不是不用开 grad accmu 了」）**
- **Grad accumulation（梯度累积）**：我们这套 varlen 训练**从来没用过**（accum=1）。global batch 是靠「per-GPU token budget × GPU 数」直接堆出来的，不是靠累积。所以**没有「关掉」这一说**。
- **Grad checkpointing（激活重算，`GC` 开关）**：这才是之前为 700m 开的东西（`GC=1`），纯粹因为 A100/H100 只有 80 GB。**B200 有 180–192 GB，775M 完全装得下 → 设 `GC=0`**，省掉 backward 的重算，快 ~25–30%。
  - 显存粗算（bf16 混合 + AdamW + EMA）：模型 bf16 ~1.5 GB + Adam(m,v) fp32 ~6.2 GB + EMA ~1.5 GB + 梯度 ~1.5 GB + 激活（varlen，per-GPU 29k tokens）≈ 十几 GB。180 GB 绰绰有余。
  - **已固化进 config**：`configs/inference/gpt_16k_base.yaml` 的 `model.grad_checkpointing` 默认改为 `false`；`train_b200.sh` 也显式 `GC=0`。（80GB 的旧 job 仍可用 `--env GC=1` 覆盖，互不影响。）

**②「固定 global batch 下如何最快用满 1-node B200」**——这正是目标：
- **用满 8 卡**：global batch 固定 232448，8 卡时每卡只做 `232448/8=29056` tokens，per-GPU 工作量最小 → **每步 wall-clock 最快**（数据并行）。同样 global batch，8×B200 ≈ 4×B200 的 2 倍速。
- **`GC=0`**：拿 B200 的大显存换掉激活重算 → backward 快 ~25–30%。
- **数据放本地 NVMe + 多 worker**（`NUM_WORKERS=12`）：B200 算得快，别让 dataloader 拖后腿。
- per-GPU 29k tokens 只用 ~20GB/180GB，**这是好事**（每卡活少=快）；固定 global batch 下，多出来的显存无法再换成更多速度（除非改 global batch 或减卡数，都与目标冲突）。
- torch.compile 在 H100 无收益；Blackwell 上可选再测。
- 想和现有 `gen_700m` 曲线可比就**锁死 `MAX_TOKEN_GLOBAL=232448`**；若哪天想加大 batch，调大它（等价改了优化超参，注意）。

**③ 吞吐**：bf16 下 B200 约 H100 的 2–2.5×。

**④ flash-attn**：代码用 FA2 的 `flash_attn_varlen_func`（block-diagonal causal，无 padding），2.8.3 API 稳定；§1 验证过 sm_100 kernel 即可，无需改代码。

**⑤ 多机**：单机 8 卡最省心。多台 B200 加 `--num_machines N --machine_rank R --main_process_ip <rank0> --main_process_port <port>`。

---

## 6. 冒烟自检（起训后应看到）
```
[train] size=xlarge params=775.xM world=8 ... start_step=0 steps=400000
step 0 loss ~9.x acc ~0.00 lr ... <tok/s>
```
从头训练看到 `start_step=0` + loss 从 ~9.x（ln(codebook=16384)≈9.7）开始下降 = 正常。

---

## 7. 附录：数据是怎么传到 HF 的（复现 / 重传用）

pretok 数据在 relay（有 S3 访问权的节点）上，通过 `S3 → localssd → HF` 上传到 `yuchengm/quadtok_data`。三个踩过的坑，重传时照做即可：

1. **必须用 WRITE 权限的 HF token**。只读 token 能 `whoami`、也能 `create_repo(exist_ok=True)`（对已存在 repo 是 no-op，会假装成功），但真正上传/commit 会 **`403 Forbidden`**。去 [settings/tokens](https://huggingface.co/settings/tokens) 建 **Write**（或 fine-grained 勾 `yuchengm/quadtok_data` 写权限）。
2. **用 `hf_transfer` 加速**。纯 huggingface_hub 上传只有 ~5 MB/s 且会 stall（卡在 11% 不动）；装 `hf_transfer`（Rust 并行分块 + 重试）后 **23 GB ~150s 传完**。venv 无 pip 时：`python3 -m pip install --target=<dir> hf_transfer` + `PYTHONPATH=<dir>` + `export HF_HUB_ENABLE_HF_TRANSFER=1`。
3. **用 `upload_folder`（批量 LFS），别用 `upload_large_folder`**。后者逐文件调 API，1470 个文件瞬间打爆 HF 的 **1000 请求/5min** 限流（429）；`upload_folder` 一次 commit、批量 LFS，只几次 API 调用。

核心上传代码（token 从 env 读，别写进进程参数，免得 `ps` 泄露）：
```python
import os
from huggingface_hub import HfApi          # 需 env: HF_TOKEN + HF_HUB_ENABLE_HF_TRANSFER=1
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    repo_id="yuchengm/quadtok_data", repo_type="dataset",
    folder_path="<local>/pretok_2level", path_in_repo="pretok_2level",
    commit_message="add pretok_2level tars")
# 校验：list_repo_files(...) 里 pretok_2level/*.tar 应为 1470
```

> 小贴士：进度别看 `/proc/<pid>/io` 的 `wchar`——hf_transfer 在 Rust 线程里传，不计入 python 进程的 io，会显得"卡住"。以 `list_repo_files` 的文件数为准。

---

*数据*：HF `yuchengm/quadtok_data` → `pretok_2level/`（**1470 tars, 23.4 GB**）｜ 原始 S3 `s3://g3i-data/yuchengm/quadtok_pretok_2level_center_hflip/`
