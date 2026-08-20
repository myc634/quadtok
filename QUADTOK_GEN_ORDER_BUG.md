# QuadTok Generation Bug 调试总结：Pretokenization Token 顺序错位

**日期**：2026-08-19
**分支**：`base-update`（remote `github.com/myc634/quadtok.git`）
**结论**：`gen_100m` 生成效果差（FID ~45 vs drive gen FID ~6）的根因**不是**插值/预处理，而是 **pretokenization 输出的 token 序列顺序错了**——tensor-native 加速版按 *slot/空间 patch 序* 输出 token，而 generator（AR）需要 `_get_ordered_nodes` 的 *BFS/nested 序*。两者对 lod-3 差一个 `coarse_split_permutation`。

---

## 1. 症状

- `gen_100m`（small, 111M, base-update 新 pretok 训练）生成图 FID ~45，明显差。
- Google-Drive 的 `drive_gen`（base, 947M, 原版 pipeline 训练）生成正常，FID ~6。
- 早期排查一度怀疑 CFG scheduler、EMA、预处理插值（BILINEAR vs center_crop_arr），均**不是**主因。

## 2. 关键调试思路（决定性）

> 用一个**已知 good 的模型（drive gen）当"探针"**去测整条 pipeline。

流程：`image → tokenizer.encode → guided 搜索出 tree → 每个 node 抽 VQ code → 打包 varlen → gen.forward_varlen → CE loss`。

判据：codebook=16384，随机 CE = `ln(16384) ≈ 9.70`。已知 good 的 drive gen 若在当前 codebase 上能到 **CE ≈ 7**，说明 pipeline 正确；若接近 9.7（随机），说明 pipeline 有 bug。

这个指标把"tokenizer 好不好"和"pipeline 对不对"解耦开了——**唯一、可证伪**。

## 3. 调试过程与测量

用 `scripts/pipeline_val_loss.py`（本次新写，走 `_get_ordered_nodes` 正确顺序）+ 早期 `val_loss.py`（走我 pretok 的 tensor 顺序）交叉测量，得到完美交叉：

| 模型 | 我 pretok 的顺序（`active_to_padded` slot/空间序） | 正确顺序（`_get_ordered_nodes` BFS/nested） |
|---|---|---|
| **drive gen**（good, FID 6，训在正确序上） | **9.55**（≈随机, acc 0.76%） | **7.06**（acc 3.67%）✅ |
| **gen_100m**（弱, FID 45，训在错序上） | **7.63**（acc 3.0%） | **10.01**（比随机还差, acc 0.26%） |

**解读**：
- drive 训在正确序 → 正确序上 good（7.06），错序上懵（9.55）。
- gen_100m 训在错序 → 错序上 good（7.63），正确序上懵（10.01，比随机差=它把错误顺序"背死"了）。
- ⇒ **两个模型训在不同的 token 顺序上**，`gen_100m` 学的是错序，解码约定对不上 → 坏图。

排除项（都不是主因）：
- **插值**：原版 pretok（`extract_code_searchquadtree.py` → `QuadtreeImageTransform`）用 `center_crop_arr(281)+TenCrop(256)`；我的也是 `center_crop_arr`。两者插值一致。recon 的 BILINEAR vs center_crop_arr 差 0.5 PSNR 只是**评测口径**，与 generation token 无关。
- **tokenizer**：同一个 drive tokenizer；recon rFID 1.50 可复现（见下，正是它掩盖了 bug）。
- **CFG / EMA / 模型大小**：都排除或非主因。

## 4. 根因

### 4.1 出错的代码

`scripts/bench256_tensor.py`（commit **365c386** 引入的 tensor-native 加速搜索；pretok 在 **f8fc1c9**）：

```python
def build_slot_maps(device):
    lod3, lod4 = 64, 256
    SLOT_LOD   = cat([full(64,3), full(256,4)])          # lod-major ✓
    SLOT_PATCH = cat([arange(64), arange(256)])          # ✗ patch = 空间序 0..63
    PARENT4    = ((j//16//2)*8 + (j%16//2))              # lod4→lod3 空间父映射

def active_to_padded(active, SLOT_LOD, SLOT_PATCH):
    order = argsort(active.int(), dim=1, descending=True, stable=True)  # ✗ 按 slot 序
    sel   = order[:, :max_seq]
    lod_pad = SLOT_LOD[sel]; pat_pad = SLOT_PATCH[sel]   # ⇒ 输出 = 空间 patch 序
```

`pretokenize_train.py` 用 `active_to_padded` + `_select_optimize_core` 抽 code 并保存，于是**每个样本的 token 序列都是"空间 patch 升序"**。

### 4.2 generator 期望的顺序

`QuadtreeGPT.generate()` 与原版 pretok `extract_code_searchquadtree.py` 都用：

```python
search_nodes = model._get_ordered_nodes(search_tree)     # BFS 遍历，按 lod 分组
search_nodes = [n for n in search_nodes if n.lod_level >= 3]
```

`_get_ordered_nodes` = BFS + lod-major。lod-3 的 BFS/nested 顺序**≠** 空间 `arange(64)` 顺序，两者差一个 `coarse_split_permutation`（probe256 里正确处理了：`new_target_nodes = [target_nodes[i] for i in inverse_permutation(coarse_split_permutation())]`）。

`forward_varlen` 给位置 P 的输入是 `tok_emb(token_{P-1}) + struct[P-1]`，RoPE=P，位置 0 是 cls——**AR 完全依赖序列顺序**（模型不吃"当前要预测哪个 node"的 struct，只靠固定顺序知道 P 对应哪个 node）。顺序一变，条件全错 → 近随机。

### 4.3 为什么 bug 躲过了 tokenizer 评测（最阴险的一点）

`_decode_optimize_core(z, lod_pad, pat_pad, seqlens)` **按 (lod, patch) 把 token 散射到网格**再解码——**与序列顺序无关**。所以：

- **recon 评测（rFID/PSNR/IS）对 token 顺序不敏感** → tensor-native 版复现出 rFID 1.50/PSNR 20.3，看着完全正常。
- **generation（AR）对 token 顺序高度敏感** → 顺序错位直接致命。

⇒ 加速重写只在"顺序无关"的 recon 路径上验证过，`argsort(active)` 用了 slot/空间序而漏了 `coarse_split_permutation`，且**没有任何 recon 指标能暴露它**，直到用 AR 模型（drive gen）当探针才现形。

## 5. 修复

把 pretok 的 token **输出顺序**改成 `_get_ordered_nodes`（BFS/nested）：

- **方案 A（保速度，推荐）**：在 `build_slot_maps` 里预计算一个 `SLOT_RANK[320]`（每个 slot 在满树 `_get_ordered_nodes` 顺序中的名次），`active_to_padded` 改成按 `(¬active, SLOT_RANK)` 排序而非按 slot index 排序。tensor 化、O(1) 重排、保持 ~114 img/s。
- **方案 B（最稳）**：直接走 node-based `_get_ordered_nodes` + `_forward_optimize`（与 `extract_code_searchquadtree.py` / `pipeline_val_loss.py` 一致，已验证 drive → 7.06），牺牲部分速度换绝对正确。

**验收标准**：用修好的 pretok 产出 tar → 经训练 loader → drive gen `forward_varlen` → **CE ≈ 7.06**（而非 9.55）。

然后需**重新 pretokenize 全量 ImageNet-train + 重训 gen_100m/300m/700m**（旧的 `s3://.../quadtok_pretok_2level_center_hflip/` 是错序数据，作废）。

## 6. 经验教训

1. **加速重写务必做端到端等价性验证**，尤其当新路径改变了数据布局/顺序时。recon 指标（顺序无关）不能替代 generation 验证（顺序相关）。
2. **"用已知 good 的模型当探针"** 是定位 pipeline bug 的利器：把"模型质量"和"数据/格式正确性"解耦，给出唯一、可证伪的指标（CE 是否接近随机）。
3. **交叉验证**（drive/gen_100m × 正确序/错序）能一次性坐实"两模型训在不同顺序上"，排除一切歧义。

---

**关键文件**：`scripts/pipeline_val_loss.py`（探针）、`scripts/bench256_tensor.py`（bug）、`scripts/pretokenize_train.py`（保存错序）、`modeling/mar.py::QuadtreeGPT.{forward_varlen,generate,_get_ordered_nodes}`、`scripts/extract_code_searchquadtree.py`（原版正确顺序参考）。
