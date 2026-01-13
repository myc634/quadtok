import random
import numpy as np
from tqdm import tqdm # 用于显示进度条
import time
from modeling.utils import build_quadtree, build_random_quadtree, QuadTreeNode

def _create_and_assign_children(node, patches_per_side_list):
    """
    辅助函数：为一个节点创建4个子节点，并计算它们的 patch_index。
    """
    current_depth = node.lod_level
    child_lod_idx = current_depth + 1
    
    # 检查是否会超过最大深度
    if child_lod_idx >= len(patches_per_side_list):
        # 这种情况不应该在主逻辑中发生，但作为安全检查
        return [] 

    parent_patches_per_side = patches_per_side_list[current_depth]
    child_patches_per_side = patches_per_side_list[child_lod_idx]

    parent_row, parent_col = divmod(node.patch_index, parent_patches_per_side)
    child_start_row, child_start_col = parent_row * 2, parent_col * 2

    top_left_idx = child_start_row * child_patches_per_side + child_start_col
    child_indices = [
        top_left_idx, 
        top_left_idx + 1, 
        top_left_idx + child_patches_per_side, 
        top_left_idx + child_patches_per_side + 1
    ]

    new_leaves = []
    for child_index in child_indices:
        child_node = QuadTreeNode(lod_level=child_lod_idx, patch_index=child_index)
        node.children.append(child_node)
        new_leaves.append(child_node)
    return new_leaves

def _build_guaranteed_recursive(node, current_depth, guaranteed_depth, patches_per_side_list, available_leaves_list):
    """
    (Phase 1 辅助函数) 递归构建确定性的树，直到 guaranteed_depth。
    """
    if current_depth == guaranteed_depth:
        available_leaves_list.append(node)
        return 0 # 这个节点是叶子

    # 必须扩展
    
    # 检查是否能合法地扩展（防止 guaranteed_depth > max_depth)
    if current_depth + 1 >= len(patches_per_side_list):
        available_leaves_list.append(node) # 无法扩展，将其视为叶子
        return 0

    _create_and_assign_children(node, patches_per_side_list)
    internal_nodes_count = 1 
    
    for child in node.children:
        internal_nodes_count += _build_guaranteed_recursive(
            child, current_depth + 1, guaranteed_depth, patches_per_side_list, available_leaves_list
        )
    return internal_nodes_count

def build_uniformly_random_quadtree_by_size(patches_per_side_list, num_internal_nodes, guaranteed_depth=0):
    """
    (步骤 2: 随机增长算法)
    
    使用“随机增长算法”构建一个随机四叉树，确保在所有具有
    `num_internal_nodes` 个内部分支节点的树结构中均匀采样。
    """
    
    local_patches_per_side = sorted(patches_per_side_list)
    max_possible_depth = len(patches_per_side_list) - 1
    
    if guaranteed_depth > max_possible_depth:
        guaranteed_depth = max_possible_depth

    root_node = QuadTreeNode(lod_level=0, patch_index=0)
    
    if num_internal_nodes == 0:
        return root_node

    available_leaves = []
    num_guaranteed_nodes = 0
    
    if guaranteed_depth > 0:
        num_guaranteed_nodes = _build_guaranteed_recursive(
            root_node, 0, guaranteed_depth, local_patches_per_side, available_leaves
        )
    else:
        available_leaves.append(root_node)

    if num_internal_nodes < num_guaranteed_nodes:
        # print(f"警告: 'num_internal_nodes' ({num_internal_nodes}) 不足以满足 'guaranteed_depth' "
        #       f"({guaranteed_depth}) 所需的 {num_guaranteed_nodes} 个节点。")
        num_internal_nodes = num_guaranteed_nodes

    num_nodes_to_add = num_internal_nodes - num_guaranteed_nodes
    nodes_added = 0
    
    # --- Phase 2: "随机增长算法" ---
    while nodes_added < num_nodes_to_add and available_leaves:
        
        node_to_split_idx = random.randrange(len(available_leaves))
        node_to_split = available_leaves[node_to_split_idx]

        # 检查：如果选中的叶子已经达到了最大深度，它就不能再分裂了
        if node_to_split.lod_level >= max_possible_depth:
            available_leaves.pop(node_to_split_idx)
            continue 

        # 从列表中移除 (O(1) 方式)
        available_leaves[node_to_split_idx] = available_leaves[-1]
        available_leaves.pop()

        # 分裂节点
        new_leaves = _create_and_assign_children(node_to_split, local_patches_per_side)
        
        available_leaves.extend(new_leaves)
        nodes_added += 1

    # if nodes_added < num_nodes_to_add:
    #     print(f"警告: 达到了最大深度限制，只添加了 {nodes_added} / {num_nodes_to_add} 个随机节点。")
        
    return root_node

# -------------------------------------------------------------------
# 解决方案：结合随机大小 + 均匀采样
# -------------------------------------------------------------------

def build_random_quadtree(patches_per_side_list, guaranteed_depth=0, min_nodes=1, max_nodes=50):
    """
    (步骤 1 + 2 的包装器)
    
    构建一个随机四叉树。树的“大小”（内部分支节点数）将从
    [min_nodes, max_nodes] 范围内随机选择。
    
    然后，对于选定的该大小，将从所有可能的树结构中均匀采样一个。
    """
    
    # --- 步骤 1: 随机选择一个“复杂度” (内部节点总数 I) ---
    # 确保 min_nodes 至少能满足 guaranteed_depth
    
    # 先计算 guaranteed_depth 至少需要多少节点
    min_guaranteed_nodes = 0
    if guaranteed_depth > 0:
        # (4^guaranteed_depth - 1) / 3
        min_guaranteed_nodes = (4**guaranteed_depth - 1) // 3
    
    # 确保我们的随机范围下限是合理的
    effective_min_nodes = max(min_nodes, min_guaranteed_nodes)
    effective_max_nodes = max(effective_min_nodes, max_nodes)
    
    # 从 [min, max] 范围内随机选择一个目标节点数 I
    # (使用 randint, 两端都包含)
    chosen_num_internal_nodes = random.randint(effective_min_nodes, effective_max_nodes)

    # --- 步骤 2: 调用均匀采样算法 ---
    # 使用我们随机选择的 I，去生成一个在该大小上均匀采样的树
    return build_uniformly_random_quadtree_by_size(
        patches_per_side_list,
        chosen_num_internal_nodes,
        guaranteed_depth
    )

def _copy_subtree_to_depth(complete_node, max_depth):
    new_node = QuadTreeNode(complete_node.lod_level, complete_node.patch_index)
    
    if complete_node.lod_level < max_depth and complete_node.children:
        for complete_child in complete_node.children:
            new_child = _copy_subtree_to_depth(complete_child, max_depth)
            new_node.children.append(new_child)
            
    return new_node


def _get_nodes_at_level(start_node, target_level):
    if start_node.lod_level == target_level:
        return [start_node]
    if start_node.lod_level > target_level or not start_node.children:
        return []
    descendant_nodes = []
    for child in start_node.children:
        descendant_nodes.extend(_get_nodes_at_level(child, target_level))
    return descendant_nodes

def _collect_all_nodes_recursive(node, max_depth, nodes_by_lod):
    """
    一个辅助函数，用于递归收集一个树中所有LOD的节点。
    
    参数:
    node: 当前开始的节点
    max_depth: 要收集的最大深度 (LOD level)
    nodes_by_lod: 一个列表的列表， e.g., [ [], [], ... ]
                  其中 nodes_by_lod[i] 是一个包含所有LOD i的节点的列表。
    """
    if node.lod_level > max_depth:
        return
    
    # 将当前节点添加到其LOD对应的列表中
    nodes_by_lod[node.lod_level].append(node)
    
    # 递归访问所有子节点
    for child in node.children:
        _collect_all_nodes_recursive(child, max_depth, nodes_by_lod)



def _build_probabilistic_recursive(node, max_depth, guaranteed_depth, expansion_probs, patches_per_side_list):
    """
    递归辅助函数，使用每层可调的概率来构建树。
    """
    current_depth = node.lod_level
    
    if current_depth >= max_depth:
        return # 达到最大深度，停止

    expand = False
    if current_depth < guaranteed_depth:
        expand = True
    else:
        # 检查 expansion_probs 列表中是否有为当前深度定义的概率
        prob_index = current_depth - guaranteed_depth
        if prob_index < len(expansion_probs):
            if random.random() < expansion_probs[prob_index]:
                expand = True
        # 如果没有定义概率 (例如，列表比树的深度短)，则默认不扩展

    if expand:
        new_nodes = _create_and_assign_children(node, patches_per_side_list)
        for child_node in new_nodes:
            _build_probabilistic_recursive(
                child_node, 
                max_depth, 
                guaranteed_depth, 
                expansion_probs, 
                patches_per_side_list
            )

def build_probabilistic_quadtree(patches_per_side_list, guaranteed_depth, expansion_probs):
    """
    构建一个随机四叉树，使用一个概率列表来控制非保证深度的扩展。
    
    参数:
    patches_per_side_list: [1, 2, 4, ...]
    guaranteed_depth: 保证扩展的深度 (e.g., 3)
    expansion_probs: 一个概率列表。
                     prob[0] = LOD (guaranteed_depth) 的扩展概率
                     prob[1] = LOD (guaranteed_depth + 1) 的扩展概率
                     ...
    """
    local_patches_per_side = sorted(patches_per_side_list)
    max_possible_depth = len(patches_per_side_list) - 1
    
    if guaranteed_depth > max_possible_depth:
        guaranteed_depth = max_possible_depth

    root_node = QuadTreeNode(lod_level=0, patch_index=0)

    _build_probabilistic_recursive(
        root_node, 
        max_possible_depth,
        guaranteed_depth,
        expansion_probs,
        local_patches_per_side
    )
        
    return root_node



if __name__ == "__main__":
    
    # --- 1. 定义模拟参数 ---
    # 为了更快看到结果，可以先用 1,000,000 (一百万) 跑一下
    N_RUNS = 10_000_0 
    PATCHES_PER_SIDE = [1, 2, 4, 8, 16, 32]
    GUARANTEED_DEPTH = 3
    TUNED_PROBS = [0.25, 0.2]
    
    MAX_LOD = len(PATCHES_PER_SIDE) - 1
    
    # --- 2. 初始化累加器 ---
    total_lod_counts = np.zeros(MAX_LOD + 1, dtype=np.int64)

    print(f"🚀 开始模拟...")
    print(f"总运行次数: {N_RUNS:,}") # 使用逗号分隔符，更易读
    print(f"保证深度 (Guaranteed Depth): {GUARANTEED_DEPTH}")
    print(f"LOD配置 (Patches per side): {PATCHES_PER_SIDE}")
    print("-" * 30)
    
    start_time = time.time()

    # --- 3. 运行模拟循环 ---
    for _ in tqdm(range(N_RUNS), desc="模拟进度"):
        root_node = build_probabilistic_quadtree(PATCHES_PER_SIDE, GUARANTEED_DEPTH, TUNED_PROBS)
        nodes_by_lod_this_run = [[] for _ in range(MAX_LOD + 1)]
        _collect_all_nodes_recursive(root_node, MAX_LOD, nodes_by_lod_this_run)
        
        for lod in range(MAX_LOD + 1):
            total_lod_counts[lod] += len(nodes_by_lod_this_run[lod])

    end_time = time.time()
    print(f"\n✅ 模拟完成! 总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 30)

    # --- 4. 计算并打印结果 (百分比) ---
    
    # 计算所有生成的节点的总数
    grand_total_nodes = total_lod_counts.sum()
    
    print(f"在 {N_RUNS:,} 次模拟中，总共生成了 {grand_total_nodes:,} 个节点。\n")
    print(f"每棵树大概有 {grand_total_nodes/N_RUNS:,} 个节点。\n")
    
    print("📊 节点频率分析 (百分比):")
    
    if grand_total_nodes > 0:
        # 计算每个LOD的百分比
        lod_percentages = (total_lod_counts / grand_total_nodes) * 100
        
        for lod, percentage in enumerate(lod_percentages):
            # 格式化输出，同时显示百分比和原始数量
            print(f"  LOD {lod}: {percentage:>7.4f} %  (共 {total_lod_counts[lod]:,} 个节点)")
            
        # 做一个简单的校验，确保总和约为100%
        print("-" * 30)
        print(f"校验: 百分比总和 = {lod_percentages.sum():.2f}%")
    else:
        print("没有生成任何节点。")