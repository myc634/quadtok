import random
import numpy as np
class QuadTreeNode:
    def __init__(self, lod_level, patch_index):
        self.lod_level = lod_level
        self.patch_index = patch_index
        self.children = []
        self.node_feature = None

    def to_dict(self):
        return {
            "lod_level": self.lod_level,
            "patch_index": self.patch_index,
            "children": [child.to_dict() for child in self.children]
        }


def _build_recursive(node, current_lod_idx, patches_per_side_list):
    if current_lod_idx >= len(patches_per_side_list) - 1:
        return

    parent_lod_idx = current_lod_idx
    child_lod_idx = current_lod_idx + 1

    parent_patches_per_side = patches_per_side_list[parent_lod_idx]
    child_patches_per_side = patches_per_side_list[child_lod_idx]

    parent_row = node.patch_index // parent_patches_per_side
    parent_col = node.patch_index % parent_patches_per_side

    child_start_row = parent_row * 2
    child_start_col = parent_col * 2

    top_left_idx = child_start_row * child_patches_per_side + child_start_col
    top_right_idx = top_left_idx + 1
    bottom_left_idx = (child_start_row + 1) * child_patches_per_side + child_start_col
    bottom_right_idx = bottom_left_idx + 1
    
    child_indices = [top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx]

    for child_index in child_indices:
        child_node = QuadTreeNode(lod_level=child_lod_idx, patch_index=child_index)
        node.children.append(child_node)
        _build_recursive(child_node, child_lod_idx, patches_per_side_list)


def build_quadtree(patches_per_side_list):
    patches_per_side_list.sort(reverse=False)

    root_node = QuadTreeNode(lod_level=0, patch_index=0)
    _build_recursive(root_node, 0, patches_per_side_list)
        
    return root_node

def _build_final_random_recursive(node, current_depth, guaranteed_depth, max_possible_depth, patches_per_side_list):
    if current_depth >= max_possible_depth:
        return

    expand = False
    if current_depth < guaranteed_depth:
        expand = True
    else:
        prob_to_expand = (max_possible_depth - current_depth) / (max_possible_depth - current_depth + 1.0)
        
        if random.random() < prob_to_expand:
            expand = True
    if expand:
        parent_lod_idx = current_depth
        child_lod_idx = current_depth + 1

        parent_patches_per_side = patches_per_side_list[parent_lod_idx]
        child_patches_per_side = patches_per_side_list[child_lod_idx]

        parent_row, parent_col = divmod(node.patch_index, parent_patches_per_side)
        child_start_row, child_start_col = parent_row * 2, parent_col * 2

        top_left_idx = child_start_row * child_patches_per_side + child_start_col
        child_indices = [top_left_idx, top_left_idx + 1, top_left_idx + child_patches_per_side, top_left_idx + child_patches_per_side + 1]

        for child_index in child_indices:
            child_node = QuadTreeNode(lod_level=child_lod_idx, patch_index=child_index)
            node.children.append(child_node)

            _build_final_random_recursive(child_node, child_lod_idx, guaranteed_depth, max_possible_depth, patches_per_side_list)

def build_random_quadtree(patches_per_side_list, guaranteed_depth=0):
    local_patches_per_side = sorted(patches_per_side_list)
    max_possible_depth = len(patches_per_side_list) - 1
    if guaranteed_depth > max_possible_depth:
        guaranteed_depth = max_possible_depth

    root_node = QuadTreeNode(lod_level=0, patch_index=0)

    _build_final_random_recursive(
        root_node, 
        0, # current_depth
        guaranteed_depth,
        max_possible_depth,
        local_patches_per_side
    )
        
    return root_node


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
    if node.lod_level > max_depth:
        return
    nodes_by_lod[node.lod_level].append(node)
    for child in node.children:
        _collect_all_nodes_recursive(child, max_depth, nodes_by_lod)

