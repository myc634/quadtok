import random
import numpy as np
from collections import deque, defaultdict
# from collections import defaultdict


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


def _create_and_assign_children(node, patches_per_side_list):

    current_depth = node.lod_level
    child_lod_idx = current_depth + 1
    
    if child_lod_idx >= len(patches_per_side_list):
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
    if current_depth == guaranteed_depth:
        available_leaves_list.append(node)
        return 0 

    if current_depth + 1 >= len(patches_per_side_list):
        available_leaves_list.append(node) 
        return 0

    _create_and_assign_children(node, patches_per_side_list)
    internal_nodes_count = 1 
    
    for child in node.children:
        internal_nodes_count += _build_guaranteed_recursive(
            child, current_depth + 1, guaranteed_depth, patches_per_side_list, available_leaves_list
        )
    return internal_nodes_count

def build_uniformly_random_quadtree_by_size(patches_per_side_list, num_internal_nodes, guaranteed_depth=0):
    
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
        num_internal_nodes = num_guaranteed_nodes

    num_nodes_to_add = num_internal_nodes - num_guaranteed_nodes
    nodes_added = 0
    
    while nodes_added < num_nodes_to_add and available_leaves:
        
        node_to_split_idx = random.randrange(len(available_leaves))
        node_to_split = available_leaves[node_to_split_idx]

        if node_to_split.lod_level >= max_possible_depth:
            available_leaves.pop(node_to_split_idx)
            continue 

        available_leaves[node_to_split_idx] = available_leaves[-1]
        available_leaves.pop()

        new_leaves = _create_and_assign_children(node_to_split, local_patches_per_side)
        
        available_leaves.extend(new_leaves)
        nodes_added += 1
        
    return root_node


def build_random_quadtree(patches_per_side_list, guaranteed_depth=0, min_nodes=1, max_nodes=50):

    min_guaranteed_nodes = 0
    if guaranteed_depth > 0:
        min_guaranteed_nodes = (4**guaranteed_depth - 1) // 3

    effective_min_nodes = max(min_nodes, min_guaranteed_nodes)
    effective_max_nodes = max(effective_min_nodes, max_nodes)

    chosen_num_internal_nodes = random.randint(effective_min_nodes, effective_max_nodes)
    return build_uniformly_random_quadtree_by_size(
        patches_per_side_list,
        chosen_num_internal_nodes,
        guaranteed_depth
    )


def _build_probabilistic_recursive(node, max_depth, guaranteed_depth, expansion_probs, patches_per_side_list):
    current_depth = node.lod_level
    
    if current_depth >= max_depth:
        return 

    expand = False
    if current_depth < guaranteed_depth:
        expand = True
    else:
        prob_index = current_depth - guaranteed_depth
        if prob_index < len(expansion_probs):
            if random.random() < expansion_probs[prob_index]:
                expand = True
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



def get_ordered_nodes(root_node, num_lod):
    if not root_node:
        return []
    nodes_by_lod = {i: [] for i in range(num_lod)}
    queue = [root_node]
    
    while queue:
        node = queue.pop(0)
        if node.lod_level < num_lod:
            nodes_by_lod[node.lod_level].append(node)
        for child in node.children:
            queue.append(child)
    
    ordered_nodes = []
    for i in range(num_lod):
        ordered_nodes.extend(nodes_by_lod[i])
        
    return ordered_nodes

def _get_parent_patch_index(child_lod_idx, child_patch_idx, patches_per_side_list):
    """
    Calculates the parent's LOD and patch index from a child's.
    """
    if child_lod_idx == 0:
        return None # Root has no parent

    parent_lod_idx = child_lod_idx - 1
    
    if parent_lod_idx < 0 or child_lod_idx >= len(patches_per_side_list):
        # This case should not happen with valid inputs
        return None 

    parent_patches_per_side = patches_per_side_list[parent_lod_idx]
    child_patches_per_side = patches_per_side_list[child_lod_idx]

    # Find child's grid position
    child_row = child_patch_idx // child_patches_per_side
    child_col = child_patch_idx % child_patches_per_side
    
    # Find parent's grid position
    parent_row = child_row // 2
    parent_col = child_col // 2
    
    # Convert parent grid position back to patch index
    parent_patch_idx = parent_row * parent_patches_per_side + parent_col
    
    return (parent_lod_idx, parent_patch_idx)

def build_tree_from_decision_nodes(current_decision_nodes, patches_per_side_list):
    """
    Reconstructs a QuadTreeNode hierarchical tree from the flat
    current_decision_nodes dictionary.
    """
    
    # Ensure patches_per_side_list is sorted, just like in build_quadtree
    patches_per_side_list.sort(reverse=False)
    
    node_map = {} # Maps (lod, patch_idx) -> QuadTreeNode object

    # 1. Create all QuadTreeNode objects for every active node
    for lod_level, patch_indices in current_decision_nodes.items():
        for patch_index in patch_indices:
            key = (lod_level, patch_index)
            # Create node if it doesn't exist (shouldn't be duplicates, but safe)
            if key not in node_map:
                node_map[key] = QuadTreeNode(lod_level, patch_index)

    root_node = None
    
    # 2. Link children to their parents
    for (lod_level, patch_index), node in node_map.items():
        if lod_level == 0:
            # This is the root node
            root_node = node
            continue
        
        # Find the key for this node's parent
        parent_key = _get_parent_patch_index(lod_level, patch_index, patches_per_side_list)
        
        if parent_key and parent_key in node_map:
            # If the parent exists in our map, add this node as its child
            parent_node = node_map[parent_key]
            parent_node.children.append(node)
        elif parent_key:
            # This case means a child exists but its parent was not in the
            # current_decision_nodes. This shouldn't happen if the optimization
            # logic is correct, but it means this node is an "orphan".
            pass 
            
    # The root node now contains the full hierarchy
    return root_node


def tree_to_decision_nodes_dict(tree_root, num_lod):
    """
    Convert tree root to decision_nodes dictionary format.
    Returns: dict {lod_idx: [list of QuadTreeNode]}
    """
    decision_nodes = defaultdict(list)
    queue = [tree_root]
    
    while queue:
        node = queue.pop(0)
        if node.lod_level < num_lod:
            decision_nodes[node.lod_level].append(node)
        for child in node.children:
            queue.append(child)
    
    return dict(decision_nodes)