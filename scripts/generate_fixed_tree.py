"""
Generate a fixed quadtree structure and save it in the format used by pretokenize_quadtree.py
"""
import os
import sys
from pathlib import Path
import json
import pickle
from collections import defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from modeling.utils import build_quadtree, get_ordered_nodes, build_probabilistic_quadtree


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


def generate_status_data(tree_root, num_lod, patches_per_side_list):
    """
    Generate status_data in the same format as pretokenize_quadtree.py
    Returns: list of status values for each node in ordered_full_nodes
    """
    # Build full tree to get ordered_full_nodes
    full_tree_root = build_quadtree(patches_per_side_list)
    ordered_full_nodes = get_ordered_nodes(full_tree_root, num_lod)
    num_total_nodes = len(ordered_full_nodes)
    
    # Build node_to_idx_map
    node_to_idx_map = {
        (node.lod_level, node.patch_index): i 
        for i, node in enumerate(ordered_full_nodes)
    }
    
    # Build child_to_parent_map
    child_to_parent_map = {}
    for parent_node in ordered_full_nodes:
        for child_node in parent_node.children:
            child_key = (child_node.lod_level, child_node.patch_index)
            if child_key in node_to_idx_map:
                child_to_parent_map[child_key] = parent_node
    
    # Extract present nodes from tree_root
    decision_nodes = tree_to_decision_nodes_dict(tree_root, num_lod)
    present_nodes_set = set()
    for lod, nodes in decision_nodes.items():
        for node in nodes:
            present_nodes_set.add((node.lod_level, node.patch_index))
    
    # Initialize status list
    status_list = [-1] * num_total_nodes
    
    # Mark present nodes with 1
    for i, node in enumerate(ordered_full_nodes):
        if (node.lod_level, node.patch_index) in present_nodes_set:
            status_list[i] = 1
    
    return status_list


def process_tree_for_saving(tree_dict):
    """
    Convert tree dictionary to saveable format (same as pretokenize_quadtree.py)
    """
    processed_tree = defaultdict(list)
    for lod_idx, nodes in sorted(tree_dict.items()):
        for node in nodes:
            processed_tree[lod_idx].append({
                'patch_index': node.patch_index,
                'lod_level': node.lod_level
            })
    return dict(processed_tree)


def main():
    # Parameters
    patches_per_side_list = [1, 2, 4, 8, 16, 32]
    guaranteed_depth = 3
    expansion_probs = [0.3, 0.4]
    num_lod = len(patches_per_side_list)
    
    # Generate tree
    print(f"Generating quadtree with patches_per_side_list={patches_per_side_list}")
    print(f"guaranteed_depth={guaranteed_depth}, expansion_probs={expansion_probs}")
    
    tree_root = build_probabilistic_quadtree(
        patches_per_side_list, 
        guaranteed_depth=guaranteed_depth, 
        expansion_probs=expansion_probs
    )
    
    # Convert to decision_nodes format (final_tree)
    final_tree = tree_to_decision_nodes_dict(tree_root, num_lod)
    
    # Generate status_data
    status_data = generate_status_data(tree_root, num_lod, patches_per_side_list)
    
    # Process tree for saving
    processed_final_tree = process_tree_for_saving(final_tree)
    
    # Create output dictionary
    output_dict = {
        'status_data': status_data,
        'final_tree': processed_final_tree
    }
    
    # Save to file
    output_path = Path(parent_dir) / "fixed_quadtree_low.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)
    
    print(f"\nFixed quadtree saved to: {output_path}")
    print(f"Status data length: {len(status_data)}")
    print(f"Status distribution: -1: {status_data.count(-1)}, 0: {status_data.count(0)}, 1: {status_data.count(1)}")
    print(f"Final tree has {len(processed_final_tree)} LOD levels")
    for lod_idx in sorted(processed_final_tree.keys()):
        print(f"  LOD {lod_idx}: {len(processed_final_tree[lod_idx])} nodes")
    
    # Also save as JSON for easy inspection
    json_output_path = Path(parent_dir) / "fixed_quadtree_low.json"
    # Convert to JSON-serializable format
    json_dict = {
        'status_data': status_data,
        'final_tree': processed_final_tree
    }
    with open(json_output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"Also saved as JSON to: {json_output_path}")


if __name__ == "__main__":
    main()

