"""
Test script to count nodes generated with different expansion_probs settings.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import random

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from modeling.utils import build_probabilistic_quadtree, get_ordered_nodes


def count_nodes_in_tree(tree_root, num_lod):
    """Count nodes in the tree by LOD level."""
    nodes_by_lod = defaultdict(int)
    
    queue = [tree_root]
    while queue:
        node = queue.pop(0)
        if node.lod_level < num_lod:
            nodes_by_lod[node.lod_level] += 1
        for child in node.children:
            queue.append(child)
    
    return dict(nodes_by_lod)


def tree_to_decision_nodes_dict(tree_root, num_lod):
    """Convert tree root to decision_nodes dictionary format."""
    decision_nodes = defaultdict(list)
    queue = [tree_root]
    
    while queue:
        node = queue.pop(0)
        if node.lod_level < num_lod:
            decision_nodes[node.lod_level].append(node)
        for child in node.children:
            queue.append(child)
    
    return dict(decision_nodes)


def test_expansion_probs(patches_per_side_list, guaranteed_depth, expansion_probs, num_runs=10):
    """
    Test different expansion_probs and count nodes.
    
    Args:
        patches_per_side_list: List of patch counts per side for each LOD
        guaranteed_depth: Guaranteed depth for the tree
        expansion_probs: List of expansion probabilities
        num_runs: Number of random runs to average over
    
    Returns:
        Statistics dictionary
    """
    num_lod = len(patches_per_side_list)
    
    all_node_counts = []
    all_nodes_by_lod = []
    
    for run_idx in range(num_runs):
        # Generate tree
        tree_root = build_probabilistic_quadtree(
            patches_per_side_list,
            guaranteed_depth=guaranteed_depth,
            expansion_probs=expansion_probs
        )
        
        # Count total nodes
        decision_nodes = tree_to_decision_nodes_dict(tree_root, num_lod)
        total_nodes = sum(len(nodes) for nodes in decision_nodes.values())
        all_node_counts.append(total_nodes)
        
        # Count nodes by LOD
        nodes_by_lod = count_nodes_in_tree(tree_root, num_lod)
        all_nodes_by_lod.append(nodes_by_lod)
    
    # Calculate statistics
    stats = {
        'expansion_probs': expansion_probs,
        'total_nodes': {
            'min': min(all_node_counts),
            'max': max(all_node_counts),
            'mean': sum(all_node_counts) / len(all_node_counts),
            'median': sorted(all_node_counts)[len(all_node_counts) // 2],
            'all': all_node_counts
        },
        'nodes_by_lod': defaultdict(dict)
    }
    
    # Calculate statistics for each LOD
    for lod in range(num_lod):
        lod_counts = [nodes_by_lod.get(lod, 0) for nodes_by_lod in all_nodes_by_lod]
        stats['nodes_by_lod'][lod] = {
            'min': min(lod_counts),
            'max': max(lod_counts),
            'mean': sum(lod_counts) / len(lod_counts),
            'median': sorted(lod_counts)[len(lod_counts) // 2],
            'all': lod_counts
        }
    
    return stats


def main():
    # Parameters
    patches_per_side_list = [1, 2, 4, 8, 16, 32]
    guaranteed_depth = 3
    num_lod = len(patches_per_side_list)
    num_runs = 2000  # Number of random runs to average over
    
    # Test different expansion_probs
    expansion_probs_to_test = [
        [0.3, 0.2],
        [0.3, 0.2],
        [0.3, 0.4],
    ]
    
    print("=" * 80)
    print(f"Testing expansion_probs with patches_per_side_list={patches_per_side_list}")
    print(f"guaranteed_depth={guaranteed_depth}, num_runs={num_runs}")
    print("=" * 80)
    print()
    
    all_results = []
    
    for expansion_probs in expansion_probs_to_test:
        print(f"\nTesting expansion_probs={expansion_probs}")
        print("-" * 80)
        
        stats = test_expansion_probs(
            patches_per_side_list,
            guaranteed_depth,
            expansion_probs,
            num_runs=num_runs
        )
        
        all_results.append(stats)
        
        # Print results
        print(f"Total nodes: min={stats['total_nodes']['min']}, max={stats['total_nodes']['max']}, "
              f"mean={stats['total_nodes']['mean']:.2f}, median={stats['total_nodes']['median']}")
        
        print("Nodes by LOD:")
        for lod in range(num_lod):
            lod_stats = stats['nodes_by_lod'][lod]
            print(f"  LOD {lod}: min={lod_stats['min']}, max={lod_stats['max']}, "
                  f"mean={lod_stats['mean']:.2f}, median={lod_stats['median']}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Expansion Probs':<20} {'Min Nodes':<12} {'Max Nodes':<12} {'Mean Nodes':<12} {'Median Nodes':<12}")
    print("-" * 80)
    
    for stats in all_results:
        exp_probs_str = str(stats['expansion_probs'])
        print(f"{exp_probs_str:<20} {stats['total_nodes']['min']:<12} {stats['total_nodes']['max']:<12} "
              f"{stats['total_nodes']['mean']:<12.2f} {stats['total_nodes']['median']:<12}")
    
    # Save results to file
    output_path = Path(parent_dir) / "tree_expansion_stats.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Tree Expansion Statistics\n")
        f.write(f"patches_per_side_list={patches_per_side_list}\n")
        f.write(f"guaranteed_depth={guaranteed_depth}, num_runs={num_runs}\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_results:
            f.write(f"\nexpansion_probs={stats['expansion_probs']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total nodes: min={stats['total_nodes']['min']}, max={stats['total_nodes']['max']}, "
                   f"mean={stats['total_nodes']['mean']:.2f}, median={stats['total_nodes']['median']}\n")
            f.write("Nodes by LOD:\n")
            for lod in range(num_lod):
                lod_stats = stats['nodes_by_lod'][lod]
                f.write(f"  LOD {lod}: min={lod_stats['min']}, max={lod_stats['max']}, "
                       f"mean={lod_stats['mean']:.2f}, median={lod_stats['median']}\n")
    
    print(f"\nDetailed statistics saved to: {output_path}")


if __name__ == "__main__":
    main()

