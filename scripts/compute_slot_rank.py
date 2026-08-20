"""Compute SLOT_RANK = rank of each 1344-slot in the generator's canonical order
(_get_ordered_nodes / tree_to_decision_nodes_dict = BFS-by-lod). Proves it is a permutation
of 0..1343 and DIFFERS from the spatial slot order (the current pretok bug)."""
import sys, numpy as np
REPO = "/sensei-fs-3/users/yuchengm/code/quadtok/3level"
sys.path.insert(0, REPO)
from modeling.utils import build_quadtree, tree_to_decision_nodes_dict

NPS = [1, 2, 4, 8, 16, 32]; NUM_LOD = 6
OFF = {3: 0, 4: 64, 5: 320}          # slot offset per lod (SLOT_LOD/SLOT_PATCH convention)

root = build_quadtree(NPS)            # FULL tree (all nodes to lod5)
final = tree_to_decision_nodes_dict(root, NUM_LOD)   # BFS-by-lod, == _get_ordered_nodes order
order = []
for lod in range(NUM_LOD):
    for nd in final.get(lod, []):
        if nd.lod_level >= 3:
            order.append((nd.lod_level, nd.patch_index))
print("total tokens (lod>=3):", len(order), "(expect 1344 = 64+256+1024)")

SLOT_RANK = np.full(1344, -1, dtype=np.int64)
for rank, (lod, patch) in enumerate(order):
    SLOT_RANK[OFF[lod] + patch] = rank
print("is permutation of 0..1343:", sorted(SLOT_RANK.tolist()) == list(range(1344)))
print("slots where BFS-rank == slot-index:", int((SLOT_RANK == np.arange(1344)).sum()),
      "/1344  (<1344 => order DIFFERS from current spatial pretok => bug real)")
print("lod3 first 16 slots -> BFS rank:", SLOT_RANK[:16].tolist())
print("lod4 first 8 slots (64..71) -> BFS rank:", SLOT_RANK[64:72].tolist())
np.save("/sensei-fs-3/users/yuchengm/code/quadtok/3level/scripts/slot_rank_3level.npy", SLOT_RANK)
print("SAVED scripts/slot_rank_3level.npy")
