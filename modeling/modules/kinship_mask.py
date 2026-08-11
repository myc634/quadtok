"""Generalized quadtree Kinship attention mask (arbitrary depth) for flex_attention.

Rule (generalization of the original 2-level get_causal_mask, paper-consistent):
  For query node q (lod Lq) and key node k (lod Lk):
    - Lk <  Lq  -> attend      (a finer node sees ALL coarser-level nodes)
    - Lk == Lq  -> attend iff Lq is the coarsest token level (full within it),
                   else iff same immediate parent (siblings)
    - Lk >  Lq  -> do NOT attend

  e.g. 3-level: lod5 attends all lod3 + all lod4 + lod5 same-parent siblings;
                lod4 attends all lod3 + lod4 siblings; lod3 attends all lod3.

Verified (tests/): reproduces the original 2-level dense mask exactly over random
trees, and flex_attention with this mask matches nn.MultiheadAttention(dense mask)
to <2e-7 while reusing the same MHA weights (warm-start safe).
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask


def node_parent(patch_index, lod, num_patch_side_list):
    """Parent patch index (at lod-1) of a node, using the 2x2 quadtree layout."""
    s = num_patch_side_list[lod]
    row, col = patch_index // s, patch_index % s
    s_parent = num_patch_side_list[lod - 1]
    return (row // 2) * s_parent + (col // 2)


def dense_kinship_can_attend(lods, patches, num_patch_side_list):
    """[L, L] bool, True == q CAN attend k. lods/patches are python lists. (reference/testing)"""
    L = len(lods)
    lods_t = torch.tensor(lods)
    min_lod = int(lods_t.min().item())
    parent = torch.tensor([
        node_parent(patches[i], lods[i], num_patch_side_list) if lods[i] > min_lod else -1
        for i in range(L)
    ])
    lq = lods_t.view(-1, 1); lk = lods_t.view(1, -1)
    pq = parent.view(-1, 1); pk = parent.view(1, -1)
    finer = lk < lq
    same_level = (lk == lq) & ((lq == min_lod) | (pq == pk))
    return finer | same_level


def make_kinship_mask_mod(lods_tensor, parent_tensor, min_lod):
    """flex_attention mask_mod closure. lods_tensor/parent_tensor: (L,) long on device."""
    def mask_mod(b, h, q_idx, kv_idx):
        lq = lods_tensor[q_idx]; lk = lods_tensor[kv_idx]
        finer = lk < lq
        same_level = (lk == lq) & ((lq == min_lod) | (parent_tensor[q_idx] == parent_tensor[kv_idx]))
        return finer | same_level
    return mask_mod


def parent_indices_vec(lod_t, patch_t, num_patch_side_list, min_lod):
    """Vectorised parent patch-index per node (at lod-1); -1 for the coarsest token level."""
    npsl = torch.tensor(num_patch_side_list, device=lod_t.device, dtype=torch.long)
    s = npsl[lod_t]
    s_parent = npsl[(lod_t - 1).clamp(min=0)]
    row = patch_t // s
    col = patch_t % s
    parent = (row // 2) * s_parent + (col // 2)
    return torch.where(lod_t > min_lod, parent, torch.full_like(parent, -1))


def build_kinship_block_mask(lod_t, patch_t, num_patch_side_list, device):
    """flex_attention BlockMask for one (batch-shared) quadtree token sequence (DECODER).

    lod_t / patch_t: (L,) long tensors, ordered by lod (lod-ascending BFS).
    Broadcast across all heads/batch (B=H=None).
    """
    L = int(lod_t.shape[0])
    min_lod = int(lod_t.min())
    parent_t = parent_indices_vec(lod_t, patch_t, num_patch_side_list, min_lod)
    mask_mod = make_kinship_mask_mod(lod_t, parent_t, min_lod)
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)


def make_selector_mask_mod(lods_full, parent_full, min_lod, num_latent):
    """flex mask_mod for the SELECTOR: [num_latent latent prefix] + [tree tokens].
    latent<->latent full; tree->latent full; tree->tree kinship; latent->tree none.
    lods_full/parent_full: (total_len,) long on device; latent positions use lod=-1.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        q_lat = q_idx < num_latent
        k_lat = kv_idx < num_latent
        lq = lods_full[q_idx]; lk = lods_full[kv_idx]
        finer = lk < lq
        same_level = (lk == lq) & ((lq == min_lod) | (parent_full[q_idx] == parent_full[kv_idx]))
        tree_tree = finer | same_level
        tree_q = torch.where(k_lat, torch.ones_like(tree_tree), tree_tree)  # tree sees all latent
        return torch.where(q_lat, k_lat, tree_q)                            # latent sees only latent
    return mask_mod


def build_selector_block_mask(num_latent, lod_t, patch_t, num_patch_side_list, device):
    """flex_attention BlockMask for the selector sequence [latent(num_latent) + tree tokens].
    lod_t / patch_t: (L,) long tensors for the tree tokens."""
    Lt = int(lod_t.shape[0])
    total = num_latent + Lt
    min_lod = int(lod_t.min())
    parent_tree = parent_indices_vec(lod_t, patch_t, num_patch_side_list, min_lod)
    lods_full = torch.full((total,), -1, device=device, dtype=torch.long)
    parent_full = torch.full((total,), -1, device=device, dtype=torch.long)
    lods_full[num_latent:] = lod_t
    parent_full[num_latent:] = parent_tree
    mask_mod = make_selector_mask_mod(lods_full, parent_full, min_lod, num_latent)
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device)


def dense_selector_can_attend(num_latent, lods, patches, num_patch_side_list):
    """[total,total] bool reference for the selector mask (testing)."""
    tree = dense_kinship_can_attend(lods, patches, num_patch_side_list)
    Lt = len(lods); total = num_latent + Lt
    m = torch.zeros(total, total, dtype=torch.bool)
    m[:num_latent, :num_latent] = True          # latent<->latent
    m[num_latent:, :num_latent] = True          # tree->latent
    m[num_latent:, num_latent:] = tree          # tree->tree kinship
    return m
