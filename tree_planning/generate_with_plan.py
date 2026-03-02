"""
用 pkl plan 的 lod_indices / patch_indices 做生成，不调用 mar 里建树逻辑。
仅支持 batch_size=1。供 inference_generator_from_plan_pkl 使用。
"""
import torch
from modeling.utils import QuadTreeNode
from modeling.generate_utils import sample


def get_ordered_nodes_from_indices(lod_indices, patch_indices):
    """从 (lod_indices, patch_indices) 得到 ordered_nodes（QuadTreeNode 列表）。"""
    if torch.is_tensor(lod_indices):
        lod_indices = lod_indices.cpu().tolist()
    if torch.is_tensor(patch_indices):
        patch_indices = patch_indices.cpu().tolist()
    return [
        QuadTreeNode(lod_level=int(lod_idx), patch_index=int(patch_idx))
        for lod_idx, patch_idx in zip(lod_indices, patch_indices)
    ]


@torch.no_grad()
def generate_with_plan(
    generator,
    condition,
    plan_dict,
    guidance_scale=3.0,
    guidance_decay="constant",
    guidance_scale_pow=3.0,
    randomize_temperature=4.5,
    softmax_temperature_annealing=False,
):
    """
    用 plan 里的 tree（lod_indices, patch_indices）做一步到位的 AR 生成。
    condition: (1,) long tensor, class_id
    plan_dict: dict with "lod_indices" and "patch_indices", each (1, seq_len) or (seq_len,) on same device as condition
    返回: result_tokens (1, seq_len), ordered_nodes (list of QuadTreeNode)
    """
    device = condition.device
    lod_indices = plan_dict["lod_indices"].to(device=device, dtype=torch.long)
    patch_indices = plan_dict["patch_indices"].to(device=device, dtype=torch.long)
    if lod_indices.dim() == 1:
        lod_indices = lod_indices.unsqueeze(0)
        patch_indices = patch_indices.unsqueeze(0)
    bsz = 1
    assert condition.shape[0] == 1 and lod_indices.shape[0] == 1, "Only batch_size=1 supported"
    max_seq_len = lod_indices.shape[1]
    tree_dict = dict(lod_indices=lod_indices, patch_indices=patch_indices)

    result_tokens = torch.zeros(bsz, max_seq_len, device=device, dtype=torch.long)
    token_indices_embedding = generator.get_token_indices_embedding(tree_dict)

    if guidance_scale > 1.0:
        cond_null = torch.ones_like(condition, device=device) * generator.num_classes
        cond_combined = torch.cat([condition, cond_null])
        bsz_cfg = 2
    else:
        cond_combined = condition
        bsz_cfg = bsz
    class_embedding = generator.cls_embedding(cond_combined, train=False)

    with torch.device(device):
        generator.setup_caches(max_batch_size=bsz_cfg, max_seq_length=max_seq_len, dtype=class_embedding.dtype)

    x = class_embedding.repeat(1, generator.cls_token_num, 1)
    cur_freqs_cis = generator.freqs_cis[: generator.cls_token_num].unsqueeze(0).repeat(bsz_cfg, 1, 1, 1)
    input_pos = torch.arange(0, x.shape[1], device=device)
    cache_position = generator.cls_token_num

    for step in range(max_seq_len):
        token_logits = generator.forward_inference(x, cur_freqs_cis, input_pos)

        if guidance_scale != 1.0:
            if guidance_decay == "linear":
                cfg_iter = 1 + (guidance_scale - 1) * step / max_seq_len
            elif guidance_decay == "constant":
                cfg_iter = guidance_scale
            elif guidance_decay == "power-cosine":
                scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / max_seq_len) ** scale_pow) * torch.pi)) * 0.5
                cfg_iter = (guidance_scale - 1) * scale_step + 1
            elif guidance_decay == "lod_scheduler":
                cfg_iter = generator.cfg_lod_scheduler(guidance_scale, step, lod_indices[:, step].float().mean())
            else:
                cfg_iter = guidance_scale
            cond_logits, uncond_logits = torch.chunk(token_logits, 2, dim=0)
            logits = uncond_logits + cfg_iter * (cond_logits - uncond_logits)
        else:
            logits = token_logits

        incides = sample(logits, randomize_temperature)[0]
        result_tokens[:, step] = incides.clone().squeeze(1)[:1]

        if step == max_seq_len - 1:
            break

        token_latent = generator.tok_embeddings(incides)
        token_latent += token_indices_embedding[:, step].unsqueeze(1)
        if guidance_scale > 1.0:
            token_latent = torch.cat([token_latent, token_latent], dim=0)
        x = token_latent

        cur_freqs_cis = generator.freqs_cis[cache_position : cache_position + 1].unsqueeze(0).repeat(bsz_cfg, 1, 1, 1)
        input_pos = torch.arange(cache_position, cache_position + 1, device=device)
        cache_position += 1

    generator.remove_caches()

    ordered_nodes = get_ordered_nodes_from_indices(lod_indices[0], patch_indices[0])
    return result_tokens, ordered_nodes
