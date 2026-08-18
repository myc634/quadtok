import torch
import torch.nn as nn
from typing import Optional, List
from torch.nn import functional as F
try:
    from flash_attn import flash_attn_varlen_func
except Exception:
    flash_attn_varlen_func = None

def batch_apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (bs, seq_len, head_dim // 2, 2)
    bs, seq_len, n_head, head_dim = x.shape
    xshaped = x.float().reshape(
        *x.shape[:-1], head_dim // 2, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)

    freqs_cis = freqs_cis.view(
        bs, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for rotary positional embedding.
    
    Args:
        dim (int): The dimension of the head.
        end (int): The maximum sequence length.
        theta (float, optional): The base for the geometric progression. Defaults to 10000.0.

    Returns:
        torch.Tensor: A tensor of shape (end, dim // 2, 2) containing the precomputed
                      cosine and sine values.
    """
    #
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()
    
    # freqs_cis is of shape (seq_len, head_dim // 2, 2)
    # it contains the real and imaginary parts of the complex numbers
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # We need to reshape it to (seq_len, dim // 2, 2) to match batch_apply_rotary_emb
    freqs_cis = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    
    return freqs_cis

def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(
        self, dim: int, ffn_dim_multiplier: int, multiple_of: int, ffn_dropout_p: float
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: int,
        attn_dropout_p: float,
        resid_dropout_p: float,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_kv_dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        # qknorm
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        """
        during inference:
        Args:
            x: [bsz, seqlen, dim], input tensor.
            freqs_cis: [bsz, seqlen, head_dim // 2, 2], used to apply rotary emb.
            input_pos: [seqlen], used to update KV cache.
            mask: [bsz, seqlen, seqlen], used to mask out attention weights.
        """
        if cu_seqlens is not None:
            # ---- VARLEN packed path (flash-attn); x: [1, T, dim] or [T, dim] ----
            assert flash_attn_varlen_func is not None, "flash_attn not installed"
            xin = x[0] if x.dim() == 3 else x                    # [T, dim]
            T = xin.shape[0]
            kv_size = self.n_kv_head * self.head_dim
            xq, xk, xv = self.wqkv(xin).split([self.dim, kv_size, kv_size], dim=-1)
            xq = xq.view(1, T, self.n_head, self.head_dim)
            xk = xk.view(1, T, self.n_kv_head, self.head_dim)
            xv = xv.view(1, T, self.n_kv_head, self.head_dim)
            xq = self.q_norm(xq); xk = self.k_norm(xk)
            if freqs_cis is not None:
                xq = batch_apply_rotary_emb(xq, freqs_cis)
                xk = batch_apply_rotary_emb(xk, freqs_cis)
            xq = xq[0].to(xv.dtype); xk = xk[0].to(xv.dtype); xv = xv[0]   # [T, H, hd]
            out = flash_attn_varlen_func(
                xq, xk, xv, cu_seqlens.to(torch.int32), cu_seqlens.to(torch.int32),
                int(max_seqlen), int(max_seqlen),
                dropout_p=self.attn_dropout_p if self.training else 0.0, causal=True)
            out = out.reshape(T, self.dim)
            out = self.resid_dropout(self.wo(out))
            return out.unsqueeze(0) if x.dim() == 3 else out

        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # this part is modified from LLaMAGen
        if freqs_cis is not None:
            xq = batch_apply_rotary_emb(xq, freqs_cis)
            xk = batch_apply_rotary_emb(xk, freqs_cis)
        else:
            xq = xq
            xk = xk

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        # this part is modified from LLaMAGen
        if self.kv_cache is not None:
            # [b, n_head, max_seq_len, head_dim]
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            
            # assuming that all the samples in a batch have the same input_pos
            max_pos = torch.max(input_pos) + 1
            keys = keys[:, :, :max_pos]
            values = values[:, :, :max_pos]
            if mask is not None:
                mask = mask[:, :, :, :max_pos]
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=(
                True if mask is None else False
            ),  # is_causal=False is for KV cache or explicit mask
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


""" Cloned from LLaMAGen: only the attention uses our customized version
"""
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim=4096,
        n_head=32,
        n_kv_head=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        attn_dropout_p=0.0,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        drop_path=0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim, n_head, n_kv_head, attn_dropout_p, resid_dropout_p
        )
        self.feed_forward = FeedForward(
            dim, ffn_dim_multiplier, multiple_of, ffn_dropout_p
        )
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = None,
        mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask, cu_seqlens, max_seqlen)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out
