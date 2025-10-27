"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
from typing import Optional
from einops.layers.torch import Rearrange
from modeling.modules.attention import RopeTransformerBlock, precompute_freqs_cis
from torch.utils.checkpoint import checkpoint
from modeling.utils import _get_nodes_at_level, QuadTreeNode, build_quadtree
import time
from collections import defaultdict



def scatter_patches(
    canvas: torch.Tensor,
    patches: torch.Tensor,
    batch_coords: torch.Tensor,
    y_starts: torch.Tensor,
    x_starts: torch.Tensor,
    patch_size: int
):
    N, C, _, _ = patches.shape
    device = patches.device
    patch_y_offsets = torch.arange(patch_size, device=device).view(patch_size, 1)
    patch_x_offsets = torch.arange(patch_size, device=device).view(1, patch_size)

    y_dest = y_starts.view(N, 1, 1) + patch_y_offsets
    x_dest = x_starts.view(N, 1, 1) + patch_x_offsets

    canvas[batch_coords[:, None, None], :, y_dest, x_dest] = patches.permute(0, 2, 3, 1).contiguous()
    
    return canvas


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ):
        return self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attention_mask, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None
    ):
        attn_output = self.attention(x=self.ln_1(x), attention_mask=attention_mask, key_padding_mask=key_padding_mask)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')



def drop_path(x, drop_prob: float = 0., training: bool = False):
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
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens, needs_width_reduction=True):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.strict_length_assertion = config.model.vq_model.get("strict_length_assertion", True)
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        if self.strict_length_assertion:
            assert H == 1 and W == self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        else:
            # relaxed assertion condition
            assert H == 1 and W <= self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x

class QuadTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.out_proj = nn.Linear(self.width, self.width)

    def forward(self, pixel_values):

        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 1: ]
        x = self.ln_post(x)
        x = self.out_proj(x)

        return x

class QuadTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.strict_length_assertion = config.model.vq_model.get("strict_length_assertion", True)
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.token_size = config.model.selector.token_size
        self.train_policy = config.model.get("train_policy", False)

        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.patch_size_list = config.model.selector.patch_size_list
        self.num_lod = len(config.model.selector.num_patch_side_list)
        # self.decoder_token_size = config.model.vq_model.decoder_token_size

        self.full_tree_root = build_quadtree(self.num_patch_side_list)
        ordered_full_nodes = self._get_ordered_nodes(self.full_tree_root)
        self.ordered_full_nodes = ordered_full_nodes

        lod_len_mapping = defaultdict(int)
        for node in self.ordered_full_nodes:
            lod_len_mapping[node.lod_level] += 1
        total_nodes_count = 0
        self.lod_start_indices = {}
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            num_nodes_at_lod = lod_len_mapping[lod_idx] 
            self.lod_start_indices[lod_idx] = total_nodes_count
            total_nodes_count += total_patches
 

        self.width = {
                "small": 512,
                "base": 768,
                "large": 512,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5

        self.token_incides_embedding_dict = nn.ModuleDict()
        self.max_seq_len = 0
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            self.max_seq_len += total_patches
            self.token_incides_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.width)

        scale = self.width ** -0.5
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.max_seq_len, self.width))

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        # self.attn_out = nn.Linear(self.width, self.decoder_token_size)

        self.decoder_channels = [self.width // (2**i) for i in range(self.num_lod)]

        self.latent_unpatchers = nn.ModuleDict()
        for i in range(self.num_lod):
            patch_size = self.patch_size_list[i]
            out_channels = self.decoder_channels[i]
            self.latent_unpatchers[str(i)] = nn.Sequential(
                nn.Linear(self.width, out_channels * patch_size * patch_size),
                Rearrange('b (c p1 p2) -> b c p1 p2', p1=patch_size, p2=patch_size)
            )

        self.upsamplers = nn.ModuleList()
        for i in range(self.num_lod - 1):
            if i < 4:
                in_channels = self.decoder_channels[i]
                out_channels = self.decoder_channels[i+1]
                self.upsamplers.append(
                    nn.Sequential(#nn.RMSNorm(in_channels),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    nn.GELU(),
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
                )
            else:
                in_channels = self.decoder_channels[i]
                out_channels = self.decoder_channels[i+1]
                self.upsamplers.append(
                    nn.Sequential(#nn.RMSNorm(in_channels),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=32, num_channels=in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                )

        self.conv_out = nn.Conv2d(self.decoder_channels[-1], 3, 3, padding=1, bias=True)

    def _get_ordered_nodes(self, root_node):
        if not root_node:
            return []
        nodes_by_lod = {i: [] for i in range(self.num_lod)}
        queue = [root_node]
        
        while queue:
            node = queue.pop(0)
            if node.lod_level < self.num_lod:
                nodes_by_lod[node.lod_level].append(node)
            for child in node.children:
                queue.append(child)
        
        ordered_nodes = []
        for i in range(self.num_lod):
            ordered_nodes.extend(nodes_by_lod[i])
            
        return ordered_nodes

    def hierarchical_latent_decode(self, tree_structure, batch_size):
        device = self.latent_token_positional_embedding.device
        dtype = self.latent_token_positional_embedding.dtype
        ordered_nodes = self._get_ordered_nodes(tree_structure)
        nodes_by_lod = {i: [] for i in range(self.num_lod)}
        for node in ordered_nodes:
            nodes_by_lod[node.lod_level].append(node)

        previous_feature_map = None
        for lod_idx in range(self.num_lod):
            channels = self.decoder_channels[lod_idx]
            patch_size = self.patch_size_list[lod_idx]
            if lod_idx == 0:
                upsampled_map = torch.zeros(batch_size, channels, patch_size, patch_size, device=device, dtype=dtype)
            else:
                upsampler = self.upsamplers[lod_idx - 1]
                upsampled_map = upsampler(previous_feature_map)

            current_lod_patch_canvas = torch.zeros(batch_size, channels, upsampled_map.shape[-2], upsampled_map.shape[-1], device=device, dtype=dtype)
            nodes_in_lod = nodes_by_lod[lod_idx]
            if nodes_in_lod:
                unpatch_fn = self.latent_unpatchers[str(lod_idx)]
                for node in nodes_in_lod:
                    latent_patch = unpatch_fn(node.node_feature)
                    
                    num_patches_per_side = self.num_patch_side_list[lod_idx]
                    row, col = divmod(node.patch_index, num_patches_per_side)
                    
                    y_start, x_start = row * patch_size, col * patch_size
                    y_end, x_end = y_start + patch_size, x_start + patch_size

                    current_lod_patch_canvas[:, :, y_start:y_end, x_start:x_end] = latent_patch

            final_lod_feature_map = upsampled_map + current_lod_patch_canvas
            previous_feature_map = final_lod_feature_map
            
        return previous_feature_map

    def hierarchical_decode_vectorized(self, features, lods, indices):

        batch_size = features.shape[0]
        device = features.device
        dtype = features.dtype
        previous_feature_map = None

        for lod_idx in range(self.num_lod):
            channels = self.decoder_channels[lod_idx]
            patch_size = self.patch_size_list[lod_idx]

            if lod_idx == 0:
                side_len = self.num_patch_side_list[0] * self.patch_size_list[0]
                upsampled_map = torch.zeros(batch_size, channels, side_len, side_len, device=device, dtype=dtype)
            else:
                upsampled_map = self.upsamplers[lod_idx - 1](previous_feature_map)
            current_lod_patch_canvas = torch.zeros_like(upsampled_map)

            lod_mask = (lods == lod_idx) & (lods != -1)
            if lod_mask.any():
                batch_coords = torch.nonzero(lod_mask, as_tuple=True)[0]
                features_lod = features[lod_mask]
                patch_idx_lod = indices[lod_mask]
                

                unpatched_features = self.latent_unpatchers[str(lod_idx)](features_lod)

                num_patches_per_side = self.num_patch_side_list[lod_idx]

                rows = patch_idx_lod // num_patches_per_side
                cols = patch_idx_lod % num_patches_per_side

                y_starts, x_starts = rows * patch_size, cols * patch_size
                
                current_lod_patch_canvas = scatter_patches(
                    current_lod_patch_canvas,
                    unpatched_features.to(current_lod_patch_canvas.dtype),
                    batch_coords,
                    y_starts,
                    x_starts,
                    patch_size
                )

            final_lod_feature_map = upsampled_map + current_lod_patch_canvas
            previous_feature_map = final_lod_feature_map
            
        return previous_feature_map

    def hierarchical_latent_decode_by_dict(self, feature, tree_dict, all_prob):
        batch_size = feature.shape[0]
        device = self.latent_token_positional_embedding.device
        dtype = self.latent_token_positional_embedding.dtype

        previous_feature_map = None
        for lod_idx in range(self.num_lod):
            # lod_start_idx = self.lod_start_indices[lod_idx]
            channels = self.decoder_channels[lod_idx]
            patch_size = self.patch_size_list[lod_idx]
            if lod_idx == 0:
                upsampled_map = torch.zeros(batch_size, channels, patch_size, patch_size, device=device, dtype=dtype)
            else:
                upsampler = self.upsamplers[lod_idx - 1]
                upsampled_map = upsampler(previous_feature_map)

            current_lod_patch_canvas = torch.zeros(batch_size, channels, upsampled_map.shape[-2], upsampled_map.shape[-1], device=device, dtype=dtype)

            node_idx_lod = tree_dict[lod_idx]
            lod_start_idx = sum(len(tree_dict[used_lod_idx]) for used_lod_idx in range(lod_idx))
            if len(node_idx_lod) != 0:
                unpatch_fn = self.latent_unpatchers[str(lod_idx)]
                for feat_idx, node_idx in enumerate(node_idx_lod):
                    latent_patch = unpatch_fn(feature[:, lod_start_idx + feat_idx]) * all_prob[lod_start_idx + feat_idx]
                    
                    num_patches_per_side = self.num_patch_side_list[lod_idx]
                    row, col = divmod(node_idx, num_patches_per_side)
                    
                    y_start, x_start = row * patch_size, col * patch_size
                    y_end, x_end = y_start + patch_size, x_start + patch_size

                    current_lod_patch_canvas[:, :, y_start:y_end, x_start:x_end] = latent_patch

            final_lod_feature_map = upsampled_map + current_lod_patch_canvas
            previous_feature_map = final_lod_feature_map
            
        return previous_feature_map

    def hierarchical_latent_decode_by_actions(self, feature, action_dict, prob_dict):
        batch_size = feature.shape[0]
        device = self.latent_token_positional_embedding.device
        dtype = self.latent_token_positional_embedding.dtype
        breakpoint()
        previous_feature_map = None
        for lod_idx in range(self.num_lod):
            # lod_start_idx = self.lod_start_indices[lod_idx]
            channels = self.decoder_channels[lod_idx]
            patch_size = self.patch_size_list[lod_idx]
            if lod_idx == 0:
                upsampled_map = torch.zeros(batch_size, channels, patch_size, patch_size, device=device, dtype=dtype)
            else:
                upsampler = self.upsamplers[lod_idx - 1]
                upsampled_map = upsampler(previous_feature_map)

            current_lod_patch_canvas = torch.zeros(batch_size, channels, upsampled_map.shape[-2], upsampled_map.shape[-1], device=device, dtype=dtype)

            node_idx_lod = tree_dict[lod_idx]
            lod_start_idx = sum(len(tree_dict[used_lod_idx]) for used_lod_idx in range(lod_idx))
            if len(node_idx_lod) != 0:
                unpatch_fn = self.latent_unpatchers[str(lod_idx)]
                for feat_idx, node_idx in enumerate(node_idx_lod):
                    latent_patch = unpatch_fn(feature[:, lod_start_idx + feat_idx]) * all_prob[lod_start_idx + feat_idx]
                    
                    num_patches_per_side = self.num_patch_side_list[lod_idx]
                    row, col = divmod(node_idx, num_patches_per_side)
                    
                    y_start, x_start = row * patch_size, col * patch_size
                    y_end, x_end = y_start + patch_size, x_start + patch_size

                    current_lod_patch_canvas[:, :, y_start:y_end, x_start:x_end] = latent_patch

            final_lod_feature_map = upsampled_map + current_lod_patch_canvas
            previous_feature_map = final_lod_feature_map
            
        return previous_feature_map
    
    def update_features_in_tree(
        self,
        updated_features,
        tree_structure
    ):
        ordered_nodes = self._get_ordered_nodes(tree_structure)
        for i, node in enumerate(ordered_nodes):
            feature_slice = updated_features[:, i, :]
            node.node_feature = feature_slice

    def _forward_reconstruction(self, z_quantized, tree_structure):
        batch_size, seq_len, _ = z_quantized.shape
        z_quantized = self.decoder_embed(z_quantized)
        ordered_nodes = self._get_ordered_nodes(tree_structure)

        lod_embeddings = []
        for node in ordered_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            device = self.token_incides_embedding_dict[str(lod_idx)].weight.device
            index_tensor = torch.tensor([index], dtype=torch.long, device=device)
            embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
            lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

        x = z_quantized + flat_token_sequence + self.latent_token_positional_embedding[:seq_len]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        # x = self.attn_out(x)
        self.update_features_in_tree(x, tree_structure)

        upsampled_latent = self.hierarchical_latent_decode(tree_structure, batch_size)

        reconstructd_image = self.conv_out(upsampled_latent)
        return reconstructd_image
    
    def _forward_optimize(self, z_quantized, decision_node, actions_list, action_idx_mapping):
        batch_size, seq_len, _ = z_quantized.shape
        z_quantized = self.decoder_embed(z_quantized)
        lod_embeddings = []
        actions, opt_probs = actions_list
        for lod_idx in range(self.num_lod):
            if lod_idx in decision_node.keys():
                decision_idx_list = decision_node[lod_idx]
                for index in decision_idx_list:
                    device = self.token_incides_embedding_dict[str(lod_idx)].weight.device
                    index_tensor = torch.tensor([index], dtype=torch.long, device=device)
                    embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
                    lod_embeddings.append(embedding)
            elif lod_idx == max(decision_node.keys()) + 1:
                index_tensor = torch.arange(actions.shape[0], dtype=torch.long, device=device)
                embedding_readout = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
                embedding_readout = embedding_readout * actions.unsqueeze(-1)
                embedding = embedding_readout[actions.bool()]
                lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

        all_prob = torch.ones(seq_len, device=device)
        all_prob[-opt_probs.shape[0]:] = opt_probs

        x = z_quantized + flat_token_sequence + self.latent_token_positional_embedding[:seq_len]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        # x = self.attn_out(x)

        tmp_tree_dict = copy.deepcopy(decision_node)
        tmp_tree_dict.update({max(decision_node.keys()) + 1:  torch.tensor(action_idx_mapping, device=actions.device)[actions.bool()].tolist()})
        # tree_structure = construct_tree_from_dict(tmp_tree_dict, self.patch_size_list)
        # self.update_features_in_tree(x, tree_structure)

        upsampled_latent = self.hierarchical_latent_decode_by_dict(x, tmp_tree_dict, all_prob)

        reconstructd_image = self.conv_out(upsampled_latent)
        return reconstructd_image

    def _forward_policy(self, z_quantized, policy_result, prob_result):
        batch_size, seq_len, _ = z_quantized.shape
        z_quantized = self.decoder_embed(z_quantized)
        lod_embeddings = []
        for lod_idx in range(self.num_lod):
            actions = policy_result[lod_idx]
            index_tensor = torch.arange(actions.shape[0], dtype=torch.long, device=device)
            embedding_readout = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
            embedding_readout = embedding_readout * actions.unsqueeze(-1)
            embedding = embedding_readout[actions.bool()]
            lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)
        x = z_quantized + flat_token_sequence + self.latent_token_positional_embedding[:seq_len]

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        
        upsampled_latent = self.hierarchical_latent_decode_by_actions(x, policy_result, prob_result)
        reconstructed_image = self.conv_out(upsampled_latent)
        
        return reconstructed_image


    def forward(self, z_quantized, tree_structure):
        if self.train_policy:
            return self._forward_policy(z_quantized, tree_structure)
        else:
            return self._forward_reconstruction(z_quantized, tree_structure)
        


class QuadTokSelctor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.grad_checkpointing = config.model.grad_checkpointing

        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.image_size = config.dataset.preprocessing.crop_size
        self.num_lod = len(config.model.selector.num_patch_side_list)

        self.model_size = config.model.vq_model.vit_enc_model_size
        self.token_size = config.model.selector.token_size

        self.train_policy = config.model.get("train_policy", False)

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.token_incides_embedding_dict = nn.ModuleDict()
        self.max_seq_len = 256
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            self.max_seq_len += total_patches
            self.token_incides_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.width)

        scale = self.width ** -0.5
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.max_seq_len, self.width))

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))

        self.ln_post = nn.LayerNorm(self.width)
        self.out_proj = nn.Linear(self.width, self.token_size)


    def _get_ordered_nodes(self, root_node):
        if not root_node:
            return []
        nodes_by_lod = {i: [] for i in range(self.num_lod)}
        queue = [root_node]
        
        while queue:
            node = queue.pop(0)
            if node.lod_level < self.num_lod:
                nodes_by_lod[node.lod_level].append(node)
            for child in node.children:
                queue.append(child)
        
        ordered_nodes = []
        for i in range(self.num_lod):
            ordered_nodes.extend(nodes_by_lod[i])
            
        return ordered_nodes

    def _forward_reconstruction(self, latent_feats, tree_structure):
        batch_size = latent_feats.shape[0]
        ordered_nodes = self._get_ordered_nodes(tree_structure)

        lod_embeddings = []
        for node in ordered_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            device = self.token_incides_embedding_dict[str(lod_idx)].weight.device
            index_tensor = torch.tensor([index], dtype=torch.long, device=device)
            embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
            lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_len = flat_token_sequence.shape[1]
        flat_token_sequence += self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([latent_feats, flat_token_sequence], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, -seq_len:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        return x
    
    def _forward_policy(self, latent_feats, policy_result):
        batch_size = latent_feats.shape[0]
        device = latent_feats.device
        lod_embeddings = []
        for lod_idx in range(self.num_lod):
            actions = policy_result[lod_idx]
            index_tensor = torch.arange(actions.shape[1], dtype=torch.long, device=device)
            embedding_readout = self.token_incides_embedding_dict[str(lod_idx)](index_tensor).unsqueeze(0).repeat(batch_size, 1, 1)
            embedding_readout = embedding_readout * actions
            embedding = embedding_readout[actions.bool().squeeze(-1)]
            lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_len = flat_token_sequence.shape[1]
        flat_token_sequence += self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([latent_feats, flat_token_sequence], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, -seq_len:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        
        return policy_result

    def _forward_optimize(self, latent_feats, decision_node, actions):
        batch_size = latent_feats.shape[0]
        lod_embeddings = []
        for lod_idx in range(self.num_lod):
            if lod_idx in decision_node.keys():
                decision_idx_list = decision_node[lod_idx]
                for index in decision_idx_list:
                    device = self.token_incides_embedding_dict[str(lod_idx)].weight.device
                    index_tensor = torch.tensor([index], dtype=torch.long, device=device)
                    embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
                    lod_embeddings.append(embedding)
            elif lod_idx ==  max(decision_node.keys()) + 1:
                index_tensor = torch.arange(actions.shape[0], dtype=torch.long, device=device)
                embedding_readout = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
                embedding_readout = embedding_readout * actions.unsqueeze(-1)
                embedding = embedding_readout[actions.bool()]
                lod_embeddings.append(embedding)

        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        flat_token_sequence = flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

        seq_len = flat_token_sequence.shape[1]
        flat_token_sequence += self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([latent_feats, flat_token_sequence], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, -seq_len:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        return x


    def forward(self, latent_feats, tree_structure=None, policy_output=None):
        if self.train_policy:
            return self._forward_policy(latent_feats, policy_output)
        else:
            return self._forward_reconstruction(latent_feats, tree_structure)
