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
from modeling.modules.attention import TransformerBlock, precompute_freqs_cis
from torch.utils.checkpoint import checkpoint
from modeling.utils import _get_nodes_at_level, QuadTreeNode, build_quadtree
import time
from collections import defaultdict
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
import time
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

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
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
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

class NerfBlock(nn.Module):
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=4):
        super().__init__()
        self.param_generator1 = nn.Linear(hidden_size_s, 2*hidden_size_x**2*mlp_ratio, bias=True)
        self.norm = nn.RMSNorm(hidden_size_x, eps=1e-6)
        self.mlp_ratio = mlp_ratio
    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x*self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x*self.mlp_ratio, hidden_size_x)

        # normalize fc1
        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        # normalize fc2
        normalized_fc2_param1 = torch.nn.functional.normalize(fc2_param1, dim=-2)
        # mlp 1
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, normalized_fc2_param1)
        x = x + res_x
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

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)
        # self.input_proj = BoxNerfEmbedder(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        # nerf_blocks = []
        # for i in range(num_res_blocks):
        #     nerf_blocks.append(NerfBlock(
        #         model_channels, model_channels
        #     ))

        self.res_blocks = nn.ModuleList(res_blocks)
        # self.nerf_blocks = nn.ModuleList(nerf_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, pos):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for res_block in self.res_blocks:
                x = checkpoint(res_block, x, y)
        else:
            for res_block in self.res_blocks:
                x = res_block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, pos, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c, pos)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class SimpleMLPAdaLNBox(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.num_freq = 8

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.freq_dim = 4 * self.num_freq * 2 
        self.input_proj = nn.Linear(in_channels + self.freq_dim, model_channels)
        # self.input_proj = BoxNerfEmbedder(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        # nerf_blocks = []
        # for i in range(num_res_blocks // 2):
        #     nerf_blocks.append(NerfBlock(
        #         model_channels, model_channels
        #     ))

        self.res_blocks = nn.ModuleList(res_blocks)
        # self.nerf_blocks = nn.ModuleList(nerf_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c, pos):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """

        # encode pos into freq
        freqs = torch.linspace(1.0, self.num_freq, self.num_freq, device=x.device, dtype=x.dtype)
        pos = pos.unsqueeze(-1)
        freqs = freqs.view(1, 1, -1)
        pos_bands = pos * freqs * torch.pi
        fourier_features = torch.cat([torch.sin(pos_bands), torch.cos(pos_bands)], dim=-1)

        x = self.input_proj(torch.cat([x, fourier_features.flatten(-2, -1).contiguous()], dim=-1))

        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for res_block in self.res_blocks:
                x = checkpoint(res_block, x, y)
        else:
            for res_block in self.res_blocks:
                x = res_block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, pos, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c, pos)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    

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

        # self.ordered_full_nodes = self._get_ordered_nodes(build_quadtree(self.num_patch_side_list))
        # ordered_lod_indices = [node.lod_level for node in self.ordered_full_nodes]
        # ordered_patch_indices = [node.patch_index for node in self.ordered_full_nodes]
        # self.register_buffer("ordered_lod_indices", torch.tensor(ordered_lod_indices, dtype=torch.long))
        # self.register_buffer("ordered_patch_indices", torch.tensor(ordered_patch_indices, dtype=torch.long))

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
    
    def _get_all_node_embeddings(self, batch_size, device):
        lod_embeddings = []
        for node in self.ordered_full_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            index_tensor = torch.tensor([index], dtype=torch.long, device=device)
            embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
            lod_embeddings.append(embedding)
        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        return flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

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

    def hierarchical_latent_decode_batch(self, tree_structure_list, batch_size):
        """
        Batch version of hierarchical_latent_decode for multiple tree structures.
        All trees should have the same structure (same number of nodes at each LOD).
        
        Args:
            tree_structure_list: List of tree structures with node.node_feature already set
            batch_size: Number of trees (should equal len(tree_structure_list))
        
        Returns:
            Feature maps with shape (batch_size, C, H, W)
        """
        device = self.latent_token_positional_embedding.device
        dtype = self.latent_token_positional_embedding.dtype

        # Get ordered_nodes for all trees and organize by LOD
        all_nodes_by_lod = {lod_idx: [] for lod_idx in range(self.num_lod)}
        for tree_structure in tree_structure_list:
            ordered_nodes = []
            for key, value in tree_structure.items():
                ordered_nodes.extend(value)

            nodes_by_lod = {i: [] for i in range(self.num_lod)}
            for node in ordered_nodes:
                nodes_by_lod[node.lod_level].append(node)
            for lod_idx in range(self.num_lod):
                all_nodes_by_lod[lod_idx].append(nodes_by_lod[lod_idx])
        

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
            # Process all trees for this LOD
            nodes_in_lod_list = all_nodes_by_lod[lod_idx]
            if any(nodes_in_lod_list):  # If any tree has nodes at this LOD
                unpatch_fn = self.latent_unpatchers[str(lod_idx)]
                num_patches_per_side = self.num_patch_side_list[lod_idx]
                
                # Process each tree in the batch
                st_for_loop = time.time()
                for tree_idx, nodes_in_lod in enumerate(nodes_in_lod_list):
                    if nodes_in_lod:
                        # Collect features for all nodes in this tree at this LOD
                        features_list = [node.node_feature.squeeze(0) for node in nodes_in_lod]  # Remove batch dim
                        features_tensor = torch.stack(features_list, dim=0)  # (num_nodes_in_lod, D)
                        
                        # Batch unpatch: (num_nodes_in_lod, C, patch_size, patch_size)
                        unpatched_features = unpatch_fn(features_tensor)
                        
                        # Use vectorized scatter for better efficiency
                        # Collect all patch indices and positions
                        patch_indices = torch.tensor([node.patch_index for node in nodes_in_lod], dtype=torch.long, device=device)
                        rows = patch_indices // num_patches_per_side
                        cols = patch_indices % num_patches_per_side
                        y_starts = rows * patch_size
                        x_starts = cols * patch_size
                        
                        # Use scatter_patches for vectorized operation
                        batch_coords = torch.full((len(nodes_in_lod),), tree_idx, dtype=torch.long, device=device)
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

    def hierarchical_latent_decode_by_mask(self, feature, attn_padding_mask):
        device = self.latent_token_positional_embedding.device
        dtype = self.latent_token_positional_embedding.dtype
        batch_size = feature.shape[0]

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
            

            lod_node_mask = (self.ordered_lod_indices == lod_idx) 
            lod_node_indices = torch.nonzero(lod_node_mask, as_tuple=False).squeeze(-1)  # Indices in ordered_full_nodes
            if len(lod_node_indices) > 0:
                features_lod = feature[:, lod_node_indices, :]  # (batch_size, num_nodes_in_lod, feature_dim
                mask_lod = ~attn_padding_mask[:, lod_node_indices]  # True for valid, False for masked
                
                features_lod = features_lod * mask_lod.unsqueeze(-1).float()  # (batch_size, num_nodes_in_lod, feature_dim)
                
                patch_indices_lod = self.ordered_patch_indices[lod_node_indices]  # (num_nodes_in_lod,)
                
                # Reshape for latent_unpatcher: (batch_size * num_nodes_in_lod, feature_dim)
                batch_size_lod, num_nodes_lod, feature_dim = features_lod.shape
                features_flat = features_lod.view(batch_size_lod * num_nodes_lod, feature_dim)
                
                # Apply unpatch: (batch_size * num_nodes_in_lod, C, H, W)
                unpatched_flat = self.latent_unpatchers[str(lod_idx)](features_flat)
                
                # Reshape back: (batch_size, num_nodes_in_lod, C, H, W)
                out_channels = self.decoder_channels[lod_idx]
                patch_size = self.patch_size_list[lod_idx]
                unpatched_features = unpatched_flat.view(batch_size_lod, num_nodes_lod, out_channels, patch_size, patch_size)
                
                num_patches_per_side = self.num_patch_side_list[lod_idx]
                
                rows = patch_indices_lod // num_patches_per_side  # (num_nodes_in_lod,)
                cols = patch_indices_lod % num_patches_per_side   # (num_nodes_in_lod,)
                
                y_starts = rows * patch_size  # (num_nodes_in_lod,)
                x_starts = cols * patch_size  # (num_nodes_in_lod,)
                
                valid_mask = mask_lod  # (batch_size, num_nodes_in_lod)
                
                if valid_mask.any():
                    batch_indices, feature_indices = torch.nonzero(valid_mask, as_tuple=True)  # Both are (num_valid_total,)
                    
                    valid_unpatched = unpatched_features[batch_indices, feature_indices]  # (num_valid_total, C, H, W)
                    
                    valid_y_starts = y_starts[feature_indices]  # (num_valid_total,)
                    valid_x_starts = x_starts[feature_indices]  # (num_valid_total,)
                    
                    current_lod_patch_canvas = scatter_patches(
                        current_lod_patch_canvas,
                        valid_unpatched.to(current_lod_patch_canvas.dtype),
                        batch_indices,
                        valid_y_starts,  # (num_valid_total,)
                        valid_x_starts,  # (num_valid_total,)
                        patch_size
                    )

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
    
    def _forward_optimize(self, z_quantized, tree_structure_list):
        """
        Batch version of _forward_reconstruction for optimization.
        
        Args:
            z_quantized: Quantized tokens for each tree (batch_size, seq_len, D) or (1, seq_len, D) if single image
            tree_structure_list: List of tree structures, each tree should have the same number of nodes after BFS
        
        Returns:
            Reconstructed images with shape (len(tree_structure_list), 3, H, W)
        """
        # Get batch size from tree_structure_list
        batch_size = len(tree_structure_list)
        device = z_quantized.device
        D = self.latent_token_positional_embedding.shape[-1]
        dtype = z_quantized.dtype
        
        # Get ordered_nodes for all trees
        ordered_nodes_list = []
        updated_tree_structure_list = []
        for tree_structure in tree_structure_list:
            ordered_nodes = []
            new_tree_structure = {}
            for key, value in tree_structure.items():
                if isinstance(value[0], QuadTreeNode):
                    ordered_nodes.extend(value)
                elif isinstance(value[0], dict):
                    value_nodes = []
                    for node_dict in value:
                        node = QuadTreeNode(node_dict['lod_level'], node_dict['patch_index'])
                        value_nodes.append(node)
                    ordered_nodes.extend(value_nodes)
                    new_tree_structure[key] = value_nodes
                else:
                    NotImplementedError("Only QuadTreeNode is supported for now")
            ordered_nodes_list.append(ordered_nodes)
            if len(new_tree_structure.keys()) > 0:
                updated_tree_structure_list.append(new_tree_structure)

        if len(updated_tree_structure_list) > 0:
            tree_structure_list = updated_tree_structure_list

        seq_lengths = [len(nodes) for nodes in ordered_nodes_list]
        max_seq_len_tree = max(seq_lengths) if seq_lengths else 0
        padded_patch_indices = torch.full((batch_size, max_seq_len_tree), 0, dtype=torch.long, device=device)
        padded_lod_indices = torch.full((batch_size, max_seq_len_tree), -1, dtype=torch.long, device=device)

        for b, (ordered_nodes, length) in enumerate(zip(ordered_nodes_list, seq_lengths)):
            if length > 0:
                patch_indices = torch.tensor([node.patch_index for node in ordered_nodes], dtype=torch.long, device=device)
                lod_indices = torch.tensor([node.lod_level for node in ordered_nodes], dtype=torch.long, device=device)
                padded_patch_indices[b, :length] = patch_indices
                padded_lod_indices[b, :length] = lod_indices
        raw_embeddings = torch.zeros(batch_size, max_seq_len_tree, D, device=device, dtype=dtype)
        
        for lod_idx_str, embedding_layer in self.token_incides_embedding_dict.items():
            lod_idx_int = int(lod_idx_str)
            mask_this_lod = (padded_lod_indices == lod_idx_int)
            indices_to_lookup = padded_patch_indices[mask_this_lod]      
            embeddings = embedding_layer(indices_to_lookup)
            
            mask_this_lod_expanded = mask_this_lod.unsqueeze(-1)
            raw_embeddings.masked_scatter_(mask_this_lod_expanded, embeddings.to(dtype))

        seq_lengths_tensor = torch.tensor(seq_lengths, device=device)
        padding_mask = torch.arange(max_seq_len_tree, device=device).expand(batch_size, max_seq_len_tree) >= seq_lengths_tensor.unsqueeze(1)
            
        # Embed z_quantized: (batch_size, seq_len, D)
        token_pe = self.latent_token_positional_embedding[:max_seq_len_tree]
        z_quantized_embedded = self.decoder_embed(z_quantized)
        
        # Add positional embeddings to token sequence
        flat_token_sequence = raw_embeddings + token_pe
        if z_quantized_embedded.shape[1] != flat_token_sequence.shape[1]:
            z_quantized_embedded = z_quantized_embedded[:, :flat_token_sequence.shape[1], :]
        x = z_quantized_embedded + flat_token_sequence
        x.masked_fill_(padding_mask.unsqueeze(-1), 0.0)

        # Apply transformer
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x, key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        # Update features in trees for hierarchical decode
        for i, (tree_structure, ordered_nodes) in enumerate(zip(tree_structure_list, ordered_nodes_list)):
            for j, node in enumerate(ordered_nodes):
                node.node_feature = x[i, j:j+1, :]  # (1, D)
        
        # Batch hierarchical decode
        upsampled_latent = self.hierarchical_latent_decode_batch(tree_structure_list, batch_size)
        reconstructed_image = self.conv_out(upsampled_latent)
        
        return reconstructed_image

    def _forward_policy(self, z_quantized, action_dict, attn_padding_mask):
        batch_size, seq_len, _ = z_quantized.shape
        device = z_quantized.device
        z_quantized = self.decoder_embed(z_quantized)
        flat_token_sequence = self._get_all_node_embeddings(batch_size, device)
        seq_len = flat_token_sequence.shape[1]

        token_mask = ~attn_padding_mask 

        valid_pos_indices = torch.cumsum(token_mask.long(), dim=1) - 1  # (batch_size, seq_len)
        valid_pos_indices = valid_pos_indices * token_mask.long()  # Set invalid positions to 0
        valid_pos_indices = torch.clamp(valid_pos_indices, min=0)  # Ensure non-negative

        max_valid_pos = valid_pos_indices.max().item() + 1 if token_mask.any() else 0
        
        if max_valid_pos > 0:
            pos_emb = self.latent_token_positional_embedding[:max_valid_pos]
            pos_emb_batched = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, max_valid_pos, D)
            valid_pos_indices_expanded = valid_pos_indices.unsqueeze(-1).expand(-1, -1, pos_emb.shape[-1])  # (batch_size, seq_len, D)

            token_pos_embeddings = torch.gather(
                pos_emb_batched,
                dim=1,
                index=valid_pos_indices_expanded
            )
            token_pos_embeddings = token_pos_embeddings * token_mask.unsqueeze(-1).float()
        else:
            token_pos_embeddings = torch.zeros_like(flat_token_sequence)
        
        flat_token_sequence = flat_token_sequence + token_pos_embeddings
        x = z_quantized + flat_token_sequence

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x, key_padding_mask=attn_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        
        upsampled_latent = self.hierarchical_latent_decode_by_mask(x, attn_padding_mask)
        reconstructed_image = self.conv_out(upsampled_latent)
        
        return reconstructed_image


    def forward(self, z_quantized, tree_structure):
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

        self.repa_param = config.losses.get("repa_param", None)

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

        if self.repa_param is not None:
            self.ln_align = nn.LayerNorm(self.width)
            self.out_proj_align = nn.Linear(self.width, 768)

        self.ordered_full_nodes = self._get_ordered_nodes(build_quadtree(self.num_patch_side_list))


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

    def _get_all_node_embeddings(self, batch_size, device):
        lod_embeddings = []
        for node in self.ordered_full_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            if str(lod_idx) in self.token_incides_embedding_dict.keys():
                index_tensor = torch.tensor([index], dtype=torch.long, device=device)
                embedding = self.token_incides_embedding_dict[str(lod_idx)](index_tensor)
                lod_embeddings.append(embedding)
        
        # (num_total_nodes, D)
        flat_token_sequence = torch.cat(lod_embeddings, dim=0)
        # (B, num_total_nodes, D)
        return flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

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
        if self.repa_param is not None:
            align_x = x[:, seq_len:]
            x = x[:, -seq_len:]
            x = self.ln_post(x)
            x = self.out_proj(x)

            x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
            return x, self.out_proj_align(self.ln_align(align_x))
        else:
            x = x[:, -seq_len:]
            x = self.ln_post(x)
            x = self.out_proj(x)

            x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
            return x
    
    def _forward_policy(self, latent_feats, attn_padding_mask):
        batch_size = latent_feats.shape[0]
        device = latent_feats.device

        flat_token_sequence = self._get_all_node_embeddings(batch_size, device)
        seq_len = flat_token_sequence.shape[1]

        full_padding_mask = torch.cat([torch.zeros(batch_size, latent_feats.shape[1], dtype=torch.bool, device=device), attn_padding_mask], dim=1)
        token_mask = ~attn_padding_mask 

        valid_pos_indices = torch.cumsum(token_mask.long(), dim=1) - 1  # (batch_size, seq_len)
        valid_pos_indices = valid_pos_indices * token_mask.long()  # Set invalid positions to 0
        valid_pos_indices = torch.clamp(valid_pos_indices, min=0)  # Ensure non-negative

        max_valid_pos = valid_pos_indices.max().item() + 1 if token_mask.any() else 0
        
        if max_valid_pos > 0:
            pos_emb = self.latent_token_positional_embedding[:max_valid_pos]
            pos_emb_batched = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, max_valid_pos, D)
            valid_pos_indices_expanded = valid_pos_indices.unsqueeze(-1).expand(-1, -1, pos_emb.shape[-1])  # (batch_size, seq_len, D)

            token_pos_embeddings = torch.gather(
                pos_emb_batched,
                dim=1,
                index=valid_pos_indices_expanded
            )
            token_pos_embeddings = token_pos_embeddings * token_mask.unsqueeze(-1).float()
        else:
            token_pos_embeddings = torch.zeros_like(flat_token_sequence)
        
        flat_token_sequence = flat_token_sequence + token_pos_embeddings
        x = torch.cat([latent_feats, flat_token_sequence], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x, key_padding_mask=full_padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, -seq_len:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        
        return x

    def _forward_optimize(self, latent_feats, tree_structure_list, batch_expand_list=None):
        if latent_feats.ndim == 2:
            latent_feats = latent_feats.unsqueeze(0)  # (1, num_latent_tokens, D)
        
        # Get batch size from tree_structure_list
        batch_size = len(tree_structure_list)
        device = latent_feats.device
        D = self.latent_token_positional_embedding.shape[-1]
        dtype = latent_feats.dtype
        
        # Get ordered_nodes for all trees
        ordered_nodes_list = []
        for tree_structure in tree_structure_list:
            ordered_nodes = []
            for key, value in tree_structure.items():
                ordered_nodes.extend(value)
            ordered_nodes_list.append(ordered_nodes)

        seq_lengths = [len(nodes) for nodes in ordered_nodes_list]
        max_seq_len_tree = max(seq_lengths) if seq_lengths else 0
        padded_patch_indices = torch.full((batch_size, max_seq_len_tree), 0, dtype=torch.long, device=device)
        padded_lod_indices = torch.full((batch_size, max_seq_len_tree), -1, dtype=torch.long, device=device)

        for b, (ordered_nodes, length) in enumerate(zip(ordered_nodes_list, seq_lengths)):
            if length > 0:
                patch_indices = torch.tensor([node.patch_index for node in ordered_nodes], dtype=torch.long, device=device)
                lod_indices = torch.tensor([node.lod_level for node in ordered_nodes], dtype=torch.long, device=device)
                padded_patch_indices[b, :length] = patch_indices
                padded_lod_indices[b, :length] = lod_indices
        raw_embeddings = torch.zeros(batch_size, max_seq_len_tree, D, device=device, dtype=dtype)
        
        for lod_idx_str, embedding_layer in self.token_incides_embedding_dict.items():
            lod_idx_int = int(lod_idx_str)
            mask_this_lod = (padded_lod_indices == lod_idx_int)
            indices_to_lookup = padded_patch_indices[mask_this_lod]      
            embeddings = embedding_layer(indices_to_lookup)
            
            mask_this_lod_expanded = mask_this_lod.unsqueeze(-1)
            raw_embeddings.masked_scatter_(mask_this_lod_expanded, embeddings.to(dtype))

        seq_lengths_tensor = torch.tensor(seq_lengths, device=device)
        tree_mask = torch.arange(max_seq_len_tree, device=device).expand(batch_size, max_seq_len_tree) >= \
                    seq_lengths_tensor.unsqueeze(1)
        token_pe = self.latent_token_positional_embedding[:max_seq_len_tree]
        flat_token_sequence = raw_embeddings + token_pe.unsqueeze(0)
        
        flat_token_sequence.masked_fill_(tree_mask.unsqueeze(-1), 0.0)

        if latent_feats.shape[0] != batch_size:
            assert batch_expand_list is not None
            batch_expand_list_tensor = torch.tensor(batch_expand_list, device=device, dtype=torch.long)
            latent_feats = torch.repeat_interleave(latent_feats, batch_expand_list_tensor, dim=0)

        latent_mask = torch.zeros(batch_size, latent_feats.shape[1], dtype=torch.bool, device=device)
        padding_mask = torch.cat([latent_mask, tree_mask], dim=1)

        x = torch.cat([latent_feats, flat_token_sequence], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x, key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = x[:, latent_feats.shape[1]:]
        x = self.ln_post(x)
        x = self.out_proj(x)

        x = x.permute(0, 2, 1).unsqueeze(2).contiguous()
        return x


    def forward(self, latent_feats, tree_structure=None, policy_output=None):
        return self._forward_reconstruction(latent_feats, tree_structure)

class BoxNerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs=8, num_coords=4):
        super().__init__()
        self.max_freqs = max_freqs
        self.num_coords = num_coords
        self.freq_dim = num_coords * max_freqs * 2 
        
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + self.freq_dim, hidden_size_input, bias=True),
        )

    def compute_fourier_features(self, boxes):
        device = boxes.device
        dtype = boxes.dtype
        
        freqs = torch.linspace(1.0, self.max_freqs, self.max_freqs, device=device, dtype=dtype)
        x = boxes.unsqueeze(-1) 
        f = freqs.view(1, 1, 1, -1)
        x_bands = x * f * torch.pi # (B, N, 4, max_freqs)

        fourier_features = torch.cat([torch.sin(x_bands), torch.cos(x_bands)], dim=-1)
        fourier_features = fourier_features.view(boxes.shape[0], boxes.shape[1], -1)
        
        return fourier_features

    def forward(self, x, boxes):
        box_emb = self.compute_fourier_features(boxes)
     
        x_input = torch.cat([x, box_emb], dim=-1)
        out = self.embedder(x_input)
        return out