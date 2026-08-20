from functools import partial
from torch._tensor import Tensor
from typing import Any
import pickle
import numpy as np
import json
import copy
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from modeling.modules.base_model import BaseModel
from modeling.modules.attention import TransformerBlock, precompute_freqs_cis, batch_apply_rotary_emb, precompute_freqs_cis_2d, find_multiple, KVCache, RMSNorm
from modeling.modules.blocks import LabelEmbedder
from modeling.generate_utils import sample
from timm.models.vision_transformer import Block

from modeling.modules.losses import DiffLoss
from modeling.utils import build_quadtree, build_probabilistic_quadtree, tree_to_decision_nodes_dict, QuadTreeNode, _get_parent_patch_index
from collections import defaultdict

def interleave_tokens(seq1, seq2):
    """ Interleave two sequences """
    result = torch.zeros_like(torch.cat((seq1, seq2), dim=1))
    result[:, ::2] = seq1
    result[:, 1::2] = seq2
    return result

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

def calculate_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()


class MAR(BaseModel):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_config = config.model

        self.model_size = config.model.model_size

        self.embed_dim = {
                "base": 768,
                "large": 1024,
                "xlarge": 1280,
            }[self.model_size]

        self.depth = {
                "base": 12,
                "large": 16,
                "xlarge": 20,
            }[self.model_size]

        self.num_heads = {
                "base": 12,
                "large": 16,
                "xlarge": 16,
            }[self.model_size]
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = 1
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = config.tokenizer.vae_embed_dim

        self.patch_size = config.model.patch_size
        self.seq_h = self.seq_w = config.dataset.preprocessing.crop_size // config.tokenizer.vae_stride // config.model.patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = config.tokenizer.vae_embed_dim * (config.model.patch_size**2)
        self.grad_checkpointing = config.model.grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = config.model.class_num
        self.class_emb = nn.Embedding(config.model.class_num, self.embed_dim)
        self.label_drop_prob = config.model.label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, self.embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm((config.model.mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, self.embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.buffer_size = config.model.buffer_size
        # self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.embed_dim))

        # self.encoder_blocks = nn.ModuleList([
        #     Block(self.embed_dim, self.num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
        #           proj_drop=config.model.proj_dropout, attn_drop=config.model.attn_dropout) for _ in range(self.depth)])
    
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.proj_dropout, attn_dropout_p=config.model.attn_dropout) for _ in range(self.depth)])
        self.encoder_norm = norm_layer(self.embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, self.embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.proj_dropout, attn_dropout_p=config.model.attn_dropout) for _ in range(self.depth)])

        self.decoder_norm = norm_layer(self.embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        # we use rope for all the attention layers
        freqs_cis = precompute_freqs_cis(self.embed_dim // self.num_heads, self.seq_len + self.buffer_size)
        self.register_buffer("freqs_cis", freqs_cis)

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(self.config)

        self.diffusion_batch_mul = self.config.model.diffusion_batch_mul

        # generation params
        self.generator_config = config.model.generator

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        # torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        # x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        freqs_cis = self.freqs_cis.unsqueeze(0).repeat(bsz, 1, 1, 1)[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim // (self.num_heads * 2), 2)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, freqs_cis)
        else:
            for block in self.encoder_blocks:
                x = block(x, freqs_cis)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad# + self.decoder_pos_embed_learned
        freqs_cis = self.freqs_cis.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x, freqs_cis)
        else:
            for block in self.decoder_blocks:
                x = block(x, freqs_cis)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        print(f"Latent Mean: {target.mean().item()}")
        print(f"Latent Std:  {target.std().item()}")
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss, loss_dict = self.diffloss(z=z, target=target, mask=mask)
        return loss, loss_dict

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss, loss_dict = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss, loss_dict

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):

        bsz = condition.shape[0]
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

        indices = list(range(self.generator_config.num_iter))
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if condition is not None:
                class_embedding = self.class_emb(condition)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not guidance_scale == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / self.generator_config.num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= self.generator_config.num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not guidance_scale == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if guidance_decay == "linear":
                cfg_iter = 1 + (guidance_scale - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif guidance_decay == "constant":
                cfg_iter = guidance_scale
            else:
                raise NotImplementedError
            breakpoint()
            sampled_token_latent = self.diffloss.sample(z, randomize_temperature, cfg_iter)
            if not guidance_scale == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

class CausalMAR(BaseModel):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_config = config.model.generator

        self.model_size = config.model.generator.model_size

        self.embed_dim = {
                "base": 768,
                "large": 1024,
                "xlarge": 1280,
            }[self.model_size]

        self.depth = {
                "base": 24,
                "large": 32,
                "xlarge": 40,
            }[self.model_size]

        self.num_heads = {
                "base": 12,
                "large": 16,
                "xlarge": 16,
            }[self.model_size]
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = 1  # base-update FFN fix: SwiGLU FeedForward already does 4*dim; passing 4 double-counted -> 4x oversized
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = config.tokenizer.vae_embed_dim

        self.patch_size = config.model.patch_size
        self.seq_h = self.seq_w = config.model.image_size // config.tokenizer.vae_stride // config.model.patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = config.tokenizer.vae_embed_dim * (config.model.patch_size**2)
        self.head_dim = self.embed_dim // self.num_heads
        self.grad_checkpointing = config.model.grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = config.model.class_num
        self.class_emb = nn.Embedding(config.model.class_num, self.embed_dim)
        self.label_drop_prob = config.model.label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, self.embed_dim))

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, self.embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.buffer_size = config.model.buffer_size
    
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.proj_dropout, attn_dropout_p=config.model.attn_dropout) for _ in range(self.depth)])

        self.out_norm = norm_layer(self.embed_dim)
        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        # we use rope for all the attention layers
        grid_size = int(self.seq_len ** 0.5)
        freqs_cis = precompute_freqs_cis_2d(grid_size, self.head_dim, cls_token_num = self.buffer_size)
        self.register_buffer("freqs_cis", freqs_cis)

        # position instruction params
        self.pos_instruct_embeddings = nn.Parameter(torch.randn(1, self.embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(self.config)

        self.diffusion_batch_mul = self.config.model.diffusion_batch_mul

        # generation params
        self.generator_config = config.model.generator

        

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.pos_instruct_embeddings, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def get_position_instruction_tokens(self, token_order):
        position_instruct_tokens = self.pos_instruct_embeddings.view(1, 1, self.num_heads, self.head_dim)
        position_instruct_tokens = position_instruct_tokens.repeat(token_order.shape[0], self.seq_len, 1, 1) # [1, block_size, n_head, dim // n_head]
        
        # apply rotary embedding
        position_instruct_freqs_cis = self.freqs_cis[self.buffer_size:].clone().to(token_order.device)[token_order]
        position_instruct_tokens = batch_apply_rotary_emb(position_instruct_tokens, position_instruct_freqs_cis)
        position_instruct_tokens = position_instruct_tokens.view(token_order.shape[0], self.seq_len, self.embed_dim).contiguous()
        return position_instruct_tokens

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
    #     return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, self.head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)


    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss, loss_dict = self.diffloss(z=z, target=target, mask=None)
        return loss, loss_dict

    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        bs, seq_len, in_dim = x.shape
        gt_latents = x.clone().detach()

        # prepare random order here
        token_order = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_order = token_order.unsqueeze(0).repeat(bs, 1)
        for i in range(bs):
            token_order[i] = token_order[i][torch.randperm(seq_len)]
        token_order = token_order.contiguous()

        # premute the input tokens and the target tokens
        x = torch.gather(x, 1, token_order.unsqueeze(-1).repeat(1, 1, in_dim)).contiguous() # [bsz, seq_len, in_dim]
        gt_latents = torch.gather(gt_latents, 1, token_order.unsqueeze(-1).repeat(1, 1, in_dim)).squeeze(-1).contiguous() # [bsz, seq_len, in_dim]

        position_instruction_tokens = self.get_position_instruction_tokens(token_order)

        if self.training:
            drop_latent_mask = torch.rand(bs) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
            cond_embeddings = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)


        token_embeddings = self.z_proj_ln(self.z_proj(x))
        z = torch.cat(
            (cond_embeddings, interleave_tokens(position_instruction_tokens, token_embeddings)),
            dim=1
        )

        token_freqs_cis = self.freqs_cis[self.buffer_size:].clone().to(token_order.device)[token_order]
        freqs_cis = torch.cat((self.freqs_cis[:self.buffer_size].unsqueeze(0).repeat(bs, 1, 1, 1), interleave_tokens(token_freqs_cis, token_freqs_cis)), dim=1)


        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                z = checkpoint(block, z, freqs_cis)
        else:
            for block in self.blocks:
                z = block(z, freqs_cis)

        z = self.out_norm(z)
        z = z[:, self.buffer_size::2].contiguous()
        # diffloss
        loss, loss_dict = self.forward_loss(z=z, target=gt_latents)

        return loss, loss_dict

    def forward_inference(self, 
                        x: torch.Tensor, 
                        freqs_cis: torch.Tensor, 
                        input_pos: torch.Tensor):
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.blocks:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.out_norm(h)
        return h

    def remove_caches(self):
        for l in self.blocks:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):

        bsz = condition.shape[0]

        token_order = torch.arange(self.seq_len, device=condition.device, dtype=torch.long)
        token_order = token_order.unsqueeze(0).repeat(bsz, 1)
        token_order = token_order.contiguous()
        for i in range(bsz):
            token_order[i] = token_order[i][torch.randperm(self.seq_len)]
        token_order = token_order.contiguous()

        result_tokens = torch.zeros((bsz, self.seq_len, self.token_embed_dim), dtype=torch.float32, device=condition.device)

        position_instruction_tokens = self.get_position_instruction_tokens(token_order)
        img_token_freq_cis = self.freqs_cis[self.buffer_size:].clone().to(token_order.device)[token_order]

        class_embedding = self.class_emb(condition)

        if not guidance_scale == 1.0:
            img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis])
            position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens])
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            bsz *= 2

        max_seq_len = self.buffer_size + self.seq_len * 2
        with torch.device(condition.device):
            self.setup_caches(max_batch_size=bsz, max_seq_length=max_seq_len, dtype=class_embedding.dtype)

        cur_inference_step = 0
        num_query_token_cur_step = 1 # how many tokens to decode at this step
        num_query_token_next_step = 1 # for every step, we only query 1 token
        query_token_idx_cur_step = 0 # the index of the first token to decode at this step

        class_embedding = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        x = torch.cat([class_embedding, position_instruction_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], dim=1)
        cur_freqs_cis = torch.cat([self.freqs_cis[:self.buffer_size].unsqueeze(0).repeat(bsz, 1, 1, 1), img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step]], dim=1)
        input_pos = torch.arange(0, x.shape[1], device=condition.device)
                                   
        while query_token_idx_cur_step <= self.seq_len - num_query_token_cur_step and query_token_idx_cur_step <= self.seq_len - 1:
            model_pred = self.forward_inference(x, cur_freqs_cis, input_pos)

            if not guidance_scale == 1.0:
                if guidance_decay == "linear":
                    cfg_iter = 1 + (guidance_scale - 1) * query_token_idx_cur_step / self.seq_len
                elif guidance_decay == "constant":
                    cfg_iter = guidance_scale

            sampled_token_latent = self.diffloss.sample(model_pred[:, -num_query_token_cur_step:].squeeze(1), randomize_temperature, cfg_iter)
            if not guidance_scale == 1.0:
                token_latent = sampled_token_latent[:bsz // 2].unsqueeze(1)
            else:
                token_latent = sampled_token_latent.unsqueeze(1)

            result_tokens[:, query_token_idx_cur_step : query_token_idx_cur_step + num_query_token_cur_step] = token_latent.clone()

            if not guidance_scale == 1.0:
                token_latent = torch.cat([token_latent, token_latent], dim=0)

            query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step

            total_len = 2 * num_query_token_cur_step - 1 + num_query_token_next_step
            if query_token_idx_next_step == self.seq_len:
                break

            x = torch.zeros(bsz, total_len, self.embed_dim, dtype=x.dtype, device=token_latent.device)

            x[:, :1] = self.z_proj_ln(self.z_proj(token_latent))

            next_position_instruction_tokens = position_instruction_tokens[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            x[:, 1:] = next_position_instruction_tokens

            cur_freqs_cis = torch.zeros((bsz, total_len, *self.freqs_cis.shape[-2:]), 
                            dtype=cur_freqs_cis.dtype, device=token_latent.device)

            cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step : query_token_idx_cur_step + 1]
            next_freq_cis = img_token_freq_cis[:, query_token_idx_next_step : query_token_idx_next_step + num_query_token_next_step]
            cur_freqs_cis[:, 1:] = next_freq_cis

            query_token_idx_cur_step = query_token_idx_next_step 
            

            last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step]
            input_pos = torch.arange(last_input_pos + 1, last_input_pos + 1 + total_len, device=token_latent.device, dtype=torch.long)
            num_query_token_cur_step = num_query_token_next_step

        self.remove_caches()
        reverse_permutation = torch.argsort(token_order, dim=-1).long().unsqueeze(-1).repeat(1, 1, self.token_embed_dim)
        tokens = torch.gather(result_tokens, 1, reverse_permutation)

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

class QuadtreeMAR(BaseModel):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_config = config.model.generator

        self.model_size = config.model.generator.model_size

        self.embed_dim = {
                "base": 768,
                "large": 1024,
                "xlarge": 1280,
            }[self.model_size]

        self.depth = {
                "base": 24,
                "large": 32,
                "xlarge": 40,
            }[self.model_size]

        self.num_heads = {
                "base": 12,
                "large": 16,
                "xlarge": 16,
            }[self.model_size]
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = 1  # base-update FFN fix: SwiGLU FeedForward already does 4*dim; passing 4 double-counted -> 4x oversized
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = config.model.vq_model.token_size

        self.patch_size = config.model.generator.patch_size

        self.seq_len = 1365 # maimum quadtree token number
        self.token_embed_dim = config.model.vq_model.token_size * (config.model.generator.patch_size**2)
        self.head_dim = self.embed_dim // self.num_heads
        self.grad_checkpointing = config.model.grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = config.model.generator.class_num
        self.class_emb = nn.Embedding(config.model.generator.class_num, self.embed_dim)
        self.label_drop_prob = config.model.generator.label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, self.embed_dim))

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, self.embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.buffer_size = config.model.generator.buffer_size
    
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.generator.proj_dropout, attn_dropout_p=config.model.generator.attn_dropout) for _ in range(self.depth)])

        self.out_norm = norm_layer(self.embed_dim)
        
        # --------------------------------------------------------------------------
        # Binary classification head for predicting whether to expand children
        # self.expand_pred_head = nn.Linear(self.embed_dim, 1, bias=True)

        # --------------------------------------------------------------------------
        # Quadtree position embeddings (learnable tokens for each LOD and patch index)
        self.num_patch_side_list = config.model.generator.num_patch_side_list
        self.num_lod = len(self.num_patch_side_list)
        
        # Create learnable embeddings for each LOD level and patch index
        # self.token_indices_embedding_dict = nn.ModuleDict()
        # for lod_idx, num_patches in enumerate(self.num_patch_side_list):
        #     total_patches = num_patches ** 2
        #     self.token_indices_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.embed_dim)
        
        # Build full quadtree structure for node mapping
        self.full_tree_root = build_quadtree(self.num_patch_side_list)
        self.ordered_full_nodes = self._get_ordered_nodes(self.full_tree_root)
        
        # Create mapping from node (lod, patch_idx) to BFS index
        self.node_to_idx_map = {
            (node.lod_level, node.patch_index): i 
            for i, node in enumerate(self.ordered_full_nodes)
        }
        
        # Guaranteed depth: all nodes with lod <= guaranteed_depth are always present
        self.guaranteed_depth = config.model.generator.guaranteed_depth
        
        # Create LOD level tensor for each node in ordered_full_nodes
        # This will be used to filter which nodes to predict expand for
        node_lod_levels = torch.tensor(
            [node.lod_level for node in self.ordered_full_nodes],
            dtype=torch.long
        )
        self.register_buffer("node_lod_levels", node_lod_levels)

        # *2 because of interleaving position and token
        max_total_seq_len = self.buffer_size + self.seq_len * 2
        freqs_cis = precompute_freqs_cis(self.head_dim, max_total_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, 512, self.embed_dim))
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, 378, self.embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(self.config)

        self.diffusion_batch_mul = self.config.model.generator.diffusion_batch_mul
        

    def _get_ordered_nodes(self, root_node):
        """Get ordered nodes in BFS order, grouped by LOD level"""
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

    def _tree_structure_to_status(self, tree_structure, device):
        """
        Convert a tree_structure (quadtree root) to status list and length.
        Reference: pretokenize_quadtree.py lines 390-422
        !! Fake status is used: as long as the node in tree_structure, the status should >= 0
        
        Args:
            tree_structure: Quadtree root node
            device: Device to create tensors on
            
        Returns:
            status: Tensor of shape (1, num_total_nodes) with values:
                - 1: Node is present in the tree
                - 0: Node is not present but its parent is present (potential but not chosen)
                - -1: Node is not present and its parent is also not present
            length: Scalar tensor with the number of active nodes (status >= 0)
        """
        # Extract all nodes from tree_structure, organized by LOD level
        decision_nodes = defaultdict(list)
        queue = [tree_structure]
        while queue:
            node = queue.pop(0)
            if node.lod_level < self.num_lod:
                decision_nodes[node.lod_level].append(node)
            for child in node.children:
                queue.append(child)
        
        # Build present_nodes_set
        present_nodes_set = set()
        for lod, nodes in decision_nodes.items():
            for node in nodes:
                present_nodes_set.add((node.lod_level, node.patch_index))
        
        # Initialize status list
        num_total_nodes = len(self.ordered_full_nodes)
        status_list = [-1] * num_total_nodes
        
        # Mark present nodes with 1
        for i, node in enumerate(self.ordered_full_nodes):
            if (node.lod_level, node.patch_index) in present_nodes_set:
                status_list[i] = 1

        # Count active nodes (status >= 0)
        length = sum(1 for s in status_list if s >= 0)
        
        # Convert to tensor
        status_tensor = torch.tensor([status_list], dtype=torch.long, device=device)
        length_tensor = torch.tensor([length], dtype=torch.long, device=device)
        
        return status_tensor, length_tensor

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        
        # # Initialize quadtree position embeddings
        # for lod_idx_str, embedding_layer in self.token_indices_embedding_dict.items():
        #     torch.nn.init.normal_(embedding_layer.weight, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def build_expand_labels_from_status(self, status, lengths, max_seq_len, device):
        """
        Build ground truth expand labels from status tensor (fully vectorized, no loops).
        Only builds expand labels for nodes with guaranteed_depth <= lod < num_lod - 1.
        
        Args:
            status: Tensor of shape (batch_size, num_total_nodes)
                   status=0: node exists but doesn't expand
                   status=1: node exists and expands
                   status=-1: node doesn't exist
            lengths: Tensor of shape (batch_size,) with actual sequence lengths
            max_seq_len: Maximum sequence length for padding
            device: Device to create tensors on
        Returns:
            gt_expand_labels: Tensor of shape (batch_size, max_seq_len) with 0/1 labels
            gt_expand_mask: Boolean tensor of shape (batch_size, max_seq_len) marking valid positions
                           for expand prediction
        """
        batch_size = status.shape[0]
        
        # Create LOD mask: only predict expand for nodes in specific LOD range
        # - Nodes with lod <= guaranteed_depth - 1: their children are guaranteed, no need to predict
        # - Nodes with lod == num_lod - 1: no children, cannot expand
        # - Nodes with guaranteed_depth <= lod < num_lod - 1: need to predict expand
        lod_mask = (self.node_lod_levels >= self.guaranteed_depth) & (self.node_lod_levels < self.num_lod - 1)
        lod_mask = lod_mask.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_total_nodes)
        
        # Create active mask: True for nodes with status >= 0 (all active nodes in sequence)
        active_mask = (status >= 0)  # (batch_size, num_total_nodes)
        
        cumsum_positions = torch.cumsum(active_mask.long(), dim=1) - 1  # (batch_size, num_total_nodes)
        cumsum_positions = cumsum_positions * active_mask.long()  # Zero out inactive nodes
        
        lengths_expanded = lengths.unsqueeze(1)  # (batch_size, 1)
        position_valid = (cumsum_positions < lengths_expanded) & active_mask & lod_mask
        
        # Initialize output tensors
        gt_expand_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.float32, device=device)
        valid_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=device)
        
        # Get batch indices and node indices for valid positions
        batch_idx, node_idx = torch.where(position_valid)
        
        if len(batch_idx) > 0:
            # Get the sequence positions for these nodes
            positions = cumsum_positions[batch_idx, node_idx]
            # Get the status values
            values = status[batch_idx, node_idx].float()
            
            # Scatter values to their positions using advanced indexing
            gt_expand_labels[batch_idx, positions] = values
            valid_mask[batch_idx, positions] = True
        
        return gt_expand_labels, valid_mask
    
    def get_position_instruction_tokens_from_status(self, status, lengths):
        """
        Get position instruction tokens based on the status and ordered_full_nodes.
        Args:
            status: Tensor of shape (batch_size, num_total_nodes) containing node status
                   status=0: node exists but doesn't expand children
                   status=1: node exists and expands children  
                   status=-1: node is not activated (not in tree)
            lengths: Tensor of shape (batch_size,) containing actual sequence lengths
        Returns:
            position_instruct_tokens: Tensor of shape (batch_size, max_seq_len, embed_dim)
                                     where padding positions are zeros
        """
        batch_size = status.shape[0]
        device = status.device
        max_seq_len = lengths.max().item()
        num_total_nodes = len(self.ordered_full_nodes)
        
        # Step 1: Pre-compute position embeddings for ALL nodes (only once, efficiently)
        # Group nodes by LOD level for batch embedding lookup
        all_position_embeddings = torch.zeros(num_total_nodes, self.embed_dim, 
                                             device=device, dtype=torch.float32)
        
        for lod_level_str, embedding_layer in self.token_indices_embedding_dict.items():
            lod_level = int(lod_level_str)
            # Find all nodes at this LOD level
            lod_nodes_indices = []
            lod_patch_indices = []
            
            for node_idx, node in enumerate(self.ordered_full_nodes):
                if node.lod_level == lod_level:
                    lod_nodes_indices.append(node_idx)
                    lod_patch_indices.append(node.patch_index)
            
            if len(lod_nodes_indices) > 0:
                # Batch lookup embeddings for all nodes at this LOD level
                patch_indices_tensor = torch.tensor(lod_patch_indices, dtype=torch.long, device=device)
                embeddings = embedding_layer(patch_indices_tensor)  # (num_nodes_at_lod, embed_dim)
                
                # Assign to corresponding positions in all_position_embeddings
                for i, node_idx in enumerate(lod_nodes_indices):
                    all_position_embeddings[node_idx] = embeddings[i]
        
        # Step 2: Create indices for gathering
        # For each batch, find active nodes (status >= 0) and gather their embeddings
        indices = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        
        # Create active mask (status >= 0 means node is in the tree)
        active_mask = status >= 0  # (batch_size, num_total_nodes)
        
        # Build indices for gathering (only this small loop remains)
        for b in range(batch_size):
            active_indices = active_mask[b].nonzero(as_tuple=True)[0]
            seq_len = min(lengths[b].item(), len(active_indices))
            if seq_len > 0:
                indices[b, :seq_len] = active_indices[:seq_len]

        position_instruct_tokens = all_position_embeddings[indices]
        
        return position_instruct_tokens

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
    #     return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, self.head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)


    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        """
        Setup KV caches for all transformer blocks.
        
        Args:
            max_batch_size: Maximum batch size for generation
            max_seq_length: Maximum sequence length
            dtype: Data type for cache tensors
        """
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, self.head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)

    def remove_caches(self):
        """Remove KV caches from all transformer blocks."""
        for l in self.blocks:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward_inference(self, 
                        x: torch.Tensor, 
                        freqs_cis: torch.Tensor, 
                        input_pos: torch.Tensor):
        """
        Forward pass for inference with KV cache.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            freqs_cis: RoPE frequencies
            input_pos: Position indices for current tokens
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.blocks:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.out_norm(h)
        return h

    def forward_loss(self, z, target, padding_mask=None):
        """
        Compute loss with optional padding mask.
        Args:
            z: Predicted tokens of shape (batch_size, seq_len, embed_dim)
            target: Target tokens of shape (batch_size, seq_len, embed_dim)
            padding_mask: Boolean mask of shape (batch_size, seq_len) where True indicates padding
        """

        target = target[(~padding_mask)].repeat(self.diffusion_batch_mul, 1)
        z = z[(~padding_mask)].repeat(self.diffusion_batch_mul, 1)
        loss, loss_dict = self.diffloss(z=z, target=target, pos=None, mask=None)
        return loss, loss_dict

    def forward(self, input_tokens, target_tokens, lod_idx, labels):
        """
        Forward pass for QuadtreeMAR.
        Args:
            input_tokens: Input tokens (z_quantized) of shape (batch_size, max_seq_len - 1, token_embed_dim)
               Padded positions have value -1.0
            target_tokens: Target tokens (z_quantized) of shape (batch_size, max_seq_len, token_embed_dim)
               Padded positions have value -1.0
            tree_dict: Dictionary containing:
                - 'status': Tensor of shape (batch_size, num_total_nodes)
                - 'lengths': Tensor of shape (batch_size,) with actual sequence lengths
                - 'tree': List of tree structures (one per batch item)
            labels: Class labels of shape (batch_size,)
        """
        
        # class embed
        class_embedding = self.class_emb(labels)

        bs, max_seq_len, in_dim = target_tokens.shape
        device = target_tokens.device

                # Build LOD-based block causal mask
        # lod_idx shape: (seq_len,) - each element represents the LOD level of that token position
        token_seq_len = input_tokens.shape[1]  # max_seq_len - 1

        lod_idx_batch = lod_idx.unsqueeze(0).repeat(bs, 1)  # (bs, token_seq_len)

        lod_idx_expanded_i = lod_idx_batch.unsqueeze(2)  # (bs, token_seq_len, 1)
        lod_idx_expanded_j = lod_idx_batch.unsqueeze(1)  # (bs, 1, token_seq_len)
        
        # Can attend if lod_idx[j] <= lod_idx[i]
        lod_mask = (lod_idx_expanded_j <= lod_idx_expanded_i)  # (bs, token_seq_len, token_seq_len)
        
        # Convert to attention mask format: 0 for can attend, -inf for cannot attend
        # For scaled_dot_product_attention, we use 0 for visible and -inf for masked
        lod_attention_mask = torch.where(
            lod_mask,
            torch.zeros_like(lod_mask, dtype=torch.float32),
            torch.full_like(lod_mask, float('-inf'), dtype=torch.float32)
        )
        
        # Add buffer positions: buffer tokens can see all tokens, all tokens can see buffer
        # Total sequence length = buffer_size + token_seq_len
        total_seq_len = self.buffer_size + token_seq_len
        full_attention_mask = torch.zeros(bs, total_seq_len, total_seq_len, device=device, dtype=torch.float32)
        
        # Buffer positions (0 to buffer_size-1): can see everything
        full_attention_mask[:, :self.buffer_size, :] = 0.0
        
        # All positions can see buffer
        full_attention_mask[:, :, :self.buffer_size] = 0.0
        
        # Token positions (buffer_size to total_seq_len-1): use LOD mask
        full_attention_mask[:, self.buffer_size:, self.buffer_size:] = lod_attention_mask

        full_attention_mask = full_attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # Random class label dropout during training
        if self.training:
            drop_latent_mask = torch.rand(bs) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(target_tokens.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        
        cond_embeddings = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        
        # Project input tokens
        token_embeddings = self.z_proj_ln(self.z_proj(input_tokens))

        # parent_token_embeddings = self.z_proj_ln(self.z_proj(parent_tokens))
        # Interleave position tokens and token embeddings
        z = torch.cat(
            (cond_embeddings, token_embeddings),
            dim=1
        )

        freqs_cis = self.freqs_cis[:total_seq_len]
        freqs_cis = freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1)
    
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                z = checkpoint(block, z, freqs_cis, None, None, use_reentrant=False)
        else:
            for block in self.blocks:
                z = block(z, freqs_cis, start_pos=None, mask=None)


         # add position aware for diffusion head

        z = self.out_norm(z)
        z = z + self.diffusion_pos_embed_learned[:, :total_seq_len]

        # Compute reconstruction loss only on valid (non-padding) positions
        diff_loss, loss_dict = self.forward_loss(z=z, target=target_tokens, padding_mask=torch.zeros(bs, total_seq_len, dtype=torch.bool))
        
        # Combine losses
        total_loss = diff_loss #+ expand_loss * self.config.losses.expand_loss_weight
        # loss_dict['expand_loss'] = expand_loss.mean().detach()
        # loss_dict['expand_acc'] = correct.mean().detach()
        loss_dict['total_loss'] = total_loss.mean().detach()
        loss_dict['z'] = z.detach() 
        
        return total_loss, loss_dict

    def forward_inference(self, 
                        x: torch.Tensor, 
                        freqs_cis: torch.Tensor, 
                        input_pos: torch.Tensor):
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.blocks:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.out_norm(h)
        return h

    def remove_caches(self):
        for l in self.blocks:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        """
        Generate tokens following quadtree structure with autoregressive generation.
        Input sequence: [cls, pos_emb[1,0]+feature[parent[1,0]], pos_emb[1,1]+feature[parent[1,1]], ...]
        Target sequence: [feature[0,0], feature[1,0], feature[1,1], ...]
        Each sample in the batch is generated independently.
        
        Args:
            condition: Class labels of shape (batch_size,)
            guidance_scale: CFG guidance scale
            guidance_decay: "constant" or "linear"
            randomize_temperature: Temperature for sampling
        
        Returns:
            result_tokens: Generated tokens of shape (batch_size, max_seq_len, token_embed_dim)
            tree_list: List of generated tree structures
        """
        bsz = condition.shape[0]
        device = condition.device
        
        # tree_root = build_probabilistic_quadtree(
        #     self.num_patch_side_list, 
        #     guaranteed_depth=3, 
        #     expansion_probs=[0.3, 0.3]
        # )
        # final_tree = tree_to_decision_nodes_dict(tree_root, 6)
        # tree_list = [copy.deepcopy(final_tree) for _ in range(bsz)]
        # max_seq_len = 0
        # for lod_idx in final_tree.keys():
        #     max_seq_len += len(final_tree[lod_idx])
        with open("/mnt/petrelfs/jianglihan/my_code/quadtok/fixed_quadtree_low.json", 'r') as f:
            tree_dict_json = json.load(f)
        final_tree_dict = {}
        lod_incides = []
        for lod_idx, node_dict in tree_dict_json['final_tree'].items():
            nodes = []
            for node_info in node_dict:
                node = QuadTreeNode(
                    patch_index=node_info['patch_index'],
                    lod_level=node_info['lod_level']
                )
                nodes.append(node)
                lod_incides.append(int(lod_idx))
            final_tree_dict[int(lod_idx)] = nodes
        lod_incides = torch.tensor(lod_incides, device=device, dtype=torch.long)
        tree_list = [copy.deepcopy(final_tree_dict) for _ in range(bsz)]

        max_seq_len = (torch.tensor(tree_dict_json['status_data']) == 1).sum().item()
        
        class_embedding = self.class_emb(condition)
        result_tokens = torch.zeros(bsz, max_seq_len, self.token_embed_dim, device=condition.device)

        if not guidance_scale == 1.0:
            class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
            bsz *= 2

        with torch.device(condition.device):
            self.setup_caches(max_batch_size=bsz, max_seq_length=max_seq_len, dtype=class_embedding.dtype)

        x = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        cur_freqs_cis = self.freqs_cis[:self.buffer_size].unsqueeze(0).repeat(bsz, 1, 1, 1)
        input_pos = torch.arange(0, x.shape[1], device=condition.device)
        cache_position = self.buffer_size

        decoded_token_num = 0

        for decoding_lod_idx in tqdm(range(lod_incides.max().item() + 1)):

            model_output = self.forward_inference(x, cur_freqs_cis, input_pos)
            decoding_token_num = (lod_incides == decoding_lod_idx).sum().item()

            model_output = model_output + self.diffusion_pos_embed_learned[:, decoded_token_num: decoded_token_num + decoding_token_num]
            # breakpoint()
            sampled_token_latent = self.diffloss.sample(model_output.flatten(0, 1), None, randomize_temperature, guidance_scale).view(bsz, decoding_token_num, -1)

            if not guidance_scale == 1.0:
                token_latent = sampled_token_latent[:bsz // 2]
            else:
                token_latent = sampled_token_latent

            # breakpoint()
            result_tokens[:, decoded_token_num: decoded_token_num + decoding_token_num] = token_latent.clone()

            if not guidance_scale == 1.0:
                token_latent = torch.cat([token_latent, token_latent], dim=0)

            token_latent = self.z_proj_ln(self.z_proj(token_latent))

            decoded_token_num += decoding_token_num

            next_lod_idx = decoding_lod_idx + 1
            if next_lod_idx > lod_incides.max().item():
                break

            next_lod_nodes = final_tree_dict[next_lod_idx]
            next_decoding_num = len(next_lod_nodes)
            
            # Build mapping: parent (lod, patch_idx) -> index in current_lod_nodes list
            current_lod_nodes = final_tree_dict[decoding_lod_idx]
            parent_key_to_idx = {
                (node.lod_level, node.patch_index): idx 
                for idx, node in enumerate(current_lod_nodes)
            }
            
            # Build ordered list of parent embeddings for next LOD tokens
            # Each child token uses its parent's embedding
            expanded_token_embeddings = []
            
            for child_node in next_lod_nodes:
                # Get parent of this child
                parent_key = _get_parent_patch_index(
                    child_node.lod_level,
                    child_node.patch_index,
                    self.num_patch_side_list
                )
                
                if parent_key is not None and parent_key in parent_key_to_idx:
                    # Get parent's position in current LOD
                    parent_idx_in_lod = parent_key_to_idx[parent_key]
                    # Get parent's embedding (already duplicated for CFG if needed)
                    parent_embedding = token_latent[:, parent_idx_in_lod:parent_idx_in_lod+1, :]  # (bsz_cfg, 1, embed_dim)
                    expanded_token_embeddings.append(parent_embedding)
                else:
                    # Should not happen in valid tree structure
                    raise ValueError(f"Child node {child_node.lod_level}, {child_node.patch_index} has no valid parent")
            
            x = torch.cat(expanded_token_embeddings, dim=1)  # (bsz_cfg, next_decoding_num, embed_dim)
            # Calculate positions for next LOD tokens
            next_positions = torch.arange(
                cache_position, 
                cache_position + next_decoding_num, 
                device=device
            )
            
            cur_freqs_cis = self.freqs_cis[next_positions].unsqueeze(0).repeat(bsz, 1, 1, 1)
            input_pos = next_positions
            cache_position += next_decoding_num

        self.remove_caches()
        return result_tokens, tree_list

class QuadtreeGPT(BaseModel):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_config = config.model.generator

        self.model_size = config.model.generator.model_size

        self.embed_dim = {
                "small": 768,
                "base": 1024,
                "large": 1536, # normal: 1280, wider: 1536
                "xlarge": 1280,  # LlamaGen-XL 775M
            }[self.model_size]

        self.depth = {
                "small": 12,
                "base": 24,
                "large": 24, # normal: 36, wider: 24
                "xlarge": 36,  # LlamaGen-XL
            }[self.model_size]

        self.num_heads = {
                "small": 12,
                "base": 16,
                "large": 16, # normal: 20, wider: 16
                "xlarge": 20,  # LlamaGen-XL
            }[self.model_size]
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = self.model_config.get("mlp_ratio", 1)  # config override (old drive ckpt=4); default 1 = base-update fix
        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = config.model.vq_model.token_size

        self.patch_size = config.model.generator.patch_size

        self.seq_len = 512 # maimum quadtree token number
        self.token_embed_dim = config.model.vq_model.token_size * (config.model.generator.patch_size**2)
        self.head_dim = self.embed_dim // self.num_heads
        self.grad_checkpointing = config.model.grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = config.model.generator.class_num
        self.cls_token_num = 1
        self.label_drop_prob = config.model.generator.label_drop_prob
        self.cls_embedding = LabelEmbedder(config.model.generator.class_num, self.embed_dim, self.label_drop_prob)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.tok_embeddings = nn.Embedding(config.model.vq_model.codebook_size, self.embed_dim)
        self.tok_dropout = nn.Dropout(config.model.generator.token_drop_prob)
    
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.generator.ffn_drop_prob, attn_dropout_p=config.model.generator.attn_dropout, resid_dropout_p=config.model.generator.resid_drop_prob) for _ in range(self.depth)])

        self.out_norm = RMSNorm(self.embed_dim, eps=1e-5)
        self.output = nn.Linear(self.embed_dim, config.model.vq_model.codebook_size, bias=False)


        # --------------------------------------------------------------------------
        # Quadtree position embeddings (learnable tokens for each LOD and patch index)
        self.num_patch_side_list = config.model.generator.num_patch_side_list
        self.num_lod = len(self.num_patch_side_list)
        
        # for lod_idx, num_patches in enumerate(self.num_patch_side_list):
        total_patches = self.num_patch_side_list[-1] ** 2
        self.token_indices_embedding = nn.Embedding(total_patches, self.embed_dim)
        self.lod_incides_embedding = nn.Embedding(self.num_lod, self.embed_dim)

        max_total_seq_len = 1 + self.seq_len * 2
        freqs_cis = precompute_freqs_cis(self.head_dim, max_total_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)


        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

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
    

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
    #     return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.blocks:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.num_heads, self.head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)


    def remove_caches(self):
        """Remove KV caches from all transformer blocks."""
        for l in self.blocks:
            l.attention.kv_cache = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def forward_inference(self, 
                        x: torch.Tensor, 
                        freqs_cis: torch.Tensor, 
                        input_pos: torch.Tensor):
        """
        Forward pass for inference with KV cache.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            freqs_cis: RoPE frequencies
            input_pos: Position indices for current tokens
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        bs = x.shape[0]
        mask = self.causal_mask[:bs, None, input_pos]
        h = x
        for layer in self.blocks:
            h = layer(h, freqs_cis, start_pos=input_pos, mask=mask)
        h = self.out_norm(h)
        logits = self.output(h)
        return logits

    def input_preprocess(self, input_tokens, target_tokens, tree_dict):
        valid_mask = target_tokens != -1
        input_tokens[input_tokens == -1] = 0
        
        return input_tokens, valid_mask
    
    def get_token_indices_embedding(self, tree_dict):

        lod_indices = tree_dict['lod_indices']  # (batch_size, max_seq_len)
        patch_indices = tree_dict['patch_indices']  # (batch_size, max_seq_len)
        
        batch_size, max_seq_len = lod_indices.shape
        device = lod_indices.device
        dtype = self.token_indices_embedding.weight.dtype
        
        valid_mask = lod_indices != -1  # (batch_size, max_seq_len)
        embeddings = torch.zeros(batch_size, max_seq_len, self.embed_dim, 
                                device=device, dtype=dtype)
        max_lod = self.num_lod - 1
        lookup_batch = torch.zeros(batch_size, max_lod + 1, 
                                  self.num_patch_side_list[-1] ** 2, 
                                  dtype=torch.long, device=device)
        lookup_batch.fill_(-1)
        
        b_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, max_seq_len)
        s_idx = torch.arange(max_seq_len, device=device).view(1, -1).expand(batch_size, -1)

        valid_b = b_idx[valid_mask]
        valid_lod = lod_indices[valid_mask]
        valid_patch = patch_indices[valid_mask]
        valid_s: Tensor = s_idx[valid_mask]

        lookup_batch[valid_b, valid_lod, valid_patch] = valid_s
        
        for lod_level in range(max_lod, -1, -1):
            lod_mask = (lod_indices == lod_level) & valid_mask
            
            if lod_level == max_lod:
                lod_patch_indices = patch_indices[lod_mask]  # (num_nodes_at_lod,)
                lod_embeddings = self.token_indices_embedding(lod_patch_indices)  # (num_nodes_at_lod, embed_dim)
                batch_indices, seq_indices = torch.where(lod_mask)
                embeddings[batch_indices, seq_indices] = lod_embeddings
            else:
                batch_indices, seq_indices = torch.where(lod_mask)
                num_nodes = len(batch_indices)

                parent_patch_indices = patch_indices[batch_indices, seq_indices]  # (num_nodes,)

                parent_patches_per_side = self.num_patch_side_list[lod_level]
                child_patches_per_side = self.num_patch_side_list[lod_level + 1]
                parent_rows = parent_patch_indices // parent_patches_per_side  # (num_nodes,)
                parent_cols = parent_patch_indices % parent_patches_per_side   # (num_nodes,)
                
                child_start_rows = parent_rows * 2  # (num_nodes,)
                child_start_cols = parent_cols * 2  # (num_nodes,)
                
                child_top_left = child_start_rows * child_patches_per_side + child_start_cols  # (num_nodes,)
                child_top_right = child_top_left + 1  # (num_nodes,)
                child_bottom_left = (child_start_rows + 1) * child_patches_per_side + child_start_cols  # (num_nodes,)
                child_bottom_right = child_bottom_left + 1  # (num_nodes,)
                
                child_patch_indices_all = torch.stack([
                    child_top_left, child_top_right, child_bottom_left, child_bottom_right
                ], dim=1)  # (num_nodes, 4)
                
                child_lod = lod_level + 1

                child_seq_indices = lookup_batch[batch_indices.unsqueeze(1).expand(-1, 4), child_lod, child_patch_indices_all]  # (num_nodes, 4)
                
                child_mask = child_seq_indices != -1  # (num_nodes, 4)
                batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, 4)  # (num_nodes, 4)
                
                safe_child_seq_indices = torch.clamp(child_seq_indices, min=0)  # (num_nodes, 4)
                child_embeddings_all = embeddings[batch_indices_expanded, safe_child_seq_indices]  # (num_nodes, 4, embed_dim)

                child_embeddings_masked = child_embeddings_all * child_mask.unsqueeze(-1)  # (num_nodes, 4, embed_dim)
                valid_counts = child_mask.sum(dim=1, keepdim=True)  # (num_nodes, 1)
                valid_counts = torch.clamp(valid_counts, min=1)  # Avoid division by zero
                parent_embeddings = child_embeddings_masked.sum(dim=1) / valid_counts  # (num_nodes, embed_dim)
                
                embeddings[batch_indices, seq_indices] = parent_embeddings
        lod_embeddings = self.lod_incides_embedding(lod_indices.clamp(0))
        return embeddings + lod_embeddings

    def forward(self, input_tokens, target_tokens, tree_dict, labels):
        """
        Forward pass for QuadtreeMAR.
        Args:
            input_tokens: Input tokens (z_quantized) of shape (batch_size, max_seq_len - 1, token_embed_dim)
               Padded positions have value -1.0
            target_tokens: Target tokens (z_quantized) of shape (batch_size, max_seq_len, token_embed_dim)
               Padded positions have value -1.0
            tree_dict: Dictionary containing:
                - 'status': Tensor of shape (batch_size, num_total_nodes)
                - 'lengths': Tensor of shape (batch_size,) with actual sequence lengths
                - 'tree': List of tree structures (one per batch item)
            labels: Class labels of shape (batch_size,)
        """
        bs, max_seq_len = target_tokens.shape
        device = target_tokens.device
        # get valid mask
        input_tokens, valid_mask = self.input_preprocess(input_tokens, target_tokens, tree_dict)

        cond_embeddings = self.cls_embedding(labels, train=self.training)[:,:self.cls_token_num]
        token_embeddings = self.tok_embeddings(input_tokens)
        token_indices_embedding = self.get_token_indices_embedding(tree_dict)

        token_embeddings = torch.cat((cond_embeddings, token_embeddings + token_indices_embedding[:, :-1]), dim=1)
        z = self.tok_dropout(token_embeddings)

        freqs_cis = self.freqs_cis[:max_seq_len]
        freqs_cis = freqs_cis.unsqueeze(0).repeat(bs, 1, 1, 1)
    
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.blocks:
                z = checkpoint(block, z, freqs_cis, None, None, use_reentrant=False)
        else:
            for block in self.blocks:
                z = block(z, freqs_cis, start_pos=None, mask=None)
         # add position aware for diffusion head

        z = self.out_norm(z)
        token_logits = self.output(z).float()

        total_loss = F.cross_entropy(token_logits[valid_mask].contiguous().float(), target_tokens[valid_mask].contiguous(), reduction="mean")

        loss_dict = {}

        pred_tokens = torch.argmax(token_logits, dim=-1)
        acc = (pred_tokens == target_tokens)[valid_mask].float().mean()

        loss_dict['total_loss'] = total_loss.mean().detach()
        loss_dict['token_logits'] = token_logits.detach()
        loss_dict['acc'] = acc.detach()
        return total_loss, loss_dict

    def forward_varlen(self, code_indices, lod_indices, patch_indices, labels,
                       cu_seqlens, seqlens, max_seqlen):
        """Varlen (packed, NO padding) training forward for flash_attn_varlen.
        Packed inputs (concatenation of N samples, T = sum of seqlens):
          code_indices/lod_indices/patch_indices : (T,) long
          cu_seqlens : (N+1,) int32   seqlens : (N,)   labels : (N,)   max_seqlen : int
        Per sample the AR sequence is [cls, tok(c0)+s0, ..., tok(c_{L-2})+s_{L-2}] predicting
        [c0, ..., c_{L-1}]. The parent-aware structural embedding reuses the existing padded
        get_token_indices_embedding (pad -> embed -> unpad); the transformer runs fully packed."""
        device = code_indices.device
        T = int(code_indices.shape[0])
        N = int(seqlens.shape[0])
        cu = cu_seqlens.to(torch.int64)
        seqlens = seqlens.to(torch.int64)
        sample_id = torch.repeat_interleave(torch.arange(N, device=device), seqlens)      # (T,)
        pos_in_sample = torch.arange(T, device=device) - cu[:-1][sample_id]                # (T,)

        # structural embedding: pad packed -> (N, max_seqlen), reuse padded path, unpad
        pad_lod = torch.full((N, max_seqlen), -1, dtype=torch.long, device=device)
        pad_patch = torch.zeros((N, max_seqlen), dtype=torch.long, device=device)
        pad_lod[sample_id, pos_in_sample] = lod_indices.long()
        pad_patch[sample_id, pos_in_sample] = patch_indices.long()
        struct_pad = self.get_token_indices_embedding(
            {"lod_indices": pad_lod, "patch_indices": pad_patch})                          # (N, max_seqlen, D)
        struct = struct_pad[pad_lod != -1]                                                 # (T, D), packed order

        # token embedding + AR input (cls prefix at sample starts, else previous token)
        tok = self.tok_embeddings(code_indices)                                            # (T, D)
        code_plus_struct = tok + struct
        shifted = torch.zeros_like(code_plus_struct)
        shifted[1:] = code_plus_struct[:-1]                                                # prev token (same sample by construction)
        cls_emb = self.cls_embedding(labels, train=self.training)[:, 0]                     # (N, D)
        is_first = torch.zeros(T, dtype=torch.bool, device=device)
        is_first[cu[:-1]] = True
        x = torch.where(is_first.unsqueeze(-1), cls_emb[sample_id], shifted)                # (T, D)
        x = self.tok_dropout(x)

        # per-sample RoPE + packed varlen transformer
        freqs_cis = self.freqs_cis[pos_in_sample]                                          # (T, head_dim//2, 2)
        cu32 = cu_seqlens.to(torch.int32)
        ms = int(max_seqlen)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block.forward_varlen, x, freqs_cis, cu32, ms, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block.forward_varlen(x, freqs_cis, cu32, ms)

        x = self.out_norm(x)
        logits = self.output(x).float()                                                    # (T, vocab)
        loss = F.cross_entropy(logits, code_indices.long(), reduction="mean")
        with torch.no_grad():
            acc = (logits.argmax(-1) == code_indices).float().mean()
        return loss, {"total_loss": loss.detach(), "acc": acc.detach()}

    def cfg_lod_scheduler(self, cfg_scale_base, step, current_lod):
        if current_lod <= 1:
            return cfg_scale_base 
        elif current_lod <= 3:
            start_w, end_w = cfg_scale_base, 1.2
            return max(end_w, start_w - (step - 21) * 0.02)
        else:
            return 1.1

    @torch.no_grad()
    def generate(self,
                 condition,
                 guidance_scale=3.0,
                 guidance_decay="linear",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        """
        Generate tokens following quadtree structure with autoregressive generation.
        Input sequence: [cls, pos_emb[1,0]+feature[parent[1,0]], pos_emb[1,1]+feature[parent[1,1]], ...]
        Target sequence: [feature[0,0], feature[1,0], feature[1,1], ...]
        Each sample in the batch is generated independently.
        
        Args:
            condition: Class labels of shape (batch_size,)
            guidance_scale: CFG guidance scale
            guidance_decay: "constant" or "linear"
            randomize_temperature: Temperature for sampling
        
        Returns:
            result_tokens: Generated tokens of shape (batch_size, max_seq_len, token_embed_dim)
            tree_list: List of generated tree structures
        """
        bsz = condition.shape[0]
        device = condition.device

        # Step 1: Build tree
        tree_root = build_probabilistic_quadtree(
            self.num_patch_side_list, 
            guaranteed_depth=3, 
            expansion_probs=[0.75]
        )
        final_tree = tree_to_decision_nodes_dict(tree_root, self.num_lod)
        lod_indices, patch_incides = [], []
        for lod_idx, nodes in final_tree.items():
            if lod_idx >= 3:
                for node in nodes:
                    lod_indices.append(node.lod_level)
                    patch_incides.append(node.patch_index)

        lod_indices = torch.tensor(lod_indices, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        patch_incides = torch.tensor(patch_incides, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        tree_dict = dict(lod_indices=lod_indices, patch_indices=patch_incides)
        max_seq_len = lod_indices.shape[1]

        result_tokens = torch.zeros(bsz, max_seq_len, device=device, dtype=torch.long)

        # Step2: Prepare Position Embeddings and CFG
        token_indices_embedding = self.get_token_indices_embedding(tree_dict)
        # permutation_indices = self.shuffle_tokens_within_lod(bsz, max_seq_len, tree_dict, device)
        # batch_indices = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, max_seq_len)

        # token_indices_embedding = token_indices_embedding[batch_indices, permutation_indices]

        if guidance_scale > 1.0:
            cond_null = torch.ones_like(condition) * self.num_classes
            cond_combined = torch.cat([condition, cond_null])
            bsz *= 2
        else:
            cond_combined = condition
        class_embedding = self.cls_embedding(cond_combined, train=False)

        with torch.device(condition.device):
            self.setup_caches(max_batch_size=bsz, max_seq_length=max_seq_len, dtype=class_embedding.dtype)

        x = class_embedding.repeat(1, self.cls_token_num, 1)
        cur_freqs_cis = self.freqs_cis[:self.cls_token_num].unsqueeze(0).repeat(bsz, 1, 1, 1)
        input_pos = torch.arange(0, x.shape[1], device=condition.device)
        cache_position = self.cls_token_num

        for step in range(max_seq_len):

            token_logitis = self.forward_inference(x, cur_freqs_cis, input_pos)

            if not guidance_scale == 1.0:
                if guidance_decay == "linear":
                    cfg_iter = 1 + (guidance_scale - 1) * (step) / max_seq_len
                elif guidance_decay == "constant":
                    cfg_iter = guidance_scale
                elif guidance_decay == "power-cosine":
                    scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                    scale_step = (1 - torch.cos(((step / max_seq_len) ** scale_pow) * torch.pi)) * 1/2
                    cfg_iter = (guidance_scale - 1) * scale_step + 1
                elif guidance_decay == "lod_scheduler":
                    cfg_iter = self.cfg_lod_scheduler(guidance_scale, step, lod_indices[:, step].float().mean())

            else:
                cfg_iter = guidance_scale
            if not guidance_scale == 1.0:
                cond_logits, uncond_logits = torch.chunk(token_logitis, 2, dim=0)
                logits = uncond_logits + cfg_iter * (cond_logits - uncond_logits)

                # std_c = cond_logits.std(dim=-1, keepdim=True)
                # std_cfg = logits.std(dim=-1, keepdim=True)
                # logits = logits * (std_c / std_cfg)
            else:
                logits = token_logitis
            entropy = calculate_entropy(logits)
            # breakpoint()
            # print(f"Step {step}, LOD: {lod_indices[:, step].float().mean()}, entropy: {entropy}")
            incides = sample(logits, randomize_temperature)[0]
            result_tokens[:, step] = incides.clone().squeeze(1)

            if step == max_seq_len - 1:
                break

            token_latent = self.tok_embeddings(incides)

            token_latent += token_indices_embedding[:, step].unsqueeze(1)
            if not guidance_scale == 1.0:
                token_latent = torch.cat([token_latent, token_latent], dim=0)

            x = token_latent

            cur_freqs_cis = self.freqs_cis[cache_position:cache_position + 1].unsqueeze(0).repeat(bsz, 1, 1, 1)
            input_pos = torch.arange(cache_position, cache_position + 1, device=condition.device)
            cache_position += 1
        self.remove_caches()
        # reverse_permutation = torch.argsort(permutation_indices, dim=1, stable=True)  # (original_bsz, max_seq_len)
        
        # batch_indices_reverse = torch.arange(condition.shape[0], device=device).unsqueeze(1).expand(-1, max_seq_len)
        # result_tokens_original = result_tokens[batch_indices_reverse, reverse_permutation]

        ori_ordered_nodes = self._get_ordered_nodes(tree_root)
        ordered_nodes = []
        for node in ori_ordered_nodes:
            if node.lod_level >= 3:
                ordered_nodes.append(node)

        return result_tokens, ordered_nodes