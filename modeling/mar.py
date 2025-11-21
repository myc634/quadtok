from functools import partial
from typing import Any

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from modeling.modules.base_model import BaseModel
from modeling.modules.attention import TransformerBlock, precompute_freqs_cis, batch_apply_rotary_emb, precompute_freqs_cis_2d, find_multiple, KVCache

from timm.models.vision_transformer import Block

from modeling.modules.losses import DiffLoss
from modeling.utils import build_quadtree
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
        mlp_ratio = 4

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
        mlp_ratio = 4

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
        mlp_ratio = 4

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
        self.token_indices_embedding_dict = nn.ModuleDict()
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            self.token_indices_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.embed_dim)
        
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

        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

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

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        
        # Initialize quadtree position embeddings
        for lod_idx_str, embedding_layer in self.token_indices_embedding_dict.items():
            torch.nn.init.normal_(embedding_layer.weight, std=.02)

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
        loss, loss_dict = self.diffloss(z=z, target=target, mask=None)
        return loss, loss_dict

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
        # Extract tree information
        status = tree_dict['status']
        lengths = tree_dict['lengths']
        
        # class embed
        class_embedding = self.class_emb(labels)

        bs, max_seq_len, in_dim = target_tokens.shape
        device = target_tokens.device
        
        # Create padding mask: True for padding positions, False for valid positions
        # Shape: (batch_size, max_seq_len)
        padding_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Get position instruction tokens based on status and ordered_full_nodes
        # Padding positions will have zero embeddings
        position_instruction_tokens = self.get_position_instruction_tokens_from_status(
            status, lengths
        )
        
        # Random class label dropout during training
        if self.training:
            drop_latent_mask = torch.rand(bs) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(target_tokens.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        
        cond_embeddings = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        
        # Project input tokens
        token_embeddings = self.z_proj_ln(self.z_proj(input_tokens))

        # Interleave position tokens and token embeddings
        z = torch.cat(
            (cond_embeddings, position_instruction_tokens[:, 1:] + token_embeddings),
            dim=1
        )

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
        
        z = z + position_instruction_tokens + self.diffusion_pos_embed_learned[:, :z.shape[1]]
        # expand_logits = self.expand_pred_head(z).squeeze(-1)  # (bs, max_seq_len)
        
        # gt_expand_labels, gt_expand_mask = self.build_expand_labels_from_status(
        #     status, lengths, max_seq_len, device
        # )

        # valid_mask = gt_expand_mask & (~padding_mask)  # Valid and not padding

        

        # Compute reconstruction loss only on valid (non-padding) positions
        diff_loss, loss_dict = self.forward_loss(z=z, target=target_tokens, padding_mask=padding_mask)
        
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
        
        # Build parent-child relationship mapping (shared across all samples)
        num_nodes = len(self.ordered_full_nodes)
        child_to_parent = {}
        for parent_idx, parent_node in enumerate(self.ordered_full_nodes):
            for child in parent_node.children:
                child_key = (child.lod_level, child.patch_index)
                if child_key in self.node_to_idx_map:
                    child_idx = self.node_to_idx_map[child_key]
                    child_to_parent[child_idx] = parent_idx
        
        # Find root node index (lod_level=0, patch_index=0)
        root_node_idx = None
        for node_idx, node in enumerate(self.ordered_full_nodes):
            if node.lod_level == 0 and node.patch_index == 0:
                root_node_idx = node_idx
                break
        assert root_node_idx is not None, "Root node not found"
        
        # Initialize result storage
        result_tokens_list = []  # List of tensors, one per sample
        tree_list = []
        
        # Generate for each sample independently
        for b in range(bsz):
            # Get class label for this sample
            sample_class_label = condition[b:b+1]
            
            # Class embedding
            class_embedding = self.class_emb(sample_class_label)
            cond_embeddings = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
            
            # Prepare for CFG
            if guidance_scale != 1.0:
                cond_embeddings_cfg = torch.cat([
                    cond_embeddings, 
                    self.fake_latent.unsqueeze(0).repeat(1, self.buffer_size, 1)
                ], dim=0)
                effective_bsz = 2
            else:
                cond_embeddings_cfg = cond_embeddings
                effective_bsz = 1
            
            # Initialize result storage for this sample
            sample_tokens = torch.zeros(self.seq_len, self.token_embed_dim, device=device)
            node_tokens = {}  # Map node_idx -> token feature
            
            # Initialize status tensor for this sample
            sample_status = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            
            # Mark guaranteed nodes (lod <= guaranteed_depth - 1) as "will expand"
            for node_idx, node in enumerate(self.ordered_full_nodes):
                if node.lod_level <= self.guaranteed_depth - 1:
                    sample_status[node_idx] = 1
            
            # Setup KV cache for this sample
            # Sequence length: buffer_size + (max_seq_len - 1) for input sequence
            max_seq_len = self.buffer_size + (self.seq_len - 1)
            with torch.device(device):
                self.setup_caches(max_batch_size=effective_bsz, max_seq_length=max_seq_len, 
                                dtype=cond_embeddings_cfg.dtype)
            
            # Initialize: pass class embedding through model to initialize KV cache
            input_pos_init = torch.arange(0, self.buffer_size, device=device)
            freqs_cis_init = self.freqs_cis[:self.buffer_size].unsqueeze(0).repeat(effective_bsz, 1, 1, 1)
            z_class = self.forward_inference(cond_embeddings_cfg, freqs_cis_init, input_pos_init)
            
            z_root_output = z_class[:, -1]  # (effective_bsz, embed_dim) - use last class token position
            
            # Sample root token
            if guidance_scale != 1.0:
                cfg_iter = guidance_scale if guidance_decay == "constant" else 1.0
                sampled_root = self.diffloss.sample(z_root_output, randomize_temperature, cfg_iter)
                assert sampled_root is not None
                root_token = sampled_root[0]  # Take conditional sample
            else:
                sampled_root = self.diffloss.sample(z_root_output, randomize_temperature, 1.0)
                assert sampled_root is not None
                root_token = sampled_root[0]
            
            # Store root token (target_tokens[0])
            sample_tokens[0] = root_token
            node_tokens[root_node_idx] = root_token
            # Root node always expands (guaranteed_depth >= 0)
            sample_status[root_node_idx] = 1
            
            # Project root token to embedding space for use as parent feature
            root_token_embedding = self.z_proj_ln(self.z_proj(root_token.unsqueeze(0).unsqueeze(0)))  # (1, 1, embed_dim)
            if guidance_scale != 1.0:
                root_token_embedding_cfg = root_token_embedding.repeat(2, 1, 1)
            else:
                root_token_embedding_cfg = root_token_embedding
            
            # Track sequence position in target sequence (starts from 1, since root is at position 0)
            seq_position = 1
            
            sample_tree = {}
            
            # Generate tokens LOD by LOD (starting from lod=1, since root is lod=0)
            for lod_idx in range(1, self.num_lod):
                # Get all nodes at current LOD
                lod_nodes = [(node_idx, node) for node_idx, node in enumerate(self.ordered_full_nodes) 
                            if node.lod_level == lod_idx]
                if not lod_nodes:
                    continue
                
                used_nodes = []
                
                # Generate tokens for each node at this LOD
                for node_idx, node in lod_nodes:
                    # Check if should generate (guaranteed or parent expanded)
                    if lod_idx > self.guaranteed_depth:
                        parent_idx = child_to_parent[node_idx]
                        if sample_status[parent_idx] != 1:
                            continue  # Skip if parent didn't expand
                    
                    used_nodes.append(node)
                    
                    # Get parent node and its feature
                    parent_idx = child_to_parent[node_idx]
                    parent_token = node_tokens[parent_idx]
                    # Project parent token to embedding space
                    parent_token_embedding = self.z_proj_ln(self.z_proj(parent_token.unsqueeze(0).unsqueeze(0)))  # (1, 1, embed_dim)
                    if guidance_scale != 1.0:
                        parent_token_embedding = parent_token_embedding.repeat(2, 1, 1)
                    
                    # Get position embedding for this node
                    node_lod = node.lod_level
                    node_patch_idx = node.patch_index
                    position_embedding = self.token_indices_embedding_dict[str(node_lod)](
                        torch.tensor([node_patch_idx], device=device, dtype=torch.long)
                    ).unsqueeze(0)  # (1, 1, embed_dim)
                    
                    if guidance_scale != 1.0:
                        position_embedding_cfg = position_embedding.repeat(2, 1, 1)
                    else:
                        position_embedding_cfg = position_embedding
                    
                    # Prepare input: pos_emb + feature[parent]
                    # Input sequence position: buffer_size + (seq_position - 1)
                    # seq_position starts from 1 (root is at position 0 in target sequence)
                    # So input sequence position is buffer_size + (seq_position - 1)
                    input_seq_pos = self.buffer_size + (seq_position - 1)
                    
                    # Input is position embedding + parent feature
                    current_input = position_embedding_cfg + parent_token_embedding
                    
                    input_pos = torch.tensor([input_seq_pos], device=device)
                    freqs_cis_current = self.freqs_cis[input_pos].unsqueeze(0).repeat(effective_bsz, 1, 1, 1)
                    
                    # Forward through model
                    z = self.forward_inference(current_input, freqs_cis_current, input_pos)
                    z_output = z[:, -1]  # (effective_bsz, embed_dim)
                    
                    # Sample token with CFG
                    if guidance_scale != 1.0:
                        if guidance_decay == "linear":
                            cfg_iter = 1.0 + (guidance_scale - 1.0) * (1.0 - seq_position / self.seq_len)
                        else:
                            cfg_iter = guidance_scale
                        sampled_token_latent = self.diffloss.sample(z_output, randomize_temperature, cfg_iter)
                        assert sampled_token_latent is not None
                        token_latent = sampled_token_latent[0]  # Take conditional sample
                    else:
                        sampled_token = self.diffloss.sample(z_output, randomize_temperature, 1.0)
                        assert sampled_token is not None
                        token_latent = sampled_token[0]
                    
                    # Store token (target_tokens[seq_position])
                    sample_tokens[seq_position] = token_latent
                    node_tokens[node_idx] = token_latent
                    
                    # Mark node status and predict expand if needed
                    if lod_idx < self.guaranteed_depth:
                        # Guaranteed to expand
                        sample_status[node_idx] = 1
                    else:
                        # Need to predict expand
                        expand_logit = self.expand_pred_head(z_output[0:1].unsqueeze(0))  # (1, 1, 1)
                        expand_prob = torch.sigmoid(expand_logit.squeeze())
                        should_expand = torch.bernoulli(expand_prob).bool()
                        sample_status[node_idx] = 1 if should_expand else 0
                    
                    # Increment sequence position
                    seq_position += 1
                    
                    if seq_position >= self.seq_len:
                        break
                
                if seq_position >= self.seq_len:
                    break
                
                sample_tree[lod_idx] = used_nodes
            
            # Clean up KV cache for this sample
            self.remove_caches()
            
            # Truncate to actual generated length and store
            result_tokens_list.append(sample_tokens[:seq_position])
            tree_list.append(sample_tree)
        
        # Stack results - pad to max length in batch
        max_len = max(tokens.shape[0] for tokens in result_tokens_list) if result_tokens_list else 0
        result_tokens = torch.zeros(bsz, max_len, self.token_embed_dim, device=device)
        
        for b in range(bsz):
            seq_len = result_tokens_list[b].shape[0]
            result_tokens[b, :seq_len] = result_tokens_list[b]

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
        mlp_ratio = 4

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
        self.token_embedding = nn.Embedding(config.model.vq_model.codebook_size, self.embed_dim)
        self.buffer_size = config.model.generator.buffer_size
    
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, ffn_dim_multiplier=mlp_ratio,
                  ffn_dropout_p=config.model.generator.proj_dropout, attn_dropout_p=config.model.generator.attn_dropout) for _ in range(self.depth)])

        self.out_norm = norm_layer(self.embed_dim)
        self.output = nn.Linear(self.embed_dim, config.model.vq_model.codebook_size, bias=False)
        
        # --------------------------------------------------------------------------
        # Binary classification head for predicting whether to expand children
        # self.expand_pred_head = nn.Linear(self.embed_dim, 1, bias=True)

        # --------------------------------------------------------------------------
        # Quadtree position embeddings (learnable tokens for each LOD and patch index)
        self.num_patch_side_list = config.model.generator.num_patch_side_list
        self.num_lod = len(self.num_patch_side_list)
        
        # Create learnable embeddings for each LOD level and patch index
        self.token_indices_embedding_dict = nn.ModuleDict()
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            self.token_indices_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.embed_dim)
        
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


        self.initialize_weights()
        

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

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.token_embedding.weight, std=.02)
        
        # Initialize quadtree position embeddings
        for lod_idx_str, embedding_layer in self.token_indices_embedding_dict.items():
            torch.nn.init.normal_(embedding_layer.weight, std=.02)

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
        loss, loss_dict = self.diffloss(z=z, target=target, mask=None)
        return loss, loss_dict

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
        # Extract tree information
        status = tree_dict['status']
        lengths = tree_dict['lengths']

        # class embed
        class_embedding = self.class_emb(labels)

        bs, max_seq_len = target_tokens.shape
        device = target_tokens.device
        
        # Create padding mask: True for padding positions, False for valid positions
        # Shape: (batch_size, max_seq_len)
        padding_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Get position instruction tokens based on status and ordered_full_nodes
        # Padding positions will have zero embeddings
        position_instruction_tokens = self.get_position_instruction_tokens_from_status(
            status, lengths
        )
        
        # Random class label dropout during training
        if self.training:
            drop_latent_mask = torch.rand(bs) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(target_tokens.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        
        cond_embeddings = class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1)
        
        valid_mask = input_tokens >= 0  # (B, N)
        safe_indices = torch.where(valid_mask, input_tokens, torch.zeros_like(input_tokens))
        token_embeddings = self.token_embedding(safe_indices)  # (B, N, C
        token_embeddings = token_embeddings * valid_mask.unsqueeze(-1)  # (B, N, C)

        z = torch.cat(
            (cond_embeddings, position_instruction_tokens[:, 1:] + token_embeddings),
            dim=1
        )

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
        token_logits = self.output(z)

        valid_mask = target_tokens >= 0

        total_loss = F.cross_entropy(token_logits[valid_mask], target_tokens[valid_mask], reduction="mean")
        loss_dict = {}
        # Combine losses

        loss_dict['total_loss'] = total_loss.mean().detach()
        loss_dict['token_logits'] = token_logits.detach() 
        return total_loss, loss_dict