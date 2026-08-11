"""This file contains the model definition of TiTok.

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
"""

from typing import Any
import torch
import torch.nn as nn
from einops import rearrange
import copy
import random
from modeling.modules.base_model import BaseModel
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder, QuadTokEncoder, QuadTokDecoder, QuadTokSelctor, ResidualAttentionBlock
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
import json
from omegaconf import OmegaConf
from pathlib import Path
import time
from modeling.utils import build_quadtree, build_random_quadtree, build_probabilistic_quadtree
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class QuadTok(BaseModel):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        # This should be False for stage1 and True for stage2.
        self.quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
        self.encoder = QuadTokEncoder(config)
        self.decoder = QuadTokDecoder(config)
        self.selector = QuadTokSelctor(config)

        self.num_lod = len(config.model.selector.num_patch_side_list)
        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.patch_size_list = config.model.selector.patch_size_list
        # Per-LOD expansion probabilities for the training-time probabilistic quadtree.
        # 2-level uses [p] (lod3->lod4); 3-level uses [p0, p1] (lod3->lod4, lod4->lod5).
        self.expansion_probs = list(config.model.selector.get("expansion_probs", [0.75]))
        self.guaranteed_depth = config.model.selector.get("guaranteed_depth", 3)

        self.repa_param = config.losses.get("repa_param", None)

        self.train_policy = config.model.get("train_policy", False)
        if not self.train_policy:
            self.apply(self._init_weights)

        if self.quantize_mode == "vq":
            self.quantize = VectorQuantizer(
                codebook_size=config.model.vq_model.codebook_size,
                token_size=config.model.vq_model.token_size,
                commitment_cost=config.model.vq_model.commitment_cost,
                use_l2_norm=config.model.vq_model.use_l2_norm,)
        elif self.quantize_mode == "vae":
            self.quantize = DiagonalGaussianDistribution
        else:
            raise NotImplementedError
        
        
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def encode(self, x):
        latent_feats = self.encoder(pixel_values=x)
        return latent_feats
    
    def decode(self, z_quantized, tree_structure):
        decoded = self.decoder(z_quantized, tree_structure)
        return decoded

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

    def _forward_reconstruction(self, x):
        latent_feats = self.encode(x)
        # 2-level: expansion_probs=[p] (lod3->lod4). 3-level: [p0, p1] (lod3->lod4, lod4->lod5).
        tree_structure = build_probabilistic_quadtree(
            self.num_patch_side_list, guaranteed_depth=self.guaranteed_depth,
            expansion_probs=self.expansion_probs)
        ori_ordered_nodes = self._get_ordered_nodes(tree_structure)
        ordered_nodes = []
        for node in ori_ordered_nodes:
            if node.lod_level >= 3:
                ordered_nodes.append(node)
        if self.repa_param is not None: # 
            z, zs = self.selector(latent_feats, ordered_nodes)
        else:
            z = self.selector(latent_feats, ordered_nodes)
            zs = None

        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            result_dict = dict(zs=zs)
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict["posteriors"] = posteriors
        decoded = self.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), ordered_nodes)
        
        return decoded, result_dict

    def decoding_pre_selector(self, latent_feats, action_dict, attn_padding_mask):
        z = self.selector._forward_policy(latent_feats, attn_padding_mask)
        if self.quantize_mode == "vq":
            z_quantized, _ = self.quantize(z)
        elif self.quantize_mode == "vae":
            z_quantized = self.quantize(z).sample()
        decoded = self.decoder._forward_policy(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), action_dict, attn_padding_mask)

        return decoded

    def _forward_policy(self, x, policy_output):
        latent_feats = self.encode(x)
        policy_output = self.selector(latent_feats, tree_structure=None, policy_output=policy_output) # 0.5s
        
        result_dict = dict(policy_output=policy_output)
        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(policy_output['z_to_quantize'])
            z_quantized = posteriors.sample()
            result_dict["posteriors"] = posteriors
        decoded = self.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), policy_output) # 0.7s
        
        return decoded, result_dict, 
    
    def forward(self, x, policy_output=None):
        return self._forward_reconstruction(x)



class PolicyQuadTok(BaseModel):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config

        self.model_size = config.policy_model.model_scale
        self.token_size = config.model.selector.token_size
        self.train_policy = config.model.get("train_policy", False)

        self.num_lod = len(config.model.selector.num_patch_side_list)
        self.num_patch_side_list = config.model.selector.num_patch_side_list
        self.patch_size_list = config.model.selector.patch_size_list

        self.guaranteed_depth = config.policy_model.get("guaranteed_depth", 3)

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

        self.policy_token_incides_embedding_dict = nn.ModuleDict()
        self.latent_seq_len = 256
        self.max_seq_len = self.latent_seq_len
        self.lod_start_indices = {}
        self.lod_node_counts = {} 

        self.full_tree_root = build_quadtree(self.num_patch_side_list)
        ordered_full_nodes = self._get_ordered_nodes(self.full_tree_root)
        self.ordered_full_nodes = ordered_full_nodes


        lod_len_mapping = defaultdict(int)
        for node in self.ordered_full_nodes:
            lod_len_mapping[node.lod_level] += 1

        total_nodes_count = 0
        for lod_idx, num_patches in enumerate(self.num_patch_side_list):
            total_patches = num_patches ** 2
            num_nodes_at_lod = lod_len_mapping[lod_idx] 
            self.max_seq_len += total_patches
            if lod_idx != len(self.num_patch_side_list):
                self.policy_token_incides_embedding_dict[str(lod_idx)] = nn.Embedding(total_patches, self.width)
            self.lod_start_indices[lod_idx] = total_nodes_count
            self.lod_node_counts[lod_idx] = num_nodes_at_lod
            total_nodes_count += total_patches

        scale = self.width ** -0.5
        self.policy_latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.max_seq_len, self.width))

        self.policy_ln_pre = nn.LayerNorm(self.width)
        self.policy_transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.policy_transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.policy_net = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, 1)) 
        
        node_to_idx_map = {
            (node.lod_level, node.patch_index): i 
            for i, node in enumerate(ordered_full_nodes)
        }
        self.num_total_nodes = len(ordered_full_nodes)
        parent_indices_list = [-1] * self.num_total_nodes
        lod_list = [0] * self.num_total_nodes
        patch_index_list = [0] * self.num_total_nodes
        self.parent_bfs_to_child_patch_map = defaultdict(list)

        for parent_idx, parent_node in enumerate(ordered_full_nodes):
            lod_list[parent_idx] = parent_node.lod_level
            patch_index_list[parent_idx] = parent_node.patch_index
            for child_node in parent_node.children:
                child_idx = node_to_idx_map.get((child_node.lod_level, child_node.patch_index))
                self.parent_bfs_to_child_patch_map[parent_idx].append(child_node.patch_index)
                if child_idx is not None:
                    parent_indices_list[child_idx] = parent_idx

        parent_indices = torch.tensor(parent_indices_list, dtype=torch.long)
        full_tree_lods = torch.tensor(lod_list, dtype=torch.long)
        full_tree_indices = torch.tensor(patch_index_list, dtype=torch.long)

        self.register_buffer('parent_indices', parent_indices)
        self.register_buffer('full_tree_lods', full_tree_lods)
        self.register_buffer('full_tree_indices', full_tree_indices)

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
        policy_lod_embeddings = []
        for node in self.ordered_full_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            if str(lod_idx) in self.policy_token_incides_embedding_dict.keys():
                index_tensor = torch.tensor([index], dtype=torch.long, device=device)
                embedding = self.policy_token_incides_embedding_dict[str(lod_idx)](index_tensor)
                policy_lod_embeddings.append(embedding)
        
        # (num_total_nodes, D)
        policy_flat_token_sequence = torch.cat(policy_lod_embeddings, dim=0)
        # (B, num_total_nodes, D)
        return policy_flat_token_sequence.unsqueeze(0).repeat(batch_size, 1, 1)

    @torch.no_grad()
    def rollout_trajs(self, image_latents_grouped):

        batch_size, latent_seq_len, _ = image_latents_grouped.shape
        device = image_latents_grouped.device
        max_depth = self.num_lod
        trajectories = [defaultdict(list) for _ in range(batch_size)]

        current_kept_mask = torch.zeros(batch_size, self.num_total_nodes, dtype=torch.bool, device=device)
        current_kept_mask[:, 0] = True 

        image_latents_with_pos = image_latents_grouped + \
            self.policy_latent_token_positional_embedding[:latent_seq_len]
        all_node_embeddings = self._get_all_node_embeddings(batch_size, device)
        all_node_embeddings = all_node_embeddings + \
            self.policy_latent_token_positional_embedding[latent_seq_len : latent_seq_len + self.num_total_nodes]
        
        # we need to first fill the input with the guaranteed nodes
        num_tree_nodes = 0
        for lod_idx in range(self.guaranteed_depth):
            num_tree_nodes += self.lod_node_counts[lod_idx]
        guaranteed_node_embeddings = all_node_embeddings[:, :num_tree_nodes, :]
        current_transformer_input = torch.cat([image_latents_with_pos, guaranteed_node_embeddings], dim=1)
        # now we need to fill the padding mask with the guaranteed nodes
        current_padding_mask = torch.cat([torch.zeros(batch_size, latent_seq_len, dtype=torch.bool, device=device), torch.zeros(batch_size, num_tree_nodes + self.lod_node_counts[self.guaranteed_depth], dtype=torch.bool, device=device)], dim=1)
        current_kept_mask[:, :num_tree_nodes + self.lod_node_counts[self.guaranteed_depth]] = True

        action_logprobs = torch.ones_like(current_kept_mask) * -10000.0
        action_dict = {}
        for lod_idx in range(self.guaranteed_depth, max_depth - 1):
            lod_start_idx = self.lod_start_indices[lod_idx]
            num_nodes_at_lod: Any = self.lod_node_counts[lod_idx]
            
            if num_nodes_at_lod == 0:
                continue
                
            node_indices_at_lod = torch.arange(
                lod_start_idx, lod_start_idx + num_nodes_at_lod, device=device
            )
            
            node_embeddings_at_lod = all_node_embeddings[:, node_indices_at_lod, :]
            state_sequence = torch.cat([current_transformer_input, node_embeddings_at_lod], dim=1)

            policy_input = self.policy_ln_pre(state_sequence)
            policy_input = policy_input.permute(1, 0, 2)
            for layer in self.policy_transformer:
                policy_input = layer(policy_input, key_padding_mask=current_padding_mask)
            policy_input = policy_input.permute(1, 0, 2)

            node_features_at_lod = policy_input[:, -num_nodes_at_lod:, :]
            logits = self.policy_net(node_features_at_lod)
            probs = torch.sigmoid(logits)

            parent_indices_at_lod = self.parent_indices[node_indices_at_lod]
            clamped_parent_indices = torch.clamp(parent_indices_at_lod, min=0)
            parents_kept_mask = torch.gather(
                current_kept_mask, 1, clamped_parent_indices.expand(batch_size, -1)
            )
            valid_mask = parents_kept_mask.unsqueeze(-1).float()
            
            dist = torch.distributions.Bernoulli(probs=probs)
            actions_t = dist.sample() 
            actions_t = actions_t * valid_mask 

            log_probs_t_vec = dist.log_prob(actions_t)
            log_probs_t_masked = log_probs_t_vec * valid_mask

            action_dict[lod_idx] = dict(actions=actions_t.squeeze(-1), log_probs=log_probs_t_masked.squeeze(-1))

            current_kept_mask[:, node_indices_at_lod] = (actions_t.squeeze(-1).bool())
            current_transformer_input = state_sequence.detach()
            # actions need to be expeanded for next lod accroding to the parent indices
            child2parent_incides = self.parent_indices[torch.arange(
                self.lod_start_indices[lod_idx + 1], self.lod_start_indices[lod_idx + 1] + self.lod_node_counts[lod_idx + 1], device=device
            )].unsqueeze(0).repeat(batch_size, 1) - lod_start_idx

            child_actions_t = torch.gather(actions_t.squeeze(-1), 1, child2parent_incides) 
            current_padding_mask = torch.cat([current_padding_mask, ~child_actions_t.bool()], dim=1)

        return action_dict, current_padding_mask[:, latent_seq_len:]

    def forward(self, latent_feats, action_dict):

        batch_size = latent_feats.shape[0]
        device = latent_feats.device
        latent_seq_len = latent_feats.shape[1]

        num_guaranteed_nodes, all_seq_len = 0, 0
        for lod_idx in range(max(action_dict.keys()) + 1):
            if lod_idx < self.guaranteed_depth:
                num_guaranteed_nodes += self.lod_node_counts[lod_idx]
            all_seq_len += self.lod_node_counts[lod_idx]

        all_node_embeddings = self._get_all_node_embeddings(batch_size, device)
        all_node_embeddings = all_node_embeddings + \
            self.policy_latent_token_positional_embedding[self.latent_seq_len : self.latent_seq_len + self.num_total_nodes]
        latent_feats_with_pos = latent_feats + \
            self.policy_latent_token_positional_embedding[:latent_seq_len]

        sequence_parts = [latent_feats_with_pos]
        sequence_parts.append(all_node_embeddings[:, :num_guaranteed_nodes, :])

        padding_mask = torch.zeros(batch_size, all_seq_len + latent_seq_len, dtype=torch.bool, device=device)
        padding_mask[:, :num_guaranteed_nodes + latent_seq_len] = True

        for lod_idx in sorted(action_dict.keys()):
            if lod_idx < self.guaranteed_depth:
                continue
            lod_start_idx = self.lod_start_indices[lod_idx]
            num_nodes_at_lod = self.lod_node_counts[lod_idx]
            
            if num_nodes_at_lod == 0:
                continue
            
            node_indices = torch.arange(lod_start_idx, lod_start_idx + num_nodes_at_lod, device=device)
            actions = action_dict[lod_idx]['actions']  # (batch_size, num_nodes_at_lod)
            padding_mask[:, node_indices] = actions.bool()
            
            # Add this lod's nodes to sequence
            sequence_parts.append(all_node_embeddings[:, node_indices, :])

        
        # Concatenate sequence and padding mask
        x = torch.cat(sequence_parts, dim=1)
        total_seq_len = x.shape[1]

        attention_mask = torch.zeros(total_seq_len, total_seq_len, dtype=torch.bool, device=device)
        pos_to_lod = torch.zeros(total_seq_len, dtype=torch.long, device=device)
        pos_to_lod[:latent_seq_len] = -1
        
        # Guaranteed nodes
        current_pos = latent_seq_len
        for lod_idx in range(max(action_dict.keys()) + 1):
            num_nodes = self.lod_node_counts[lod_idx]
            pos_to_lod[current_pos:current_pos + num_nodes] = lod_idx
            current_pos += num_nodes


        i_indices = torch.arange(total_seq_len, device=device).view(-1, 1)
        j_indices = torch.arange(total_seq_len, device=device).view(1, -1)
        lod_i = pos_to_lod.view(-1, 1)
        lod_j = pos_to_lod.view(1, -1)
        attention_mask = (
            (j_indices > i_indices) & 
            (lod_i != -1) & 
            (lod_j != -1) & 
            (lod_j > lod_i)
        )
        
        # Forward through transformer
        x = self.policy_ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for layer in self.policy_transformer:
            x = layer(x, key_padding_mask=padding_mask, attention_mask=attention_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_node_logits = x[:, latent_seq_len:, :]
        logits = self.policy_net(x_node_logits)

        return logits[:, num_guaranteed_nodes:].squeeze(-1), padding_mask[:, latent_seq_len + num_guaranteed_nodes:]


    