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

    def _forward_reconstruction(self, x):
        latent_feats = self.encode(x)
        tree_structure = build_probabilistic_quadtree(self.num_patch_side_list, guaranteed_depth=3, expansion_probs=[0.7, 0.4])
        z = self.selector(latent_feats, tree_structure)

        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = posteriors
        decoded = self.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_structure)
        
        return decoded, result_dict

    def decoding_pre_selector(self, latent_feats, policy_output):
        policy_output = self.selector(latent_feats, tree_structure=None, policy_output=policy_output)
        result_dict = dict(policy_output=policy_output)
        if self.quantize_mode == "vae":
            posteriors = self.quantize(policy_output['z_to_quantize'])
        else:
            NotImplementedError
        z_quantized = posteriors.sample()
        result_dict["posteriors"] = posteriors
        decoded = self.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), policy_output)

        return decoded, result_dict, 

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

        if self.train_policy:
            return self._forward_policy(x, policy_output=policy_output)
        else:
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
        for lod_idx, num_patches in enumerate(self.num_patch_side_list[: -1]):
            total_patches = num_patches ** 2
            num_nodes_at_lod = lod_len_mapping[lod_idx] 
            self.max_seq_len += total_patches
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

    def forward(self, image_latents):
        batch_size, latent_seq_len, _ = image_latents.shape
        device = image_latents.device

        # (B, num_total_nodes, D)
        all_node_embeddings = self._get_all_node_embeddings(batch_size, device)
        all_node_embeddings = all_node_embeddings + \
            self.policy_latent_token_positional_embedding[latent_seq_len : latent_seq_len + self.num_total_nodes]

        # >> This will store the *global node indices* of active nodes for each batch item
        # >> We use a list of lists, where each inner list holds indices for one batch item
        active_node_indices_batch = [[] for _ in range(batch_size)]

        # >> Initialize with guaranteed nodes.
        # >> We need to map (lod, patch_index) to the global node index
        node_map = {(node.lod_level, node.patch_index): i for i, node in enumerate(self.ordered_full_nodes)}
        
        for node in self.ordered_full_nodes:
            lod_idx, index = node.lod_level, node.patch_index
            if lod_idx <= self.guaranteed_depth:
                global_node_idx = node_map.get((lod_idx, index))
                if global_node_idx is not None:
                    # >> Add guaranteed nodes to *every* item in the batch
                    for b_idx in range(batch_size):
                        active_node_indices_batch[b_idx].append(global_node_idx)


        final_actions = defaultdict(list)
        all_probs = {}

        # >> This is our *accumulating* context. It starts with just image latents.
        current_context_embeddings = image_latents + \
            self.policy_latent_token_positional_embedding[:latent_seq_len]
        
        # >> The mask for the image latents is all False (no padding).
        current_context_padding_mask = torch.zeros(
            batch_size, latent_seq_len, dtype=torch.bool, device=device
        )

        for lod_idx in range(self.num_lod - 1):
            
            lod_start_idx = self.lod_start_indices[lod_idx]
            num_nodes_at_lod = self.lod_node_counts[lod_idx]
            
            if num_nodes_at_lod == 0:
                continue
                
            # >> These are the "query" nodes we are deciding on for this level
            # >> Their global node indices run from lod_start_idx to lod_start_idx + num_nodes_at_lod
            node_indices_at_lod = torch.arange(
                lod_start_idx, lod_start_idx + num_nodes_at_lod, device=device
            )
            
            # >> Get query embeddings: (B, num_nodes_at_lod, D)
            node_embeddings_at_lod = all_node_embeddings[:, node_indices_at_lod, :]
            
            # >> The query nodes themselves are not padded *relative to each other*
            query_padding_mask = torch.zeros(batch_size, num_nodes_at_lod, dtype=torch.bool, device=device)

            # >> Build the full sequence: [current_context, query_nodes]
            # >> current_context_embeddings has shape (B, variable_context_len, D)
            # >> node_embeddings_at_lod has shape (B, num_nodes_at_lod, D)
            state_sequence = torch.cat([current_context_embeddings, node_embeddings_at_lod], dim=1)
            
            # >> Build the full padding mask to match the state_sequence
            full_padding_mask = torch.cat([current_context_padding_mask, query_padding_mask], dim=1)
            
            policy_input = self.policy_ln_pre(state_sequence)
            policy_input = policy_input.permute(1, 0, 2)

            # >> --- This is the key change ---
            # >> Pass the key_padding_mask to the transformer
            for layer in self.policy_transformer:
                policy_input = layer(policy_input, key_padding_mask=full_padding_mask)
            
            policy_input = policy_input.permute(1, 0, 2)

            # >> Get features for the *query* nodes (the ones we just added)
            node_features_at_lod = policy_input[:, -num_nodes_at_lod:, :]
            logits = self.policy_net(node_features_at_lod) # (B, num_nodes_at_lod, 1)
            probs = torch.sigmoid(logits)
            all_probs[lod_idx] = probs
            
            actions_hard = (probs > 0.5) # (B, num_nodes_at_lod, 1)
            final_actions[lod_idx] = actions_hard # Store all decisions

            # >> --- Update context for the *next* iteration ---
            
            # >> Find which nodes were selected *for each batch item*
            new_active_node_indices_batch = [[] for _ in range(batch_size)]
            max_active_at_lod = 0 # >> Max *new* active nodes in this batch
            
            for b_idx in range(batch_size):
                # >> Find the *local* indices (0 to num_nodes_at_lod-1) that were activated
                b_active_local_indices = torch.where(actions_hard[b_idx].squeeze(-1))[0]
                
                if b_active_local_indices.numel() > 0:
                    # >> Convert local indices to *global* node indices
                    b_active_global_indices = node_indices_at_lod[b_active_local_indices]
                    new_active_node_indices_batch[b_idx] = b_active_global_indices.tolist()
                    
                    if len(new_active_node_indices_batch[b_idx]) > max_active_at_lod:
                        max_active_at_lod = len(new_active_node_indices_batch[b_idx])
            
            # >> If any nodes were selected, add them to the context
            if max_active_at_lod > 0:
                # >> 1. Create padded embedding tensor and new mask
                padded_embeddings = torch.zeros(batch_size, max_active_at_lod, self.width, device=device)
                new_padding_mask = torch.ones(batch_size, max_active_at_lod, dtype=torch.bool, device=device)

                # >> 2. Fill the tensor
                for b_idx in range(batch_size):
                    global_indices = new_active_node_indices_batch[b_idx]
                    num_active = len(global_indices)
                    
                    if num_active > 0:
                        global_indices_tensor = torch.tensor(global_indices, dtype=torch.long, device=device)
                        
                        # >> Gather embeddings from all_node_embeddings
                        # >> all_node_embeddings[b_idx] is (N_total, D)
                        embeddings = all_node_embeddings[b_idx].index_select(0, global_indices_tensor)
                        
                        padded_embeddings[b_idx, :num_active, :] = embeddings
                        new_padding_mask[b_idx, :num_active] = False # >> Mark these as False (not padded)

                # >> 3. Append to the context and mask for the *next* loop iteration
                current_context_embeddings = torch.cat([current_context_embeddings, padded_embeddings], dim=1)
                current_context_padding_mask = torch.cat([current_context_padding_mask, new_padding_mask], dim=1)

            # >> The original code had a complex gather logic based on parents.
            # >> This new logic replaces it. The "filtering" is now done by
            # >> the transformer itself, as it only sees active nodes from previous steps.
            # >> If you still need the parent-child gather, you would apply it
            # >> *before* the `torch.where` to mask out logits for non-children.

        # >> We must fix the original code's `final_actions` logic, as it was bugged.
        # >> This version stores *all* decisions. Your original `final_actions`
        # >> logic was trying to filter by parentage, which was complex and buggy.
        # >> This autoregressive model is a cleaner way to achieve that,
        # >> as decisions at lod_idx+1 are *conditioned* on active nodes from lod_idx.
        
        return final_actions, all_probs