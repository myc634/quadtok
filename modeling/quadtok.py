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
from modeling.modules.blocks import TiTokEncoder, TiTokDecoder, QuadTokEncoder, QuadTokDecoder, QuadTokSelctor
from modeling.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
import json
from omegaconf import OmegaConf
from pathlib import Path

from modeling.utils import build_quadtree, build_random_quadtree


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

    
    def forward(self, x):
        latent_feats = self.encode(x)
        tree_structure = build_random_quadtree(self.num_patch_side_list, 2)

        z = self.selector(latent_feats, tree_structure)

        if self.quantize_mode == "vq":
            z_quantized, result_dict = self.quantize(z)
        elif self.quantize_mode == "vae":
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            result_dict = posteriors
        decoded = self.decode(z_quantized.permute(0, 3, 2, 1).squeeze(2).contiguous(), tree_structure)
        
        return decoded, result_dict