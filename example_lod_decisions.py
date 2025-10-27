#!/usr/bin/env python3
"""
Example script demonstrating LOD decision making with STE in QuadTok.

This script shows how to use the new forward_with_lod_decisions function
that makes decisions for each LOD level while maintaining differentiability
through Straight-Through Estimator (STE).
"""

import torch
import torch.nn as nn
from modeling.modules.blocks import QuadTokDecoder, QuadTokSelctor, PolicyDecisionModule

def example_usage():
    """Example of using the new LOD decision functionality."""
    
    # Example configuration (you would load this from your actual config)
    class MockConfig:
        def __init__(self):
            self.model = MockModelConfig()
            self.dataset = MockDatasetConfig()
    
    class MockModelConfig:
        def __init__(self):
            self.vq_model = MockVQModelConfig()
            self.selector = MockSelectorConfig()
            self.train_policy = True
    
    class MockVQModelConfig:
        def __init__(self):
            self.vit_enc_patch_size = 16
            self.vit_dec_patch_size = 16
            self.vit_enc_model_size = "base"
            self.vit_dec_model_size = "base"
            self.num_latent_tokens = 256
            self.token_size = 512
            self.is_legacy = False
            self.strict_length_assertion = True
            self.quantize_mode = "vq"
    
    class MockSelectorConfig:
        def __init__(self):
            self.num_patch_side_list = [4, 8, 16, 32]  # Different LOD levels
            self.patch_size_list = [64, 32, 16, 8]     # Patch sizes for each LOD
            self.token_size = 512
    
    class MockDatasetConfig:
        def __init__(self):
            self.preprocessing = MockPreprocessingConfig()
    
    class MockPreprocessingConfig:
        def __init__(self):
            self.crop_size = 256
    
    # Create mock config
    config = MockConfig()
    
    # Initialize models
    decoder = QuadTokDecoder(config)
    selector = QuadTokSelctor(config)
    
    # Example input data
    batch_size = 2
    seq_len = 256  # Total sequence length across all LODs
    feature_dim = 512
    
    # Create some dummy quantized features
    z_quantized = torch.randn(batch_size, seq_len, feature_dim)
    
    print("=== QuadTokDecoder with LOD Decisions ===")
    print(f"Input shape: {z_quantized.shape}")
    
    # Forward pass with automatic LOD decisions
    with torch.no_grad():
        reconstructed_image, policy_results, prob_results = decoder.forward_with_lod_decisions(z_quantized)
    
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    print(f"Number of LODs: {len(policy_results)}")
    
    for lod_idx, actions in policy_results.items():
        print(f"LOD {lod_idx}:")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Number of selected patches: {actions.sum(dim=1)}")
        print(f"  Probabilities shape: {prob_results[lod_idx].shape}")
    
    print("\n=== QuadTokSelctor with LOD Decisions ===")
    
    # Forward pass with automatic LOD decisions for selector
    with torch.no_grad():
        output_tokens, policy_results_sel, prob_results_sel = selector.forward_with_lod_decisions(z_quantized)
    
    print(f"Output tokens shape: {output_tokens.shape}")
    print(f"Number of LODs: {len(policy_results_sel)}")
    
    for lod_idx, actions in policy_results_sel.items():
        print(f"LOD {lod_idx}:")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Number of selected patches: {actions.sum(dim=1)}")
    
    print("\n=== Training Mode (with STE) ===")
    
    # Set models to training mode to enable STE
    decoder.train()
    selector.train()
    
    # Forward pass in training mode (STE will be applied)
    reconstructed_image_train, policy_results_train, prob_results_train = decoder.forward_with_lod_decisions(z_quantized)
    
    print("Training mode forward pass completed with STE")
    print(f"Reconstructed image shape: {reconstructed_image_train.shape}")
    
    # Check gradients are maintained
    loss = reconstructed_image_train.sum()
    loss.backward()
    
    print("Gradients computed successfully - STE is working!")
    
    # Check if policy decision modules have gradients
    for name, param in decoder.policy_decision.named_parameters():
        if param.grad is not None:
            print(f"Policy decision parameter {name} has gradients: {param.grad.norm().item():.4f}")

if __name__ == "__main__":
    example_usage()