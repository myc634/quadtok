#!/usr/bin/env python3
"""
Example usage of LOD decision making with STE (Straight-Through Estimator) in QuadTok.

This example demonstrates how to use the new LOD decision functionality that allows
each LOD level to decide the next LOD's actions while maintaining differentiability
through the use of Gumbel-Softmax and Straight-Through Estimator.
"""

import torch
import torch.nn as nn
from modeling.modules.blocks import QuadTokDecoder, QuadTokSelctor, gumbel_softmax, straight_through_estimator


def example_lod_decision_workflow():
    """
    Example workflow showing how to use LOD decisions with STE.
    """
    # Configuration parameters
    batch_size = 4
    seq_len = 256
    feature_dim = 768
    num_lod = 4
    num_patch_side_list = [4, 8, 16, 32]  # Example LOD levels
    
    # Create dummy input features
    z_quantized = torch.randn(batch_size, seq_len, feature_dim)
    
    # Initialize decoder (you would use your actual config)
    # decoder = QuadTokDecoder(config)
    
    print("=== LOD Decision Workflow Example ===")
    print(f"Input shape: {z_quantized.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Number of LOD levels: {num_lod}")
    print(f"Patch side list: {num_patch_side_list}")
    
    # Example 1: Using Gumbel-Softmax for differentiable discrete sampling
    print("\n1. Gumbel-Softmax Example:")
    logits = torch.randn(batch_size, 16)  # 16 possible actions
    
    # Soft sampling (differentiable)
    soft_actions = gumbel_softmax(logits, temperature=1.0, hard=False)
    print(f"Soft actions shape: {soft_actions.shape}")
    print(f"Soft actions sum per sample: {soft_actions.sum(dim=1)}")
    
    # Hard sampling with STE (differentiable in backward pass)
    hard_actions = gumbel_softmax(logits, temperature=1.0, hard=True)
    print(f"Hard actions shape: {hard_actions.shape}")
    print(f"Hard actions sum per sample: {hard_actions.sum(dim=1)}")
    
    # Example 2: Straight-Through Estimator
    print("\n2. Straight-Through Estimator Example:")
    probs = torch.softmax(logits, dim=-1)
    hard_samples = torch.argmax(probs, dim=-1).float()
    
    # Apply STE
    ste_output = straight_through_estimator(probs, hard_samples)
    print(f"STE output shape: {ste_output.shape}")
    print(f"STE output (forward): {ste_output}")
    print(f"STE output (backward): {probs}")  # Same as probs for gradients
    
    # Example 3: LOD Decision Making
    print("\n3. LOD Decision Making Example:")
    
    # Simulate LOD decision process
    action_dict = {}
    prob_dict = {}
    
    for lod_idx in range(num_lod):
        num_patches = num_patch_side_list[lod_idx] ** 2
        patch_logits = torch.randn(batch_size, num_patches)
        
        # Make decisions using Gumbel-Softmax
        actions = gumbel_softmax(patch_logits, temperature=1.0, hard=True)
        probs = torch.softmax(patch_logits, dim=-1)
        
        action_dict[lod_idx] = actions
        prob_dict[lod_idx] = probs
        
        print(f"LOD {lod_idx}: {num_patches} patches, {actions.sum(dim=1).mean():.2f} active on average")
    
    # Example 4: Hierarchical Decoding with Actions
    print("\n4. Hierarchical Decoding with Actions:")
    print("This would use the hierarchical_latent_decode_by_actions function")
    print("to reconstruct images based on the LOD decisions made above.")
    
    return action_dict, prob_dict


def example_training_loop():
    """
    Example training loop showing how to use LOD decisions in training.
    """
    print("\n=== Training Loop Example ===")
    
    # Dummy parameters
    batch_size = 2
    seq_len = 128
    feature_dim = 512
    
    # Create dummy data
    z_quantized = torch.randn(batch_size, seq_len, feature_dim, requires_grad=True)
    target_images = torch.randn(batch_size, 3, 256, 256)
    
    # Simulate decoder forward pass with LOD decisions
    # In practice, you would use: reconstructed_image, action_dict, prob_dict = decoder.forward_with_lod_decisions(z_quantized)
    
    # For this example, we'll simulate the process
    reconstructed_image = torch.randn(batch_size, 3, 256, 256)
    
    # Compute loss (e.g., reconstruction loss)
    reconstruction_loss = nn.MSELoss()(reconstructed_image, target_images)
    
    # Add regularization for LOD decisions (optional)
    # This encourages sparsity in LOD decisions
    lod_regularization = 0.0
    # for lod_idx in range(num_lod):
    #     lod_regularization += action_dict[lod_idx].sum()  # L1 regularization
    
    total_loss = reconstruction_loss + 0.01 * lod_regularization
    
    print(f"Reconstruction loss: {reconstruction_loss.item():.4f}")
    print(f"LOD regularization: {lod_regularization:.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Backward pass (gradients flow through STE)
    total_loss.backward()
    
    print(f"Gradient norm for z_quantized: {z_quantized.grad.norm().item():.4f}")
    print("✓ Gradients successfully computed through STE!")


def example_inference_mode():
    """
    Example inference mode showing how to use LOD decisions during inference.
    """
    print("\n=== Inference Mode Example ===")
    
    # During inference, you might want to use deterministic decisions
    # instead of stochastic sampling
    
    batch_size = 1
    seq_len = 64
    feature_dim = 256
    
    z_quantized = torch.randn(batch_size, seq_len, feature_dim)
    
    # For inference, you can use temperature=0.1 for more deterministic decisions
    # or directly use argmax for completely deterministic decisions
    
    logits = torch.randn(batch_size, 16)
    
    # Deterministic inference
    deterministic_actions = torch.argmax(logits, dim=-1).float()
    print(f"Deterministic actions: {deterministic_actions}")
    
    # Or use very low temperature for near-deterministic
    near_deterministic = gumbel_softmax(logits, temperature=0.1, hard=True)
    print(f"Near-deterministic actions: {near_deterministic}")
    
    print("✓ Inference mode ready!")


if __name__ == "__main__":
    print("QuadTok LOD Decision Making with STE - Example Usage")
    print("=" * 60)
    
    # Run examples
    action_dict, prob_dict = example_lod_decision_workflow()
    example_training_loop()
    example_inference_mode()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("\nKey features demonstrated:")
    print("1. Gumbel-Softmax for differentiable discrete sampling")
    print("2. Straight-Through Estimator for maintaining gradients")
    print("3. LOD decision making at multiple levels")
    print("4. Hierarchical decoding with actions")
    print("5. Training and inference modes")