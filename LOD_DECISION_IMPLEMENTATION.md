# LOD Decision Implementation with STE

## Overview

This document describes the implementation of Level-of-Detail (LOD) decision making with Straight-Through Estimator (STE) in the QuadTok model. The implementation allows each LOD level to decide the next LOD's actions while maintaining differentiability through the use of Gumbel-Softmax and Straight-Through Estimator.

## Key Features

### 1. Straight-Through Estimator (STE)
- **Function**: `straight_through_estimator(probs, hard_samples)`
- **Purpose**: Maintains gradients through discrete sampling operations
- **Usage**: Uses hard samples in forward pass but soft probabilities in backward pass

### 2. Gumbel-Softmax Sampling
- **Function**: `gumbel_softmax(logits, temperature=1.0, hard=False, dim=-1)`
- **Purpose**: Provides differentiable discrete sampling
- **Parameters**:
  - `temperature`: Controls the sharpness of the distribution (lower = more deterministic)
  - `hard`: Whether to use hard sampling with STE
  - `dim`: Dimension along which to apply softmax

### 3. LOD Decision Making
- **Function**: `make_lod_decisions(features, temperature=1.0, hard=True)`
- **Purpose**: Makes decisions for each LOD level using Gumbel-Softmax
- **Returns**: 
  - `action_dict`: Dictionary mapping LOD index to action tensors
  - `prob_dict`: Dictionary mapping LOD index to probability tensors

### 4. Hierarchical Decoding with Actions
- **Function**: `hierarchical_latent_decode_by_actions(feature, action_dict, prob_dict)`
- **Purpose**: Reconstructs images based on LOD decisions
- **Features**: 
  - Processes each LOD level with corresponding actions
  - Applies probabilities to features
  - Handles batch processing efficiently

## Modified Classes

### QuadTokDecoder
- Added `make_lod_decisions()` method for LOD decision making
- Added `forward_with_lod_decisions()` method for complete workflow
- Fixed `hierarchical_latent_decode_by_actions()` to handle batch decisions
- Updated `_forward_policy()` to accept batch decisions

### QuadTokSelctor
- Added `make_lod_decisions_selector()` method
- Updated `_forward_policy()` to handle probability outputs
- Enhanced forward method to support policy and probability outputs

## Usage Examples

### Training Mode
```python
# Make LOD decisions with STE
action_dict, prob_dict = decoder.make_lod_decisions(features, temperature=1.0, hard=True)

# Forward pass with decisions
reconstructed_image = decoder.forward_with_lod_decisions(z_quantized, temperature=1.0, hard=True)

# Compute loss and backpropagate
loss = criterion(reconstructed_image, target_image)
loss.backward()  # Gradients flow through STE
```

### Inference Mode
```python
# Use lower temperature for more deterministic decisions
action_dict, prob_dict = decoder.make_lod_decisions(features, temperature=0.1, hard=True)

# Or use completely deterministic decisions
actions = torch.argmax(logits, dim=-1).float()
```

## Key Benefits

1. **Differentiability**: STE allows gradients to flow through discrete decisions
2. **Flexibility**: Supports both training (stochastic) and inference (deterministic) modes
3. **Efficiency**: Batch processing for multiple LOD levels
4. **Scalability**: Works with different numbers of LOD levels and patch sizes

## Implementation Details

### Gradient Flow
- Forward pass: Uses hard discrete decisions
- Backward pass: Uses soft continuous probabilities
- STE bridges the gap between discrete and continuous representations

### Temperature Annealing
- Start with high temperature (e.g., 1.0) for exploration during training
- Gradually decrease temperature for more deterministic decisions
- Use very low temperature (e.g., 0.1) or argmax for inference

### Batch Processing
- All operations support batch processing
- Actions are computed per batch item
- Hierarchical decoding handles variable numbers of active patches

## Error Fixes

1. **Import Issues**: Fixed missing `Attention` class import
2. **Variable References**: Fixed undefined variables in `hierarchical_latent_decode_by_actions`
3. **Device Handling**: Added proper device management throughout
4. **Type Annotations**: Fixed type issues with optional parameters
5. **Constant Redefinition**: Fixed `ATTENTION_MODE` redefinition issue

## Future Enhancements

1. **Adaptive Temperature**: Implement temperature annealing during training
2. **Multi-Objective Loss**: Add regularization terms for LOD decisions
3. **Attention Mechanisms**: Integrate attention-based LOD decision making
4. **Dynamic LOD**: Support for variable numbers of LOD levels
5. **Memory Optimization**: Optimize memory usage for large batch sizes

## Testing

Run the example script to test the implementation:
```bash
python example_lod_decision.py
```

This will demonstrate:
- Gumbel-Softmax sampling
- Straight-Through Estimator
- LOD decision making
- Training and inference modes
- Gradient flow verification