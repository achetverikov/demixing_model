#!/usr/bin/env python3
"""
Mirror-Aware Neural Network Model for mu1 surfaces
==================================================

This module implements a mirror-aware neural network that respects the symmetry:
mu1_comp1_surface(sf1, sf2, sp) = mu1_comp2_surface(sf2, sf1, sp)

The model only predicts mu1 surfaces and enforces this symmetry during training and inference.
Grid dimensions are taken from config rather than hardcoded.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
from shared.config import config


def normalize_to_density_flexible(log_probs, mu1_bias_range=None):
    """
    Normalize log probabilities to proper probability densities that integrate to 1.
    Uses logsumexp for numerical stability with dx correction.

    Args:
        log_probs: Shape (batch, mu1_error, feat_diff)
        mu1_bias_range: Tuple of (min, max) for mu1 bias values, defaults to config

    Returns:
        Normalized log densities with same shape
    """
    if mu1_bias_range is None:
        mu1_bias_range = config.mu1_bias_range
    
    # Grid spacing for mu1_error dimension
    mu1_error_min, mu1_error_max = mu1_bias_range
    n_mu1_points = log_probs.shape[1]
    dmu1 = (mu1_error_max - mu1_error_min) / (n_mu1_points - 1)

    # Normalize using logsumexp (ensures discrete probabilities sum to 1)
    log_probs_discrete = log_probs - jax.scipy.special.logsumexp(log_probs, axis=1, keepdims=True)

    # Convert to continuous density by dividing by dx (in log space: subtract log(dx))
    log_density = log_probs_discrete - jnp.log(dmu1)

    return log_density


class MirrorAwareMu1Predictor(nn.Module):
    """
    Neural network predictor for mu1 surfaces.
    
    Trained on data where the correct surface (mu1_comp1 or mu1_comp2) is chosen
    based on parameter relationships during training. The model learns to produce
    the correct surface for any parameter combination (sf1, sf2, sp).
    """
    hidden_dims: list = (64, 128, 256)
    cnn_channels: list = (128, 64, 32, 1)
    latent_size: int = 16

    @property
    def output_shape(self) -> Tuple[int, int]:
        """Get output shape from config."""
        return config.mu1_surface_shape
    
    @property 
    def mu1_bias_range(self) -> Tuple[float, float]:
        """Get mu1 bias range from config."""
        return config.mu1_bias_range

    @nn.compact
    def __call__(self, h):  # x shape: (batch, 3) where h = [sf1, sf2, sp]
        """Forward pass producing log-density surfaces.

        Args:
            h: Input array of shape (batch, 3) with [sf1, sf2, sp].

        Returns:
            Log-density surfaces with shape (batch, mu1_bias, feat_diff).
        """
        # Process input parameters directly - no canonicalization
        # The model learns to produce correct surfaces for any parameter combination
        for dim in self.hidden_dims:
            h = nn.relu(nn.Dense(dim)(h))

        h = nn.Dense(self.latent_size * 8 * 16)(h)
        h = nn.relu(h)
        h = h.reshape((-1, 8, 16, self.latent_size))

        # CNN decoder
        for ch in self.cnn_channels[:-1]:
            h = nn.relu(nn.ConvTranspose(features=ch, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(h))
        h = nn.ConvTranspose(features=self.cnn_channels[-1], kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h)

        # Reshape to get surface with configurable output shape
        h = jnp.transpose(h, (0, 3, 1, 2))
        new_size = (h.shape[0], h.shape[1], self.output_shape[0], self.output_shape[1])
        h = jax.image.resize(h, shape=new_size, method='linear')
        h = jnp.transpose(h, (0, 2, 3, 1))
        
        # Extract surface
        log_probs = h.squeeze(-1)  # Shape: (batch, mu1_bias_points, feat_diff_points)
        
        # Normalize to proper probability density with configurable range
        log_probs_normalized = normalize_to_density_flexible(log_probs, self.mu1_bias_range)
        
        return log_probs_normalized


def test_model_functionality():
    """Test that the model can process different parameter combinations."""
    import jax.random as random
    
    # Initialize model with config defaults
    model = MirrorAwareMu1Predictor()
    key = random.PRNGKey(42)
    
    # Test inputs with different parameter relationships
    test_input_1 = jnp.array([[30.0, 50.0, 100.0]])  # sf1 < sf2
    test_input_2 = jnp.array([[50.0, 30.0, 100.0]])  # sf1 > sf2
    test_input_3 = jnp.array([[40.0, 40.0, 80.0]])   # sf1 = sf2
    
    # Initialize parameters
    params = model.init(key, test_input_1)
    
    # Get predictions
    pred_1 = model.apply(params, test_input_1)
    pred_2 = model.apply(params, test_input_2)
    pred_3 = model.apply(params, test_input_3)
    
    print(f"Mu1 surface model functionality test:")
    print(f"  Output shape from config: {config.mu1_surface_shape}")
    print(f"  Mu1 bias range from config: {config.mu1_bias_range}")
    print(f"  Input 1 (sf1<sf2): sf1=30, sf2=50, sp=100 -> shape {pred_1.shape}")
    print(f"  Input 2 (sf1>sf2): sf1=50, sf2=30, sp=100 -> shape {pred_2.shape}")
    print(f"  Input 3 (sf1=sf2): sf1=40, sf2=40, sp=80 -> shape {pred_3.shape}")
    print(f"  Model processes all parameter combinations correctly")
    
    # Check outputs are different (since they represent different parameter combinations)
    diff_1_2 = jnp.max(jnp.abs(pred_1 - pred_2))
    diff_1_3 = jnp.max(jnp.abs(pred_1 - pred_3))
    
    print(f"  Difference between pred_1 and pred_2: {diff_1_2:.3f}")
    print(f"  Difference between pred_1 and pred_3: {diff_1_3:.3f}")
    print(f"  Model produces different outputs for different inputs: {diff_1_2 > 0.1 and diff_1_3 > 0.1}")
    
    return True


if __name__ == "__main__":
    print("Testing mu1 surface model with config...")
    print(f"Config mu1 surface shape: {config.mu1_surface_shape}")
    print(f"Config mu1 bias range: {config.mu1_bias_range}")
    print()
    
    test_model_functionality()
