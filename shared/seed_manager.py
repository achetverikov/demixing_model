"""Seed management utilities for reproducible simulations."""

import os

import jax
import numpy as np


class SeedManager:
    """Manages random seeds throughout."""

    def __init__(self, base_seed=None, quiet = False):
        """Initialize the seed manager.

        Args:
            base_seed: Optional base seed; if None, uses OS entropy.
            quiet: If True, suppress seed generation logs.
        """
        if base_seed is None:
            # Generate random seed from OS entropy
            base_seed = int.from_bytes(os.urandom(4), byteorder='big') % (2 ** 31)

        self.base_seed = base_seed
        self.current_seed = base_seed
        self.seed_counter = 0
        self.quiet = quiet

        print(f"SeedManager initialized with base_seed: {base_seed}")

    def get_seed(self, purpose="general"):
        """Get a new seed for a specific purpose."""
        self.seed_counter += 1
        new_seed = (self.base_seed + self.seed_counter * 1000) % (2 ** 31)
        if not self.quiet:
            print(f"  Generated seed {new_seed} for: {purpose}")
        return new_seed

    def get_jax_key(self, purpose="general"):
        """Get a JAX random key."""
        seed = self.get_seed(purpose)
        return jax.random.PRNGKey(seed)

