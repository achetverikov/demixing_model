"""Shared configuration defaults and grid helpers."""

from dataclasses import dataclass
from typing import Tuple
import jax, os, platform
from shared.seed_manager import SeedManager
from datetime import datetime

# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
machine_id = platform.node()  # Gets hostname
cache_dir = os.path.expanduser(f"~/.cache/jax_cache_{machine_id}")
# print(cache_dir)
os.makedirs(cache_dir, exist_ok=True)

jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 2)  # Lower threshold

# Enable XLA caching properly
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') +
    ' --xla_gpu_autotune_level=4'  # Cache GPU kernel tuning
)
print('Jax config updated to use persistent cache in /tmp/jax_cache')
print(f'Jax is running on device {jax.devices()}')
@dataclass
class Config:
    surfaces_folder: str = './likelihood_surfaces_10k'
    samples_folder: str = './samples_10k'
    checkpoints_dir: str = 'checkpoints'
    param_range_low: float = 10.0
    param_range_high: float = 200.0
    param_step: float = 10.0
    
    # Grid parameters for likelihood surface computation
    feat_diff_step: int = 2
    mu1_bias_step: int = 2  
    mu2_bias_step: int = 6
    feat_diff_range: Tuple[int, int] = (0, 180)
    mu1_bias_range: Tuple[int, int] = (-180, 180)
    mu2_bias_range: Tuple[int, int] = (-498, 498)

    n_samples: int = 100

    seed = SeedManager(int(datetime.now().timestamp()))

    @property
    def data_range(self):
        """Return the configured data range bounds."""
        return (self.data_range_low, self.data_range_high)
    
    @property 
    def feat_diff_grid_size(self) -> int:
        """Number of feature difference grid points."""
        return (self.feat_diff_range[1] - self.feat_diff_range[0]) // self.feat_diff_step + 1
    
    @property
    def mu1_bias_grid_size(self) -> int:
        """Number of mu1 bias grid points."""
        return (self.mu1_bias_range[1] - self.mu1_bias_range[0]) // self.mu1_bias_step + 1
    
    def create_grid(self, param_name: str):
        """Create grid using arange with step size for given parameter.
        
        Args:
            param_name: One of 'feat_diff', 'mu1_bias', 'mu2_bias'
        """
        import jax.numpy as jnp
        
        # Get the range and step attributes dynamically
        range_attr = f"{param_name}_range"
        step_attr = f"{param_name}_step"
        
        if not hasattr(self, range_attr) or not hasattr(self, step_attr):
            raise ValueError(f"Unknown parameter name: {param_name}. Must be one of 'feat_diff', 'mu1_bias', 'mu2_bias'")
        
        param_range = getattr(self, range_attr)
        param_step = getattr(self, step_attr)
        
        return jnp.arange(param_range[0], param_range[1] + param_step, param_step)
    
    @property
    def mu1_surface_shape(self) -> Tuple[int, int]:
        """Shape of mu1 surfaces: (mu1_bias_points, feat_diff_points)."""
        return (self.mu1_bias_grid_size, self.feat_diff_grid_size)

config = Config()
