"""
Dual-Component Bias Analysis for Gaussian Mixture Models

This module provides functions for simulating and analyzing bias in dual-component
Gaussian mixture models, computing empirical likelihood surfaces, and estimating
bias probabilities for parameter fitting.

Key Functions:
- simulate_dual_component_bias_distribution: Core simulation function
- compute_empirical_likelihood_surface: Generate likelihood surfaces using step sizes
- estimate_bias_probability: Estimate probability densities for specific bias values
- process_csv_probabilities: Batch process CSV data for probability analysis

Global Settings:
- wrap_2nd: If True, treats spatial dimension as circular; if False, treats as linear
"""

import time
import jax
import jax.numpy as jnp
from jax.scipy import stats
import numpy as np
from functools import partial
from typing import Optional
import jax_fit_functions as jf  # uses the module in this repo
from jax_fit_functions import ResCol  # Import column index enum
import matplotlib.pyplot as plt


@partial(jax.jit, static_argnames=['n_simulations', 'n_samples', 'return_full_results',
                                   'fix_weights', 'algorithm', 'diagonal_covariance'])
def simulate_dual_component_bias_distribution(key, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff,
                                              n_simulations=1000, n_samples=100, return_full_results=False,
                                              fix_weights=False, algorithm='EM',
                                              diagonal_covariance: bool = True):
    """
    Simulate bias distributions for both components in a dual-component Gaussian mixture model.

    Args:
        key: JAX random key
        sd_feat1: Standard deviation for feature component 1
        sd_feat2: Standard deviation for feature component 2
        sd_spat: Spatial standard deviation (both components)
        feat_diff: Component separation in feature dimension
        spat_diff: Component separation in spatial dimension
        n_simulations: Number of simulation runs
        n_samples: Samples per simulation
        return_full_results: If True, return full results array in addition to biases
        fix_weights: If True, constrain mixture weights to be equal during fitting (default False)
        algorithm: Which fitting algorithm to use ('EM', 'VBEM' for MAP, or 'VBEM_MIX' to use mixture-of-modes means)

    Returns:
        If return_full_results is False:
            tuple: (mu_1_bias, mu_2_bias) arrays of shape (n_simulations, 2) for both components
        If return_full_results is True:
            tuple: (mu_1_bias, mu_2_bias, full_results)
                   full_results shape (n_simulations, 2, len(jf.RESULT_COLUMNS)) - consistent structure always includes r_est
                   columns (see jf.RESULT_COLUMNS): comp_id, orig_comp_id, weight, mu1_est, mu2_est,
                            sigma1_est, sigma2_est, r_est (0.0 for diagonal), true_mu1, true_mu2,
                            true_sigma1, true_sigma2, true_sigma1_flipped, n_samples,
                            sample_mu1, sample_mu2, sample_sigma1, sample_sigma2,
                            weight_mix, mu1_mix, mu2_mix
    """
    algorithm = algorithm.upper()
    keys = jax.random.split(key, n_simulations)

    def single_simulation(single_key):
        """Run one simulation and fit for a single random key."""
        true_mu1, true_mu2 = jnp.array([-0.5, 0.5]) * feat_diff, jnp.array([-0.5, 0.5]) * spat_diff
        true_sigma1, true_sigma2 = jnp.array([sd_feat1, sd_feat2]), jnp.array([sd_spat, sd_spat])
        # Returns array shape (2, len(jf.RESULT_COLUMNS)) - always consistent structure with r_est column
        return jf.jax_generate_and_fit(
            single_key, true_mu1, true_mu2, true_sigma1, true_sigma2,
            weights=0.5, n_samples=n_samples, algorithm=algorithm,
            fix_weights=fix_weights, diagonal_covariance=diagonal_covariance
        )

    # Vmap over all simulations -> results shape: (n_simulations, 2, len(jf.RESULT_COLUMNS))
    results = jax.vmap(single_simulation)(keys)

    # Extract fitted and true parameters using named column indices (zero overhead)
    if algorithm == 'VBEM_MIX':
        mean_1 = results[:, :, ResCol.mu1_mix]
        mean_2 = results[:, :, ResCol.mu2_mix]
    else:
        mean_1 = results[:, :, ResCol.mu1_est]
        mean_2 = results[:, :, ResCol.mu2_est]
    true_mu1 = results[:, :, ResCol.true_mu1]
    true_mu2 = results[:, :, ResCol.true_mu2]

    # Compute bias: bias = -sign(true_mu) * (fitted - true)
    # Result shape: (n_simulations, 2) for both components
    mu_1_bias = -jnp.sign(true_mu1) * (jf.angular_difference_jit(mean_1, true_mu1) if jf.wrap_1st else (mean_1 - true_mu1))
    mu_2_bias = -jnp.sign(true_mu2) * (jf.angular_difference_jit(mean_2, true_mu2) if jf.wrap_2nd else (mean_2 - true_mu2))

    if return_full_results:
        results = jnp.concatenate([results, mu_1_bias[..., None], mu_2_bias[..., None]], axis = -1)  # Append biases as last two columns
        return mu_1_bias, mu_2_bias, results
    else:
        return mu_1_bias, mu_2_bias


def estimate_probability_density(bias_samples, query_bias, bandwidth=None, method='kde'):
    """
    Estimate probability density at query point using KDE or histogram.

    Args:
        bias_samples: Empirical bias samples from simulations
        query_bias: Point(s) to estimate density at
        bandwidth: KDE bandwidth (None for auto)
        method: 'kde' or 'histogram'

    Returns:
        float/array: Probability density at query point(s)
    """
    if method == 'kde':
        kde = stats.gaussian_kde(bias_samples)
        if bandwidth is not None: kde.set_bandwidth(bandwidth)
        return kde(query_bias)
    elif method == 'histogram':
        hist, bin_edges = jnp.histogram(bias_samples, bins='auto', density=True)
        bin_idx = jnp.clip(jnp.digitize(query_bias, bin_edges) - 1, 0, len(hist) - 1)
        return hist[bin_idx]
    else:
        raise ValueError("Method must be 'kde' or 'histogram'")


def estimate_bias_probability(key, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff, query_bias,
                             component=1, dimension='mu1', n_simulations=1000, n_samples=100, bandwidth=None, method='kde'):
    """
    Estimate probability of observing specific bias value for given component and dimension.

    Args:
        key: JAX random key
        sd_feat1: Standard deviation for feature component 1
        sd_feat2: Standard deviation for feature component 2
        sd_spat: Spatial standard deviation
        feat_diff: Component separation in feature dimension
        spat_diff: Component separation in spatial dimension
        query_bias: Bias value to estimate probability for
        component: Component number (1 or 2)
        dimension: 'mu1' (feature) or 'mu2' (spatial)
        n_simulations: Number of simulation runs
        n_samples: Samples per simulation
        bandwidth: KDE bandwidth (None for auto)
        method: Density estimation method ('kde' or 'histogram')

    Returns:
        dict: Contains 'probability_density', 'bias_samples', and 'statistics'
    """
    mu_1_bias, mu_2_bias = simulate_dual_component_bias_distribution(key, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff, n_simulations, n_samples)
    bias_samples = (mu_1_bias if dimension == 'mu1' else mu_2_bias)[:, component - 1]
    bias_samples_np = np.array(bias_samples)

    return {
        'probability_density': estimate_probability_density(bias_samples, query_bias, bandwidth, method),
        'bias_samples': bias_samples,
        'statistics': {k: getattr(np, k)(bias_samples_np) for k in ['mean', 'std', 'median', 'min', 'max']} | {'n_samples': len(bias_samples_np)}
    }



def process_csv_probabilities(csv_file_path, sd_feat1=30.0, sd_feat2=30.0, sd_spat=40.0, spat_diff=80.0,
                              component=1, dimension='mu1', n_simulations=1000, n_samples=100, random_seed=42):
    """
    Process CSV file to estimate bias probabilities using 'atddr' as feat_diff and 'bias_to_distr_corr' as query values.

    Args:
        csv_file_path: Path to CSV with required columns: 'atddr', 'bias_to_distr_corr'
        sd_feat1: Standard deviation for feature component 1
        sd_feat2: Standard deviation for feature component 2
        sd_spat: Spatial standard deviation
        spat_diff: Component separation in spatial dimension
        component: Which component to analyze (1 or 2)
        dimension: Which dimension to analyze ('mu1' or 'mu2')
        n_simulations: Number of simulation runs per unique atddr
        n_samples: Samples per simulation
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (df_with_probabilities, summary_stats_df, distributions_cache)
    """
    import pandas as pd

    df = pd.read_csv(csv_file_path).dropna(subset=['bias_to_distr_corr', 'atddr'])
    print(f"Processing {len(df)} rows, analyzing component {component}, dimension {dimension}")

    unique_atddr = df['atddr'].unique()
    distributions_cache, key = {}, jax.random.PRNGKey(random_seed)

    print(f"Simulating distributions for {len(unique_atddr)} unique atddr values...")
    for feat_diff in sorted(unique_atddr):
        key, subkey = jax.random.split(key)
        mu_1_bias, mu_2_bias = simulate_dual_component_bias_distribution(subkey, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff, n_simulations, n_samples)
        bias_samples = (mu_1_bias if dimension == 'mu1' else mu_2_bias)[:, component - 1]
        distributions_cache[feat_diff] = np.array(bias_samples)
        print(f"  atddr={feat_diff}: mean={bias_samples.mean():.3f}, std={bias_samples.std():.3f}")

    df['probability_density'] = [estimate_probability_density(distributions_cache[row['atddr']], row['bias_to_distr_corr'], method='kde')
                                 for _, row in df.iterrows()]

    summary_stats = []
    for feat_diff in sorted(unique_atddr):
        bias_samples = distributions_cache[feat_diff]
        rows_subset = df[df['atddr'] == feat_diff]
        summary_stats.append({
            'atddr': feat_diff, 'component': component, 'dimension': dimension, 'n_data_rows': len(rows_subset),
            **{f'distribution_{k}': getattr(np, k)(bias_samples) for k in ['mean', 'std', 'median', 'min', 'max']},
            'n_simulated_samples': len(bias_samples),
            **{f'data_bias_{k}': getattr(rows_subset['bias_to_distr_corr'], k)() for k in ['mean', 'std']},
            'avg_probability_density': rows_subset['probability_density'].mean()
        })

    print(f"Probability density range: {df['probability_density'].min():.6f} - {df['probability_density'].max():.6f}")
    return df, pd.DataFrame(summary_stats), distributions_cache


def compute_empirical_likelihood_surface(sd_feat1, sd_feat2, sd_spat, feat_diff_step=2, mu1_bias_step=2, mu2_bias_step=6,
                                         feat_diff_range=(4, 180), mu1_bias_range=(-180, 180), mu2_bias_range=(-498, 498),
                                         n_simulations=1000, n_samples=100, random_seed=42):
    """
    Compute empirical likelihood surfaces for all component/dimension combinations using step-based grids.

    Args:
        sd_feat1: Standard deviation for feature component 1
        sd_feat2: Standard deviation for feature component 2
        sd_spat: Spatial standard deviation
        feat_diff_step: Step size for feature difference grid (default: 2)
        mu1_bias_step: Step size for mu1 bias grid (default: 2)
        mu2_bias_step: Step size for mu2 bias grid (default: 6)
        feat_diff_range: Range for feature difference (default: (4, 180))
        mu1_bias_range: Range for mu1 bias (default: (-180, 180))
        mu2_bias_range: Range for mu2 bias (default: (-498, 498))
        n_simulations: Number of simulation runs per feat_diff
        n_samples: Samples per simulation
        random_seed: Random seed for reproducibility

    Returns:
        Surface: Object containing 4 likelihood surfaces (mu1_comp1, mu1_comp2, mu2_comp1, mu2_comp2)

    Note:
        - Uses CircularGaussianKDE for mu1 (always circular)
        - Uses CircularGaussianKDE for mu2 if wrap_2nd=True, else normal Gaussian KDE
        - Grid sizes determined by: arange(start, stop + step, step)
    """
    from shared.utils import Surface

    # Create grids using step sizes
    feat_diff_vals = jnp.arange(feat_diff_range[0], feat_diff_range[1] + feat_diff_step, feat_diff_step)
    mu1_bias_vals = jnp.arange(mu1_bias_range[0], mu1_bias_range[1] + mu1_bias_step, mu1_bias_step)
    mu2_bias_vals = jnp.arange(mu2_bias_range[0], mu2_bias_range[1] + mu2_bias_step, mu2_bias_step)

    n_feat_diff, n_mu1_bias, n_mu2_bias = len(feat_diff_vals), len(mu1_bias_vals), len(mu2_bias_vals)
    print(f"Computing empirical surfaces: {n_feat_diff} × {n_mu1_bias} × {n_mu2_bias} points, params: sd_feat1={sd_feat1}, sd_feat2={sd_feat2}, sd_spat={sd_spat}")

    # Initialize surfaces
    surfaces = {f'{dim}_comp{comp}': jnp.zeros((n_mu1_bias if dim == 'mu1' else n_mu2_bias, n_feat_diff))
                for dim in ['mu1', 'mu2'] for comp in [1, 2]}

    key = jax.random.PRNGKey(random_seed)
    total_start_time = time.time()
    for i, feat_diff in enumerate(feat_diff_vals):
        if i % 10 == 0: print(f"    Processing feat_diff {i + 1}/{n_feat_diff}: {feat_diff:.1f}; elapsed time: {time.time() - total_start_time}")

        key, subkey = jax.random.split(key)
        mu_1_bias, mu_2_bias = simulate_dual_component_bias_distribution(subkey, sd_feat1, sd_feat2, sd_spat, feat_diff, 42.0, n_simulations, n_samples)

        # Process each component and dimension
        for comp_idx, comp in enumerate([1, 2]):
            # Mu1 bias (always circular)
            kde_mu1 = CircularGaussianKDE(mu_1_bias[:, comp_idx])
            surfaces[f'mu1_comp{comp}'] = surfaces[f'mu1_comp{comp}'].at[:, i].set(kde_mu1.logpdf(mu1_bias_vals))

            # Mu2 bias (circular if wrap_2nd, otherwise normal Gaussian)
            if wrap_2nd:
                kde_mu2 = CircularGaussianKDE(mu_2_bias[:, comp_idx])
            else:
                kde_mu2 = stats.gaussian_kde(mu_2_bias[:, comp_idx])
            surfaces[f'mu2_comp{comp}'] = surfaces[f'mu2_comp{comp}'].at[:, i].set(kde_mu2.logpdf(mu2_bias_vals))

    print(f"Total time: {time.time() - total_start_time}")
    print("Log-likelihood ranges:", {k: f"{v.min():.3f} to {v.max():.3f}" for k, v in surfaces.items()})

    return {'surface' : Surface(feat_diff_grid=feat_diff_vals, mu1_bias_grid=mu1_bias_vals, mu2_bias_grid=mu2_bias_vals,
                   mu1_comp1_surface=surfaces['mu1_comp1'], mu1_comp2_surface=surfaces['mu1_comp2'],
                   mu2_comp1_surface=surfaces['mu2_comp1'], mu2_comp2_surface=surfaces['mu2_comp2']),
    'mu_1_bias' : mu_1_bias, 'mu_2_bias' : mu_2_bias}



class CircularGaussianKDE:
    """
    Circular Gaussian KDE for angular data with periodic boundary conditions.

    Handles circular data by wrapping samples across multiple periods to ensure
    proper density estimation near boundaries (e.g., -180°/+180°).

    Args:
        samples: 1D array of angular samples
        n_wraps: Number of periodic wraps to consider (default: 5)

    Methods:
        logpdf(theta): Compute log probability density
        __call__(theta): Compute probability density (exp of logpdf)
    """
    def __init__(self, samples, n_wraps=5):
        """Initialize the KDE with samples and wrap count.

        Args:
            samples: 1D array of angular samples.
            n_wraps: Number of periodic wraps to consider.
        """
        self.samples = jnp.asarray(samples)
        self.n_wraps = n_wraps
        self.bandwidth = jnp.std(samples) * (len(samples) ** (-1 / 5))

    def logpdf(self, theta):
        """Compute log probability density with circular wrapping."""
        theta = jnp.asarray(theta)
        half_wraps = self.n_wraps // 2
        wrap_offsets = jnp.arange(-half_wraps, -half_wraps + self.n_wraps) * 360
        wrapped_samples = self.samples[:, None] + wrap_offsets[None, :]

        theta_expanded, wrapped_expanded = theta[:, None, None], wrapped_samples[None, :, :]
        log_densities = -0.5 * ((theta_expanded - wrapped_expanded) / self.bandwidth) ** 2
        log_norm = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.bandwidth)

        return jax.scipy.special.logsumexp(log_densities, axis=(1, 2)) + log_norm - jnp.log(len(self.samples))

    def __call__(self, theta):
        """Compute probability density (convenience method)."""
        return jnp.exp(self.logpdf(theta))



# Legacy compatibility functions
def simulate_mu1_bias_distribution(key, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff, n_simulations=1000, n_samples=100):
    """
    Legacy wrapper: returns only mu1 bias for component 1 (original behavior).
    For new code, use simulate_dual_component_bias_distribution instead.
    """
    mu_1_bias, _ = simulate_dual_component_bias_distribution(key, sd_feat1, sd_feat2, sd_spat, feat_diff, spat_diff, n_simulations, n_samples)
    return mu_1_bias[:, 0]



if __name__ == "__main__":
    key = jax.random.PRNGKey(554)

    # Test dual-component function
    mu1_bias, mu2_bias = simulate_dual_component_bias_distribution(key, 200, 200, 200, 1, 42)
    for i, comp in enumerate([1, 2]):
        print(f'Component {comp} - Mu1: {mu1_bias[:, i].min():.3f}/{mu1_bias[:, i].max():.3f}, Mu2: {mu2_bias[:, i].min():.3f}/{mu2_bias[:, i].max():.3f}')

    # Test legacy compatibility
    legacy_result = simulate_mu1_bias_distribution(key, 200, 200, 200, 1, 42)
    print(f'Legacy compatibility - Mu1 bias (comp 1): {legacy_result.min():.3f}/{legacy_result.max():.3f}')
    # Test with a single small surface first
    # Basic usage with default parameters
    surface = compute_empirical_likelihood_surface(
        sd_feat1=60.0,    # Standard deviation for feature component 1
        sd_feat2=115.0,   # Standard deviation for feature component 2
        sd_spat=82.0,     # Spatial standard deviation
        n_simulations=10000
    )

    surface = compute_empirical_likelihood_surface(
        sd_feat1=60.0,  # Standard deviation for feature component 1
        sd_feat2=115.0,  # Standard deviation for feature component 2
        sd_spat=82.0,  # Spatial standard deviation
        n_simulations=10000,
        random_seed=32
    )

    surface = compute_empirical_likelihood_surface(
        sd_feat1=60.0,  # Standard deviation for feature component 1
        sd_feat2=115.0,  # Standard deviation for feature component 2
        sd_spat=82.0,  # Spatial standard deviation
        n_simulations=10000,
        random_seed=22
    )
    log_surface = np.asarray(surface.get_surf(1, 1, log=True))
    print(f"Surface shape: {log_surface.shape}")
    print(f"Surface range: {log_surface.min():.3f} to {log_surface.max():.3f}")

    plt.figure()
    plt.contourf(log_surface, levels=10)
    plt.colorbar()
    plt.show()
    # Plot all combinations

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Plot all component/dimension combinations
    combinations = [(1, 'mu1'), (2, 'mu1'), (1, 'mu2'), (2, 'mu2')]
    titles = ['Component 1 - Feature Bias', 'Component 2 - Feature Bias',
              'Component 1 - Spatial Bias', 'Component 2 - Spatial Bias']

    for i, ((comp, dim), title) in enumerate(zip(combinations, titles)):
        row, col = i // 2, i % 2

        # Get the surface data
        log_surface = np.asarray(surface.get_surf(1 if dim == 'mu1' else 2, comp, log=True))
        bias_grid = np.asarray(surface.get_bias_grid(1 if dim == 'mu1' else 2))
        feat_diff_grid = np.asarray(surface.feat_diff_grid)
        # Create meshgrid and plot
        feat_diff_mesh, bias_mesh = np.meshgrid(feat_diff_grid, bias_grid)
        # im = axes[row, col].contourf(feat_diff_mesh, bias_mesh, log_surface, levels=20, cmap='viridis')
        im = plt.imshow(log_surface, aspect='auto', origin='lower')
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('feat_diff')
        axes[row, col].set_ylabel(f'{dim}_bias')
        plt.colorbar(im, ax=axes[row, col])

    plt.tight_layout()
    plt.show()
