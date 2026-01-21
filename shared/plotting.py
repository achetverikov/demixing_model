"""Plotting helpers for surface visualization and comparisons."""

import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import jax.numpy as jnp
import numpy as np
from shared.surface_folder_parsing import load_filtered_surfaces
import random

from shared.surface_functions import entropy_smooth_columns_with_mask

def _extract_surface_data(surface, dimension=1, component=1):
    """
    Extract surface data from either Surface object or tuple format.
    
    Parameters:
    -----------
    surface : Surface object or tuple
        Either a Surface object with grids and surfaces, or legacy tuple format
    dimension : int
        1 for mu1, 2 for mu2 (default: 1)
    component : int 
        1 or 2 for component selection (default: 1)
        
    Returns:
    --------
    tuple : (feat_diff_meshgrid, bias_meshgrid, log_surface_data)
    """
    # Import here to avoid circular import
    try:
        from shared.utils import Surface
        from shared.utils import AveragedSurface
        
        # Check if it's a Surface object (including AveragedSurface)
        if isinstance(surface, (Surface, AveragedSurface)):
            feat_diff_1d = surface.feat_diff_grid
            
            if dimension == 1:
                bias_1d = surface.mu1_bias_grid
                if component == 1:
                    log_surface_data = surface.mu1_comp1_surface
                else:
                    log_surface_data = surface.mu1_comp2_surface
            else:  # dimension == 2
                bias_1d = surface.mu2_bias_grid
                if component == 1:
                    log_surface_data = surface.mu2_comp1_surface
                else:
                    log_surface_data = surface.mu2_comp2_surface
            
            # Create meshgrids compatible with contour plotting
            # Surface data is shaped as (bias_points, feat_diff_points)
            # For matplotlib contour: X should be feat_diff, Y should be bias
            feat_diff_meshgrid, bias_meshgrid = jnp.meshgrid(feat_diff_1d, bias_1d, indexing='xy')
            
            # Verify shapes match
            expected_shape = (len(bias_1d), len(feat_diff_1d))
            if log_surface_data.shape != expected_shape:
                print(f"Warning: Surface shape {log_surface_data.shape} != expected {expected_shape}")
                print(f"Meshgrid shapes: feat_diff={feat_diff_meshgrid.shape}, bias={bias_meshgrid.shape}")
            
            return feat_diff_meshgrid, bias_meshgrid, log_surface_data
    except ImportError:
        pass
    
    # Legacy tuple format: (feat_diff_grid, mu1_error_grid, log_likelihood_surface, likelihood_surface)
    if isinstance(surface, (tuple, list)) and len(surface) >= 3:
        feat_diff_grid, bias_grid, log_surface_data = surface[0], surface[1], surface[2]
        return feat_diff_grid, bias_grid, log_surface_data
    
    raise ValueError(f"Unsupported surface format: {type(surface)}. Expected Surface object or tuple.")

def plot_surface_comparison(surface1: Union[Tuple, 'Surface'], surface2: Union[Tuple, 'Surface'],
                           titles: Tuple[str, str] = ("Surface 1", "Surface 2"),
                           figsize: Tuple[int, int] = (18, 6),
                           do_exp: bool = True,
                           same_color_axis: bool = True,
                           show_plot: bool = True,
                           save_path: Optional[str] = None,
                           dimension: int = 1,
                           component: int = 1) -> plt.Figure:
    """
    Plot two surfaces side by side with difference plot.
    Now supports both Surface objects and legacy tuple format.

    Parameters:
    -----------
    surface1, surface2 : Surface object or tuple
        Either Surface objects with grids and surfaces, or legacy tuple format:
        (feat_diff_grid, mu1_error_grid, log_likelihood_surface, likelihood_surface)
    titles : tuple
        Titles for the two surfaces
    figsize : tuple
        Figure size (width, height)
    do_exp : bool
        Whether to exponentiate log surfaces for display
    same_color_axis : bool
        Whether to use same color axis for both plots
    show_plot : bool
        Whether to display the plot
    save_path : str, optional
        Path to save the plot
    dimension : int
        1 for mu1, 2 for mu2 (only used for Surface objects)
    component : int
        1 or 2 for component selection (only used for Surface objects)

    Returns:
    --------
    plt.Figure
        The created figure
    """
    
    # Extract data using the helper function
    feat_diff_grid1, mu1_error_grid1, surface_data1 = _extract_surface_data(surface1, dimension, component)
    feat_diff_grid2, mu1_error_grid2, surface_data2 = _extract_surface_data(surface2, dimension, component)
    if do_exp:
        surface_data1 = jnp.exp(surface_data1)
        surface_data2 = jnp.exp(surface_data2)
    # Check if grids are compatible for difference calculation
    grids_compatible = (feat_diff_grid1.shape == feat_diff_grid2.shape and
                       np.allclose(feat_diff_grid1, feat_diff_grid2, atol=1e-6) and
                       np.allclose(mu1_error_grid1, mu1_error_grid2, atol=1e-6))

    # Create subplots
    if grids_compatible:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plot_difference = True
    else:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2/3, figsize[1]))
        plot_difference = False

    vmin = min(surface_data1.min(), surface_data2.min())
    vmax = max(surface_data1.max(), surface_data2.max())

    # Collect all data and labels in lists
    feat_diff_grids = [feat_diff_grid1, feat_diff_grid2]
    mu1_error_grids = [mu1_error_grid1, mu1_error_grid2]
    log_likelihoods = [surface_data1, surface_data2]

    for i in range(2):
        if same_color_axis:
            im = axes[i].contourf(feat_diff_grids[i], mu1_error_grids[i], log_likelihoods[i],
                              levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        else:
            im = axes[i].contourf(feat_diff_grids[i], mu1_error_grids[i], log_likelihoods[i],
                                  levels=20, cmap='viridis')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('feat_diff')
        axes[i].set_ylabel('mu1_error')
        if not same_color_axis:
            plt.colorbar(im, ax=axes[i])

        if i == 0:
            im_for_colorbar = im  # save for colorbar
    if same_color_axis:
        fig.colorbar(im_for_colorbar, ax=[axes[0], axes[1]], orientation='vertical', fraction=0.02, pad=0.04)

    # Add surface statistics
    # stats2 = compute_surface_stats(surface2)
    # stats_text2 = f'Range: {stats2["log_likelihood_min"]:.2f} to {stats2["log_likelihood_max"]:.2f}'
    # axes[1].text(0.02, 0.98, stats_text2, transform=axes[1].transAxes,
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    #             verticalalignment='top')

    # Plot difference if grids are compatible
    if plot_difference:
        difference = surface_data1 - surface_data2
        max_abs_diff = np.max(np.abs(difference))

        im3 = axes[2].contourf(feat_diff_grid1, mu1_error_grid1, difference,
                              levels=20, cmap='RdBu_r',
                              vmin=-max_abs_diff, vmax=max_abs_diff)
        axes[2].set_title(f'Difference ({titles[0]} - {titles[1]})')
        axes[2].set_xlabel('feat_diff')
        axes[2].set_ylabel('mu1_error')
        plt.colorbar(im3, ax=axes[2])

        # Add difference statistics
        rmse = np.sqrt(np.mean(difference ** 2))
        mae = np.mean(np.abs(difference))
        correlation = np.corrcoef(surface_data1.flatten(), surface_data2.flatten())[0, 1]

        diff_stats = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nCorr: {correlation:.3f}'
        axes[2].text(0.02, 0.98, diff_stats, transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Surface comparison saved: {save_path}")
    if show_plot:
        plt.show()
    return fig

