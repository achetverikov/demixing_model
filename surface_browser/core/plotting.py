"""
Surface Plotting
===============

Concise plotting with builder pattern and smart defaults.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# Add parent directory to path to import Surface class
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from shared.mu1_axis import sign_masks


class Plot:
    """Fluent interface for creating surface plots."""
    
    def __init__(self, surface=None):
        self.surface = surface
        self.fig = None
        self._expectation_data = None
    
    @staticmethod
    def compute_expectation(surface, dimension, component, circular=True):
        """Compute E[bias] as function of feat_diff for given dimension and component.
        
        Args:
            surface: Surface object containing likelihood data
            dimension: Which dimension (1 or 2) 
            component: Which component (1 or 2)
            circular: If True, use circular statistics for angular data. If False, use linear expectation.
        """
        feat_diff_values = surface.feat_diff_grid
        bias_values = surface.get_bias_grid(dimension)
        log_likelihood_surface = surface.get_surf(dimension, component, log=True)
        
        # Convert to likelihood
        likelihood_surface = np.exp(log_likelihood_surface)
        
        expected_bias = []
        for i in range(len(feat_diff_values)):
            prob_profile = likelihood_surface[:, i] / np.sum(likelihood_surface[:, i])
            
            if circular:
                # Circular expectation using complex representation
                angles_rad = np.deg2rad(bias_values)
                complex_exp = np.exp(1j * angles_rad)
                weighted_complex = np.sum(prob_profile * complex_exp)
                expected_angle_rad = np.angle(weighted_complex)
                expectation = np.rad2deg(expected_angle_rad)
            else:
                # Linear expectation (original method)  
                expectation = np.sum(bias_values * prob_profile)
                
            expected_bias.append(expectation)
        
        return feat_diff_values, np.array(expected_bias)
    
    @staticmethod
    def compute_asymmetry(surface, dimension, component):
        """Compute asymmetry as (sum(p>0)-sum(p<0))*100."""
        feat_diff_values = surface.feat_diff_grid
        bias_values = surface.get_bias_grid(dimension)
        log_likelihood_surface = surface.get_surf(dimension, component, log=True)
        
        # Convert to likelihood
        likelihood_surface = np.exp(log_likelihood_surface)
        
        asymmetry_values = []
        for i in range(len(feat_diff_values)):
            prob_profile = likelihood_surface[:, i] / np.sum(likelihood_surface[:, i])
            
            # Simple asymmetry: (sum(p>0)-sum(p<0))*100.  On the circular mu1
            # axis both 0 and the antipode (-180) are sign-ambiguous and are
            # excluded from both sides; mu2 is a linear axis and only 0 is.
            if dimension == 1:
                pos_mask, neg_mask = (np.asarray(m) for m in sign_masks(bias_values))
            else:
                pos_mask, neg_mask = bias_values > 0, bias_values < 0
            prob_positive = np.sum(prob_profile[pos_mask])
            prob_negative = np.sum(prob_profile[neg_mask])
            asymmetry = (prob_positive - prob_negative) * 100
            asymmetry_values.append(asymmetry)
        
        return feat_diff_values, np.array(asymmetry_values)
    
    def multi_component_heatmap(self, title="", show_expectation=True, use_log=True, auto_zoom=False, prob_threshold=0.0001):
        """Create 2x2 grid showing all 4 component surfaces."""
        subplot_titles = [
            'Mu1 Component 1', 'Mu1 Component 2',
            'Mu2 Component 1', 'Mu2 Component 2'
        ]
        
        if use_log:
            subplot_titles = [f'{title} (Log)' for title in subplot_titles]
        
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles
        )
        
        # Calculate zoom ranges for each surface if auto_zoom is enabled
        zoom_ranges = {}
        if auto_zoom:
            for dim in [1, 2]:
                for comp in [1, 2]:
                    zoom_ranges[(dim, comp)] = self._calculate_zoom_range(dim, comp, prob_threshold, use_log)
        
        # Plot all 4 combinations
        for dim in [1, 2]:
            for comp in [1, 2]:
                row = dim
                col = comp
                
                surface_data = self.surface.get_surf(dim, comp, log=use_log)
                bias_grid = self.surface.get_bias_grid(dim)
                
                self.fig.add_trace(go.Heatmap(
                    z=surface_data,
                    x=self.surface.feat_diff_grid,
                    y=bias_grid,
                    colorscale='Viridis' if use_log else 'Plasma',
                    showscale=(dim == 1 and comp == 1),  # Only show colorbar for first plot
                    hovertemplate='feat_diff: %{x}<br>bias: %{y}<br>value: %{z:.3g}<extra></extra>'
                ), row=row, col=col)
                
                if show_expectation:
                    self._add_expectation_line(dim, comp, row, col)
                
                # Apply auto-zoom if enabled
                if auto_zoom and (dim, comp) in zoom_ranges:
                    y_min, y_max = zoom_ranges[(dim, comp)]
                    self.fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        
        self.fig.update_layout(title=title, height=800)
        self.fig.update_xaxes(title_text="feat_diff")
        self.fig.update_yaxes(title_text="bias", row=1)
        self.fig.update_yaxes(title_text="bias", row=2)
        
        return self
    
    def heatmap_3d(self, dimension=1, component=1, use_log=True, title=""):
        """Create 3D surface plot for specific dimension and component."""
        z_data = self.surface.get_surf(dimension, component, log=use_log)
        bias_grid = self.surface.get_bias_grid(dimension)
        
        z_title = "Log-Likelihood" if use_log else "Likelihood"
        bias_label = f"mu{dimension}_bias"
        
        self.fig = go.Figure(data=[go.Surface(
            z=z_data,
            x=self.surface.feat_diff_grid,
            y=bias_grid,
            colorscale='Viridis' if use_log else 'Plasma',
            hovertemplate='feat_diff: %{x}<br>' + bias_label + ': %{y}<br>value: %{z:.3g}<extra></extra>'
        )])
        
        self.fig.update_layout(
            title=f"{title} - Mu{dimension} Component {component}",
            scene=dict(
                xaxis_title="feat_diff",
                yaxis_title=bias_label, 
                zaxis_title=z_title
            ),
            height=600
        )
        
        return self
    
    def _add_expectation_line(self, dimension, component, row, col):
        """Add expectation line to specific subplot."""
        # Use circular statistics for mu1 (dimension 1), linear for mu2 (dimension 2)
        circular = (dimension == 1)
        feat_diff_exp, bias_exp = self.compute_expectation(self.surface, dimension, component, circular=circular)
        
        expectation_trace = dict(
            x=feat_diff_exp,
            y=bias_exp,
            mode='lines',
            name=f'E[mu{dimension}_comp{component}]',
            line=dict(color='white', width=2),
            showlegend=(row == 1 and col == 1)  # Only show legend for first plot
        )
        
        self.fig.add_trace(go.Scatter(**expectation_trace), row=row, col=col)
    
    def _calculate_zoom_range(self, dimension, component, prob_threshold, use_log):
        """Calculate Y-axis zoom range based on probability threshold for this specific surface component."""
        # Get the surface data for this specific dimension and component
        if use_log:
            surface_data = self.surface.get_surf(dimension, component, log=True)
            # Convert to probability for threshold calculation
            prob_surface = np.exp(surface_data)
        else:
            prob_surface = self.surface.get_surf(dimension, component, log=False)
        
        bias_grid = self.surface.get_bias_grid(dimension)
        
        # Find rows where maximum probability across feat_diff exceeds threshold
        # Surfaces are already normalized probability distributions
        max_prob_per_bias = np.max(prob_surface, axis=1)
        
        # If all values are very small, use a percentile-based approach
        if np.max(max_prob_per_bias) < prob_threshold:
            # Use top 10% of values as the region of interest
            threshold_percentile = np.percentile(max_prob_per_bias, 90)
            valid_bias_indices = np.where(max_prob_per_bias >= threshold_percentile)[0]
        else:
            valid_bias_indices = np.where(max_prob_per_bias >= prob_threshold)[0]
        
        if len(valid_bias_indices) == 0:
            # Fallback: if no values exceed threshold, use full range
            return bias_grid.min(), bias_grid.max()
        
        # Get the bias range for valid indices
        min_bias_idx = valid_bias_indices.min()
        max_bias_idx = valid_bias_indices.max()
        
        y_min = bias_grid[min_bias_idx]
        y_max = bias_grid[max_bias_idx]
        
        # Make the zoom range symmetric around zero
        grid_min, grid_max = bias_grid.min(), bias_grid.max()
        
        # Find the maximum absolute distance from zero needed to include the probability region
        max_abs_distance = max(abs(y_min), abs(y_max))
        
        # Ensure minimum distance of 10 degrees from zero
        min_distance_from_zero = 10.0
        final_distance = max(max_abs_distance, min_distance_from_zero)
        
        # Create symmetric range around zero
        y_min_symmetric = -final_distance
        y_max_symmetric = final_distance
        
        # Ensure we don't go outside the actual bias grid bounds
        y_min = max(y_min_symmetric, grid_min)
        y_max = min(y_max_symmetric, grid_max)
        
        # If grid constraints break symmetry, expand the other side if possible
        if y_min > -final_distance and y_max < grid_max:
            # Can't go far enough negative, try to expand positive
            available_positive = grid_max
            y_max = min(available_positive, abs(y_min))
        elif y_max < final_distance and y_min > grid_min:
            # Can't go far enough positive, try to expand negative  
            available_negative = grid_min
            y_min = max(available_negative, -abs(y_max))
        
        return y_min, y_max
    
    def show(self):
        """Return the plotly figure."""
        return self.fig


class MultiPlot:
    """Handles multiple surface plotting."""
    
    @staticmethod
    def comparison(surfaces, titles, dimension=1, component=1, use_log=True, show_expectation=True):
        """Create comparison grid of surfaces for specific dimension/component."""
        n_surfaces = len(surfaces)
        cols = min(3, n_surfaces)
        rows = (n_surfaces + cols - 1) // cols
        
        surface_type = "Log-Likelihood" if use_log else "Likelihood"
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{title}<br>Mu{dimension} Comp{component} {surface_type}" for title in titles]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (surface, title) in enumerate(zip(surfaces, titles)):
            row, col = i // cols + 1, i % cols + 1
            
            # Get surface data
            z_data = surface.get_surf(dimension, component, log=use_log)
            bias_grid = surface.get_bias_grid(dimension)
            
            # Add heatmap
            fig.add_trace(go.Heatmap(
                z=z_data,
                x=surface.feat_diff_grid,
                y=bias_grid,
                colorscale='Viridis' if use_log else 'Plasma',
                showscale=(i == 0),
                hovertemplate='feat_diff: %{x}<br>bias: %{y}<br>value: %{z:.3g}<extra></extra>'
            ), row=row, col=col)
            
            # Add expectation line
            if show_expectation:
                # Use circular statistics for mu1 (dimension 1), linear for mu2 (dimension 2)
                circular = (dimension == 1)
                feat_diff_exp, bias_exp = Plot.compute_expectation(surface, dimension, component, circular=circular)
                fig.add_trace(go.Scatter(
                    x=feat_diff_exp,
                    y=bias_exp,
                    mode='lines',
                    name=f'E[{title}]',
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=(i < 3)
                ), row=row, col=col)
        
        fig.update_layout(title=f"Surface Comparison (Mu{dimension} Comp{component} {surface_type})", height=400 * rows)
        fig.update_xaxes(title_text="feat_diff")
        fig.update_yaxes(title_text=f"mu{dimension}_bias")
        
        return fig
    
    @staticmethod
    def expectation_curves(surfaces, titles, surface_rows, show_asymmetry=False):
        """Create expectation or asymmetry comparison plot for all dimensions and components."""
        if show_asymmetry:
            subplot_titles = ['Mu1 Component 1', 'Mu1 Component 2', 'Mu2 Component 1', 'Mu2 Component 2']
            plot_title = "Asymmetry Curves: Skewness vs feat_diff"
            y_label_prefix = "Asymmetry[mu"
        else:
            subplot_titles = ['Mu1 Component 1', 'Mu1 Component 2', 'Mu2 Component 1', 'Mu2 Component 2']
            plot_title = "Expectation Curves: E[bias] vs feat_diff"
            y_label_prefix = "E[mu"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles
        )
        
        colors = px.colors.qualitative.Set1
        
        for dim in [1, 2]:
            for comp in [1, 2]:
                row, col = dim, comp
                
                for i, (surface, title, surface_row) in enumerate(zip(surfaces, titles, surface_rows)):
                    if show_asymmetry:
                        feat_diff_values, y_values = Plot.compute_asymmetry(surface, dim, comp)
                        hover_y_label = f"Asymmetry[mu{dim}_bias]"
                    else:
                        # Use circular statistics for mu1 (dimension 1), linear for mu2 (dimension 2)
                        circular = (dim == 1)
                        feat_diff_values, y_values = Plot.compute_expectation(surface, dim, comp, circular=circular)
                        hover_y_label = f"E[mu{dim}_bias]"
                    
                    fig.add_trace(go.Scatter(
                        x=feat_diff_values,
                        y=y_values,
                        mode='lines',
                        name=title if dim == 1 and comp == 1 else None,
                        showlegend=(dim == 1 and comp == 1),
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'{title}<br>feat_diff: %{{x:.1f}}<br>{hover_y_label}: %{{y:.3f}}<br>' +
                                     f'sd_feat1: {surface_row["sd_feat1"]:.1f}<br>' +
                                     f'sd_feat2: {surface_row["sd_feat2"]:.1f}<br>' +
                                     f'sd_spat: {surface_row["sd_spat"]:.1f}<extra></extra>'
                    ), row=row, col=col)
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
        
        fig.update_layout(
            title=plot_title,
            height=800
        )
        
        fig.update_xaxes(title_text="feat_diff")
        fig.update_yaxes(title_text=f"{y_label_prefix}1_bias]", row=1)
        fig.update_yaxes(title_text=f"{y_label_prefix}2_bias]", row=2)
        
        return fig


# Convenience functions for one-liners
def plot_surface(surface, title="", show_expectation=True, use_log=True, auto_zoom=False, prob_threshold=0.0001):
    """One-liner for single surface plot showing all 4 components."""
    return Plot(surface).multi_component_heatmap(title, show_expectation, use_log, auto_zoom, prob_threshold).show()

def plot_3d(surface, dimension=1, component=1, title="", use_log=True):
    """One-liner for 3D surface plot."""
    return Plot(surface).heatmap_3d(dimension, component, use_log, title).show()

def compare_surfaces(surfaces, titles, dimension=1, component=1, use_log=True, show_expectation=True):
    """One-liner for surface comparison."""
    return MultiPlot.comparison(surfaces, titles, dimension, component, use_log, show_expectation)

def compare_expectations(surfaces, titles, surface_rows, show_asymmetry=False):
    """One-liner for expectation comparison."""
    return MultiPlot.expectation_curves(surfaces, titles, surface_rows, show_asymmetry)
