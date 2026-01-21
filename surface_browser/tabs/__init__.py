"""
Tab Components
=============

Base classes and implementations for all tab handlers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from core.plotting import plot_surface, plot_3d, compare_surfaces, compare_expectations
from core.surface_selector import SurfaceSelector, SingleSurfaceSelector
from components.sidebar import show_surface_info, show_debug_info
from browser_utils.url_state import state_manager


class BaseTab(ABC):
    """Base class for all tab handlers."""

    def __init__(self, data_manager, filtered_df):
        self.data_manager = data_manager
        self.filtered_df = filtered_df

    @abstractmethod
    def render(self):
        """Render the tab content."""
        pass

    def check_min_surfaces(self, min_count=1):
        """Check if enough surfaces are available."""
        if len(self.filtered_df) < min_count:
            st.warning(f"Need at least {min_count} surface(s). Current: {len(self.filtered_df)}")
            return False
        return True


class SingleTab(BaseTab):
    """Single surface viewer tab."""

    def render(self):
        st.header("Single Surface Viewer")

        if not self.check_min_surfaces(1):
            return

        # Surface selection
        selector = SingleSurfaceSelector(self.filtered_df)
        selected_surface = selector.render()

        # Load surface
        surface = self.data_manager.load_surface(selected_surface)
        if surface is None:
            return

        # Get parameters for title (from surface row data)
        params = {
            'sd_feat1': selected_surface['sd_feat1'],
            'sd_feat2': selected_surface['sd_feat2'],
            'sd_spat': selected_surface['sd_spat']
        }

        # Display options
        st.subheader("Display Options")
        
        # Initialize session state for display options
        if 'single_show_expectation' not in st.session_state:
            st.session_state.single_show_expectation = True
        if 'single_use_log' not in st.session_state:
            st.session_state.single_use_log = False
        if 'single_show_debug' not in st.session_state:
            st.session_state.single_show_debug = False
        if 'single_plot_type' not in st.session_state:
            st.session_state.single_plot_type = "2D Heatmaps"
        if 'single_auto_zoom' not in st.session_state:
            st.session_state.single_auto_zoom = True
        if 'single_prob_threshold' not in st.session_state:
            st.session_state.single_prob_threshold = 0.0001
            
        col1, col2, col3 = st.columns(3)

        with col1:
            plot_type = st.radio(
                "Plot Type",
                ["2D Heatmaps", "3D Surface"],
                horizontal=True,
                index=["2D Heatmaps", "3D Surface"].index(st.session_state.single_plot_type),
                key="single_plot_type_radio"
            )
            st.session_state.single_plot_type = plot_type

        with col2:
            show_expectation = st.checkbox(
                "Show E[bias]", 
                value=st.session_state.single_show_expectation, 
                key="single_show_expectation_cb"
            )
            st.session_state.single_show_expectation = show_expectation
            
            use_log = st.checkbox(
                "Use Log-Likelihood", 
                value=st.session_state.single_use_log, 
                key="single_use_log_cb"
            )
            st.session_state.single_use_log = use_log
            
            show_debug = st.checkbox(
                "Debug info", 
                value=st.session_state.single_show_debug, 
                key="single_show_debug_cb"
            )
            st.session_state.single_show_debug = show_debug

        with col3:
            auto_zoom = st.checkbox(
                "Auto-zoom Y-axis", 
                value=st.session_state.single_auto_zoom, 
                key="single_auto_zoom_cb",
                help="Zoom Y-axis to regions with probability above threshold"
            )
            st.session_state.single_auto_zoom = auto_zoom
            
            if auto_zoom:
                threshold_options = {
                    "0.1 (10%)": 0.1,
                    "0.01 (1%)": 0.01,
                    "0.001 (0.1%)": 0.001,
                    "0.0001 (0.01%)": 0.0001,
                    "0.00001 (0.001%)": 0.00001,
                    "0.000001 (0.0001%)": 0.000001
                }
                
                # Find current value in options
                current_val = st.session_state.single_prob_threshold
                current_key = None
                for key, val in threshold_options.items():
                    if abs(val - current_val) < 1e-8:
                        current_key = key
                        break
                
                if current_key is None:
                    current_key = "0.0001 (0.01%)"  # Default fallback
                
                selected_key = st.selectbox(
                    "Prob threshold",
                    options=list(threshold_options.keys()),
                    index=list(threshold_options.keys()).index(current_key),
                    key="single_prob_threshold_select",
                    help="Minimum probability threshold for zoom region"
                )
                
                prob_threshold = threshold_options[selected_key]
                st.session_state.single_prob_threshold = prob_threshold
            else:
                prob_threshold = st.session_state.single_prob_threshold

        # Show debug info if requested
        if show_debug:
            st.text(surface.summary())

        # Create and show plot
        title = f"sd_feat1={params['sd_feat1']:.1f}, sd_feat2={params['sd_feat2']:.1f}, sd_spat={params['sd_spat']:.1f}"

        if plot_type == "2D Heatmaps":
            fig = plot_surface(surface, title, show_expectation, use_log, auto_zoom, prob_threshold)
        else:
            # Initialize session state for 3D options
            if 'single_3d_dimension' not in st.session_state:
                st.session_state.single_3d_dimension = 1
            if 'single_3d_component' not in st.session_state:
                st.session_state.single_3d_component = 1
                
            # For 3D, let user select dimension and component
            col3, col4 = st.columns(2)
            with col3:
                dimension = st.selectbox(
                    "Dimension", 
                    [1, 2], 
                    format_func=lambda x: f"Mu{x}", 
                    index=[1, 2].index(st.session_state.single_3d_dimension),
                    key="single_3d_dimension_sb"
                )
                st.session_state.single_3d_dimension = dimension
            with col4:
                component = st.selectbox(
                    "Component", 
                    [1, 2], 
                    index=[1, 2].index(st.session_state.single_3d_component),
                    key="single_3d_component_sb"
                )
                st.session_state.single_3d_component = component
            fig = plot_3d(surface, dimension, component, title, use_log)

        st.plotly_chart(fig, use_container_width=True)


class ComparisonTab(BaseTab):
    """Surface comparison tab."""

    def render(self):
        st.header("Surface Comparison")

        if not self.check_min_surfaces(2):
            return

        # Surface selection
        selector = SurfaceSelector(self.filtered_df, key_prefix="compare")
        selected_surfaces = selector.render()

        if not selected_surfaces:
            return

        st.write(f"Comparing {len(selected_surfaces)} surfaces")

        # Comparison options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dimension = st.selectbox("Dimension", [1, 2], format_func=lambda x: f"Mu{x}", key="comp_dim")
        with col2:
            component = st.selectbox("Component", [1, 2], key="comp_comp")
        with col3:
            show_expectation = st.checkbox("Show E[bias]", value=True, key="comp_exp")
        with col4:
            use_log = st.checkbox("Use log-likelihood", value=True, key="comp_log")

        # Load and plot surfaces
        surfaces, titles = self.data_manager.load_surface_batch(selected_surfaces)

        if surfaces:
            fig = compare_surfaces(surfaces, titles, dimension, component, use_log, show_expectation)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No surfaces could be loaded. Check for file errors.")


class ExpectationTab(BaseTab):
    """Expectation comparison tab."""

    def render(self):
        st.header("Expectation Comparison")
        st.markdown("Compare E[bias] curves or asymmetry across parameter combinations for all dimensions and components")

        if not self.check_min_surfaces(2):
            return

        # Surface selection
        selector = SurfaceSelector(self.filtered_df, key_prefix="expectation")
        selected_surfaces = selector.render(methods=['manual', 'grid', 'random'])

        if not selected_surfaces:
            return

        # Load saved plot type
        current_state = state_manager.load_state() or {}
        saved_plot_type = current_state.get('expectation_plot_type', 'Expectation curves')
        
        # Plot type selection
        plot_type = st.radio(
            "Plot Type",
            ["Expectation curves", "Asymmetry curves"],
            index=0 if saved_plot_type == "Expectation curves" else 1,
            horizontal=True,
            key="expectation_plot_type",
            help="Choose between E[bias] curves or asymmetry (skewness) curves"
        )
        
        # Save plot type to state if changed
        if plot_type != saved_plot_type:
            state_manager.update_state({'expectation_plot_type': plot_type})
        
        show_asymmetry = plot_type == "Asymmetry curves"
        
        st.write(f"Comparing {'asymmetry' if show_asymmetry else 'expectation'} curves for {len(selected_surfaces)} surfaces")

        # Load surfaces and create plot
        surfaces, titles = self.data_manager.load_surface_batch(selected_surfaces)

        if surfaces:
            fig = compare_expectations(surfaces, titles, selected_surfaces, show_asymmetry)
            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            self._show_expectation_stats(surfaces, titles, selected_surfaces, show_asymmetry)
        else:
            st.error("No surfaces could be loaded. Check for file errors.")
    
    def _show_expectation_stats(self, surfaces, titles, surface_rows, show_asymmetry=False):
        """Show expectation or asymmetry statistics table for all dimensions and components."""
        st.subheader("Asymmetry Statistics" if show_asymmetry else "Expectation Statistics")
        
        from core.plotting import Plot
        
        stats_data = []
        for surface, title, surface_row in zip(surfaces, titles, surface_rows):
            for dim in [1, 2]:
                for comp in [1, 2]:
                    if show_asymmetry:
                        _, values = Plot.compute_asymmetry(surface, dim, comp)
                        value_label = "Asymmetry"
                    else:
                        # Use circular statistics for mu1 (dimension 1), linear for mu2 (dimension 2)
                        circular = (dim == 1)
                        _, values = Plot.compute_expectation(surface, dim, comp, circular=circular)
                        value_label = "E[bias]"
                    
                    stats_data.append({
                        'Surface': title,
                        'Dimension': f'Mu{dim}',
                        'Component': comp,
                        'sd_feat1': surface_row['sd_feat1'],
                        'sd_feat2': surface_row['sd_feat2'],
                        'sd_spat': surface_row['sd_spat'],
                        f'{value_label} Mean': np.mean(values),
                        f'{value_label} Std': np.std(values),
                        f'{value_label} Range': np.max(values) - np.min(values),
                        f'Max |{value_label}|': np.max(np.abs(values))
                    })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)


class SpaceTab(BaseTab):
    """Parameter space overview tab."""
    
    def render(self):
        st.header("Parameter Space Overview")
        
        if not self.check_min_surfaces(1):
            return
        
        # 3D scatter plot
        import plotly.express as px
        
        fig = px.scatter_3d(
            self.filtered_df,
            x='sd_feat1',
            y='sd_feat2',
            z='sd_spat',
            title="Parameter Space Coverage",
            hover_data=['filename']
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameter distributions
        st.subheader("Parameter Distributions")
        col1, col2, col3 = st.columns(3)
        
        for i, param in enumerate(['sd_feat1', 'sd_feat2', 'sd_spat']):
            with [col1, col2, col3][i]:
                fig = px.histogram(self.filtered_df, x=param, title=f"{param} Distribution")
                st.plotly_chart(fig, use_container_width=True)


class StatsTab(BaseTab):
    """Surface statistics tab."""
    
    def render(self):
        st.header("Surface Statistics")
        
        if not self.check_min_surfaces(1):
            return
        
        # Sample surfaces for analysis
        sample_size = min(10, len(self.filtered_df))
        sample_surfaces = self.filtered_df.sample(n=sample_size) if len(self.filtered_df) > sample_size else self.filtered_df
        
        stats_data = []
        progress_bar = st.progress(0)
        
        for i, (_, surface_row) in enumerate(sample_surfaces.iterrows()):
            surface = self.data_manager.load_surface(surface_row)
            if surface is not None:
                # Compute stats for all components
                all_log_likelihoods = []
                for dim in [1, 2]:
                    for comp in [1, 2]:
                        log_likelihood_surface = surface.get_surf(dim, comp, log=True)
                        all_log_likelihoods.append(log_likelihood_surface.flatten())
                
                combined_log_likelihood = np.concatenate(all_log_likelihoods)
                
                stats_data.append({
                    'sd_feat1': surface_row['sd_feat1'],
                    'sd_feat2': surface_row['sd_feat2'],
                    'sd_spat': surface_row['sd_spat'],
                    'log_likelihood_min': float(np.min(combined_log_likelihood)),
                    'log_likelihood_max': float(np.max(combined_log_likelihood)),
                    'log_likelihood_mean': float(np.mean(combined_log_likelihood)),
                    'log_likelihood_std': float(np.std(combined_log_likelihood))
                })
            
            progress_bar.progress((i + 1) / len(sample_surfaces))
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            
            st.subheader(f"Statistics from {len(stats_df)} surfaces")
            st.dataframe(stats_df)
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                import plotly.express as px
                fig = px.scatter(
                    stats_df,
                    x='sd_feat1',
                    y='log_likelihood_max',
                    color='sd_spat',
                    title="Max Log-Likelihood vs sd_feat1"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    stats_df,
                    x='log_likelihood_mean',
                    y='log_likelihood_std',
                    color='sd_feat2',
                    title="Log-Likelihood Mean vs Std"
                )
                st.plotly_chart(fig, use_container_width=True)


# Export all tab classes
__all__ = ['SingleTab', 'ComparisonTab', 'ExpectationTab', 'SpaceTab', 'StatsTab']