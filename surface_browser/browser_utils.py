"""
Utilities and Configuration
===========================

Common utilities, configuration, and helper functions.
"""

import streamlit as st
import numpy as np
import pandas as pd


# Configuration constants
DEFAULT_DIRECTORY = "../likelihood_surfaces"
DEFAULT_PLOT_HEIGHT = 600
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Surface Browser",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def format_surface_title(params):
    """Create standardized surface title from parameters."""
    return f"sd_feat1={params['sd_feat1']:.1f}, sd_feat2={params['sd_feat2']:.1f}, sd_spat={params['sd_spat']:.1f}"


def format_short_title(params):
    """Create short surface title for plots."""
    return f"sf1={params['sd_feat1']:.1f}, sf2={params['sd_feat2']:.1f}, sp={params['sd_spat']:.1f}"


def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator."""
    return numerator / denominator if denominator != 0 else default


def compute_surface_stats(surface_data):
    """Compute basic statistics for a surface."""
    _, _, log_likelihood_surface, likelihood_surface = surface_data
    
    return {
        'log_likelihood_min': float(np.min(log_likelihood_surface)),
        'log_likelihood_max': float(np.max(log_likelihood_surface)),
        'log_likelihood_mean': float(np.mean(log_likelihood_surface)),
        'log_likelihood_std': float(np.std(log_likelihood_surface)),
        'likelihood_min': float(np.min(likelihood_surface)),
        'likelihood_max': float(np.max(likelihood_surface)),
        'likelihood_mean': float(np.mean(likelihood_surface)),
        'likelihood_std': float(np.std(likelihood_surface))
    }


def validate_surface_data(surface_data):
    """Validate surface data structure."""
    if surface_data is None:
        return False, "Surface data is None"
    
    if len(surface_data) != 4:
        return False, f"Expected 4 components, got {len(surface_data)}"
    
    feat_diff_grid, mu1_error_grid, log_likelihood_surface, likelihood_surface = surface_data
    
    # Check shapes match
    if not all(arr.shape == feat_diff_grid.shape for arr in surface_data):
        return False, "Surface component shapes don't match"
    
    # Check for NaN or infinite values
    for i, arr in enumerate(surface_data):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            names = ['feat_diff_grid', 'mu1_error_grid', 'log_likelihood_surface', 'likelihood_surface']
            return False, f"Found NaN or infinite values in {names[i]}"
    
    return True, "Valid"


def create_error_message(error, context=""):
    """Create standardized error message."""
    return f"Error {context}: {str(error)}"


def with_loading(func, message="Loading..."):
    """Context manager for showing loading spinner."""
    with st.spinner(message):
        return func()


class PerformanceTimer:
    """Simple performance timer for debugging."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        st.write(f"⏱️ {self.name}: {elapsed:.2f}s")


def debug_dataframe(df, title="DataFrame Info"):
    """Show debug information about a dataframe."""
    if st.checkbox(f"Show {title}", value=False):
        st.write(f"**{title}:**")
        st.write(f"- Shape: {df.shape}")
        st.write(f"- Columns: {list(df.columns)}")
        st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        if len(df) > 0:
            st.write("- Sample data:")
            st.dataframe(df.head(3))


def get_color_palette(n_colors):
    """Get color palette for plots."""
    if n_colors <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:n_colors]
    
    # Generate additional colors if needed
    import plotly.express as px
    return px.colors.qualitative.Set1[:n_colors] if n_colors <= 10 else px.colors.qualitative.Light24[:n_colors]


def cache_key(params):
    """Generate cache key from parameters."""
    return f"{params['sd_feat1']:.1f}_{params['sd_feat2']:.1f}_{params['sd_spat']:.1f}"


def format_number(value, precision=2):
    """Format number for display."""
    if abs(value) >= 1000:
        return f"{value:.0f}"
    elif abs(value) >= 1:
        return f"{value:.{precision}f}"
    else:
        return f"{value:.{precision+1}f}"


def create_download_button(data, filename, label="Download"):
    """Create download button for data."""
    if isinstance(data, pd.DataFrame):
        csv_data = data.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv_data,
            file_name=filename,
            mime='text/csv'
        )
    else:
        st.download_button(
            label=label,
            data=str(data),
            file_name=filename,
            mime='text/plain'
        )


# Streamlit session state helpers
def get_session_state(key, default=None):
    """Get value from session state with default."""
    return st.session_state.get(key, default)


def set_session_state(key, value):
    """Set value in session state."""
    st.session_state[key] = value


def clear_session_state(*keys):
    """Clear specific keys from session state."""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]
