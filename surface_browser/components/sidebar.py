"""
Sidebar Components
=================

Reusable sidebar for directory selection and parameter filtering with performance info.
"""

import streamlit as st
from browser_utils.url_state import state_manager


def create_sidebar(data_manager, app_state=None):
    """Create sidebar and return filtered dataframe."""
    if app_state is None:
        app_state = {}
    
    st.sidebar.header("📁 Data Directory")

    # Directory input with state persistence
    directory = st.sidebar.text_input(
        "Surface Directory",
        value=app_state.get('directory', '../likelihood_surfaces_10k'),
        help="Directory containing surface pickle files",
        key="sidebar_directory"
    )
    
    # Save directory to state if changed
    if directory != app_state.get('directory'):
        state_manager.update_state({'directory': directory})

    # Load surfaces (fast - filename parsing only)
    with st.spinner("Scanning surface files..."):
        df = data_manager.load_directory(directory)

    if df is None or len(df) == 0:
        st.sidebar.error("No surfaces found")
        data_manager.suggest_filename_format()
        return None

    # Show loading performance stats
    stats = data_manager.get_surface_count_stats()
    st.sidebar.success(f"Found {stats['total_surfaces']} surfaces")

    # Show parameter space coverage
    with st.sidebar.expander("📊 Parameter Space Info"):
        st.write(f"**Coverage:**")
        st.write(f"- sd_feat1: {stats['unique_sd_feat1']} unique values")
        st.write(f"- sd_feat2: {stats['unique_sd_feat2']} unique values") 
        st.write(f"- sd_spat: {stats['unique_sd_spat']} unique values")
        st.write(f"- Loaded surfaces: {stats['cache_size']}")
        
        # Show surface type breakdown if available
        if 'regular_surfaces' in stats or 'averaged_surfaces' in stats:
            st.write(f"**Surface Types:**")
            if 'regular_surfaces' in stats:
                st.write(f"- Regular: {stats['regular_surfaces']}")
            if 'averaged_surfaces' in stats:
                st.write(f"- Averaged: {stats['averaged_surfaces']}")

        if st.button("🗑️ Clear Cache", help="Free memory by clearing loaded surfaces"):
            data_manager.clear_cache()
            st.rerun()

    # Global filters
    st.sidebar.header("🌐 Global Filters")
    
    # Equal noise only filter with state persistence
    equal_noise_only = st.sidebar.checkbox(
        "Equal noise only",
        value=app_state.get('equal_noise_only', False),
        key="equal_noise_filter",
        help="Filter surfaces to only show those where sd_feat1 == sd_feat2"
    )
    
    # Save equal noise filter state if changed
    if equal_noise_only != app_state.get('equal_noise_only'):
        state_manager.update_state({'equal_noise_only': equal_noise_only})
    
    # Apply equal noise filter first
    if equal_noise_only:
        df = df[df['sd_feat1'] == df['sd_feat2']].reset_index(drop=True)
        st.sidebar.info(f"Equal noise filter: {len(df)} surfaces remaining")
    
    # Surface type filter with state persistence
    surface_types_available = []
    if 'surface_type' in df.columns:
        surface_types_available = ['All'] + sorted(df['surface_type'].unique().tolist())
        
        surface_type_filter = st.sidebar.selectbox(
            "Surface Type",
            options=surface_types_available,
            index=0,  # Default to 'All'
            key="surface_type_filter",
            help="Filter by surface type (regular vs averaged)"
        )
        
        # Apply surface type filter
        if surface_type_filter != 'All':
            df = df[df['surface_type'] == surface_type_filter].reset_index(drop=True)
            st.sidebar.info(f"Surface type filter: {len(df)} {surface_type_filter} surfaces remaining")
    
    # Parameter filtering
    st.sidebar.header("🔍 Parameter Filters")

    param_ranges = data_manager.get_param_bounds(df)
    if not param_ranges:
        return df

    # Create sliders with proper handling for single values and state persistence
    filter_ranges = {}
    saved_filter_ranges = app_state.get('filter_ranges', {})

    for param in ['sd_feat1', 'sd_feat2', 'sd_spat']:
        if param in param_ranges:
            min_val, max_val = param_ranges[param]

            if min_val == max_val:
                st.sidebar.write(f"{param}: {min_val:.1f} (single value)")
                filter_ranges[param] = (min_val, max_val)
            else:
                # Use saved filter range or default to full range
                saved_range = saved_filter_ranges.get(param, (min_val, max_val))
                # Ensure saved range is within current bounds
                saved_range = (max(saved_range[0], min_val), min(saved_range[1], max_val))
                
                filter_ranges[param] = st.sidebar.slider(
                    f"{param} Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=saved_range,
                    step=0.1,
                    key=f"filter_{param}",
                    help=f"Filter surfaces by {param} values"
                )
    
    # Save filter ranges to state if changed
    if filter_ranges != saved_filter_ranges:
        state_manager.update_state({'filter_ranges': filter_ranges})

    # Apply filters (fast - no file loading)
    filtered_df = data_manager.filter_surfaces(df, **filter_ranges)

    # Show filter results
    if len(filtered_df) != len(df):
        st.sidebar.write(f"🔽 Filtered: {len(filtered_df)} / {len(df)} surfaces")
    else:
        st.sidebar.write(f"📋 All {len(filtered_df)} surfaces selected")

    # Performance warning for large selections
    if len(filtered_df) > 100:
        st.sidebar.warning(f"⚠️ Large selection ({len(filtered_df)} surfaces). Plots may be slow.")

    return filtered_df


def show_surface_info(params, title="Surface Parameters"):
    """Display surface parameters in a clean format."""
    st.subheader(title)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("sd_feat1", f"{params['sd_feat1']:.1f}")
    with col2:
        st.metric("sd_feat2", f"{params['sd_feat2']:.1f}")
    with col3:
        st.metric("sd_spat", f"{params['sd_spat']:.1f}")


def show_debug_info(surface_data):
    """Show debug information about surface data."""
    feat_diff_grid, mu1_error_grid, log_likelihood_surface, likelihood_surface = surface_data

    st.write("**Debug Information:**")
    st.write(f"- feat_diff_grid shape: {feat_diff_grid.shape}")
    st.write(f"- mu1_error_grid shape: {mu1_error_grid.shape}")
    st.write(f"- log_likelihood_surface shape: {log_likelihood_surface.shape}")
    st.write(f"- likelihood_surface shape: {likelihood_surface.shape}")
    st.write(f"- Resolution: {log_likelihood_surface.shape[1]} × {log_likelihood_surface.shape[0]} points")
    st.write(f"- feat_diff range: {feat_diff_grid[0, :].min():.1f} to {feat_diff_grid[0, :].max():.1f}")
    st.write(f"- mu1_error range: {mu1_error_grid[:, 0].min():.1f} to {mu1_error_grid[:, 0].max():.1f}")


def show_loading_performance(loading_time, n_surfaces):
    """Show loading performance metrics."""
    if loading_time > 1.0:
        time_per_surface = loading_time / max(n_surfaces, 1)
        st.info(f"⏱️ Loaded {n_surfaces} surfaces in {loading_time:.1f}s ({time_per_surface:.2f}s per surface)")


def warn_large_selection(n_selected, threshold=20):
    """Warn user about large selections that may be slow."""
    if n_selected > threshold:
        st.warning(f"⚠️ Loading {n_selected} surfaces may take a while. Consider using filters to reduce the selection.")
        return True
    return False