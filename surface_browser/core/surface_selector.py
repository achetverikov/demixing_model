"""
Surface Selector
===============

Reusable surface selection component with multiple methods.
"""

import streamlit as st
import random
import numpy as np
from typing import List, Dict, Any
from browser_utils.url_state import state_manager


class SurfaceSelector:
    """Reusable surface selection widget."""
    
    def __init__(self, df, key_prefix=""):
        self.df = df
        self.key_prefix = key_prefix
        self.selected_surfaces = []
    
    def render(self, methods=['manual', 'grid', 'random'], default_method='manual', 
               min_surfaces=2, max_surfaces=50):
        """Render selection interface and return selected surfaces."""

        if len(self.df) < min_surfaces:
            st.warning(f"Need at least {min_surfaces} surfaces for selection.")
            return []

        # Show selection size warning
        if len(self.df) > 100:
            st.info(f"💡 **Performance tip**: With {len(self.df)} surfaces available, consider using filters to reduce selection size for faster loading.")

        # Method selection
        method_names = {
            'manual': 'Manual Selection',
            'grid': 'Parameter Grid',
            'random': 'Random Sample'
        }

        available_methods = [method_names[m] for m in methods if m in method_names]

        selection_method = st.radio(
            "Selection Method",
            available_methods,
            horizontal=True,
            key=f"{self.key_prefix}_method"
        )

        # Route to appropriate method
        method_key = {v: k for k, v in method_names.items()}[selection_method]

        if method_key == 'manual':
            return self._manual_selection(max_surfaces)
        elif method_key == 'grid':
            return self._grid_selection()
        elif method_key == 'random':
            return self._random_selection(min_surfaces, max_surfaces)

        return []

    def _manual_selection(self, max_surfaces):
        """Manual surface selection with performance limits."""
        display_names = [
            f"sd_feat1={row['sd_feat1']:.1f}, sd_feat2={row['sd_feat2']:.1f}, sd_spat={row['sd_spat']:.1f}"
            for _, row in self.df.iterrows()
        ]

        # Limit selection size for performance
        default_selection = list(range(min(4, len(self.df))))

        selected_indices = st.multiselect(
            f"Select Surfaces (max {max_surfaces} for performance)",
            range(len(self.df)),
            format_func=lambda x: display_names[x],
            default=default_selection,
            key=f"{self.key_prefix}_manual",
            help=f"Select up to {max_surfaces} surfaces. Large selections may be slow to load."
        )

        # Enforce max selection limit
        if len(selected_indices) > max_surfaces:
            st.warning(f"⚠️ Selection limited to {max_surfaces} surfaces for performance. Using first {max_surfaces}.")
            selected_indices = selected_indices[:max_surfaces]

        return [self.df.iloc[i] for i in selected_indices]

    def _grid_selection(self):
        """Parameter grid selection."""
        st.subheader("Parameter Grid Selection")

        # Load saved state from global state manager
        state_key = f"grid_selection_{self.key_prefix}"
        current_state = state_manager.load_state() or {}
        saved_grid_state = current_state.get(state_key, {})
        
        # Optional parameter range filters
        use_filters = st.checkbox(
            "Use parameter range filters",
            value=saved_grid_state.get("use_filters", False),
            key=f"{self.key_prefix}_use_filters",
            help="Filter parameter ranges before grid selection"
        )
        
        filtered_df = self.df.copy()
        filter_ranges = {}
        
        if use_filters:
            st.write("**Parameter Range Filters:**")
            filter_cols = st.columns(3)
            
            for i, param in enumerate(['sd_feat1', 'sd_feat2', 'sd_spat']):
                with filter_cols[i]:
                    min_val = float(self.df[param].min())
                    max_val = float(self.df[param].max())
                    
                    if min_val != max_val:
                        # Get saved filter range or default to full range
                        default_range = (min_val, max_val)
                        if "filter_ranges" in saved_grid_state:
                            saved_range = saved_grid_state["filter_ranges"].get(param, default_range)
                            # Ensure saved range is within bounds
                            default_range = (max(saved_range[0], min_val), min(saved_range[1], max_val))
                        
                        filter_range = st.slider(
                            f"{param} range",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_range,
                            step=0.1,
                            key=f"{self.key_prefix}_filter_{param}",
                            help=f"Filter {param} values for grid selection"
                        )
                        
                        filter_ranges[param] = filter_range
                        
                        # Apply filter
                        filtered_df = filtered_df[
                            (filtered_df[param] >= filter_range[0]) & 
                            (filtered_df[param] <= filter_range[1])
                        ]
                    else:
                        st.write(f"{param}: {min_val:.1f} (fixed)")
                        filter_ranges[param] = (min_val, max_val)
            
            st.write(f"Filtered to {len(filtered_df)} surfaces")

        # Get unique values for each parameter from filtered data
        params = ['sd_feat1', 'sd_feat2', 'sd_spat']
        selected_values = {}

        st.write("**Grid Selection:**")
        cols = st.columns(3)
        for i, param in enumerate(params):
            with cols[i]:
                available_vals = sorted(filtered_df[param].unique())
                if len(available_vals) == 0:
                    st.warning(f"No {param} values available")
                    selected_values[param] = []
                else:
                    # Get saved grid selection or default to first 2 values
                    default_selection = available_vals[:min(2, len(available_vals))]
                    if "grid_selections" in saved_grid_state:
                        saved_selection = saved_grid_state["grid_selections"].get(param, [])
                        # Only use saved values that are still available
                        valid_saved = [v for v in saved_selection if v in available_vals]
                        if valid_saved:
                            default_selection = valid_saved
                    
                    selected_values[param] = st.multiselect(
                        f"{param} values",
                        available_vals,
                        default=default_selection,
                        key=f"{self.key_prefix}_grid_{param}"
                    )

        # Save current grid state to global state manager
        new_grid_state = {
            "use_filters": use_filters,
            "filter_ranges": filter_ranges,
            "grid_selections": selected_values
        }
        
        # Update global state
        current_state = state_manager.load_state() or {}
        current_state[state_key] = new_grid_state
        state_manager.save_state(current_state)

        # Find matching surfaces from filtered data
        selected_surfaces = []
        for _, surface in filtered_df.iterrows():
            # If no values selected for a parameter, include all available values
            match = True
            for param in params:
                if len(selected_values[param]) > 0:
                    if surface[param] not in selected_values[param]:
                        match = False
                        break
                # If selected_values[param] is empty, include all values (no filtering for this param)
            
            if match:
                selected_surfaces.append(surface)

        st.write(f"Selected {len(selected_surfaces)} surfaces from grid")
        return selected_surfaces

    def _random_selection(self, min_surfaces, max_surfaces):
        """Random surface selection with performance considerations."""
        # Adjust max for large datasets
        effective_max = min(max_surfaces, len(self.df), 20)  # Cap at 20 for performance

        n_random = st.slider(
            "Number of random surfaces",
            min_surfaces,
            effective_max,
            min(6, effective_max),
            key=f"{self.key_prefix}_random_n",
            help=f"Limited to {effective_max} for performance with large datasets"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎲 New Sample", key=f"{self.key_prefix}_random_btn"):
                # Force rerun to get new random sample
                st.rerun()

        # Generate random sample (deterministic based on session state)
        if f"{self.key_prefix}_random_seed" not in st.session_state:
            st.session_state[f"{self.key_prefix}_random_seed"] = random.randint(1, 10000)

        random.seed(st.session_state[f"{self.key_prefix}_random_seed"])
        selected_indices = random.sample(range(len(self.df)), n_random)
        selected_surfaces = [self.df.iloc[i] for i in selected_indices]

        with col2:
            st.write(f"Random sample (seed: {st.session_state[f'{self.key_prefix}_random_seed']})")

        return selected_surfaces


class SingleSurfaceSelector:
    """Simplified selector for single surface selection."""

    def __init__(self, df):
        self.df = df

    def render(self):
        """Render single surface selector with parameter-based selection."""
        # Initialize session state for parameter selection
        if 'single_sd_feat1' not in st.session_state:
            st.session_state.single_sd_feat1 = None
        if 'single_sd_feat2' not in st.session_state:
            st.session_state.single_sd_feat2 = None
        if 'single_sd_spat' not in st.session_state:
            st.session_state.single_sd_spat = None
            
        st.subheader("Surface Selection")
        
        # Toggle between dropdown and manual input
        input_mode = st.radio(
            "Selection mode:",
            ["Dropdown", "Manual input"],
            horizontal=True,
            key="single_input_mode"
        )
        
        # Three columns for the three parameters
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            if input_mode == "Dropdown":
                # Get available sd_feat1 values (filtered by other selected parameters)
                available_df = self.df.copy()
                if st.session_state.single_sd_feat2 is not None:
                    available_df = available_df[available_df['sd_feat2'] == st.session_state.single_sd_feat2]
                if st.session_state.single_sd_spat is not None:
                    available_df = available_df[available_df['sd_spat'] == st.session_state.single_sd_spat]
                
                feat1_options = sorted(available_df['sd_feat1'].unique())
                
                # Find current index
                current_idx = 0
                if st.session_state.single_sd_feat1 is not None and st.session_state.single_sd_feat1 in feat1_options:
                    current_idx = feat1_options.index(st.session_state.single_sd_feat1)
                elif len(feat1_options) > 0:
                    st.session_state.single_sd_feat1 = feat1_options[0]
                
                selected_feat1 = st.selectbox(
                    "sd_feat1",
                    options=feat1_options,
                    index=current_idx,
                    key="single_feat1_select"
                )
                st.session_state.single_sd_feat1 = selected_feat1
            else:
                # Manual input
                all_feat1_values = sorted(self.df['sd_feat1'].unique())
                min_val, max_val = min(all_feat1_values), max(all_feat1_values)
                default_val = st.session_state.single_sd_feat1 if st.session_state.single_sd_feat1 is not None else min_val
                
                selected_feat1 = st.number_input(
                    "sd_feat1",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=10.0,
                    key="single_feat1_input"
                )
                st.session_state.single_sd_feat1 = selected_feat1
        
        with col2:
            if input_mode == "Dropdown":
                # Get available sd_feat2 values (filtered by other selected parameters)
                available_df = self.df.copy()
                if st.session_state.single_sd_feat1 is not None:
                    available_df = available_df[available_df['sd_feat1'] == st.session_state.single_sd_feat1]
                if st.session_state.single_sd_spat is not None:
                    available_df = available_df[available_df['sd_spat'] == st.session_state.single_sd_spat]
                
                feat2_options = sorted(available_df['sd_feat2'].unique())
                
                # Find current index
                current_idx = 0
                if st.session_state.single_sd_feat2 is not None and st.session_state.single_sd_feat2 in feat2_options:
                    current_idx = feat2_options.index(st.session_state.single_sd_feat2)
                elif len(feat2_options) > 0:
                    st.session_state.single_sd_feat2 = feat2_options[0]
                
                selected_feat2 = st.selectbox(
                    "sd_feat2",
                    options=feat2_options,
                    index=current_idx,
                    key="single_feat2_select"
                )
                st.session_state.single_sd_feat2 = selected_feat2
            else:
                # Manual input
                all_feat2_values = sorted(self.df['sd_feat2'].unique())
                min_val, max_val = min(all_feat2_values), max(all_feat2_values)
                default_val = st.session_state.single_sd_feat2 if st.session_state.single_sd_feat2 is not None else min_val
                
                selected_feat2 = st.number_input(
                    "sd_feat2",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=10.0,
                    key="single_feat2_input"
                )
                st.session_state.single_sd_feat2 = selected_feat2
        
        with col3:
            if input_mode == "Dropdown":
                # Get available sd_spat values (filtered by other selected parameters)
                available_df = self.df.copy()
                if st.session_state.single_sd_feat1 is not None:
                    available_df = available_df[available_df['sd_feat1'] == st.session_state.single_sd_feat1]
                if st.session_state.single_sd_feat2 is not None:
                    available_df = available_df[available_df['sd_feat2'] == st.session_state.single_sd_feat2]
                
                spat_options = sorted(available_df['sd_spat'].unique())
                
                # Find current index
                current_idx = 0
                if st.session_state.single_sd_spat is not None and st.session_state.single_sd_spat in spat_options:
                    current_idx = spat_options.index(st.session_state.single_sd_spat)
                elif len(spat_options) > 0:
                    st.session_state.single_sd_spat = spat_options[0]
                
                selected_spat = st.selectbox(
                    "sd_spat",
                    options=spat_options,
                    index=current_idx,
                    key="single_spat_select"
                )
                st.session_state.single_sd_spat = selected_spat
            else:
                # Manual input
                all_spat_values = sorted(self.df['sd_spat'].unique())
                min_val, max_val = min(all_spat_values), max(all_spat_values)
                default_val = st.session_state.single_sd_spat if st.session_state.single_sd_spat is not None else min_val
                
                selected_spat = st.number_input(
                    "sd_spat",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=10.0,
                    key="single_spat_input"
                )
                st.session_state.single_sd_spat = selected_spat

        with col4:
            if st.button("🎲 Random"):
                # Select random surface from available options
                random_surface = self.df.sample(n=1).iloc[0]
                st.session_state.single_sd_feat1 = random_surface['sd_feat1']
                st.session_state.single_sd_feat2 = random_surface['sd_feat2']
                st.session_state.single_sd_spat = random_surface['sd_spat']
                st.rerun()
        
        # Find the matching surface
        if input_mode == "Dropdown":
            # Exact match for dropdown mode
            matching_surfaces = self.df[
                (self.df['sd_feat1'] == st.session_state.single_sd_feat1) &
                (self.df['sd_feat2'] == st.session_state.single_sd_feat2) &
                (self.df['sd_spat'] == st.session_state.single_sd_spat)
            ]
            
            if len(matching_surfaces) == 0:
                st.error("No surface found with selected parameters")
                return None
            elif len(matching_surfaces) > 1:
                st.warning(f"Multiple surfaces found ({len(matching_surfaces)}), using first one")
            
            return matching_surfaces.iloc[0]
        else:
            # For manual input, find the closest surface
            target_feat1 = st.session_state.single_sd_feat1
            target_feat2 = st.session_state.single_sd_feat2  
            target_spat = st.session_state.single_sd_spat
            
            # Calculate distances to all surfaces
            distances = []
            for _, row in self.df.iterrows():
                dist = abs(row['sd_feat1'] - target_feat1) + abs(row['sd_feat2'] - target_feat2) + abs(row['sd_spat'] - target_spat)
                distances.append(dist)
            
            # Find closest surface
            closest_idx = np.argmin(distances)
            closest_surface = self.df.iloc[closest_idx]
            
            # Show info about the closest match
            if distances[closest_idx] > 0:
                st.info(f"Closest surface: sd_feat1={closest_surface['sd_feat1']}, sd_feat2={closest_surface['sd_feat2']}, sd_spat={closest_surface['sd_spat']}")
            
            return closest_surface


def quick_select(df, method='random', n=4, key_prefix=""):
    """Quick selection for when you just need surfaces fast."""
    if method == 'random':
        indices = random.sample(range(len(df)), min(n, len(df)))
        return [df.iloc[i] for i in indices]
    elif method == 'first':
        return [df.iloc[i] for i in range(min(n, len(df)))]
    elif method == 'last':
        return [df.iloc[i] for i in range(max(0, len(df)-n), len(df))]
    elif method == 'spread':
        # Evenly spaced selection
        indices = [int(i * len(df) / n) for i in range(n)]
        return [df.iloc[i] for i in indices[:len(df)]]

    return []