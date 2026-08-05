#!/usr/bin/env python3
"""
Surface Browser - Main Entry Point
=================================

Streamlit app for interactive likelihood surface exploration.

Usage: streamlit run main.py
"""

import streamlit as st
import json
# data_manager puts the repo root on sys.path, so shared.* is importable after it.
from core.data_manager import SurfaceDataManager
from shared.config import averaged_surfaces_dir
from components.sidebar import create_sidebar
from tabs import SingleTab, ComparisonTab, ExpectationTab, SpaceTab, StatsTab
from browser_utils.url_state import state_manager, init_url_state_js

# Tab configuration
TABS = {
    '🔍 Single Surface': SingleTab,
    '⚖️ Compare Surfaces': ComparisonTab, 
    '📈 Expectation Curves': ExpectationTab,
    '📊 Parameter Space': SpaceTab,
    '📈 Statistics': StatsTab
}

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Surface Browser",
        page_icon="🌊", 
        layout="wide"
    )
    
    # Load saved state with defaults (must happen early)
    default_state = {
        'directory': str(averaged_surfaces_dir(20)),  # Default to averaged surfaces
        'equal_noise_only': False,
        'filter_ranges': {},
        'expectation_plot_type': 'Expectation curves'
    }
    app_state = state_manager.get_state_with_defaults(default_state)
    
    # Show compressed URL info for debugging
    if st.sidebar.button("🔗 Show URL Info", help="Debug: show current compressed state"):
        current_state = state_manager.load_state()
        if current_state:
            st.sidebar.json(current_state)
            # Show compression ratio
            original_size = len(json.dumps(current_state))
            compressed_size = len(st.query_params.get("s", ""))
            if compressed_size > 0:
                ratio = (1 - compressed_size / original_size) * 100
                st.sidebar.write(f"Compression: {ratio:.1f}% ({original_size} → {compressed_size} chars)")
        else:
            st.sidebar.write("No state in URL")
    
    st.title("🌊 Likelihood Surface Browser")
    st.markdown("Interactive exploration of computed likelihood surfaces")
    
    # Initialize data manager
    data_manager = SurfaceDataManager()
    
    # Create sidebar and load data (pass current state)
    filtered_df = create_sidebar(data_manager, app_state)
    
    if filtered_df is None or len(filtered_df) == 0:
        st.error("No surfaces found or match current filters.")
        st.info("Expected files with pattern: surface_*.pkl")
        return
    
    st.success(f"Found {len(filtered_df)} surfaces")
    
    # Create tabs and render content with state tracking
    tab_names = list(TABS.keys())
    
    # Track which tab is currently active using radio buttons
    current_tab_index = app_state.get('selected_tab', 0)
    if current_tab_index >= len(tab_names):
        current_tab_index = 0
    
    # Use radio buttons to track tab selection
    selected_tab_index = st.radio(
        "Navigate:",
        range(len(tab_names)),
        format_func=lambda x: tab_names[x],
        index=current_tab_index,
        horizontal=True,
        key="tab_selector"
    )
    
    # Save tab selection to state if changed
    if selected_tab_index != current_tab_index:
        state_manager.update_state({'selected_tab': selected_tab_index})
    
    # Render the selected tab content
    tab_name = tab_names[selected_tab_index]
    tab_class = TABS[tab_name]
    
    st.markdown(f"### {tab_name}")
    tab_handler = tab_class(data_manager, filtered_df)
    tab_handler.render()

if __name__ == "__main__":
    main()
