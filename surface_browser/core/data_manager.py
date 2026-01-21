"""
Surface Data Manager
===================

Efficient data loading with filename-based parameter extraction and lazy loading.
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
import streamlit as st
import re
import sys
import os

# Add parent directory to path to import Surface class
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Surface classes to make them available for unpickling
from shared.utils import Surface

# Import AveragedSurface from the correct path
try:
    sys.path.insert(0, parent_dir)
    from create_averaged_surfaces_from_samples import AveragedSurface
    # Make it available globally for pickle unpickling
    globals()['AveragedSurface'] = AveragedSurface
    # Also add to sys.modules to handle different import paths
    import sys
    if 'create_averaged_surfaces_from_samples' in sys.modules:
        sys.modules['__main__'].AveragedSurface = AveragedSurface
except ImportError as e:
    # Create a dummy AveragedSurface class if import fails
    class AveragedSurface(Surface):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.n_sample_files_used = kwargs.get('n_sample_files_used', 0)
            self.averaged_from_params = kwargs.get('averaged_from_params', [])
            self.kde_parameters = kwargs.get('kde_parameters', {})
    
    globals()['AveragedSurface'] = AveragedSurface
    print(f"Warning: Could not import AveragedSurface, using fallback class. Error: {e}")


class SurfaceDataManager:
    """Manages surface data loading with lazy loading and filename parsing."""

    def __init__(self):
        self.surfaces_df = None
        self.directory = None
        self._surface_cache = {}

    def load_directory(self, directory: str = "../likelihood_surfaces_10k") -> pd.DataFrame:
        """Scan directory and extract parameters from filenames (fast)."""
        directory = Path(directory)

        if self.directory == directory and self.surfaces_df is not None:
            return self.surfaces_df

        self.directory = directory
        surfaces = []

        if not directory.exists():
            return pd.DataFrame()

        # Patterns to match both regular and averaged surfaces
        regular_pattern = r'surface_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_.*\.pkl'
        averaged_pattern = r'averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl'

        # Scan both regular surfaces and averaged surfaces
        for pattern_glob, pattern_re, surface_type in [
            ("surface_*.pkl", regular_pattern, "regular"),
            ("averaged_*.pkl", averaged_pattern, "averaged")
        ]:
            for surface_file in directory.glob(pattern_glob):
                try:
                    # Try to extract parameters from filename first (fast)
                    match = re.match(pattern_re, surface_file.name)

                    if match:
                        # Parse from filename - very fast
                        sd_feat1, sd_feat2, sd_spat = map(float, match.groups())
                        surfaces.append({
                            'file': surface_file,
                            'sd_feat1': sd_feat1,
                            'sd_feat2': sd_feat2,
                            'sd_spat': sd_spat,
                            'filename': surface_file.name,
                            'source': 'filename',
                            'surface_type': surface_type
                        })
                    else:
                        # Fallback: load Surface object to get parameters (slow)
                        try:
                            with open(surface_file, 'rb') as f:
                                data = pickle.load(f)
                            
                            # Extract parameters from the data structure
                            if isinstance(data, dict) and 'parameters' in data:
                                params = data['parameters']
                                surfaces.append({
                                    'file': surface_file,
                                    'sd_feat1': params.get('sd_feat1', 0),
                                    'sd_feat2': params.get('sd_feat2', 0), 
                                    'sd_spat': params.get('sd_spat', 0),
                                    'filename': surface_file.name,
                                    'source': 'pickle',
                                    'surface_type': 'unknown'
                                })
                        except Exception:
                            # Skip corrupted files
                            continue

                except Exception:
                    continue

        self.surfaces_df = pd.DataFrame(surfaces).sort_values(['sd_feat1', 'sd_feat2', 'sd_spat'])

        # Show loading stats
        if len(surfaces) > 0:
            filename_parsed = sum(1 for s in surfaces if s.get('source') == 'filename')
            pickle_parsed = len(surfaces) - filename_parsed
            regular_count = sum(1 for s in surfaces if s.get('surface_type') == 'regular')
            averaged_count = sum(1 for s in surfaces if s.get('surface_type') == 'averaged')
            st.sidebar.caption(f"Scanned {len(surfaces)} surfaces: {regular_count} regular, {averaged_count} averaged")
            st.sidebar.caption(f"Parsed: {filename_parsed} from filename, {pickle_parsed} from file")

        return self.surfaces_df

    def load_surface(self, surface_row):
        """Load Surface object from new format."""
        file_path = str(surface_row['file'])

        # Check cache first
        if file_path in self._surface_cache:
            return self._surface_cache[file_path]

        try:
            # Custom unpickler to handle AveragedSurface import issues
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'AveragedSurface':
                        return AveragedSurface
                    return super().find_class(module, name)
            
            with open(surface_row['file'], 'rb') as f:
                data = CustomUnpickler(f).load()

            # Extract the Surface object from the checkpoint data
            if isinstance(data, dict) and 'surface' in data:
                surface = data['surface']
            else:
                surface = data  # Fallback if it's directly a Surface object

            # Cache the result
            self._surface_cache[file_path] = surface
            return surface

        except Exception as e:
            st.error(f"Error loading surface {surface_row['filename']}: {e}")
            return None

    def load_surface_batch(self, surface_rows, max_batch_size=50):
        """Load multiple Surface objects with progress bar and batch size limit."""
        if len(surface_rows) > max_batch_size:
            st.warning(f"Loading limited to {max_batch_size} surfaces for performance. Selected {len(surface_rows)}.")
            surface_rows = surface_rows[:max_batch_size]

        surfaces, titles = [], []

        # Show progress for large batches
        if len(surface_rows) > 5:
            progress_bar = st.progress(0)
            status_text = st.empty()

        for i, surface_row in enumerate(surface_rows):
            if len(surface_rows) > 5:
                status_text.text(f"Loading surface {i+1}/{len(surface_rows)}: {surface_row['filename']}")
                progress_bar.progress((i + 1) / len(surface_rows))

            surface = self.load_surface(surface_row)
            if surface is not None:
                surfaces.append(surface)
                surface_type_label = f" ({surface_row.get('surface_type', 'unknown')})"
                titles.append(f"sf1={surface_row['sd_feat1']:.1f}, sf2={surface_row['sd_feat2']:.1f}, sp={surface_row['sd_spat']:.1f}{surface_type_label}")

        # Clear progress indicators
        if len(surface_rows) > 5:
            progress_bar.empty()
            status_text.empty()

        return surfaces, titles

    def filter_surfaces(self, df: pd.DataFrame, **param_ranges) -> pd.DataFrame:
        """Filter surfaces by parameter ranges (very fast - no file loading)."""
        mask = pd.Series([True] * len(df))

        for param, (min_val, max_val) in param_ranges.items():
            if param in df.columns:
                mask &= (df[param] >= min_val) & (df[param] <= max_val)

        return df[mask]

    def get_param_bounds(self, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Get min/max ranges for parameters (very fast - no file loading)."""
        if df is None or len(df) == 0:
            return {}

        return {
            param: (float(df[param].min()), float(df[param].max()))
            for param in ['sd_feat1', 'sd_feat2', 'sd_spat']
        }

    def get_surface_count_stats(self) -> Dict[str, int]:
        """Get statistics about surface availability."""
        if self.surfaces_df is None:
            return {}

        stats = {
            'total_surfaces': len(self.surfaces_df),
            'unique_sd_feat1': len(self.surfaces_df['sd_feat1'].unique()),
            'unique_sd_feat2': len(self.surfaces_df['sd_feat2'].unique()),
            'unique_sd_spat': len(self.surfaces_df['sd_spat'].unique()),
            'cache_size': len(self._surface_cache)
        }
        
        # Add surface type counts if column exists
        if 'surface_type' in self.surfaces_df.columns:
            for surface_type in self.surfaces_df['surface_type'].unique():
                stats[f'{surface_type}_surfaces'] = len(self.surfaces_df[self.surfaces_df['surface_type'] == surface_type])
                
        return stats

    def clear_cache(self):
        """Clear the surface data cache to free memory."""
        self._surface_cache.clear()
        st.success(f"Cleared surface cache")

    def load_multiple_surfaces(self, surface_rows):
        """Load multiple surfaces efficiently with batching."""
        return self.load_surface_batch(surface_rows)

    def validate_filename_pattern(self, filename: str) -> bool:
        """Check if filename follows expected patterns."""
        regular_pattern = r'surface_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_.*\.pkl'
        averaged_pattern = r'averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl'
        return bool(re.match(regular_pattern, filename)) or bool(re.match(averaged_pattern, filename))

    def suggest_filename_format(self):
        """Show expected filename formats for user reference."""
        st.info("""
        **Expected filename formats:**
        - Regular surfaces: `surface_sf1_25.5_sf2_28.3_sp_45.2_hash.pkl`
        - Averaged surfaces: `averaged_sf1_25.5_sf2_28.3_sp_45.2.pkl`
        
        Fast loading relies on these naming conventions.
        """)