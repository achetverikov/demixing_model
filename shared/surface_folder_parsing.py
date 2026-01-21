"""Helpers for loading and filtering surface files on disk."""

import pickle
import re
from pathlib import Path
from typing import Dict
from shared.surface_functions import smooth_surface
from shared.config import config


def load_filtered_surfaces(folder: str = config.surfaces_folder, low=30, high=60):
    """Load surfaces within a parameter range.

    Args:
        folder: Folder containing surface pickle files.
        low: Lower bound for parameter filtering.
        high: Upper bound for parameter filtering.

    Returns:
        list: Loaded surface records for matching parameter ranges.
    """
    pattern = re.compile(r"surface_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_\w+\.pkl")
    surfaces_list = []
    data_list = []
    for file in Path(folder).glob("surface_sf1_*_sf2_*_sp_*.pkl"):
        match = pattern.match(file.name)
        if match:
            sf1, sf2, sp = map(float, match.groups())
            if all((low-1e-2) <= val <= (high+1e-2) for val in (sf1, sf2, sp)):
                data_list.append(file)
    print(f'Total surfaces: {len(data_list)}')
    for file in data_list:
        try:
            data = load_surface(file)
            # New format: data contains 'surface' (Surface object) and 'parameters'
            surface_obj = data["surface"]
            # For training, we need mu1_comp1_surface which has shape (180, 90)
            if surface_obj.mu1_comp1_surface.shape == (180, 90):
                surfaces_list.append(data)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    return surfaces_list


def load_surface(filename: str, smooth: bool = False) -> Dict:
    """Load precomputed empirical surfaces from pickle file."""
    print(f"Loading empirical surfaces from {filename}")

    with open(filename, 'rb') as f:
        orig_surface = pickle.load(f)
    if smooth:
        orig_surface = smooth_surface(orig_surface)
    return orig_surface
