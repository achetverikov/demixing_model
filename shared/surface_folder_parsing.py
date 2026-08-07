"""Helpers for loading and filtering surface files on disk."""

import pickle
import re
from pathlib import Path
from typing import Dict
from shared.surface_functions import smooth_surface
from shared.config import config
from shared.mu1_axis import guard_surface_mu1_axis


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
        # NOTE: deliberately NOT wrapped in a bare `except Exception`. This used
        # to gate on `shape == config.mu1_surface_shape` and *silently drop*
        # every surface of the wrong shape, so a half-done migration looked like
        # a smaller dataset rather than an error — the one failure mode that
        # looks like success. The guard raises instead, and a swallowing handler
        # here would reproduce exactly the bug it replaces.
        data = load_surface(file)
        # New format: data contains 'surface' (Surface object) and 'parameters'
        surface_obj = data["surface"]
        guard_surface_mu1_axis(surface_obj, source=str(file))
        surfaces_list.append(data)

    return surfaces_list


def load_surface(filename: str, smooth: bool = False) -> Dict:
    """Load precomputed empirical surfaces from pickle file."""
    print(f"Loading empirical surfaces from {filename}")

    with open(filename, 'rb') as f:
        orig_surface = pickle.load(f)
    if smooth:
        orig_surface = smooth_surface(orig_surface)
    return orig_surface
