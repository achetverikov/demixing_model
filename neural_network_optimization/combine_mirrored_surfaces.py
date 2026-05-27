#!/usr/bin/env python3
"""
Combine Mirrored Surfaces Script
===============================

This script combines mirrored surfaces by averaging log-likelihoods to create higher-quality surfaces.
The symmetry property: mu1_comp1_surface(sf1, sf2, sp) = mu1_comp2_surface(sf2, sf1, sp)

For each unique pair (sf1, sf2, sp):
- averaged_mu1_comp1 = average([original.mu1_comp1, mirror.mu1_comp2])
- averaged_mu1_comp2 = average([original.mu1_comp2, mirror.mu1_comp1])
- averaged_mu2 = average([original.mu2_comp1, original.mu2_comp2, mirror.mu2_comp1, mirror.mu2_comp2])

Only processes L1 surfaces (coarse grid).
"""

import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re
from dataclasses import dataclass
from shared.utils import Surface, AveragedSurface
from shared.config import config
from jax.scipy.special import logsumexp


def find_surface_file(folder: Path, sf1: float, sf2: float, sp: float) -> Optional[Path]:
    """Find surface file matching the given parameters."""
    pattern = re.compile(rf"surface_sf1_{sf1:.1f}_sf2_{sf2:.1f}_sp_{sp:.1f}_.*\.pkl")
    
    for file in folder.glob("surface_sf1_*_sf2_*_sp_*.pkl"):
        if pattern.match(file.name):
            return file
    
    return None


def load_surface_data(file_path: Path) -> Tuple[Surface, Dict]:
    """Load surface data from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['surface'], data['parameters']


def is_l1_surface(sf1: float, sf2: float, sp: float) -> bool:
    """Check if surface parameters correspond to L1 grid (coarse grid)."""
    # L1 surfaces have parameters at config.param_step intervals
    step = config.param_step
    low = config.param_range_low
    high = config.param_range_high
    
    # Check if parameters are on the coarse grid
    def is_on_grid(val):
        """Return True if the value aligns with the L1 grid step."""
        return abs(val - round((val - low) / step) * step - low) < 1e-6
    
    return is_on_grid(sf1) and is_on_grid(sf2) and is_on_grid(sp)


def average_log_likelihoods(log_likelihoods: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Average log-likelihoods using logsumexp for numerical stability.
    
    For log-likelihoods L1, L2, ..., Ln:
    average = logsumexp([L1, L2, ..., Ln]) - log(n)
    """
    if len(log_likelihoods) == 0:
        raise ValueError("Cannot average empty list of log-likelihoods")
    
    if len(log_likelihoods) == 1:
        return log_likelihoods[0]
    
    # Stack along new axis for logsumexp
    stacked = jnp.stack(log_likelihoods, axis=0)
    
    # Average in log space: logsumexp - log(n)
    averaged = logsumexp(stacked, axis=0) - jnp.log(len(log_likelihoods))
    
    return averaged


def create_averaged_surface(original_file: Path, mirror_file: Optional[Path] = None) -> AveragedSurface:
    """Create averaged surface from original and optional mirror files."""
    
    # Load original surface
    original_surface, original_params = load_surface_data(original_file)
    
    params_used = [(original_params['sd_feat1'], original_params['sd_feat2'], original_params['sd_spat'])]
    
    # Prepare surfaces for averaging
    if mirror_file and mirror_file.exists():
        # Load mirror surface
        mirror_surface, mirror_params = load_surface_data(mirror_file)
        params_used.append((mirror_params['sd_feat1'], mirror_params['sd_feat2'], mirror_params['sd_spat']))
        
        # Cross-average: mu1_comp1 with mirror's mu1_comp2, and vice versa
        averaged_mu1_comp1 = average_log_likelihoods([
            original_surface.mu1_comp1_surface,
            mirror_surface.mu1_comp2_surface
        ])
        
        averaged_mu1_comp2 = average_log_likelihoods([
            original_surface.mu1_comp2_surface,
            mirror_surface.mu1_comp1_surface
        ])
        
        # Average all mu2 surfaces together
        averaged_mu2 = average_log_likelihoods([
            original_surface.mu2_comp1_surface,
            original_surface.mu2_comp2_surface,
            mirror_surface.mu2_comp1_surface,
            mirror_surface.mu2_comp2_surface
        ])
        
        surfaces_count = 2  # Two original surfaces contributed
        
    else:
        # No mirror available, just use original
        averaged_mu1_comp1 = original_surface.mu1_comp1_surface
        averaged_mu1_comp2 = original_surface.mu1_comp2_surface
        
        # Average mu2 components from original surface
        averaged_mu2 = average_log_likelihoods([
            original_surface.mu2_comp1_surface,
            original_surface.mu2_comp2_surface
        ])
        
        surfaces_count = 1
    
    # Create averaged surface
    return AveragedSurface(
        feat_diff_grid=original_surface.feat_diff_grid,
        mu1_bias_grid=original_surface.mu1_bias_grid,
        mu2_bias_grid=original_surface.mu2_bias_grid,
        mu1_comp1_surface=averaged_mu1_comp1,
        mu1_comp2_surface=averaged_mu1_comp2,
        mu2_comp1_surface=averaged_mu2,
        mu2_comp2_surface=averaged_mu2,  # Same as comp1 for mu2
        n_sample_files_used=surfaces_count,
        averaged_from_params=params_used
    )


def get_l1_parameter_combinations(folder: Path) -> List[Tuple[float, float, float]]:
    """Get all L1 parameter combinations from surface files."""
    pattern = re.compile(r"surface_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)_.*\.pkl")
    combinations = set()
    
    for file in folder.glob("surface_sf1_*_sf2_*_sp_*.pkl"):
        match = pattern.match(file.name)
        if match:
            sf1, sf2, sp = map(float, match.groups())
            if is_l1_surface(sf1, sf2, sp):
                combinations.add((sf1, sf2, sp))
    
    return sorted(list(combinations))


def save_averaged_surface(averaged_surface: AveragedSurface, sf1: float, sf2: float, sp: float, 
                         output_folder: Path) -> None:
    """Save averaged surface to output folder."""
    
    # Create canonical filename (always sf1 <= sf2 for consistency)
    canonical_sf1 = min(sf1, sf2)
    canonical_sf2 = max(sf1, sf2)
    
    filename = f"averaged_sf1_{canonical_sf1:.1f}_sf2_{canonical_sf2:.1f}_sp_{sp:.1f}.pkl"
    output_file = output_folder / filename
    
    # Create minimal data structure (no metadata like processing times, machine_id)
    averaged_data = {
        'parameters': {
            'sd_feat1': canonical_sf1,
            'sd_feat2': canonical_sf2,
            'sd_spat': sp
        },
        'surface': averaged_surface,
        'creation_timestamp': np.datetime64('now').astype(str)
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(averaged_data, f)
    
    print(f"Saved: {filename} (averaged from {averaged_surface.n_sample_files_used} surfaces)")


def combine_all_surfaces(input_folder: str = None, output_folder: str = None) -> None:
    """Main function to combine all L1 surfaces."""
    
    if input_folder is None:
        input_folder = config.surfaces_folder
    if output_folder is None:
        output_folder = "combined_mirrored_surfaces_10k"
    
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Combining L1 surfaces from: {input_path}")
    print(f"Output folder: {output_path}")
    
    # Get all L1 parameter combinations
    combinations = get_l1_parameter_combinations(input_path)
    print(f"Found {len(combinations)} L1 parameter combinations")
    
    # Process each combination
    processed_pairs = set()
    combined_count = 0
    
    for sf1, sf2, sp in combinations:
        # Create canonical pair to avoid duplicates
        canonical_pair = (min(sf1, sf2), max(sf1, sf2), sp)
        
        if canonical_pair in processed_pairs:
            continue
        
        # Find original surface file
        original_file = find_surface_file(input_path, sf1, sf2, sp)
        if not original_file:
            print(f"Warning: Original surface not found for ({sf1}, {sf2}, {sp})")
            continue
        
        # Find mirror surface file (if different from original)
        mirror_file = None
        if sf1 != sf2:
            mirror_file = find_surface_file(input_path, sf2, sf1, sp)
        
        try:
            # Create averaged surface
            averaged_surface = create_averaged_surface(original_file, mirror_file)
            
            # Save averaged surface
            save_averaged_surface(averaged_surface, canonical_pair[0], canonical_pair[1], sp, output_path)
            
            processed_pairs.add(canonical_pair)
            combined_count += 1
            
            if combined_count % 50 == 0:
                print(f"Processed {combined_count} surfaces...")
                
        except Exception as e:
            print(f"Error processing ({sf1}, {sf2}, {sp}): {e}")
            continue
    
    print(f"\nCombination complete!")
    print(f"Total averaged surfaces created: {combined_count}")
    print(f"Original L1 surfaces: {len(combinations)}")
    print(f"Reduction factor: {len(combinations) / combined_count:.2f}")


def verify_averaged_surfaces(output_folder: str = "combined_mirrored_surfaces_10k") -> None:
    """Verify that averaged surfaces are correctly created."""
    
    output_path = Path(output_folder)
    averaged_files = list(output_path.glob("averaged_sf1_*_sf2_*_sp_*.pkl"))
    
    print(f"\nVerifying {len(averaged_files)} averaged surfaces...")
    
    for i, file in enumerate(averaged_files[:5]):  # Check first 5 files
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            
            averaged_surface = data['surface']
            params = data['parameters']
            
            print(f"\nFile {i+1}: {file.name}")
            print(f"  Parameters: sf1={params['sd_feat1']}, sf2={params['sd_feat2']}, sp={params['sd_spat']}")
            print(f"  Surfaces averaged: {averaged_surface.n_sample_files_used}")
            print(f"  Averaged from: {averaged_surface.averaged_from_params}")
            print(f"  Surface shapes: {averaged_surface.mu1_comp1_surface.shape}")
            
            # Check that mu2_comp1 and mu2_comp2 are identical
            mu2_diff = jnp.max(jnp.abs(averaged_surface.mu2_comp1_surface - averaged_surface.mu2_comp2_surface))
            print(f"  mu2_comp1 vs mu2_comp2 max diff: {mu2_diff:.2e}")
            
        except Exception as e:
            print(f"Error verifying {file.name}: {e}")
    
    print("\nVerification complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine mirrored surfaces by averaging log-likelihoods')
    parser.add_argument('--input-folder', type=str, default=None,
                        help='Input folder containing surface files')
    parser.add_argument('--output-folder', type=str, default="combined_mirrored_surfaces_10k",
                        help='Output folder for averaged surfaces')
    parser.add_argument('--verify', action='store_true',
                        help='Verify averaged surfaces after creation')
    
    args = parser.parse_args()
    
    # Combine surfaces
    combine_all_surfaces(args.input_folder, args.output_folder)
    
    # Verify if requested
    if args.verify:
        verify_averaged_surfaces(args.output_folder)
