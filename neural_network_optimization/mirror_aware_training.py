#!/usr/bin/env python3
"""
Mirror-Aware Training Script
===========================

This script trains the mirror-aware neural network on averaged mirrored surfaces.
It uses the proper symmetry-enforcing architecture and loads data from combined surfaces.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from pathlib import Path
import pickle
import re
from typing import Dict, List, Tuple
import time
from datetime import datetime
from flax.training import train_state

from mirror_aware_model import MirrorAwareMu1Predictor, normalize_to_density_flexible
from loss_functions import combined_probabilistic_loss, kl_divergence_loss, expectation_loss, smoothness_regularization
from shared.config import config
from shared.utils import save_checkpoint, cleanup_old_checkpoints, save_training_log_smart, resolve_input_path, resolve_results_path
# from shared.surface_functions import entropy_smooth_columns_with_mask
from shared.utils import AveragedSurface

def _is_valid_surface(data: Dict, filename: str) -> bool:
    """Return True if data has the expected averaged-surface structure."""
    if 'surface' not in data or 'parameters' not in data:
        print(f"Warning: Invalid data structure in {filename}")
        return False
    if not hasattr(data['surface'], 'mu1_comp1_surface'):
        print(f"Warning: Invalid surface structure in {filename}")
        return False
    return True


def _in_param_range(sf1: float, sf2: float, sp: float,
                    param_low: float, param_high: float) -> bool:
    return all((param_low - 1e-2) <= v <= (param_high + 1e-2) for v in (sf1, sf2, sp))


def _load_from_bundles(folder_path: Path, param_low: float, param_high: float) -> List[Dict]:
    """Load averaged surfaces from surface_bundle_*.pkl.gz files."""
    import gzip as _gzip
    pattern = re.compile(r"averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl")
    surfaces_list = []
    bundle_paths = sorted(folder_path.glob("surface_bundle_*.pkl.gz"))
    for bundle_path in bundle_paths:
        try:
            with _gzip.open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            if not isinstance(bundle, dict) or "surfaces" not in bundle:
                print(f"Warning: unexpected bundle format in {bundle_path.name}")
                continue
            for filename, raw_bytes in bundle["surfaces"].items():
                match = pattern.match(filename)
                if not match:
                    continue
                sf1, sf2, sp = map(float, match.groups())
                if not _in_param_range(sf1, sf2, sp, param_low, param_high):
                    continue
                try:
                    data = pickle.loads(raw_bytes)
                    if _is_valid_surface(data, filename):
                        surfaces_list.append(data)
                except Exception as e:
                    print(f"Error deserializing {filename} from {bundle_path.name}: {e}")
        except Exception as e:
            print(f"Error loading bundle {bundle_path.name}: {e}")
    return surfaces_list


def _load_from_files(folder_path: Path, param_low: float, param_high: float) -> List[Dict]:
    """Load averaged surfaces from individual averaged_sf1_*.pkl files."""
    pattern = re.compile(r"averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl")
    surfaces_list = []
    for file in folder_path.glob("averaged_sf1_*_sf2_*_sp_*.pkl"):
        match = pattern.match(file.name)
        if not match:
            continue
        sf1, sf2, sp = map(float, match.groups())
        if not _in_param_range(sf1, sf2, sp, param_low, param_high):
            continue
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if _is_valid_surface(data, file.name):
                surfaces_list.append(data)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    return surfaces_list


def load_averaged_surfaces(folder: str = "combined_mirrored_surfaces_10k",
                           param_low: float = None, param_high: float = None) -> List[Dict]:
    """Load averaged mirrored surfaces from a folder.

    Accepts either a directory of individual ``averaged_sf1_*.pkl`` files or a
    directory of ``surface_bundle_*.pkl.gz`` bundle files (auto-detected).

    Args:
        folder: Folder containing averaged surface files or bundle files.
        param_low: Lower bound for parameter filtering.
        param_high: Upper bound for parameter filtering.

    Returns:
        List of surface data dictionaries.
    """
    if param_low is None:
        param_low = config.param_range_low
    if param_high is None:
        param_high = config.param_range_high

    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Averaged surfaces folder not found: {folder}")

    if any(folder_path.glob("surface_bundle_*.pkl.gz")):
        print(f"Loading from bundles in: {folder}")
        surfaces_list = _load_from_bundles(folder_path, param_low, param_high)
    else:
        surfaces_list = _load_from_files(folder_path, param_low, param_high)

    print(f"Loaded {len(surfaces_list)} averaged surfaces from {folder}")
    return surfaces_list


def prepare_mirror_aware_data(surfaces_list: List[Dict]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare training data for mirror-aware model.
    
    For each parameter combination (sf1, sf2, sp):
    - If sf1 <= sf2: use mu1_comp1_surface as target
    - If sf1 > sf2: use mu1_comp2_surface as target
    
    This teaches the NN to produce the correct surface for ANY parameter combination.
    
    Args:
        surfaces_list: List of surface data dictionaries
        
    Returns:
        Tuple of (inputs, targets) arrays
    """
    print("Preparing mirror-aware training data...")
    
    inputs = []
    targets = []
    comp1_count = 0
    comp2_count = 0
    
    for data in surfaces_list:
        params = data['parameters']
        surface_obj = data['surface']
        
        sf1, sf2, sp = params['sd_feat1'], params['sd_feat2'], params['sd_spat']
        
        # Input: [sf1, sf2, sp] - exact parameters as given
        input_params = [sf1, sf2, sp]
        inputs.append(input_params)
        
        # Target selection based on parameter relationship
        # Use mu1_comp1_surface for canonical ordering
        target_surface = surface_obj.mu1_comp1_surface
        comp1_count += 1
        # target = entropy_smooth_columns_with_mask(target_surface, sigma=2)
        targets.append(target_surface)
        if sf1 != sf2:
            # Use mu1_comp2_surface for mirrored ordering
            input_params = [sf2, sf1, sp]
            inputs.append(input_params)
            target_surface = surface_obj.mu1_comp2_surface
            comp2_count += 1

            # Apply smoothing to target
            # target = entropy_smooth_columns_with_mask(target_surface, sigma=2)
            targets.append(target_surface)
    
    inputs = np.array(inputs)
    targets = np.stack([np.asarray(t) for t in targets], axis=0)
    
    print(f"Data preparation complete. Inputs: {inputs.shape}, Targets: {targets.shape}")
    print(f"Used mu1_comp1_surface for {comp1_count} cases (sf1 <= sf2)")
    print(f"Used mu1_comp2_surface for {comp2_count} cases (sf1 > sf2)")
    print(f"Total training examples: {len(inputs)}")
    
    return inputs, targets


def create_mirror_aware_train_state(key: jax.random.PRNGKey,
                                   learning_rate: float = 1e-3,
                                   warmup_steps: int = 1000,
                                   total_steps: int = 10000,
                                   weight_decay: float = 1e-4) -> train_state.TrainState:
    """Create training state for mirror-aware model."""

    # Initialize model
    model = MirrorAwareMu1Predictor()
    
    # Initialize parameters with dummy input
    dummy_input = jnp.ones((1, 3))
    params = model.init(key, dummy_input)
    
    # Create warmup + cosine annealing schedule
    warmup_cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=warmup_cosine_schedule,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def mirror_aware_train_step(state: train_state.TrainState,
                           batch_inputs: jnp.ndarray,
                           batch_targets: jnp.ndarray) -> Tuple[train_state.TrainState, float, Dict]:
    """Training step for mirror-aware model."""

    def loss_fn(params):
        """Compute total loss and loss components for one batch."""
        # Get predictions from mirror-aware model
        preds = state.apply_fn(params, batch_inputs)

        # The model already handles normalization, so preds are normalized log densities
        loss, loss_components = combined_probabilistic_loss(preds, batch_targets)
        return loss, (preds, loss_components)

    (loss, (preds, loss_components)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, loss_components


def train_mirror_aware_model(surfaces_folder: str = "combined_mirrored_surfaces_10k",
                            n_epochs: int = 2500,
                            batch_size: int = 64,  # Increased for better GPU utilization
                            learning_rate: float = 2e-3,
                            weight_decay: float = 1e-4,
                            save_dir: str = "neural_net_checkpoints",
                            save_every: int = 25,
                            results_dir: str = "results",
                            seed: int = 42) -> train_state.TrainState:
    """
    Train mirror-aware model on averaged surfaces.
    
    Args:
        surfaces_folder: Folder containing averaged surface files
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        
    Returns:
        Final training state
    """
    
    print("=== Mirror-Aware Surface Prediction Training ===")
    print(f"Using config: mu1_surface_shape={config.mu1_surface_shape}, mu1_bias_range={config.mu1_bias_range}")
    print(f"Optimization settings: batch_size={batch_size}, pre-compilation enabled")
    
    # Load averaged surfaces
    resolved_surfaces_folder = resolve_input_path(surfaces_folder, results_dir)
    resolved_save_dir = resolve_results_path(save_dir, results_dir)

    # param_low extends one half-step below param_range_low (e.g. 5 when step=10)
    # so L2 grid surfaces (sp=5, sf=5, etc.) are included automatically.
    surfaces_list = load_averaged_surfaces(
        folder=str(resolved_surfaces_folder),
        param_low=config.param_range_low - config.param_step // 2,
        param_high=config.param_range_high
    )
    
    if len(surfaces_list) == 0:
        raise ValueError(f"No surfaces found in {resolved_surfaces_folder}")
    
    # Prepare training data
    inputs, targets = prepare_mirror_aware_data(surfaces_list)
    n_samples = len(inputs)
    
    # Calculate training parameters
    n_batches = n_samples // batch_size
    total_steps = n_epochs * n_batches
    warmup_steps = min(total_steps // 20, 1000)
    
    print(f"Training setup:")
    print(f"  Samples: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Seed: {seed}")
    
    # Create training state
    key = jax.random.PRNGKey(seed)
    state = create_mirror_aware_train_state(
        key=key,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        weight_decay=weight_decay
    )

    # Pre-compile the training step to avoid slow first iteration
    # print("Pre-compiling training step...")
    # dummy_inputs = jnp.ones((batch_size, 3))
    # dummy_targets = jnp.ones((batch_size, config.mu1_surface_shape[0], config.mu1_surface_shape[1]))
    # _, _, _ = mirror_aware_train_step(state, dummy_inputs, dummy_targets)
    # print("JIT compilation complete!")
    
    # Create save directory
    Path(resolved_save_dir).mkdir(parents=True, exist_ok=True)
    
    # Log hyperparameters
    save_training_log_smart(
        save_dir=str(resolved_save_dir),
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_samples=n_samples,
        data_range=f"{config.param_range_low}-{config.param_range_high}",
        surfaces_folder=str(resolved_surfaces_folder),
        model_type="MirrorAwareMu1Predictor",
        loss_type="combined_probabilistic",
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        mirror_aware=True,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    
    # Training loop
    print("Starting training...")
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Shuffle data
        perm = rng.permutation(n_samples)
        batched_indices = perm[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        # Training metrics — accumulated as JAX arrays to avoid per-step GPU→CPU sync
        acc_loss = jnp.zeros(())
        acc_components = {'mse': jnp.zeros(()), 'kl': jnp.zeros(()), 'expectation': jnp.zeros(()), 'smoothness': jnp.zeros(()), 'total': jnp.zeros(())}

        # Training loop
        for batch_idx in range(n_batches):
            batch_inputs = inputs[batched_indices[batch_idx]]
            batch_targets = targets[batched_indices[batch_idx]]

            state, loss, loss_components = mirror_aware_train_step(state, batch_inputs, batch_targets)

            acc_loss += loss
            for key in acc_components:
                acc_components[key] += loss_components[key]

        # Single float() conversion per epoch — one GPU→CPU sync
        epoch_loss = float(acc_loss) / n_batches
        total_loss_components = {key: float(acc_components[key]) / n_batches for key in acc_components}
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        if epoch % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch+1}/{n_epochs}: loss = {epoch_loss:.4f} [{epoch_time:.1f}s]")
            print(f"  Components: MSE={total_loss_components['mse']:.4f}, "
                  f"KL={total_loss_components['kl']:.4f}, "
                  f"Exp={total_loss_components['expectation']:.4f}, "
                  f"Smooth={total_loss_components['smoothness']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == n_epochs - 1:
            save_checkpoint(state, epoch + 1, epoch_loss, save_dir=str(resolved_save_dir))
            cleanup_old_checkpoints(save_dir=str(resolved_save_dir), keep_last_n=3)
        
        # Log to file
        save_training_log_smart(
            save_dir=str(resolved_save_dir),
            epoch=epoch + 1,
            loss=epoch_loss,
            mse_loss=total_loss_components['mse'],
            kl_loss=total_loss_components['kl'],
            expectation_loss=total_loss_components['expectation'],
            smoothness_loss=total_loss_components['smoothness'],
            epoch_time_seconds=epoch_time
        )
    
    print(f"\nTraining completed!")
    print(f"Final loss: {epoch_loss:.4f}")
    
    return state


def check_trained_model_outputs(checkpoint_path: str) -> bool:
    """Check that both ordered component predictions are finite and normalized.

    The component pair for (a, b) is [F(a, b), F(b, a)]. These two surfaces
    generally differ when a != b; swapping the items reverses the pair rather
    than making the individual surfaces invariant.
    """
    
    print("Checking ordered component predictions from trained model...")
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Create model and restore parameters
    model = MirrorAwareMu1Predictor()
    params = checkpoint_data['params']
    
    # Test inputs
    test_input_1 = jnp.array([[30.0, 50.0, 100.0]])  # sf1=30, sf2=50
    test_input_2 = jnp.array([[50.0, 30.0, 100.0]])  # sf1=50, sf2=30 (swapped)
    
    # Get predictions
    pred_1 = model.apply(params, test_input_1)
    pred_2 = model.apply(params, test_input_2)
    
    component_diff = jnp.max(jnp.abs(pred_1 - pred_2))
    integrals = jnp.sum(jnp.exp(jnp.concatenate([pred_1, pred_2], axis=0)), axis=1)
    integrals = integrals * config.mu1_bias_step
    finite = jnp.all(jnp.isfinite(pred_1)) & jnp.all(jnp.isfinite(pred_2))
    normalized = jnp.allclose(integrals, 1.0, rtol=1e-5, atol=1e-5)
    valid = finite & normalized
    
    print(f"  Input 1: sf1=30, sf2=50, sp=100")
    print(f"  Input 2: sf1=50, sf2=30, sp=100")
    print(f"  Max component difference (not expected to be zero): {component_diff:.2e}")
    print(f"  Finite: {finite}")
    print(f"  Density integrals equal 1: {normalized}")
    
    return bool(valid)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mirror-aware surface prediction model')
    parser.add_argument('--surfaces-folder', type=str, default="averaged_surfaces_10k_20samples",
                        help='Folder containing averaged surface files')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-3,
                        help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay coefficient')
    parser.add_argument('--save-dir', type=str, default="neural_net_checkpoints_20samples",
                        help='Directory to save checkpoints')
    parser.add_argument('--test-checkpoint', type=str, default=None,
                        help='Check ordered component predictions from a checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for model initialization and epoch shuffling (default: 42)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for outputs (relative paths are placed here)')
    
    args = parser.parse_args()
    
    if args.test_checkpoint:
        checkpoint_path = resolve_input_path(args.test_checkpoint, args.results_dir)
        if not check_trained_model_outputs(str(checkpoint_path)):
            raise SystemExit(1)
    else:
        # Train model
        final_state = train_mirror_aware_model(
            surfaces_folder=args.surfaces_folder,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_dir=args.save_dir,
            results_dir=args.results_dir,
            seed=args.seed,
        )
