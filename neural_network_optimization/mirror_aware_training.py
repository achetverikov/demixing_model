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
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime
from flax.training import train_state

from mirror_aware_model import (MirrorAwareMu1Predictor, infer_native_mu1_rows,
                                normalize_to_density_flexible)
from loss_functions import LOSS_PROFILES, combined_probabilistic_loss, loss_profile
from shared.config import config, averaged_surfaces_dir
from shared.mu1_axis import guard_surface_mu1_axis
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
        # The L2 grid reaches a half-step below param_range_low; defaulting to
        # param_range_low would drop those surfaces (~7% of the set) without a word.
        param_low = config.param_grid_low
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

    # Guard the mu1 axis OUTSIDE the loaders: both wrap their per-file work in a
    # bare `except Exception` that prints and continues, so a guard raised in
    # there would be swallowed and leave a silently shorter training set — the
    # one failure mode that looks like success. Training on legacy 181-row
    # targets would teach the network to reproduce the duplicated wrap row it
    # exists to be rid of.
    for data in surfaces_list:
        guard_surface_mu1_axis(data['surface'], source=str(folder))

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
                                   weight_decay: float = 1e-4,
                                   native_mu1_rows: int = 64,
                                   output_feat_cols: Optional[int] = None) -> train_state.TrainState:
    """Create training state for mirror-aware model."""

    # Initialize model
    model = MirrorAwareMu1Predictor(
        native_mu1_rows=native_mu1_rows, output_feat_cols=output_feat_cols)
    
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


def _resample_target_feature_probabilities(targets, output_cols):
    """Endpoint-aligned interpolation of conditional probabilities."""
    input_cols = targets.shape[2]
    if input_cols == output_cols:
        return targets
    probabilities = jax.nn.softmax(targets, axis=1)
    positions = jnp.arange(output_cols) * (input_cols - 1) / (output_cols - 1)
    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, input_cols - 1)
    fraction = (positions - lower).reshape((1, 1, output_cols))
    resized = (jnp.take(probabilities, lower, axis=2) * (1 - fraction) +
               jnp.take(probabilities, upper, axis=2) * fraction)
    return jnp.log(jnp.maximum(resized, jnp.finfo(resized.dtype).tiny))


def make_mirror_aware_train_step(loss_kwargs=None, target_feat_cols=None):
    """Compile a training step for one fixed objective-ablation profile."""
    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)

    @jax.jit
    def train_step(state: train_state.TrainState,
                   batch_inputs: jnp.ndarray,
                   batch_targets: jnp.ndarray):
        def loss_fn(params):
            preds = state.apply_fn(params, batch_inputs)
            targets = _resample_target_feature_probabilities(
                batch_targets, target_feat_cols or batch_targets.shape[2])
            loss, components = combined_probabilistic_loss(
                preds, targets, **loss_kwargs)
            return loss, components

        (loss, loss_components), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_components

    return train_step


mirror_aware_train_step = make_mirror_aware_train_step()


def train_mirror_aware_model(surfaces_folder: str = "combined_mirrored_surfaces_10k",
                            n_epochs: int = 2500,
                            batch_size: int = 64,  # Increased for better GPU utilization
                            learning_rate: float = 2e-3,
                            weight_decay: float = 1e-4,
                            save_dir: str = "neural_net_checkpoints",
                            save_every: int = 25,
                            keep_checkpoints: int = 3,
                            results_dir: str = "results",
                            seed: int = 42,
                            loss_profile_name: str = "circular",
                            native_mu1_rows: int = 64,
                            training_feat_cols: int = 90,
                            init_checkpoint: Optional[str] = None,
                            epoch_offset: int = 0) -> train_state.TrainState:
    """
    Train mirror-aware model on averaged surfaces.

    These function defaults (2500 epochs, batch 64) are NOT what the CLI runs:
    the argument parser below defaults to 1500 epochs / batch 32 and passes
    them explicitly. Production training uses the CLI's 1500-epoch, batch-32
    schedule (with a validation-selected checkpoint potentially preceding the
    final epoch; see pretrained/README.md). Programmatic callers omitting these
    arguments get a different training run than the documented one.

    Args:
        surfaces_folder: Folder containing averaged surface files
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        keep_checkpoints: Number of recent checkpoints to retain; zero keeps all
        init_checkpoint: Optional checkpoint whose parameters initialize a new
            optimizer schedule. This is a warm restart, not an optimizer-state
            resume.
        epoch_offset: Epoch number assigned to the state before this run. Saved
            checkpoint numbers continue from this value.
        
    Returns:
        Final training state
    """
    
    print("=== Mirror-Aware Surface Prediction Training ===")
    if save_every < 1 or keep_checkpoints < 0 or epoch_offset < 0:
        raise ValueError(
            "save_every must be positive; keep_checkpoints and epoch_offset "
            "must be nonnegative")
    print(f"Using config: mu1_surface_shape={config.mu1_surface_shape}, mu1_bias_range={config.mu1_bias_range}")
    print(f"Optimization settings: batch_size={batch_size}, "
          f"native_mu1_rows={native_mu1_rows}, "
          f"training_feat_cols={training_feat_cols}, pre-compilation enabled")
    objective = loss_profile(loss_profile_name)
    train_step = make_mirror_aware_train_step(objective, training_feat_cols)
    component_names = [
        name for name, weight_name in (
            ('kl', 'kl_weight'),
            ('energy', 'energy_weight'),
            ('expectation', 'expectation_weight'),
            ('asymmetry', 'asymmetry_weight'),
            ('hellinger', 'hellinger_weight'),
            ('log_smoothness', 'log_smoothness_weight'),
            ('curvature', 'curvature_weight'),
            ('trajectory', 'trajectory_weight'),
            ('feature_gradient', 'feature_gradient_weight'))
        if objective.get(weight_name, 0.0)
    ]
    component_names.append('total')
    print(f"Loss profile: {loss_profile_name} {objective}")
    
    # Load averaged surfaces
    resolved_surfaces_folder = resolve_input_path(surfaces_folder, results_dir)
    resolved_save_dir = resolve_results_path(save_dir, results_dir)

    # config.param_grid_low extends one half-step below param_range_low (e.g. 5 when
    # step=10) so L2 grid surfaces (sp=5, sf=5, etc.) are included automatically.
    surfaces_list = load_averaged_surfaces(
        folder=str(resolved_surfaces_folder),
        param_low=config.param_grid_low,
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
        weight_decay=weight_decay,
        native_mu1_rows=native_mu1_rows,
        output_feat_cols=training_feat_cols,
    )
    if init_checkpoint:
        resolved_checkpoint = resolve_input_path(init_checkpoint, results_dir)
        with open(resolved_checkpoint, 'rb') as handle:
            checkpoint_data = pickle.load(handle)
        checkpoint_epoch = int(checkpoint_data['epoch'])
        if checkpoint_epoch != epoch_offset:
            raise ValueError(
                f"Initial checkpoint is epoch {checkpoint_epoch}, but "
                f"epoch_offset={epoch_offset}")
        loaded_rows = infer_native_mu1_rows(checkpoint_data['params'])
        if loaded_rows != native_mu1_rows:
            raise ValueError(
                f"Initial checkpoint uses native_mu1_rows={loaded_rows}, but "
                f"the requested model uses {native_mu1_rows}")
        if jax.tree_util.tree_structure(checkpoint_data['params']) != \
                jax.tree_util.tree_structure(state.params):
            raise ValueError("Initial checkpoint parameter tree is incompatible")
        state = state.replace(params=checkpoint_data['params'])
        print(f"Warm-started parameters from {resolved_checkpoint} "
              f"(epoch {checkpoint_epoch}); optimizer schedule restarted")

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
        data_range=f"{config.param_grid_low}-{config.param_range_high}",
        surfaces_folder=str(resolved_surfaces_folder),
        model_type="MirrorAwareMu1Predictor",
        architecture=(f"native_mu1_rows={native_mu1_rows},"
                      f"training_feat_cols={training_feat_cols}"),
        training_feat_cols=training_feat_cols,
        loss_type=loss_profile_name,
        **objective,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        init_checkpoint=str(resolved_checkpoint) if init_checkpoint else None,
        epoch_offset=epoch_offset,
        mirror_aware=True,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    # Continue the deterministic epoch-shuffle stream when warm-starting.
    for _ in range(epoch_offset):
        rng.permutation(n_samples)
    
    # Training loop
    print("Starting training...")
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        
        # Shuffle data
        perm = rng.permutation(n_samples)
        batched_indices = perm[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        # Training metrics — accumulated as JAX arrays to avoid per-step GPU→CPU sync
        acc_loss = jnp.zeros(())
        acc_components = {name: jnp.zeros(()) for name in component_names}

        # Training loop
        for batch_idx in range(n_batches):
            batch_inputs = inputs[batched_indices[batch_idx]]
            batch_targets = targets[batched_indices[batch_idx]]

            state, loss, loss_components = train_step(
                state, batch_inputs, batch_targets)

            acc_loss += loss
            for key in acc_components:
                acc_components[key] += loss_components[key]

        # Single float() conversion per epoch — one GPU→CPU sync
        epoch_loss = float(acc_loss) / n_batches
        total_loss_components = {key: float(acc_components[key]) / n_batches for key in acc_components}
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        if epoch % 10 == 0 or epoch < 10:
            absolute_epoch = epoch_offset + epoch + 1
            print(f"Epoch {absolute_epoch}/{epoch_offset + n_epochs}: "
                  f"loss = {epoch_loss:.4f} [{epoch_time:.1f}s]")
            component_text = ", ".join(
                f"{name}={value:.4f}" for name, value in total_loss_components.items()
                if name != 'total')
            print(f"  Components: {component_text}")
        
        # Save checkpoint
        absolute_epoch = epoch_offset + epoch + 1
        if absolute_epoch % save_every == 0 or epoch == n_epochs - 1:
            checkpoint_state = state
            if training_feat_cols != config.feat_diff_grid_size:
                inference_model = MirrorAwareMu1Predictor(
                    native_mu1_rows=native_mu1_rows,
                    output_feat_cols=config.feat_diff_grid_size)
                checkpoint_state = state.replace(apply_fn=inference_model.apply)
            save_checkpoint(
                checkpoint_state, absolute_epoch, epoch_loss,
                save_dir=str(resolved_save_dir))
            if keep_checkpoints:
                cleanup_old_checkpoints(
                    save_dir=str(resolved_save_dir), keep_last_n=keep_checkpoints)
        
        # Log to file
        metric_log_names = {
            'kl': 'kl_loss',
            'energy': 'circular_energy_loss',
            'expectation': 'circular_moment_loss',
            'asymmetry': 'density_asymmetry_loss',
        }
        metric_values = {
            metric_log_names.get(name, f'ablation_{name}'): value
            for name, value in total_loss_components.items() if name != 'total'
        }
        save_training_log_smart(
            save_dir=str(resolved_save_dir),
            epoch=absolute_epoch,
            loss=epoch_loss,
            **metric_values,
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
    params = checkpoint_data['params']
    model = MirrorAwareMu1Predictor(
        native_mu1_rows=infer_native_mu1_rows(params))
    
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
    parser.add_argument('--surfaces-folder', type=str, default=str(averaged_surfaces_dir(20)),
                        help='Folder containing averaged surface files or bundles '
                             '(default resolves under $DEMIXING_ARTIFACT_ROOT)')
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
    parser.add_argument('--save-every', type=int, default=25,
                        help='Save a checkpoint every N epochs (default: 25)')
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                        help='Recent checkpoints to retain; zero keeps all (default: 3)')
    parser.add_argument('--test-checkpoint', type=str, default=None,
                        help='Check ordered component predictions from a checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for model initialization and epoch shuffling (default: 42)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base directory for outputs (relative paths are placed here)')
    parser.add_argument('--loss-profile', choices=sorted(LOSS_PROFILES), default='circular',
                        help='Fixed objective profile for a controlled loss ablation')
    parser.add_argument('--native-mu1-rows', type=int, choices=(64, 128), default=64,
                        help='Decoder rows before the final periodic resize (default: 64)')
    parser.add_argument('--training-feat-cols', type=int, choices=(90, 128), default=90,
                        help='Feature columns used by the training loss (default: 90)')
    parser.add_argument('--init-checkpoint', type=str,
                        help='Warm-start parameters from this checkpoint using '
                             'a fresh optimizer schedule')
    parser.add_argument('--epoch-offset', type=int, default=0,
                        help='Epoch of --init-checkpoint; checkpoint numbering '
                             'continues from here (default: 0)')
    
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
            save_every=args.save_every,
            keep_checkpoints=args.keep_checkpoints,
            results_dir=args.results_dir,
            seed=args.seed,
            loss_profile_name=args.loss_profile,
            native_mu1_rows=args.native_mu1_rows,
            training_feat_cols=args.training_feat_cols,
            init_checkpoint=args.init_checkpoint,
            epoch_offset=args.epoch_offset,
        )
