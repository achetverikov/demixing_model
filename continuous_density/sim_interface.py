"""Thin wrapper around the production Demixing Model simulator.

The theoretical model is untouched: this module only calls
``jax_fit_main.simulate_dual_component_bias_distribution`` at *continuous*
parameter values instead of on the 5/10-degree grid, and returns the raw
per-simulation ``mu1_bias`` outcomes.  No KDE, no surface, no averaging.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from continuous_density.design import SIM_SPAT_DIFF

_SC = Path(__file__).resolve().parents[1] / "surface_computation"
if str(_SC) not in sys.path:
    sys.path.insert(0, str(_SC))

import jax_fit_main as jfm  # noqa: E402

#: Spatial separation used throughout the production grid pipeline.
SPAT_DIFF = SIM_SPAT_DIFF


def simulation_keys(key, n_rows: int, common_random_numbers: bool = False,
                    row_offset: int = 0, simulation_offset: int = 0,
                    common_random_groups=None):
    """Deterministic row keys for one simulation chunk.

    Keys depend on absolute row/chunk coordinates, not on how either axis is
    partitioned for execution.  The chunk size is recorded in the generation
    manifest because the production simulator splits each chunk key internally.
    """
    if common_random_groups is not None:
        groups = np.asarray(common_random_groups)
        if groups.shape != (n_rows,):
            raise ValueError(f'common_random_groups must have shape ({n_rows},)')
        return jnp.stack([
            jax.random.fold_in(jax.random.fold_in(key, int(group)), simulation_offset)
            for group in groups
        ])
    if common_random_numbers:
        shared = jax.random.fold_in(key, simulation_offset)
        return jnp.broadcast_to(shared, (n_rows,) + shared.shape)
    return jnp.stack([
        jax.random.fold_in(jax.random.fold_in(key, row), simulation_offset)
        for row in range(row_offset, row_offset + n_rows)
    ])


@partial(jax.jit, static_argnames=['n_simulations', 'n_samples', 'fix_weights'])
def _simulate_block(keys, design, n_simulations: int, n_samples: int,
                    fix_weights: bool):
    """Simulate one block of design rows, scanning rows like the grid pipeline.

    Args:
        keys: ``(M,)`` PRNG keys, one per design row.  Passing the *same* key in
            every row implements common random numbers (see ``simulate``).
        design: ``(M, 4)`` rows of ``[sd_feat1, sd_feat2, sd_ident, feat_diff]``.

    Returns:
        ``(M, n_simulations, 2)`` mu1_bias, component 1 then component 2.
    """

    def step(carry, inputs):
        key, row = inputs
        mu1_bias, _ = jfm.simulate_dual_component_bias_distribution(
            key, row[0], row[1], row[2], row[3], SPAT_DIFF,
            n_simulations=n_simulations, n_samples=n_samples,
            return_full_results=False, fix_weights=fix_weights,
            algorithm='EM', diagonal_covariance=True,
        )
        return carry, mu1_bias

    _, out = jax.lax.scan(step, None, (keys, design))
    return out


def simulate(key, design, n_simulations: int = 200, n_samples: int = 100,
             fix_weights: bool = False, common_random_numbers: bool = False,
             block_rows: int = 1, progress: bool = False,
             row_offset: int = 0, simulation_offset: int = 0,
             common_random_groups=None):
    """Simulate raw EM bias outcomes for a continuous parameter design.

    Args:
        design: ``(M, 4)`` array ``[sd_feat1, sd_feat2, sd_ident, feat_diff]``.
        n_simulations: EM runs per design row.
        n_samples: internal evidence samples per EM run (the observer model).
        common_random_numbers: reuse one key across all design rows, so
            ``b_j(x) = G(x, eps_j)`` shares ``eps_j`` between rows.  The EM
            initialisation indices and the mixture draws then depend only on the
            simulation index, which removes most of the Monte-Carlo roughness of
            the parameter-to-density map at the cost of correlating rows.
        block_rows: rows per compiled block; bounds device memory.
        row_offset: absolute first-row index, used to make keys invariant to
            output sharding.
        simulation_offset: absolute first-simulation index of this chunk, used
            to give independently generated, reproducible simulation chunks.

    Returns:
        ``(M, n_simulations, 2)`` float32 array of mu1_bias in degrees.
    """
    design = jnp.asarray(design, dtype=jnp.float32)
    n_rows = design.shape[0]
    out = []
    for start in range(0, n_rows, block_rows):
        block = design[start:start + block_rows]
        groups = (None if common_random_groups is None else
                  np.asarray(common_random_groups)[start:start + block.shape[0]])
        keys = simulation_keys(key, block.shape[0], common_random_numbers,
                               row_offset + start, simulation_offset, groups)
        out.append(_simulate_block(keys, block, n_simulations, n_samples,
                                   fix_weights))
        if progress:
            print(f"  simulated rows {start}-{start + block.shape[0]} / {n_rows}",
                  flush=True)
    return jnp.concatenate(out, axis=0)


def flatten_with_mirror(design, bias):
    """Expand simulator output into training rows, exploiting the mirror symmetry.

    Each simulation yields two observations: the component-1 bias at
    ``(sd_feat1, sd_feat2, ...)`` and the component-2 bias, which is the
    component-1 bias of the mirrored parameter vector.  Non-finite biases (rare
    EM failures) are dropped.

    Returns ``(x, b)`` with shapes ``(2*M*S, 4)`` and ``(2*M*S,)``.
    """
    design = np.asarray(design)
    bias = np.asarray(bias)
    mirrored = design[:, [1, 0, 2, 3]]
    x = np.concatenate([
        np.repeat(design, bias.shape[1], axis=0),
        np.repeat(mirrored, bias.shape[1], axis=0),
    ], axis=0)
    b = np.concatenate([bias[:, :, 0].ravel(), bias[:, :, 1].ravel()])
    keep = np.isfinite(b)
    return x[keep], b[keep]
