"""Thin wrapper around the production Demixing Model simulator.

Uses the same generation keys, component ordering and likelihood EM as the
native observer API at continuous parameter values. Independent parameter rows
share a GPU batch and can pool long-running starts between iteration stages.
Returns raw per-simulation ``mu_feat_bias`` outcomes, with checked convergence.
No KDE, no surface, no averaging.

Design arrays and the runtime fitting API call the third SD coordinate
``sd_idf``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from surface_computation.wnm_design import SIM_IDF_DIFF

from surface_computation import jax_fit_main as jfm
from surface_computation import em_batch
from surface_computation import likelihood_em as em

#: Identifiability separation used by the maintained Demixing Model simulator.
IDF_DIFF = SIM_IDF_DIFF


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


@partial(jax.jit, static_argnames=['n_simulations', 'n_samples'])
def generate_block(keys, design, n_simulations: int, n_samples: int):
    """Flatten rows/trials with precisely the native simulator's key splitting."""
    def row(key, point):
        trial_keys = jax.random.split(key, n_simulations)
        def trial(k):
            gen, fit = jax.random.split(k)
            observed = jfm.jf.jax_jax_mixture(gen, jnp.array([-.5,.5])*point[3],
                jnp.array([-.5,.5])*IDF_DIFF, point[:2], jnp.repeat(point[2], 2),
                n_samples=n_samples)
            return observed, fit
        return jax.vmap(trial)(trial_keys)
    x, fit = jax.vmap(row)(keys, design)
    return x.reshape((-1, n_samples, 2)), fit.reshape((-1,)+fit.shape[2:])


@partial(jax.jit, static_argnames=['n_simulations'])
def bias_readout(result, design, n_simulations):
    ordered = jax.vmap(lambda p: jfm.jf.rearrange_components(p, jnp.array([-.5,.5])*IDF_DIFF))(result[:5])
    true = jnp.repeat(design[:, 3], n_simulations)[:, None]*jnp.array([-.5,.5])
    difference = ordered[:, :, 3]-true
    if jfm.jf.wrap_feat:
        difference = jfm.jf.angular_difference_jit(ordered[:, :, 3], true)
    return (-jnp.sign(true)*difference).reshape((len(design), n_simulations, 2))


def simulate(key, design, n_simulations: int = 200, n_samples: int = 100,
             fix_weights: bool = False, common_random_numbers: bool = False,
             block_rows: int = 1, progress: bool = False,
             row_offset: int = 0, simulation_offset: int = 0,
             common_random_groups=None, em_stages=None, continuation_batch=None,
             return_diagnostics=False, em_max_iter=em.MAX_ITER, em_backend='jax'):
    """Simulate raw EM bias outcomes for a continuous parameter design.

    Args:
        design: ``(M, 4)`` array ``[sd_feat1, sd_feat2, sd_idf, feat_diff]``.
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
        em_stages: optional cumulative all-start caps ending at 5000. Unfinished
            starts are pooled across parameter rows; capped winners then extend
            up to 25000 updates. None uses the native compiled loop.
        continuation_batch: optional limit on packed continuation starts.
        return_diagnostics: also return per-trial likelihood/convergence arrays.
        em_backend: 'jax' or the optional persistent 'cuda' backend.

    Returns:
        ``(M, n_simulations, 2)`` float32 array of mu_feat_bias in degrees.
        With return_diagnostics, a tuple of this array and per-trial diagnostics,
        including the selected parameters in raw component order.
    """
    design = jnp.asarray(design, dtype=jnp.float32)
    if block_rows < 1 or n_simulations < 1:
        raise ValueError('block_rows and n_simulations must be positive')
    n_rows = design.shape[0]
    out = []
    diagnostics = []
    for start in range(0, n_rows, block_rows):
        block = design[start:start + block_rows]
        groups = (None if common_random_groups is None else
                  np.asarray(common_random_groups)[start:start + block.shape[0]])
        keys = simulation_keys(key, block.shape[0], common_random_numbers,
                               row_offset + start, simulation_offset, groups)
        x, fit_keys = generate_block(keys, block, n_simulations, n_samples)
        fitted = em_batch.fit_batch(fit_keys, x, fix_weights=fix_weights,
            wrap_axes=(jfm.jf.wrap_feat, jfm.jf.wrap_idf), stages=em_stages,
            continuation_batch=continuation_batch, max_iter=em_max_iter,
            backend=em_backend)
        em_batch.check_results(fitted)
        out.append(bias_readout(fitted, block, n_simulations))
        if return_diagnostics:
            diagnostic = dict(fitted[-1], log_likelihood=fitted[5])
            diagnostics.append(jax.tree.map(lambda v: v.reshape((len(block), n_simulations)+v.shape[1:]), diagnostic))
        if progress:
            print(f"  simulated rows {start}-{start + block.shape[0]} / {n_rows}",
                  flush=True)
    bias = jnp.concatenate(out, axis=0)
    if return_diagnostics:
        return bias, jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *diagnostics)
    return bias


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
