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

from shared import wnm as wm
from surface_computation.wnm_design import SIM_IDF_DIFF

from surface_computation import jax_fit_main as jfm
from surface_computation import em_batch, cuda_em
from surface_computation.simulation_corpus import HISTOGRAM_COUNTS
from surface_computation import likelihood_em as em

#: Identifiability separation used by the maintained Demixing Model simulator.
IDF_DIFF = SIM_IDF_DIFF

#: A call refuses its output once more than max(3, trials // 1000) selected
#: fits stay invalid after repair. Isolated failures are masked and recorded; a
#: systematic failure stops at the first call (generator tile) it reaches. The
#: allowance applies per call, not to the corpus as a whole.
MASKED_ALLOWANCE = (3, 1000)


def masked_allowance(trials):
    return max(MASKED_ALLOWANCE[0], trials // MASKED_ALLOWANCE[1])


def simulation_keys(key, n_rows: int, common_random_numbers: bool = False,
                    row_offset: int = 0, simulation_offset: int = 0,
                    common_random_groups=None):
    """Deterministic row keys for one simulation chunk.

    Row keys depend on absolute row/chunk coordinates and preserve row sharding.
    Trial draws depend on simulation chunk size: the simulator splits each chunk
    key internally. The generation manifest therefore freezes chunk geometry.
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


def repair_failed(fit_keys, x, fitted, failed, backend, max_iter, starts='random'):
    """Refit failed selected fits once; return the merged fit and repaired/masked flags.

    CUDA refits use ordinary EM with the complete evaluator. Their starts are
    regenerated for the whole original batch, with its keys and geometry, and
    then indexed, so they equal the starts the original fit used.
    The JAX backend already is ordinary EM, so its failures are masked directly.
    """
    failed = np.flatnonzero(failed)
    repaired = np.zeros(len(x), bool)
    masked = np.zeros(len(x), bool)
    if backend == 'cuda':
        theta = cuda_em.start_parameters(fit_keys, x, starts=starts)[failed]
        retry = cuda_em.fit_from_starts(x[failed], theta, max_iter=max_iter, accelerate=False)
        valid = ~np.asarray(em_batch.winner_failed(retry[-1]))
        target, source = jnp.asarray(failed[valid]), jnp.asarray(np.flatnonzero(valid))
        fitted = jax.tree.map(lambda old, new: old.at[target].set(new[source]), fitted, retry)
        repaired[failed[valid]] = True
        masked[failed[~valid]] = True
    else:
        masked[failed] = True
    return fitted, repaired, masked


def simulate(key, design, n_simulations: int = 200, n_samples: int = 100,
             fix_weights: bool = False, common_random_numbers: bool = False,
             block_rows: int = 1, progress: bool = False,
             row_offset: int = 0, simulation_offset: int = 0,
             common_random_groups=None, em_stages=None, continuation_batch=None,
             return_diagnostics=False, em_max_iter=em.MAX_ITER, em_backend='jax',
             em_starts='random'):
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
        em_starts: 'random' (225 random starts) or 'structured', the deterministic
            data-derived bank of :mod:`structured_starts` (CUDA only).

    Returns:
        ``(M, n_simulations, 2)`` float32 array of mu_feat_bias in degrees.
        A selected fit that is capped or decreased likelihood is refitted once
        (:func:`repair_failed`); if it still fails, both its biases are NaN.
        With return_diagnostics, a tuple of this array and per-trial diagnostics,
        including the selected parameters in raw component order and the
        ``repaired``/``masked`` flags.
    """
    design = jnp.asarray(design, dtype=jnp.float32)
    # This chooses a complete evaluator before fitting. Every evaluator still
    # follows current fitted SDs and retains the full wrapped-normal fallback.
    eligible = (np.min(np.asarray(design)[:, :2], axis=1) <= cuda_em.SINGLE_IMAGE_SD
                if em_backend == 'cuda' and n_samples == 500 else None)
    if block_rows < 1 or n_simulations < 1:
        raise ValueError('block_rows and n_simulations must be positive')
    n_rows = design.shape[0]
    out = []
    diagnostics = []
    allowed = masked_allowance(n_rows*n_simulations)
    n_masked = 0
    # Bound asynchronous CUDA work while amortizing host failure-mask downloads.
    validation_blocks = max(1, 4096 // (block_rows*n_simulations)) if em_backend == 'cuda' else 1
    pending = []

    def settle():
        nonlocal n_masked
        failed = np.asarray(jnp.concatenate([em_batch.winner_failed(fitted[-1])
                                             for *_, fitted in pending]))
        offset = 0
        for block, fit_keys, x, fitted in pending:
            block_failed, offset = failed[offset:offset+len(x)], offset+len(x)
            repaired = masked = np.zeros(len(x), bool)
            if block_failed.any():
                fitted, repaired, masked = repair_failed(
                    fit_keys, x, fitted, block_failed, em_backend, em_max_iter, em_starts)
                n_masked += int(masked.sum())
                if n_masked > allowed:
                    raise em_batch.EMValidationError(
                        f'EM validation failed: {n_masked} selected fits still failed after repair '
                        f'(allowed {allowed} of {n_rows*n_simulations}); no samples saved')
            shape = (len(block), n_simulations)
            bias = bias_readout(fitted, block, n_simulations)
            out.append(jnp.where(masked.reshape(shape)[..., None], jnp.nan, bias))
            if return_diagnostics:
                diagnostic = dict(fitted[-1], log_likelihood=fitted[5], repaired=repaired, masked=masked)
                diagnostics.append(jax.tree.map(lambda v: jnp.reshape(v, shape+jnp.shape(v)[1:]), diagnostic))
        pending.clear()

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
            backend=em_backend, starts=em_starts, cuda_single_image=(eligible is not None and
                bool(eligible[start:start + len(block)].all())))
        pending.append((block, fit_keys, x, fitted))
        if len(pending) >= validation_blocks or start + len(block) == n_rows:
            settle()
        if progress:
            print(f"  simulated rows {start}-{start + block.shape[0]} / {n_rows}",
                  flush=True)
    bias = jnp.concatenate(out, axis=0)
    if return_diagnostics:
        return bias, jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *diagnostics)
    return bias


#: Diagnostic counts are summed on the device in blocks of this many trials and
#: the blocks are added in int64 on the host. Per-trial counts (for example
#: likelihood-decrease events over all iterations and starts) must be
#: non-negative and below 2**31 // COUNT_BLOCK, checked before any narrowing to
#: int32, so no int32 block sum can overflow.
COUNT_BLOCK = 256
_COUNT_FIELDS = ('converged', 'capped_starts', 'nonfinite_starts', 'material_decreases',
                 'continued_winner', 'repaired', 'masked')


def bin_index(bias, bins):
    """Histogram bin of each finite bias under :func:`shared.wnm.histogram_edges`.

    A float32 estimate from the scaled angle is corrected by comparison with the
    exact float32 edges, because rounding while shifting by 180 degrees can move
    an angle next to an edge into the neighbouring bin. Angles in [-540, 540) are
    wrapped into [-180, 180); +180 belongs to bin 0. Denormal angles are flushed
    to zero by XLA, so the negative denormals just below 0 count as 0.
    """
    edges = jnp.asarray(wm.histogram_edges(bins))
    b = jnp.where(bias >= 180., bias - 360., jnp.where(bias < -180., bias + 360., bias))
    index = jnp.clip(jnp.floor((b + 180.) * (bins / 360.)).astype(jnp.int32), 0, bins - 1)
    index = index - (b < edges[index]).astype(jnp.int32) + (b >= edges[index + 1]).astype(jnp.int32)
    return jnp.mod(index, bins)


def histogram_tile(bias, diagnostic, bins):
    """One simulation chunk as histogram-corpus arrays, binned on device.

    ``bias`` is ``(rows, trials, 2)`` from :func:`simulate`; ``diagnostic`` its
    per-trial diagnostics. Returns host arrays: ``histogram (rows, 1, 2, bins)``
    counts over the bins of :func:`shared.wnm.histogram_edges` (see
    :func:`bin_index`), ``counts (rows, 1, len(HISTOGRAM_COUNTS))`` in int64 and
    ``max_iterations (rows, 1)``. Masked (NaN) trials are left out of the
    histogram counts.
    """
    if bins < 2:
        raise ValueError('histograms need at least two bins')
    if bias.shape[1] >= 2**31:
        raise ValueError('too many trials per row for int32 histogram counts')
    limit = 2**31 // COUNT_BLOCK
    for name in _COUNT_FIELDS:
        value = diagnostic[name]
        # Host arrays wider than int32 would be narrowed on the way to the device.
        if isinstance(value, np.ndarray) and value.dtype.kind in 'iu' and value.dtype.itemsize > 4:
            if value.size and (value.min() < 0 or value.max() >= limit):
                raise ValueError(f'diagnostic {name} has counts outside [0, {limit}) that could overflow '
                                 f'int32 block sums of {COUNT_BLOCK} trials')
    reduced = jax.device_get(_histogram_tile(bias, diagnostic, bins))
    if reduced['min_count'] < 0:
        raise ValueError('negative per-trial diagnostic counts')
    if reduced['max_count'] >= limit:
        raise ValueError(f'a per-trial diagnostic count of {reduced["max_count"]} could overflow '
                         f'int32 block sums of {COUNT_BLOCK} trials')
    return dict(histogram=reduced['histogram'],
                counts=reduced['count_blocks'].sum(axis=1, dtype=np.int64)[:, None],
                max_iterations=reduced['max_iterations'])


@partial(jax.jit, static_argnames=['bins'])
def _histogram_tile(bias, diagnostic, bins):
    rows, trials = bias.shape[:2]
    finite = jnp.isfinite(bias)
    index = bin_index(jnp.where(finite, bias, 0.), bins)
    cell = (jnp.arange(rows)[:, None, None]*2+jnp.arange(2)[None, None, :])*bins+index
    histogram = jnp.zeros(rows*2*bins, jnp.int32).at[cell.ravel()].add(finite.ravel().astype(jnp.int32))
    sums = dict(selected_cap_hits=~diagnostic['converged'],
                capped_losing_starts=diagnostic['capped_starts'],
                nonfinite_starts=diagnostic['nonfinite_starts'],
                material_decreases=diagnostic['material_decreases'],
                continued_winners=diagnostic['continued_winner'],
                repaired_fits=diagnostic['repaired'], masked_fits=diagnostic['masked'])
    per_trial = jnp.stack([jnp.asarray(sums[name]).astype(jnp.int32) for name in HISTOGRAM_COUNTS], axis=-1)
    padded = -(-trials // COUNT_BLOCK) * COUNT_BLOCK
    per_trial = jnp.pad(per_trial, ((0, 0), (0, padded - trials), (0, 0)))
    count_blocks = per_trial.reshape(rows, padded // COUNT_BLOCK, COUNT_BLOCK, -1).sum(axis=2)
    return dict(histogram=histogram.reshape(rows, 1, 2, bins), count_blocks=count_blocks,
                max_count=jnp.max(per_trial), min_count=jnp.min(per_trial),
                max_iterations=jnp.max(diagnostic['iterations'], axis=1)[:, None].astype(jnp.int32))


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
