"""Exhaustive density search: a second backend, not a flag on the first one.

``fit_hierarchical_grid`` cannot be adapted to read a curve cache. It generates
3-D log surfaces and collapses each to a density-asymmetry curve *inside* the JIT
region, and there is no point in that loop where an already-collapsed curve could
be substituted. So the exhaustive path is its own entry point, and the two are
kept honest by sharing the objective (`density_objective`) and this module's
return contract rather than by sharing code.

Why exhaustive search is tractable at all: **at a fixed shared parameter the
conditions are independent.** The loss of a candidate is a sum over conditions,
and a condition's term depends only on its own ``(sd_feat1, sd_feat2)``, so
minimising the sum is minimising each term separately. Taking the per-condition
minimum and adding them is therefore the *exact* joint optimum at that
``sd_spat`` -- not an approximation of it -- which turns a
``n_spat * n_pairs^n_conditions`` problem into ``n_spat * n_pairs``. Production
already relies on this factorisation (segment_min then sum); here it is what
makes a full lattice scan affordable.

"Exact" throughout means **exact on the lattice the source provides**. Nothing
here searches between lattice points.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

try:
    import density_objective
except ModuleNotFoundError:  # imported as `model_fit_to_data.<module>` from the repo root
    from model_fit_to_data import density_objective


#: Objectives this backend can score. `density_legacy` is deliberately absent:
#: it exists to reproduce published numbers, which were produced by the
#: hierarchical path, so running it through a different search would defeat its
#: only purpose.
SUPPORTED_OBJECTIVES = ("density",)


class CurveSource:
    """Where candidate density-asymmetry curves come from.

    Implementations: an in-memory source (tests, small scans), and the on-disk
    1-degree cache. Both must present curves **already centered**, with their
    means and variances alongside -- see `density_objective.ccc_loss_batched` for
    why the centering has to happen before the dot product rather than inside it.

    Attributes:
        n_points: curve length. Read from the source, never hard-coded: it
            follows from the feat_diff grid, and a source built on a different
            grid must not be silently scored against a 90-point target.
        sd_spat_values: ``(n_spat,)`` ascending shared-parameter lattice.
        feat_pairs: ``(n_pairs, 2)`` of ``(sd_feat1, sd_feat2)``, in the
            enumeration order the tie policy refers to.
    """

    n_points: int
    sd_spat_values: np.ndarray
    feat_pairs: np.ndarray

    def slab(self, spat_index: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(centered_curves, means, variances)`` for one ``sd_spat``.

        ``centered_curves`` is ``(n_pairs, n_points)``; the other two are
        ``(n_pairs,)``. Row ``i`` corresponds to ``feat_pairs[i]``.
        """
        raise NotImplementedError


class InMemoryCurveSource(CurveSource):
    """A `CurveSource` backed by an array already in memory.

    Used by the tests and by any caller small enough not to want a cache on disk.
    The on-disk cache implements the same interface, so the backend never learns
    which one it is talking to.
    """

    def __init__(self, sd_spat_values, feat_pairs, curves):
        """
        Args:
            sd_spat_values: ``(n_spat,)``, ascending.
            feat_pairs: ``(n_pairs, 2)``.
            curves: ``(n_spat, n_pairs, n_points)`` RAW (uncentered) curves;
                centering and statistics are computed once here.
        """
        self.sd_spat_values = np.asarray(sd_spat_values, dtype=float)
        self.feat_pairs = np.asarray(feat_pairs, dtype=float)
        curves = jnp.asarray(curves)
        if curves.ndim != 3 or curves.shape[:2] != (len(self.sd_spat_values), len(self.feat_pairs)):
            raise ValueError(
                f"curves must be (n_spat, n_pairs, n_points); got {curves.shape} for "
                f"{len(self.sd_spat_values)} sd_spat values and {len(self.feat_pairs)} pairs"
            )
        if np.any(np.diff(self.sd_spat_values) <= 0):
            raise ValueError("sd_spat_values must be strictly ascending: the tie policy "
                             "resolves to the smallest sd_spat, which is only meaningful "
                             "if the lattice is ordered.")
        self.n_points = int(curves.shape[2])
        self._centered, self._means = density_objective.centered_curves(curves)
        self._variances = jnp.mean(self._centered ** 2, axis=-1)

    def slab(self, spat_index):
        return self._centered[spat_index], self._means[spat_index], self._variances[spat_index]


def _score_slab(source: CurveSource, spat_index: int, target_stats) -> jnp.ndarray:
    """Losses of every candidate pair against every condition, at one ``sd_spat``.

    Returns ``(n_conditions, n_pairs)``. One matrix product per condition; the
    slab's curves and statistics are read once and reused across conditions,
    which is the whole reason the scan is cheap.
    """
    centered_curves, means, variances = source.slab(spat_index)
    losses = []
    for centered_target, target_mean, target_var in target_stats:
        losses.append(density_objective.ccc_loss_batched(
            centered_curves @ centered_target, means, variances,
            target_mean, target_var, source.n_points,
        ))
    return jnp.stack(losses)


def fit_exhaustive_density(
    source: CurveSource,
    targets,
    condition_names: Sequence[str],
    objective: str = "density",
    sd_motor: float = 0.0,
    verbosity: int = 1,
) -> Dict:
    """Exact-on-the-lattice density fit by exhaustive scan.

    Returns the **same object shape** ``fit_hierarchical_grid`` returns, because
    ``fit_model_to_data.process_subject`` consumes that shape directly and must
    not have to know which backend ran: ``best_loss``, ``shared_params``
    (``sd_spat``, ``sd_motor``) and ``condition_results`` keyed by condition name.

    Tie policy -- **exhaustive search over a lattice produces exact ties far more
    often than a hierarchical zoom does, so this is written down rather than
    left to whichever index an argmin happens to return.** The winner is the
    smallest ``sd_spat``; within it, the earliest entry in ``source.feat_pairs``.
    With the conventional enumeration (``sd_feat1`` outer, ``sd_feat2`` inner,
    both ascending) that reads as: smallest ``sd_spat``, then smallest
    ``sd_feat1``, then smallest ``sd_feat2``. Deterministic, and independent of
    how the scan is parallelised.

    Args:
        source: candidate curves, already centered (see `CurveSource`).
        targets: ``(n_conditions, n_points)`` empirical density-asymmetry curves.
        condition_names: names in the same order as ``targets``.
        objective: must be in `SUPPORTED_OBJECTIVES`.
        sd_motor: recorded in ``shared_params``; this backend does not search it
            (see `fit_exhaustive_density`'s caller for the refusal).
        verbosity: 0 silences progress output.

    Raises:
        ValueError: for an unsupported objective, a target/curve length mismatch,
            or a constant target (see `density_objective.check_targets_fittable`).
    """
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"exhaustive search supports {SUPPORTED_OBJECTIVES}, not {objective!r}. "
            "density_legacy exists to reproduce published numbers, which came from the "
            "hierarchical search; scoring it through a different search would defeat that."
        )

    targets = jnp.asarray(targets)
    if targets.ndim != 2 or targets.shape[0] != len(condition_names):
        raise ValueError(
            f"targets must be (n_conditions, n_points) matching {len(condition_names)} "
            f"condition names; got {targets.shape}"
        )
    if targets.shape[1] != source.n_points:
        raise ValueError(
            f"target curves have {targets.shape[1]} points but the curve source provides "
            f"{source.n_points}. These must come from the same feat_diff grid -- scoring "
            "across a grid mismatch silently compares different quantities."
        )
    density_objective.check_targets_fittable(targets, condition_names, objective)

    target_stats = [density_objective.target_statistics(targets[i])
                    for i in range(len(condition_names))]

    start_time = time.time()
    n_conditions = len(condition_names)
    best_total = np.inf
    best_spat_index = -1
    best_pair_indices = None
    best_losses = None

    for spat_index in range(len(source.sd_spat_values)):
        losses = _score_slab(source, spat_index, target_stats)     # (n_cond, n_pairs)
        # Per-condition minimum, then sum: EXACT at this sd_spat, because a
        # condition's loss depends only on its own feature pair.
        pair_indices = jnp.argmin(losses, axis=1)                  # first index wins ties
        condition_minima = losses[jnp.arange(n_conditions), pair_indices]
        total = float(jnp.sum(condition_minima))

        # Strict <: an sd_spat that merely ties the incumbent does not displace
        # it, so the smallest sd_spat wins. Ascending order is enforced by the
        # source.
        if total < best_total:
            best_total = total
            best_spat_index = spat_index
            best_pair_indices = np.asarray(pair_indices)
            best_losses = np.asarray(condition_minima)

    if best_spat_index < 0:
        raise ValueError("curve source is empty: no sd_spat values to scan.")

    total_time = time.time() - start_time
    sd_spat = float(source.sd_spat_values[best_spat_index])

    condition_results = {}
    for i, name in enumerate(condition_names):
        sd_feat1, sd_feat2 = source.feat_pairs[best_pair_indices[i]]
        condition_results[name] = {
            'condition_name': name,
            'sd_feat1': float(sd_feat1),
            'sd_feat2': float(sd_feat2),
            'loss': float(best_losses[i]),
            'surface_idx': None,
        }

    if verbosity > 0:
        print("=== EXHAUSTIVE DENSITY SEARCH ===")
        print(f"Lattice: {len(source.sd_spat_values)} sd_spat x {len(source.feat_pairs)} "
              f"feature pairs x {n_conditions} conditions")
        print(f"Scanned in {total_time:.1f}s; best loss {best_total:.4f} at sd_spat={sd_spat:.1f}")
        for name, entry in condition_results.items():
            print(f"  {name}: sd_feat1={entry['sd_feat1']:.1f}, "
                  f"sd_feat2={entry['sd_feat2']:.1f}, loss={entry['loss']:.4f}")

    return {
        'best_loss': best_total,
        'shared_params': {'sd_spat': sd_spat, 'sd_motor': float(sd_motor)},
        'condition_results': condition_results,
        'total_time': total_time,
        # Present so callers that read it (`process_subject` stores
        # `<method>_stage_times`) behave identically; a single scan has no stages.
        'stage_times': [total_time],
        'n_conditions': n_conditions,
        'condition_names': list(condition_names),
        'search_backend': 'exhaustive',
        'lattice_shape': (len(source.sd_spat_values), len(source.feat_pairs)),
    }
