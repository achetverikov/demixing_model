"""Fit one subject's conditions by bounded gradient search over the mixture.

The third search backend, alongside the hierarchical zoom and the exhaustive
lattice scan. It returns the **same object shape** those two return, because
``fit_model_to_data.process_subject`` consumes that shape directly and must not
have to know which backend ran -- the alternative is a third set of result
handling that drifts from the other two.

It is the only backend that searches continuous parameters rather than a
lattice, so its result carries the extra facts that only it can report: how many
starts ran, how far apart they finished, and which bounds the winner sits on.
A lattice search has no analogue of the first two, and a benchmark that compared
backends without them would be comparing a single number from each.

Nothing here decides whether gradients are the right production search. That is
the recovery panel's job, and the start budget this runs at is a placeholder
until that panel selects one -- see ``TODO.md`` item 3.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from continuous_optimizer import (build_bounds, condition_parameter_layout,
                                  minimize_continuous)
from shared import surrogate as surrogate_module
from wnm_scoring import MEAN_ONLY_METHODS, SUPPORTED_METHODS, score_all_conditions, \
    validate_feature_grid


def fit_continuous(predictor, targets, condition_names: Sequence[str], *,
                   objective: str, curve_losses, energy_score, d_circ_matrix,
                   feat_diff_grid, emp_density_weights_sd: float,
                   density_smoothing_sigma: Optional[float] = None,
                   corr_weight: float = 0.25, condition_trials=None,
                   sd_motor: float = 0.0, n_starts: int = 8, seed: int = 0,
                   verbosity: int = 1) -> Dict:
    """Bounded multistart gradient fit, in the shape the other backends return.

    Args:
        predictor: a ``WrappedMixturePredictor``. Motor noise is applied here
            rather than searched: this backend fits the noise parameters at a
            given motor SD, and a free motor SD is the caller's stage to run.
        targets: the subject's :class:`fitting_targets.FittingTargets`.
        condition_names: must match the targets' own order, which is the order
            every target array's first axis follows.
        objective: one of ``wnm_scoring.SUPPORTED_METHODS``.
        condition_trials: ``[(feat_diff, bias), ...]`` for trial-summed objectives.
        n_starts: **provisional**; see the module docstring.

    Returns:
        The ``fit_hierarchical_grid`` shape, plus ``loss_spread``, ``at_bound``,
        ``n_starts`` and ``start_losses``.
    """
    if objective not in SUPPORTED_METHODS:
        raise ValueError(f"unknown objective {objective!r}; expected one of {SUPPORTED_METHODS}")
    if tuple(condition_names) != tuple(targets.condition_names):
        raise ValueError(
            f"condition order mismatch: asked to fit {tuple(condition_names)} against targets "
            f"built for {tuple(targets.condition_names)}. Every target array is positional, so "
            "a mismatch here scores each condition against another one's data.")
    if objective in MEAN_ONLY_METHODS and sd_motor:
        raise ValueError(
            f"{objective!r} is invariant to motor noise -- a symmetric zero-mean convolution "
            "leaves the circular mean exactly where it was -- so fitting it at a non-zero "
            "sd_motor would report a number the objective cannot see.")

    # The per-evaluation path skips validation so it stays differentiable, so the
    # grid it will be evaluated on is checked once, here, before any fitting.
    validate_feature_grid(feat_diff_grid, predictor)

    if sd_motor:
        predictor = predictor.with_motor_noise(sd_motor)

    bounds_by_axis = surrogate_module.search_bounds(predictor.domain)
    n_conditions = len(condition_names)
    names = condition_parameter_layout(n_conditions, fit_motor=False)
    bounds = build_bounds(n_conditions, bounds_by_axis["sd_feat"], bounds_by_axis["sd_spat"])

    def objective_fn(parameters):
        return score_all_conditions(
            objective, predictor, targets, parameters, curve_losses=curve_losses,
            energy_score=energy_score, d_circ_matrix=d_circ_matrix,
            emp_density_weights_sd=emp_density_weights_sd,
            density_smoothing_sigma=density_smoothing_sigma, corr_weight=corr_weight,
            condition_trials=condition_trials)

    started = time.time()
    fit = minimize_continuous(objective_fn, bounds, names, n_starts=n_starts, seed=seed)
    total_time = time.time() - started

    parameters = np.asarray(fit.parameters)
    sd_spat = float(parameters[2 * n_conditions])

    # Per-condition losses at the joint solution. The fit minimises their sum, so
    # these are reported for diagnosis, not re-minimised: a per-condition loss
    # that looks bad next to a good total is how a shared parameter shows it is
    # being pulled by one condition.
    condition_results = {}
    for index, name in enumerate(condition_names):
        trials = None if condition_trials is None else [condition_trials[index]]
        one = type(targets)(
            condition_names=(name,),
            feat_indices=targets.feat_indices,
            target_bias=targets.target_bias[index][None, :],
            bias_weights=targets.bias_weights[index][None, :],
            target_density=targets.target_density[index][None, :],
            target_bias_curve=targets.target_bias_curve[index][None, :],
            target_d=targets.target_d[index][None, :, :],
            fd_weights=targets.fd_weights[index][None, :],
            bias_fd_weights=targets.bias_fd_weights[index][None, :],
            density_target_var=np.asarray(targets.density_target_var)[index][None],
            density_degenerate=np.asarray(targets.density_degenerate)[index][None],
            density_bandwidth=(targets.density_bandwidth[index],),
            near_constant_warnings=(),
        )
        per_condition = score_all_conditions(
            objective, predictor, one,
            jnp.asarray([parameters[2 * index], parameters[2 * index + 1], sd_spat]),
            curve_losses=curve_losses, energy_score=energy_score,
            d_circ_matrix=d_circ_matrix, emp_density_weights_sd=emp_density_weights_sd,
            density_smoothing_sigma=density_smoothing_sigma, corr_weight=corr_weight,
            condition_trials=trials)
        condition_results[name] = {
            'condition_name': name,
            'sd_feat1': float(parameters[2 * index]),
            'sd_feat2': float(parameters[2 * index + 1]),
            'loss': float(per_condition),
            'surface_idx': None,
        }

    if verbosity > 0:
        print("=== CONTINUOUS GRADIENT SEARCH ===")
        print(f"Objective: {objective} | conditions: {n_conditions} | starts: {n_starts} "
              f"(provisional budget)")
        print(f"Best loss {fit.loss:.6f} | spread across converged starts "
              f"{fit.loss_spread:.3e} | "
              f"{sum(s.success for s in fit.starts)}/{n_starts} converged")
        if fit.at_bound:
            print(f"  BOUNDARY: {', '.join(fit.at_bound)} — the fit wanted to leave the box")
        if fit.loss_spread > 0.1 * max(abs(fit.loss), 1e-12):
            print("  WARNING: starts disagree by more than 10% of the winning loss; the start "
                  "budget, not the objective, may be choosing this answer.")

    return {
        'best_loss': float(fit.loss),
        'shared_params': {'sd_spat': sd_spat, 'sd_motor': float(sd_motor)},
        'condition_results': condition_results,
        'total_time': total_time,
        # A gradient search has no stages; present so callers that record
        # `<method>_stage_times` behave identically across backends.
        'stage_times': [total_time],
        'n_conditions': n_conditions,
        'condition_names': list(condition_names),
        'search_backend': 'continuous',
        # Only this backend can report these, and the comparison needs them.
        'loss_spread': float(fit.loss_spread),
        'at_bound': list(fit.at_bound),
        'n_starts': int(n_starts),
        'n_converged': int(sum(s.success for s in fit.starts)),
        'start_losses': [float(s.loss) for s in fit.starts],
        'search_settings': dict(fit.settings),
    }
