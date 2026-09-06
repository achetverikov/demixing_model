"""Fit one subject's conditions by bounded gradient search over the mixture.

The third search backend, alongside the hierarchical zoom and the exhaustive
lattice scan. It returns every key ``fit_model_to_data.process_subject`` consumes,
so result handling does not fork a third way.

It is a superset, not an identical shape: like the exhaustive backend it adds
``search_backend``, which the hierarchical backend does not emit at all. Any
consumer reading that key must use ``.get`` with a default rather than indexing
it, or a hierarchical result raises. The extra continuous-only fields are listed
at the bottom of the returned dict.

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
import numpy as _np

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
                   sd_motor: float = 0.0, fit_motor: bool = False,
                   sd_motor_bounds=(0.1, 50.0), n_starts: int = 8, seed: int = 0,
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
    if condition_trials is not None and len(condition_trials) != len(condition_names):
        raise ValueError(
            f"{len(condition_trials)} trial arrays for {len(condition_names)} conditions. "
            "The sequence is positional, so a mismatch fits one condition's parameters to "
            "another's observations while every loss stays finite and plausible.")
    if objective in MEAN_ONLY_METHODS and (sd_motor or fit_motor):
        raise ValueError(
            f"{objective!r} is invariant to motor noise -- a symmetric zero-mean convolution "
            "leaves the circular mean exactly where it was -- so a motor SD fitted or fixed "
            "under it would be a number the objective cannot see. The caller skips these "
            "objectives in a motor-noise run rather than reporting an unidentified estimate.")
    if fit_motor and sd_motor:
        raise ValueError(
            "sd_motor is either searched (fit_motor=True) or held fixed, not both.")

    # The per-evaluation path skips validation so it stays differentiable, so the
    # grid it will be evaluated on is checked once, here, before any fitting.
    validate_feature_grid(feat_diff_grid, predictor)

    if sd_motor:
        predictor = predictor.with_motor_noise(sd_motor)

    bounds_by_axis = surrogate_module.search_bounds(predictor.domain)
    n_conditions = len(condition_names)
    names = condition_parameter_layout(n_conditions, fit_motor=fit_motor)
    bounds = build_bounds(n_conditions, bounds_by_axis["sd_feat"], bounds_by_axis["sd_spat"],
                          motor_bounds=tuple(sd_motor_bounds) if fit_motor else None)

    def objective_fn(parameters):
        return score_all_conditions(
            objective, predictor, targets, parameters, curve_losses=curve_losses,
            energy_score=energy_score, d_circ_matrix=d_circ_matrix,
            feat_diff_grid=feat_diff_grid,
            emp_density_weights_sd=emp_density_weights_sd,
            density_smoothing_sigma=density_smoothing_sigma, corr_weight=corr_weight,
            condition_trials=condition_trials, fit_motor=fit_motor)

    started = time.time()
    fit = minimize_continuous(objective_fn, bounds, names, n_starts=n_starts, seed=seed)
    total_time = time.time() - started

    parameters = np.asarray(fit.parameters)
    sd_spat = float(parameters[2 * n_conditions])
    fitted_motor = float(parameters[2 * n_conditions + 1]) if fit_motor else float(sd_motor)

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
        one_params = [parameters[2 * index], parameters[2 * index + 1], sd_spat]
        if fit_motor:
            one_params.append(fitted_motor)
        per_condition = score_all_conditions(
            objective, predictor, one, jnp.asarray(one_params), fit_motor=fit_motor,
            curve_losses=curve_losses, energy_score=energy_score,
            d_circ_matrix=d_circ_matrix, feat_diff_grid=feat_diff_grid,
            emp_density_weights_sd=emp_density_weights_sd,
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
        'shared_params': {'sd_spat': sd_spat, 'sd_motor': fitted_motor},
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
        # Losses alone cannot say whether the winner was a start that converged
        # or one that merely stopped, nor which starts railed. Only converged
        # starts can win, but the record has to show that rather than assert it.
        'start_outcomes': [
            {'start': [float(v) for v in s.start],
             'solution': [float(v) for v in s.solution],
             'loss': float(s.loss), 'success': bool(s.success), 'status': s.status,
             'n_iterations': int(s.n_iterations), 'n_evaluations': int(s.n_evaluations),
             'at_bound': list(s.at_bound)}
            for s in fit.starts],
        'search_settings': dict(fit.settings),
    }


class ContinuousEngine:
    """Per-subject driver for the continuous backend, shaped like the optimizer.

    ``process_subject`` reads a handful of attributes off whatever engine it is
    given -- the empirical targets, the condition names, the feature grid -- and
    then calls one fit method per objective. This exposes that same surface so the
    loop body does not branch on which backend is running; the alternative is a
    second copy of the per-subject bookkeeping, which is how two paths start
    disagreeing about what a fitted run contains.

    It holds no surfaces and loads no checkpoint of its own: it is constructed
    from an already-loaded predictor, and builds its targets through
    ``fitting_targets`` like everything else.
    """

    def __init__(self, predictor, *, curve_losses, energy_score,
                 degenerate_targets, bwcrps_condition_targets, target_bias_curve_core,
                 emp_density_weights_sd: float, density_smoothing_sigma=None,
                 density_bandwidth_rule: str = "sj", density_bandwidth_mode: str = "pooled",
                 corr_weight: float = 0.25, skip_motor_noise: bool = True,
                 n_starts: int = 8, seed: int = 0):
        import jax.numpy as _jnp
        from fitting_targets import build_fitting_targets
        from shared.config import config

        self.predictor = predictor
        self.skip_motor_noise = bool(skip_motor_noise)
        self.corr_weight = float(corr_weight)
        self.emp_density_weights_sd = float(emp_density_weights_sd)
        self.density_smoothing_sigma = density_smoothing_sigma
        self.density_bandwidth_rule = density_bandwidth_rule
        self.density_bandwidth_mode = density_bandwidth_mode
        self.n_starts = int(n_starts)
        self.seed = int(seed)

        self._build_targets = build_fitting_targets
        self._curve_losses = curve_losses
        self._energy_score = energy_score
        self._degenerate_targets = degenerate_targets
        self._bwcrps_condition_targets = bwcrps_condition_targets
        self._target_bias_curve_core = target_bias_curve_core

        self.feat_diff_grid = config.create_grid('feat_diff')
        bias_grid = config.create_grid('mu1_bias')
        self.n_mu1_bias = len(bias_grid)
        difference = _jnp.abs(bias_grid[:, None] - bias_grid[None, :])
        self.D_circ_matrix = _jnp.minimum(difference, 360.0 - difference)

        # Checked once, here: the per-evaluation path skips validation to stay
        # differentiable, so the grid every prediction will use is verified
        # against the surrogate's domain before any subject is fitted.
        validate_feature_grid(self.feat_diff_grid, predictor)

        self.targets = None
        self.condition_datasets = None
        self.condition_names = ()
        self.n_conditions = 0

    def update_dataset(self, condition_datasets):
        """Rebuild the empirical targets for one subject's conditions."""
        self.condition_datasets = condition_datasets
        self.condition_names = tuple(condition_datasets)
        self.n_conditions = len(self.condition_names)
        self.targets = self._build_targets(
            condition_datasets, feat_diff_grid=self.feat_diff_grid,
            d_circ_matrix=self.D_circ_matrix, n_mu1_bias=self.n_mu1_bias,
            emp_density_weights_sd=self.emp_density_weights_sd,
            density_bandwidth_rule=self.density_bandwidth_rule,
            density_bandwidth_mode=self.density_bandwidth_mode,
            degenerate_targets=self._degenerate_targets,
            bwcrps_condition_targets=self._bwcrps_condition_targets,
            target_bias_curve_core=self._target_bias_curve_core)
        for warning in self.targets.near_constant_warnings:
            print(warning)

    # The attribute names process_subject reads, kept identical so the loop body
    # is shared rather than duplicated.
    @property
    def unified_target_bias(self):
        return self.targets.target_bias

    @property
    def unified_bias_weights(self):
        return self.targets.bias_weights

    @property
    def unified_target_density(self):
        return self.targets.target_density

    @property
    def unified_target_bias_curve(self):
        return self.targets.target_bias_curve

    @property
    def unified_feat_indices(self):
        return self.targets.feat_indices

    def _trials(self):
        import jax.numpy as _jnp

        return [(_jnp.asarray(_np.asarray(values)[:, 0]),
                 _jnp.asarray(_np.asarray(values)[:, 1]))
                for values in self.condition_datasets.values()]

    def fit(self, fitting_method: str, sd_motor: float = 0.0, sd_motor_max: float = 50.0,
            verbosity: int = 1):
        """One objective, through the gradient search.

        When the engine was built with motor noise enabled, the motor SD is a
        searched parameter bounded above by the caller's empirical cap -- the same
        data-derived cap the surface backend uses, since the motor SD is one
        component of the total response error and cannot exceed it.
        """
        if self.targets is None:
            raise RuntimeError("call update_dataset() before fitting")
        fit_motor = not self.skip_motor_noise and not sd_motor
        return fit_continuous(
            self.predictor, self.targets, list(self.condition_names),
            objective=fitting_method, curve_losses=self._curve_losses,
            energy_score=self._energy_score, d_circ_matrix=self.D_circ_matrix,
            feat_diff_grid=self.feat_diff_grid,
            emp_density_weights_sd=self.emp_density_weights_sd,
            density_smoothing_sigma=self.density_smoothing_sigma,
            corr_weight=self.corr_weight, condition_trials=self._trials(),
            sd_motor=sd_motor, fit_motor=fit_motor,
            sd_motor_bounds=(0.1, float(sd_motor_max)),
            n_starts=self.n_starts, seed=self.seed, verbosity=verbosity)

    def evaluate(self, params_by_condition, fitting_methods):
        """Every objective's loss per condition, at fixed parameters."""
        from wnm_scoring import evaluate_condition_losses

        return evaluate_condition_losses(
            self.predictor, self.targets, params_by_condition, fitting_methods,
            curve_losses=self._curve_losses, energy_score=self._energy_score,
            d_circ_matrix=self.D_circ_matrix, feat_diff_grid=self.feat_diff_grid,
            emp_density_weights_sd=self.emp_density_weights_sd,
            density_smoothing_sigma=self.density_smoothing_sigma,
            corr_weight=self.corr_weight, condition_trials=self._trials())

    def search_spec(self) -> Dict:
        """The settings that produced the parameters, for the run fingerprint.

        The one place this description is built. It used to be duplicated at the
        fingerprint call site, which is how the two drifted: a change to the
        optimizer's tolerances or iteration cap altered the fitted parameters
        while both copies kept emitting the same five fields, so two genuinely
        different searches shared a digest and could resume into each other.

        Defaults are read from ``minimize_continuous`` itself rather than
        restated, so a change there cannot silently leave the identity behind.
        """
        import inspect

        bounds = surrogate_module.search_bounds(self.predictor.domain)
        defaults = inspect.signature(minimize_continuous).parameters
        return {
            "method": "L-BFGS-B",
            "parameterisation": "log",
            "n_starts": int(self.n_starts),
            "seed": int(self.seed),
            "sd_feat_bounds": [float(v) for v in bounds["sd_feat"]],
            "sd_spat_bounds": [float(v) for v in bounds["sd_spat"]],
            "max_iterations": int(defaults["max_iterations"].default),
            "tolerance": float(defaults["tolerance"].default),
            "gradient_tolerance": float(defaults["gradient_tolerance"].default),
            "motor": "searched" if not self.skip_motor_noise else "fixed_zero",
        }
