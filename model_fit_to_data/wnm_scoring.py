"""Every fitting objective, scored against a wrapped-normal mixture.

The public fitting command does not write only the score it was asked to fit: it
evaluates every objective at each fitted method's parameters, so a run that could
fit under one objective but not score under another would leave a results table
with some columns from one surrogate and some from another. That is why this
module covers all eight methods rather than only ``density``, and why it exists
before any WNM run is allowed to write results.

Each method keeps its own definition. The targets come from ``fitting_targets``
-- the same arrays the surface backend scores against -- and the losses come from
the same helpers the surface branches call, so what changes between the families
is the *prediction*, never the objective. Recoding a loss here, however
faithfully, would create the second copy that eventually disagrees.

What legitimately differs, and is versioned rather than hidden:

* **Likelihood.** The surface backend reads a trial's log density out of the
  180-row bias grid, at the centre of the cell the observation falls in. The
  mixture evaluates its density at the observation itself. That is a different
  observation-scoring convention, not a better implementation of the same one,
  and a head-to-head information criterion across the two needs one convention
  applied to both.
* **CRPS variants.** The surface backend renormalises grid densities into cell
  probabilities. The mixture integrates each cell exactly, which matters because
  a component can be far narrower than a 2-degree cell.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from shared.config import config
from shared.mu1_axis import bin_indices, mu1_grid

#: Methods this module can score. The public command writes all of them.
SUPPORTED_METHODS = ("likelihood", "expectation", "smoothed_exp", "density",
                     "density_legacy", "crps", "balanced_crps", "bias_weighted_crps")

#: Methods whose loss is a mean over the model's own distribution rather than a
#: sum over trials, so they say nothing when the motor SD is free and unidentified.
MEAN_ONLY_METHODS = ("expectation", "smoothed_exp")


def condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_values) -> jnp.ndarray:
    """Parameter rows for one condition, one per requested feature difference.

    The surrogate takes ``[sd_feat1, sd_feat2, sd_spat, feat_diff]`` per row, so a
    condition's curve is a batch that varies only in its last column.
    """
    feat = jnp.asarray(feat_diff_values, dtype=jnp.float32)
    return jnp.stack([jnp.broadcast_to(jnp.asarray(sd_feat1, jnp.float32), feat.shape),
                      jnp.broadcast_to(jnp.asarray(sd_feat2, jnp.float32), feat.shape),
                      jnp.broadcast_to(jnp.asarray(sd_spat, jnp.float32), feat.shape),
                      feat], axis=-1)


def _smoothing_sigma(emp_density_weights_sd, density_smoothing_sigma):
    """Grid-step sigma for the model-side feature smoother.

    ``None`` derives it from the empirical feature-weight SD exactly as the
    surface path does: the two curves have to be smoothed by the same operation
    at the same width, or the objective compares differently blurred things.
    """
    if density_smoothing_sigma is not None:
        return float(density_smoothing_sigma)
    return float(emp_density_weights_sd) / config.feat_diff_step


def predicted_asymmetry_curve(predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_grid,
                              emp_density_weights_sd, density_smoothing_sigma=None,
                              sd_motor=None):
    """Fitting-smoothed analytic signed-arc asymmetry over the feature grid.

    Validation is skipped inside the call and done once on the grid instead, by
    :func:`validate_feature_grid`. The SDs here are tracers whenever a gradient
    is being taken, so a concrete-value check inside would fail outright; the
    feature grid, which is what the layout rules actually constrain, is a fixed
    constant and can be checked before any fitting begins.
    """
    rows = condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_grid)
    return predictor.smoothed_asymmetry_curve(
        rows, _smoothing_sigma(emp_density_weights_sd, density_smoothing_sigma),
        validate=False, sd_motor=sd_motor)


def predicted_matched_density_curve(predictor, sd_feat1, sd_feat2, sd_spat,
                                    feat_diff_grid, feature_operator,
                                    density_bandwidth, sd_motor=None):
    """Signed mass after the empirical KDE and observed-design operators."""
    rows = condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_grid)
    effective_motor = predictor.sd_motor if sd_motor is None else sd_motor
    smoothing_sd = jnp.hypot(jnp.asarray(effective_motor), density_bandwidth)
    raw = predictor.signed_arc_asymmetry(
        rows, validate=False, sd_motor=smoothing_sd)
    return feature_operator @ raw


def predicted_matched_mean_bias(predictor, sd_feat1, sd_feat2, sd_spat,
                                feat_diff_grid, feature_operator, sd_motor=None):
    """Circular mean after pooling model first moments on the observed design."""
    rows = condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_grid)
    mean, resultant = predictor.mean_and_resultant(
        rows, validate=False, sd_motor=sd_motor)
    radians = jnp.radians(mean)
    real = feature_operator @ (resultant * jnp.cos(radians))
    imaginary = feature_operator @ (resultant * jnp.sin(radians))
    return jnp.degrees(jnp.arctan2(imaginary, real))


def packed_curve_loss(method, predictor, parameters, coordinates, condition_index,
                      feature_operator, target, support, density_bandwidth, *,
                      curve_losses, fit_motor=False):
    """Joint exact-coordinate loss with all participant data passed dynamically.

    The operator has shape ``(conditions, feature_bins, capacity)``. Padding
    columns are exactly zero, so their legal dummy coordinate cannot contribute.
    Keeping these arrays as explicit arguments lets one compiled optimizer serve
    every group in the same workload-derived capacity bucket.
    """
    if method not in ("density", "smoothed_exp"):
        raise ValueError(f"packed curve loss does not implement {method!r}")
    n_conditions = feature_operator.shape[0]
    shared_spatial = parameters[2 * n_conditions]
    rows = jnp.column_stack((
        parameters[2 * condition_index],
        parameters[2 * condition_index + 1],
        jnp.full_like(coordinates, shared_spatial),
        coordinates,
    ))
    fitted_motor = parameters[2 * n_conditions + 1] if fit_motor else None

    if method == "smoothed_exp":
        moment = predictor.first_moment(rows, validate=False, sd_motor=fitted_motor)
        pooled = jnp.einsum("cfu,u->cf", feature_operator, moment)
        predicted = jnp.degrees(jnp.angle(pooled))
        return jnp.sum(curve_losses(
            predicted, target, loss_type="mse", is_angular=True, weights=support))

    model_motor = fitted_motor if fit_motor else predictor.sd_motor
    kde_motor = jnp.hypot(jnp.asarray(model_motor), density_bandwidth[condition_index])
    predicted_raw = predictor.signed_arc_asymmetry(
        rows, validate=False, sd_motor=kde_motor)
    predicted = jnp.einsum("cfu,u->cf", feature_operator, predicted_raw)
    return jnp.sum(curve_losses(
        predicted, target, loss_type="ccc", is_angular=False))


def validate_feature_grid(feat_diff_grid, predictor=None):
    """Check the fixed feature grid once, before any fitting.

    The curve layout rules -- ordered, evenly spaced -- constrain the feature
    axis, which does not change during a fit. Checking it here means the
    per-evaluation path can stay traceable without giving up the guarantee.
    """
    from shared.prediction import check_curve_layout, validate_params

    grid = jnp.asarray(feat_diff_grid, dtype=jnp.float32)
    probe = condition_rows(10.0, 10.0, 10.0, grid)
    check_curve_layout(probe)
    if predictor is not None:
        # The grid must also lie inside the surrogate's domain: an out-of-domain
        # feature difference would otherwise be silently extrapolated on every
        # evaluation, since the per-evaluation path does not validate.
        validate_params(probe, domain=predictor.domain,
                        name="feature grid against the surrogate domain")


def predicted_mean_bias(predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_values,
                        sd_motor=None):
    """Analytic circular mean bias, in model degrees, at the requested features.

    Closed form, not the argmax or the expectation of a sampled grid: a mixture
    whose components straddle the wrap would give a mean that depends on where
    the grid was cut.
    """
    rows = condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_values)
    mean, _ = predictor.mean_and_resultant(rows, validate=False, sd_motor=sd_motor)
    return mean


def predicted_cell_probabilities(predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_grid,
                                 sd_motor=None):
    """``(n_bias, n_feat)`` integrated cell mass, the layout the CRPS code wants.

    Integrated rather than sampled-and-renormalised: ``min_scale`` is a quarter
    degree, well inside a 2-degree reporting cell, so a narrow component's mass
    read off the grid depends on where its peak fell within the cell.
    """
    rows = condition_rows(sd_feat1, sd_feat2, sd_spat, feat_diff_grid)
    probabilities = predictor.cell_probabilities(rows, validate=False,
                                                 sd_motor=sd_motor)  # (n_feat, n_bias)
    return probabilities.T


def trial_log_density(predictor, sd_feat1, sd_feat2, sd_spat, feat_diff, bias,
                      sd_motor=None):
    """Log density at each trial's own bias and feature difference.

    Continuous, at the observation itself. The surface backend instead reads the
    cell the observation falls in; see the module docstring on why that
    difference is a score version rather than a fix.
    """
    rows = jnp.stack([jnp.broadcast_to(jnp.asarray(sd_feat1, jnp.float32), jnp.asarray(feat_diff).shape),
                      jnp.broadcast_to(jnp.asarray(sd_feat2, jnp.float32), jnp.asarray(feat_diff).shape),
                      jnp.broadcast_to(jnp.asarray(sd_spat, jnp.float32), jnp.asarray(feat_diff).shape),
                      jnp.asarray(feat_diff, jnp.float32)], axis=-1)
    return predictor.log_density(rows, jnp.asarray(bias, jnp.float32), validate=False,
                                 sd_motor=sd_motor)


def score_condition(method, predictor, targets, condition_index, sd_feat1, sd_feat2,
                    sd_spat, *, curve_losses, ccc_or_combined_kwargs, energy_score,
                    d_circ_matrix, feat_diff_grid, emp_density_weights_sd,
                    density_smoothing_sigma=None, trials: Optional[tuple] = None,
                    sd_motor=None):
    """Loss for one condition under one objective.

    Args:
        method: one of :data:`SUPPORTED_METHODS`.
        predictor: a ``WrappedMixturePredictor``, motor noise already applied.
        targets: the :class:`fitting_targets.FittingTargets` for this subject.
        condition_index: which row of those targets to score against.
        sd_feat1, sd_feat2, sd_spat: this condition's parameters, in model degrees.
        curve_losses: ``_compute_curve_losses`` from the optimizer, injected so
            both families share one implementation of every curve loss.
        ccc_or_combined_kwargs: extra keyword arguments for the legacy density
            branch, which is the only consumer of ``corr_weight``.
        energy_score: ``bwcrps_energy_score``, injected for the same reason.
        d_circ_matrix: circular distance matrix over the bias grid.
        feat_diff_grid: the feature grid the targets were built on. Passed rather
            than rebuilt, so predictions cannot be evaluated somewhere the target
            was not.
        emp_density_weights_sd: feature-weight SD of the empirical target.
        density_smoothing_sigma: override for the model-side smoother.
        trials: ``(feat_diff, bias)`` arrays for the trial-summed objectives.

    Returns:
        Scalar loss, in that method's own units. Losses from different methods are
        deliberately not comparable.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unknown fitting method {method!r}; expected one of {SUPPORTED_METHODS}")

    # The grid is passed in, never rebuilt from config here. Targets are built on
    # whatever grid the caller used, and a differently offset grid of the same
    # length -- 1, 3, ... 179 against 2, 4, ... 180 -- would line up shape for
    # shape while every feature location was wrong, so the losses would be
    # finite, plausible, and computed between curves sampled at different places.
    feat_diff_grid = jnp.asarray(feat_diff_grid)

    if method in ("density", "density_legacy"):
        is_matched = method == "density"
        degenerate = (targets.matched_density_degenerate if is_matched
                      else targets.density_degenerate)
        if bool(np.asarray(degenerate)[condition_index]):
            raise ValueError(
                f"condition {targets.condition_names[condition_index]!r} has a constant "
                "density target; a density objective cannot be fit against it. This is "
                "scoped to the density objectives -- likelihood and CRPS are unaffected.")
        if is_matched:
            prediction_coordinates = (
                targets.prediction_coordinates
                if targets.feature_coordinate_mode == "exact"
                else feat_diff_grid)
            predicted = predicted_matched_density_curve(
                predictor, sd_feat1, sd_feat2, sd_spat, prediction_coordinates,
                targets.feature_operator[condition_index],
                targets.density_bandwidth[condition_index], sd_motor=sd_motor)
            target = targets.matched_density_target[condition_index]
        else:
            predicted = predicted_asymmetry_curve(
                predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_grid,
                emp_density_weights_sd, density_smoothing_sigma, sd_motor=sd_motor)
            target = targets.target_density[condition_index]
        loss_type = "ccc" if is_matched else "combined"
        extra = ccc_or_combined_kwargs if not is_matched else {}
        return curve_losses(predicted[None, :], target[None, :],
                            loss_type=loss_type, is_angular=False, **extra)[0]

    if method == "expectation":
        feat_values = feat_diff_grid[targets.feat_indices]
        predicted = predicted_mean_bias(predictor, sd_feat1, sd_feat2, sd_spat, feat_values,
                                        sd_motor=sd_motor)
        return curve_losses(predicted[None, :],
                            targets.target_bias[condition_index][None, :],
                            loss_type="mse", is_angular=True,
                            weights=targets.bias_weights[condition_index][None, :])[0]

    if method == "smoothed_exp":
        prediction_coordinates = (
            targets.prediction_coordinates
            if targets.feature_coordinate_mode == "exact"
            else feat_diff_grid)
        predicted = predicted_matched_mean_bias(
            predictor, sd_feat1, sd_feat2, sd_spat, prediction_coordinates,
            targets.feature_operator[condition_index], sd_motor=sd_motor)
        return curve_losses(predicted[None, :],
                            targets.target_bias_curve[condition_index][None, :],
                            loss_type="mse", is_angular=True,
                            weights=targets.smoothed_support[condition_index][None, :])[0]

    if method in ("balanced_crps", "bias_weighted_crps"):
        probabilities = predicted_cell_probabilities(
            predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_grid, sd_motor=sd_motor)
        weights = (targets.fd_weights if method == "balanced_crps"
                   else targets.bias_fd_weights)[condition_index]
        # bias weights can be all-zero for a condition; the binary support mask
        # cannot, which is why only the weighted variant floors its normaliser.
        floor = None if method == "balanced_crps" else 1e-10
        loss = energy_score(probabilities[None, :, :],
                            targets.target_d[condition_index][None, :, :],
                            weights[None, :], d_circ_matrix, norm_floor=floor)
        return loss[0, 0]

    if trials is None:
        raise ValueError(f"{method!r} is a trial-summed objective and needs trial data")
    feat_diff, bias = trials

    if method == "likelihood":
        return -jnp.sum(trial_log_density(predictor, sd_feat1, sd_feat2, sd_spat,
                                          feat_diff, bias, sd_motor=sd_motor))

    # crps: energy score at each trial's own (bias cell, feature) location, with
    # the grid indexing the surface backend uses, so only the probabilities differ.
    probabilities = predicted_cell_probabilities(
        predictor, sd_feat1, sd_feat2, sd_spat, feat_diff_grid,
        sd_motor=sd_motor)                                                # (n_bias, n_feat)
    cross = d_circ_matrix @ probabilities                                 # (n_bias, n_feat)
    second = jnp.sum(probabilities * cross, axis=0)                       # (n_feat,)
    per_cell = cross - 0.5 * second[None, :]

    bias_index = bin_indices(jnp.asarray(bias, jnp.float32))
    feat_index = jnp.clip(
        jnp.round((jnp.asarray(feat_diff, jnp.float32) - feat_diff_grid[0])
                  / config.feat_diff_step).astype(jnp.int32),
        0, len(feat_diff_grid) - 1)
    return jnp.sum(per_cell[bias_index, feat_index])


def score_all_conditions(method, predictor, targets, parameters, *, curve_losses,
                         energy_score, d_circ_matrix, feat_diff_grid,
                         emp_density_weights_sd, density_smoothing_sigma=None,
                         corr_weight=0.25, condition_trials=None, fit_motor=False):
    """Sum a method's per-condition losses, the aggregation the fitter uses.

    Conditions are summed unweighted, matching the surface backend: introducing
    trial-count weighting across conditions would change what every curve
    objective means, and is not part of this transition.

    Args:
        parameters: ``[sd_feat1_c0, sd_feat2_c0, ..., sd_spat]`` as laid out by
            ``continuous_optimizer.condition_parameter_layout``. Motor noise is
            carried by ``predictor``, not by this vector.
    """
    n_conditions = len(targets.condition_names)
    parameters = jnp.asarray(parameters)
    # One trailing entry when the motor SD is searched. It is a shared parameter
    # like sd_spat, and it is traced, so it reaches the scorers as a value rather
    # than through a predictor rebuilt per evaluation.
    expected = 2 * n_conditions + 1 + int(bool(fit_motor))
    # Exact, not "at least". A vector laid out with a trailing sd_motor would
    # otherwise have that entry silently ignored, and the fit would be scored
    # without the motor noise its own record claims it was fitted with. Motor
    # noise reaches this function through the predictor, never through here.
    if parameters.shape[0] != expected:
        raise ValueError(
            f"expected exactly {expected} parameters for {n_conditions} conditions (two "
            f"feature SDs each plus a shared spatial SD"
            + (", plus a shared motor SD" if fit_motor else "")
            + f"), got {parameters.shape[0]}."
            + ("" if fit_motor else " Motor noise is carried by the predictor unless "
                                     "fit_motor=True."))
    if condition_trials is not None and len(condition_trials) != n_conditions:
        raise ValueError(
            f"{len(condition_trials)} trial arrays for {n_conditions} conditions. The "
            "sequence is positional and is paired with the targets by index, so a length "
            "mismatch means some condition is being fitted to another's observations.")
    sd_spat = parameters[2 * n_conditions]
    sd_motor = parameters[2 * n_conditions + 1] if fit_motor else None

    total = 0.0
    for index in range(n_conditions):
        trials = None if condition_trials is None else condition_trials[index]
        total = total + score_condition(
            method, predictor, targets, index,
            parameters[2 * index], parameters[2 * index + 1], sd_spat,
            curve_losses=curve_losses,
            ccc_or_combined_kwargs={"corr_weight": corr_weight},
            energy_score=energy_score, d_circ_matrix=d_circ_matrix,
            feat_diff_grid=feat_diff_grid,
            emp_density_weights_sd=emp_density_weights_sd,
            density_smoothing_sigma=density_smoothing_sigma, trials=trials,
            sd_motor=sd_motor)
    return total


def evaluate_condition_losses(predictor, targets, params_by_condition, methods, *,
                              curve_losses, energy_score, d_circ_matrix, feat_diff_grid,
                              emp_density_weights_sd, density_smoothing_sigma=None,
                              corr_weight=0.25, condition_trials=None):
    """Every objective's loss per condition, at one fixed parameter set each.

    The mixture's counterpart to ``fit_model_to_data.evaluate_parameter_losses``,
    with the same semantics: one row of ``params_by_condition`` per condition, in
    the targets' order, as ``[sd_feat1, sd_feat2, sd_spat]`` with an optional
    fourth ``sd_motor``. Motor noise is applied per condition from that column,
    exactly as the surface version convolves each surface with its own kernel.

    This is what makes a fitted run's results interpretable: the public command
    writes *every* objective's score at each fitted method's parameters, so all of
    them have to be computable for the surrogate that produced the fit. A method
    missing here leaves a results row half-populated with no indication why.

    Mean-only objectives are evaluated rather than skipped, matching the surface
    path. They are invariant to the motor SD, but a cross-evaluation column that
    silently disappeared at non-zero motor noise would be worse than one that is
    merely insensitive.

    Returns:
        ``{method: (n_conditions,) array}``.
    """
    params_by_condition = jnp.asarray(params_by_condition)
    n_conditions = len(targets.condition_names)
    if params_by_condition.ndim != 2 or params_by_condition.shape[0] != n_conditions:
        raise ValueError(
            "params_by_condition must have shape (n_conditions, 3 or 4); got "
            f"{tuple(params_by_condition.shape)} for {n_conditions} conditions")
    if params_by_condition.shape[1] not in (3, 4):
        raise ValueError(
            f"expected 3 or 4 columns [sd_feat1, sd_feat2, sd_spat, (sd_motor)], got "
            f"{params_by_condition.shape[1]}")
    if condition_trials is not None and len(condition_trials) != n_conditions:
        raise ValueError(
            f"{len(condition_trials)} trial arrays for {n_conditions} conditions; the "
            "sequence is positional and pairs with the targets by index.")

    has_motor = params_by_condition.shape[1] >= 4
    losses = {}
    for method in methods:
        per_condition = []
        for index in range(n_conditions):
            row = params_by_condition[index]
            scorer = predictor
            if has_motor and float(row[3]) > 0:
                scorer = predictor.with_motor_noise(float(row[3]))
            trials = None if condition_trials is None else condition_trials[index]
            per_condition.append(score_condition(
                method, scorer, targets, index, row[0], row[1], row[2],
                curve_losses=curve_losses,
                ccc_or_combined_kwargs={"corr_weight": corr_weight},
                energy_score=energy_score, d_circ_matrix=d_circ_matrix,
                feat_diff_grid=feat_diff_grid,
                emp_density_weights_sd=emp_density_weights_sd,
                density_smoothing_sigma=density_smoothing_sigma, trials=trials))
        losses[method] = jnp.stack(per_condition)
    return losses
