"""The operations every consumer of a surrogate needs, defined once per family.

Fitting, cross-objective evaluation, likelihood rescoring, plotting and the
prediction APIs all ask a surrogate the same handful of questions.  Historically
each asked in its own way -- materialise a surface, then index it, integrate it,
or FFT-convolve it inline -- which is why the same quantity is computed by
several pieces of code that can drift apart.  This module is the one place those
questions are answered:

* ``log_density`` -- log density of the bias at given observations
* ``grid_log_density`` -- the same on a shared grid, for display
* ``cell_probabilities`` -- integrated mass per reporting cell, for
  distributional scores
* ``mean_and_resultant`` / ``circular_sd`` -- the first circular moment
* ``signed_arc_asymmetry`` -- ``P(0 < b < 180) - P(-180 < b < 0)``
* ``smoothed_asymmetry_curve`` -- that curve put through the density target's
  own smoother

Motor noise is applied inside this layer, by
:meth:`BiasPredictor.with_motor_noise`, so no consumer convolves a surface or
widens a variance itself.

Two concrete implementations, and deliberately no plugin system.  There are two
families and one of them is being retired.

**Grid densities are for display; probabilities are for scoring.**  A WNM
component can be far narrower than the 2-degree reporting cell, so sampling its
peak on a grid and renormalising does not give correct mass.  Distributional
scores take :meth:`cell_probabilities`, which integrates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from shared.mu1_axis import (mu1_cell_width, mu1_grid, mu1_grid_np,
                             periodic_integral, sign_masks)
from shared.utils import gaussian_curve_smoother

#: Component separation on the spatial axis, in model degrees.  Hardcoded at
#: every production call site of the simulator, and the reason d-prime and
#: ``sd_spat`` are two views of one number.
SPATIAL_SEPARATION = 42.0

#: The order every surrogate takes its parameters in.
PARAM_ORDER = ("sd_feat1", "sd_feat2", "sd_spat", "feat_diff")


def dprime_from_sd_spat(sd_spat):
    """``d' = 42 / sd_spat``.

    Only interfaces that speak d-prime should call this.  The surrogates, the
    fitter and the stored parameters all use ``sd_spat``; converting anywhere
    else invites two conventions in one pipeline.
    """
    return SPATIAL_SEPARATION / jnp.asarray(sd_spat)


def sd_spat_from_dprime(dprime):
    """Inverse of :func:`dprime_from_sd_spat`."""
    return SPATIAL_SEPARATION / jnp.asarray(dprime)


def mirror_params(params):
    """Swap ``sd_feat1`` and ``sd_feat2``, giving the component-2 problem.

    The simulator's two components are exchangeable: the component-2 bias at
    ``(sd_feat1, sd_feat2)`` is distributed exactly like the component-1 bias at
    ``(sd_feat2, sd_feat1)``.  Every surrogate predicts component 1, so this swap
    is how component 2 is obtained.  It is *not* a sign flip on the bias, and
    adding one on top of the swap would flip the attraction/repulsion reading of
    every component-2 curve.
    """
    params = jnp.asarray(params)
    return jnp.stack([params[..., 1], params[..., 0], params[..., 2], params[..., 3]],
                     axis=-1)


#: Fallback domain for an artifact that predates the measured per-parameter one.
#: Deliberately the production fitting box rather than anything wider: for an
#: artifact that does not say what it was trained on, the safe assumption is the
#: interval the fitter already searches, not a guess at the corpus.
LEGACY_DOMAIN = {"sd_feat1": (5.0, 200.0), "sd_feat2": (5.0, 200.0),
                 "sd_spat": (5.0, 200.0), "feat_diff": (0.0, 180.0)}


def validate_params(params, *, domain, name="parameters"):
    """Check a parameter batch against a surrogate's supported domain.

    Called on concrete arrays, outside any compiled kernel: inside ``jit`` the
    values are tracers and a bounds check either silently does nothing or forces
    a host callback.  Raises naming the offending column and its extreme value,
    rather than returning a surrogate's extrapolation as though it were a
    prediction.

    The domain is per parameter, because the corpus's is: ``sd_feat`` reaches
    down to 2.5 degrees while ``sd_spat`` stops at 5.0, since the latter is
    ``42/d'`` and d-prime was bounded at 8.4.  One shared interval had to be
    either the union -- claiming ``sd_spat`` coverage that does not exist -- or
    the intersection, which is what it was, and which refused roughly a fifth of
    the feature-noise region the network was actually trained on.

    Args:
        params: ``(N, 4)`` array ordered as :data:`PARAM_ORDER`.
        domain: mapping from parameter name to ``(low, high)``.
        name: what to call the batch in an error message.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[-1] != 4:
        raise ValueError(f"{name}: expected shape (N, 4) as {PARAM_ORDER}, got {params.shape}")
    if not np.all(np.isfinite(params)):
        bad = int(np.sum(~np.isfinite(params)))
        raise ValueError(f"{name}: {bad} non-finite entries")

    missing = [key for key in PARAM_ORDER if key not in domain]
    if missing:
        raise ValueError(f"{name}: supported domain is missing bounds for {missing}")

    # A search that optimises log SDs lands on a bound as exp(log(bound)), which
    # in float32 overshoots: exp(log(200)) is 200.0000153. Refusing that as
    # extrapolation would reject the optimizer's own legal endpoint, so bounds are
    # compared with a relative slack far below any real domain difference and far
    # above float32's round-trip error.
    ROUND_TRIP_SLACK = 1e-6
    for column, key in enumerate(PARAM_ORDER):
        low, high = domain[key]
        slack = ROUND_TRIP_SLACK * max(abs(low), abs(high), 1.0)
        low, high = low - slack, high + slack
        values = params[:, column]
        if values.min() < low or values.max() > high:
            raise ValueError(
                f"{name}: {key} ranges [{values.min():.6g}, {values.max():.6g}], outside the "
                f"corpus hull [{low:.6g}, {high:.6g}] this artifact was trained over. "
                "Predictions there are extrapolation, not interpolation.")


def domain_from_meta(meta) -> dict:
    """Read an artifact's per-parameter domain, tolerating the wnm/1 layout."""
    measured = (meta or {}).get("supported_domain")
    if measured:
        return {key: (float(lo), float(hi)) for key, (lo, hi) in measured.items()}
    sd_range = (meta or {}).get("supported_sd_range")
    feat_range = (meta or {}).get("supported_feat_diff_range")
    if sd_range and feat_range:
        # wnm/1 recorded one interval for all three SDs, taken from the
        # featurisation constants rather than from the corpus. Honour it as
        # given: it is what that artifact claims, and widening it here would
        # invent coverage on the artifact's behalf.
        return {"sd_feat1": tuple(sd_range), "sd_feat2": tuple(sd_range),
                "sd_spat": tuple(sd_range), "feat_diff": tuple(feat_range)}
    return dict(LEGACY_DOMAIN)


def legal_warmup_params(n_rows: int, sd_range, feat_diff_range) -> jnp.ndarray:
    """A batch of in-domain rows for compilation warm-up and batch padding.

    Compilation warm-up used to use ``jnp.ones((n, 3))`` -- an ``sd`` triple of
    1 degree, far below any supported value.  Shapes are all that compilation
    needs, so illegal values cost nothing there; they cost something as soon as
    the layer validates its inputs, and a padded row carrying them would either
    trip that check or quietly produce a prediction nobody wanted.  Padding rows
    are legal and interior for both reasons.
    """
    low, high = sd_range
    sd = float(np.exp(0.5 * (np.log(low) + np.log(high))))  # geometric middle
    feat_low, feat_high = feat_diff_range
    feat = float(0.5 * (feat_low + feat_high))
    return jnp.tile(jnp.asarray([[sd, sd, sd, feat]], dtype=jnp.float32), (n_rows, 1))


def check_curve_layout(params):
    """Check that a parameter batch really is one curve along the feature axis.

    All rows must share ``(sd_feat1, sd_feat2, sd_spat)`` and their ``feat_diff``
    must be strictly increasing and evenly spaced: the smoother's width is
    expressed in grid steps and its kernel is symmetric about each point, so a
    shuffled or unevenly sampled batch still yields a smooth-looking curve -- one
    that averaged the wrong neighbours.

    Deliberately not called from inside the prediction itself. It inspects
    concrete values, and under ``jax.grad`` or ``jax.jit`` the SD columns are
    tracers, so calling it there fails outright. An optimizer is exactly the
    caller that hits this: its feature grid is a fixed constant while the SDs
    vary, so it validates the layout once at setup and then differentiates
    freely. Validate outside the compiled kernel, not inside it.
    """
    params = np.asarray(params)
    if params.ndim != 2 or params.shape[-1] != 4:
        raise ValueError(f"expected (N, 4) parameter rows, got {params.shape}")

    triples = params[:, :3]
    if not np.allclose(triples, triples[:1]):
        raise ValueError(
            "expected one curve: all rows must share (sd_feat1, sd_feat2, sd_spat) and vary "
            "only in feat_diff. Smoothing runs along the feature axis, so a mixed batch "
            "would average unrelated points.")

    feat = np.asarray(params[:, 3], dtype=np.float64)
    if feat.size < 2:
        raise ValueError("a curve needs at least two feature points")
    steps = np.diff(feat)
    if np.any(steps <= 0):
        raise ValueError(
            "feat_diff must be strictly increasing; the smoother averages each point with its "
            f"neighbours by position, so row order is the feature axis. Got {feat[:5]}...")
    if not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-6):
        raise ValueError(
            "feat_diff must be evenly spaced: the kernel width is expressed in grid steps, so "
            f"uneven spacing silently changes the smoothing width (steps range "
            f"[{steps.min():.6g}, {steps.max():.6g}]).")


def pooled_bias_weighted_crps(probabilities, datasets, feat_grid, distance_matrix,
                             weights_sd):
    """Bias-weighted CRPS of report orders pooled into one distribution.

    The order of operations is the estimator, and the two orders are not
    interchangeable. The report orders are pooled into a single predicted
    distribution at each feature location *first*, and the energy score is
    applied to that. Scoring each order separately and averaging the scores is a
    different quantity, because the score is nonlinear in the distribution -- and
    it looks entirely plausible. Measured on the recorded fixtures the two differ
    by up to 2.68 on a pair with unequal support (600 trials against 80), which
    is larger than the differences these comparisons exist to detect.

    Takes probabilities rather than log surfaces so both families can supply
    them: the surface backend renormalises its sampled grid, the mixture
    integrates each cell exactly.

    Args:
        probabilities: ``(n_orders, n_bias, n_feat)``, normalised over the bias
            axis.
        datasets: one ``(n_trials, 2)`` array of ``[feat_diff, bias]`` per order,
            in the same order as ``probabilities``.
        feat_grid: feature-difference grid.
        distance_matrix: circular distance over the bias grid.
        weights_sd: Gaussian feature-weight SD, in model degrees.

    Returns:
        The pooled score, as a float.

    Raises:
        ValueError: when every feature location has zero weight. A pooled score
            with no identified feature locations is not a small score, it is no
            score.
    """
    from shared.config import config

    probabilities = np.asarray(probabilities, dtype=float)
    feat_grid = np.asarray(feat_grid, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    if len(datasets) != probabilities.shape[0]:
        raise ValueError(
            f"{len(datasets)} report-order datasets for {probabilities.shape[0]} predicted "
            "distributions; the two are paired positionally, so a mismatch scores one "
            "order's model against another's trials.")

    bias_low = config.mu1_bias_range[0]
    bias_step = config.mu1_bias_step
    n_bias = probabilities.shape[1]

    supports, weighted_empirical, weighted_bias = [], [], []
    for dataset in datasets:
        values = np.asarray(dataset, dtype=float)
        feat_diff, bias = values[:, 0], values[:, 1]
        kernel = np.exp(-0.5 * ((feat_grid[:, None] - feat_diff[None, :]) / weights_sd) ** 2)
        support = kernel.sum(axis=1)
        # Circular binning: wrap, never clip (the mu1_bias axis is a circle).
        bias_bin = np.mod(np.round((bias - bias_low) / bias_step).astype(int), n_bias)
        one_hot = np.zeros((len(bias), n_bias), dtype=float)
        one_hot[np.arange(len(bias)), bias_bin] = 1.0
        supports.append(support)
        weighted_empirical.append(kernel @ one_hot)
        weighted_bias.append(kernel @ bias)

    supports = np.stack(supports)
    total_support = supports.sum(axis=0)
    pred_fd = np.einsum("rf,rbf->fb", supports, probabilities)
    pred_fd /= np.maximum(total_support[:, None], 1e-10)
    empirical_fd = np.sum(weighted_empirical, axis=0) / np.maximum(total_support[:, None], 1e-10)
    target_d = empirical_fd @ distance_matrix
    mean_bias = np.sum(weighted_bias, axis=0) / np.maximum(total_support, 1e-10)
    support_mask = total_support > np.median(total_support) * 0.01
    fd_weights = mean_bias ** 2 * support_mask
    if not np.any(fd_weights > 0):
        raise ValueError(
            "pooled report-order BWCRPS is unidentified because all bias weights are zero")
    cross = np.sum(pred_fd * target_d, axis=1)
    self_energy = np.sum(pred_fd * (pred_fd @ distance_matrix), axis=1)
    return float(np.sum(fd_weights * (2 * cross - self_energy)) / np.sum(fd_weights))


@dataclass(frozen=True)
class PredictorIdentity:
    """What a prediction has to be labelled with to be interpretable later."""

    family: str
    n_samples: int
    artifact: str
    sd_motor: float = 0.0
    evaluator_version: str = "1"

    def as_dict(self) -> dict:
        return {
            "surrogate_family": self.family,
            "surrogate_n_samples": self.n_samples,
            "surrogate_artifact": self.artifact,
            "surrogate_sd_motor": float(self.sd_motor),
            "surrogate_evaluator_version": self.evaluator_version,
        }


class BiasPredictor:
    """Interface note: see the module docstring for the operation list.

    Subclasses carry their own supported domain in ``sd_range`` and
    ``feat_diff_range`` and are expected to validate before predicting.
    """

    family: str
    sd_range = (5.0, 200.0)
    feat_diff_range = (0.0, 180.0)

    def identity(self) -> PredictorIdentity:  # pragma: no cover - trivial
        raise NotImplementedError

    def with_motor_noise(self, sd_motor) -> "BiasPredictor":  # pragma: no cover
        raise NotImplementedError


class WrappedMixturePredictor(BiasPredictor):
    """The conditional wrapped-normal mixture, answered analytically throughout.

    Means, resultants, circular SD and the signed-arc asymmetry all have closed
    forms for this family, so none of them is obtained by integrating a grid.
    Motor noise is exact: the wrapped normal is closed under circular
    convolution, so component variances add and the weights and means are
    untouched.
    """

    family = "wnm"

    def __init__(self, model, variables, n_samples: int, artifact: str,
                 meta: Optional[dict] = None, sd_motor: float = 0.0,
                 n_wraps: int = 4, arc_wraps: int = 8):
        from continuous_density import wrapped_mixture_model as wm

        self._wm = wm
        self.model = model
        self.variables = variables
        self.n_samples = int(n_samples)
        self.artifact = artifact
        self.meta = dict(meta or {})
        self.sd_motor = float(sd_motor)
        self.n_wraps = int(n_wraps)
        self.arc_wraps = int(arc_wraps)
        self.domain = domain_from_meta(self.meta)
        # Kept for callers that only need a coarse box; the authority is
        # ``self.domain``, which is per parameter.
        self.sd_range = (min(self.domain[k][0] for k in ("sd_feat1", "sd_feat2", "sd_spat")),
                         max(self.domain[k][1] for k in ("sd_feat1", "sd_feat2", "sd_spat")))
        self.feat_diff_range = self.domain["feat_diff"]

    # -- identity -----------------------------------------------------------

    def identity(self) -> PredictorIdentity:
        return PredictorIdentity(family=self.family, n_samples=self.n_samples,
                                 artifact=self.artifact, sd_motor=self.sd_motor)

    def with_motor_noise(self, sd_motor) -> "WrappedMixturePredictor":
        # Variances add, so the sign is squared away: -20 and +20 would give
        # identical densities while the identity recorded -20, and a NaN would
        # propagate into every downstream number as a plausible-looking absence.
        sd_motor = float(sd_motor)
        if not np.isfinite(sd_motor):
            raise ValueError(f"sd_motor must be finite, got {sd_motor!r}")
        if sd_motor < 0:
            raise ValueError(
                f"sd_motor must be non-negative, got {sd_motor!r}: motor noise enters as a "
                "variance, so a negative SD is silently identical to its positive twin while "
                "being recorded as negative.")
        return WrappedMixturePredictor(
            self.model, self.variables, self.n_samples, self.artifact, self.meta,
            sd_motor=sd_motor, n_wraps=self.n_wraps, arc_wraps=self.arc_wraps)

    # -- the mixture --------------------------------------------------------

    def distribution(self, params, validate: bool = True, sd_motor=None):
        """Mixture parameters for a ``(N, 4)`` batch, motor noise already applied.

        Everything else in this class goes through here, so motor noise cannot be
        applied to one quantity and forgotten for another.

        Args:
            sd_motor: overrides the predictor's own motor SD for this call, and
                unlike :meth:`with_motor_noise` it may be a traced value. That is
                what lets a search treat the motor SD as a fitted parameter:
                variance addition is differentiable, but the constructor's
                validation is not, so a traced SD has to bypass the constructor
                rather than go through it. Bounds on a searched motor SD are the
                optimizer's to enforce, which it does through real bounds.
        """
        params = jnp.asarray(params, dtype=jnp.float32)
        if validate:
            validate_params(params, domain=self.domain, name=f"{self.artifact} inputs")
        dist = self.model.apply(self.variables, params)
        effective = self.sd_motor if sd_motor is None else sd_motor
        # A *concrete* zero means "no motor noise" and is skipped entirely, so the
        # no-motor case stays bit-identical to a predictor built without one
        # rather than being convolved with a zero-width kernel. This holds for an
        # explicit override as much as for the predictor's own SD: a caller that
        # passes 0.0 is asking for no motor noise, and previously that request was
        # honoured only when the predictor happened to store zero too, so
        # `sd_motor=0.0` against a motor-carrying predictor silently kept the
        # predictor's noise. A traced SD cannot be inspected and goes through the
        # convolution, which is what lets the search fit the motor SD.
        if isinstance(effective, (int, float, np.floating, np.integer)) and not effective:
            return dist
        return self._wm.add_motor_noise(dist, effective)

    def component_distribution(self, params, component: int, validate: bool = True):
        """Component 1 as given; component 2 by swapping the two feature SDs."""
        if component == 1:
            return self.distribution(params, validate)
        if component == 2:
            return self.distribution(mirror_params(params), validate)
        raise ValueError(f"component must be 1 or 2, got {component!r}")

    # -- the operations -----------------------------------------------------

    def log_density(self, params, bias, validate: bool = True, sd_motor=None):
        """Log density per model degree at one bias value per parameter row.

        Continuous: evaluated at the observation itself, not at the centre of the
        reporting cell it falls in.
        """
        dist = self.distribution(params, validate, sd_motor)
        return self._wm.mixture_logpdf(jnp.asarray(bias), dist, self.n_wraps)

    def grid_log_density(self, params, grid=None, validate: bool = True, sd_motor=None):
        """Log density on a shared grid -- for display, not for scoring."""
        grid = mu1_grid() if grid is None else jnp.asarray(grid)
        dist = self.distribution(params, validate, sd_motor)
        return self._wm.mixture_logpdf_grid(grid, dist, self.n_wraps)

    def cell_probabilities(self, params, edges=None, validate: bool = True, sd_motor=None):
        """Integrated mass per reporting cell: ``(N, len(edges) - 1)``.

        This is what distributional scores take.  Sampling the density at cell
        centres and renormalising is not equivalent when a component is narrower
        than a cell, which the mixture's ``min_scale`` of a quarter degree
        explicitly allows.

        **Computed in float32, deliberately, and that is a decision rather than
        an oversight.**  A cell far from every component has a mass that is small
        but representable in float64 and not in float32 -- a sigma-11 component
        180 degrees away holds 7.3e-60 -- so this returns exact zeros where the
        true value is merely tiny.  For a *score* that is immaterial: on the
        corpus's narrowest corner the 79 zeroed cells hold 3.2e-15 of total mass
        between them, contributing at most 6e-13 to an energy score whose values
        run from tens to thousands.  Staying in float32 keeps the path
        differentiable and in one dtype.

        Where the same underflow is *not* immaterial is a per-trial log
        likelihood, because log(0) is not a small number.  That column is
        computed separately in float64 from the mixture parameters; see
        ``postprocess_fitted_likelihoods.wnm_cell_log_probability``.
        """
        edges = self._default_edges() if edges is None else jnp.asarray(edges)
        dist = self.distribution(params, validate, sd_motor)
        weights = jnp.exp(dist['log_pi'])

        per_component = jax.vmap(
            lambda lo, hi: self._wm.wrapped_normal_interval_probability(
                dist['mu'], dist['sigma'], lo, hi, self.arc_wraps),
            out_axes=-1)(edges[:-1], edges[1:])
        return jnp.sum(weights[..., None] * per_component, axis=-2)

    def mean_and_resultant(self, params, validate: bool = True, sd_motor=None):
        return self._wm.mean_and_resultant(self.distribution(params, validate, sd_motor))

    def circular_sd(self, params, validate: bool = True, sd_motor=None):
        return self._wm.circular_sd(self.distribution(params, validate, sd_motor))

    def signed_arc_asymmetry(self, params, validate: bool = True, sd_motor=None):
        """Raw analytic ``P(0 < b < 180) - P(-180 < b < 0)``, one per row.

        Raw: this is the model's own quantity, before the density target's
        feature-axis smoother.  Keep it distinguishable from the fitting-smoothed
        curve; scoring one against a target built for the other changes the
        estimator.
        """
        return self._wm.density_asymmetry(self.distribution(params, validate, sd_motor),
                                          self.arc_wraps)

    def smoothed_asymmetry_curve(self, params, smoothing_sigma, validate: bool = True, sd_motor=None):
        """The asymmetry curve as the density objective sees it.

        ``params`` must be one curve: rows sharing an SD triple and varying only
        in ``feat_diff``, ordered along the feature grid, because the smoother
        runs along that axis and a reordered or mixed batch would smooth across
        unrelated points.  :func:`check_curve_layout` enforces that, and
        ``validate=False`` skips it for callers that differentiate through this
        -- see that function for why the check cannot live inside a traced call.
        """
        params = jnp.asarray(params, dtype=jnp.float32)
        if validate:
            check_curve_layout(params)
        return gaussian_curve_smoother(self.signed_arc_asymmetry(params, validate, sd_motor),
                                       smoothing_sigma)

    def pooled_circular_sd(self, params, bin_weights, validate: bool = True,
                           sd_motor=None):
        """Circular SD of a bin's pooled distribution, per bin.

        The plots compare a model SD against an empirical SD taken over trials
        pooled into ~10-degree feature bins. Evaluating the model at one 2-degree
        column measures a different quantity: pooling trials whose bias means
        differ adds a between-column term the single-column value has none of.
        The comparable model quantity mixes the per-column densities in the
        proportion the trials actually occupy.

        Done here without a grid. Mixing densities is linear, so the pooled first
        moment is the weight-average of the per-column first moments, and each of
        those is closed form for this family -- the same estimator the surface
        path reaches by integrating a mixed grid.

        The two agree to about 9e-05 degrees on the golden fixture, not better.
        The residual is the *grid's* error, not this one's: a 2-degree Riemann sum
        approximates the integral this computes exactly. So the analytic value is
        the estimand and the grid value approximates it, which is the right way
        round -- but the agreement should not be quoted more tightly than measured.
        (An earlier note claimed 4.6e-08; that was one column set, not the
        fixture, and it overstated the general case.)

        Args:
            params: ``(n_columns, 4)`` rows, one per feature column.
            bin_weights: ``(n_bins, n_columns)`` mixture weights, rows summing to
                1, or to 0 for a bin no trial fell in.

        Returns:
            ``(n_bins,)`` circular SDs in degrees, NaN where a bin is empty --
            matching the empirical curve's gaps rather than plotting a zero as
            though it were a measurement.
        """
        weights = jnp.asarray(bin_weights)
        moments = self._wm.circular_moment(self.distribution(params, validate, sd_motor), 1)
        pooled = weights @ moments
        mass = jnp.sum(weights, axis=-1)
        # Same clamp as the surface path, so the two families cannot disagree at
        # the extremes where a clamp is what decides the answer.
        resultant = jnp.abs(pooled) / jnp.where(mass > 0, mass, jnp.nan)
        safe = jnp.minimum(jnp.maximum(resultant, 1e-10), 1.0 - 1e-10)
        return jnp.degrees(jnp.sqrt(-2 * jnp.log(safe)))

    @staticmethod
    def _default_edges():
        """Cell edges of the production reporting grid, from its centres."""
        centres = mu1_grid_np()
        half = mu1_cell_width() / 2.0
        return jnp.asarray(np.concatenate([centres - half, [centres[-1] + half]]))


class SurfacePredictor(BiasPredictor):
    """The historical surface network, answered the way it always has been.

    Its outputs are 180-row log-density surfaces, so every quantity here is a
    grid operation: densities are read at cell centres, moments and asymmetry are
    discrete sums over the mu1 axis, and cell probability is density times cell
    width.  These are the conventions the deployed results were produced under
    and they are preserved exactly rather than upgraded -- a "better" integral
    here would silently re-score every historical fit.

    Surfaces are supplied by the caller, which already owns the compiled batching
    for them; this class does not load or run the network.
    """

    family = "surface_nn"

    def __init__(self, log_surfaces, n_samples: int, artifact: str,
                 sd_motor: float = 0.0):
        self.log_surfaces = jnp.asarray(log_surfaces)
        self.n_samples = int(n_samples)
        self.artifact = artifact
        self.sd_motor = float(sd_motor)
        if self.log_surfaces.ndim != 3:
            raise ValueError(
                f"expected (n_rows, n_mu1_bias, n_feat_diff) surfaces, got "
                f"{tuple(self.log_surfaces.shape)}")

    def identity(self) -> PredictorIdentity:
        return PredictorIdentity(family=self.family, n_samples=self.n_samples,
                                 artifact=self.artifact, sd_motor=self.sd_motor)

    def with_motor_noise(self, sd_motor):
        raise NotImplementedError(
            "Motor noise for the surface backend is an FFT convolution over the mu1 axis and "
            "is applied by the optimizer that owns the compiled kernels. Pass surfaces that "
            "already carry it, and record sd_motor here so the identity is right.")

    def grid_log_density(self, feat_index):
        """Log density down the mu1 axis at one feature-grid column."""
        return self.log_surfaces[:, :, feat_index]

    def cell_probabilities(self, feat_index):
        """Density times cell width -- the historical mass convention."""
        return jnp.exp(self.grid_log_density(feat_index)) * mu1_cell_width()

    def signed_arc_asymmetry(self, feat_indices):
        """Discrete sign-mass difference, excluding 0 and the antipode."""
        probs = jnp.exp(self.log_surfaces)[:, :, feat_indices]
        positive, negative = sign_masks()
        dx = mu1_cell_width()
        p_pos = jnp.sum(jnp.where(positive[None, :, None], probs, 0.0), axis=1) * dx
        p_neg = jnp.sum(jnp.where(negative[None, :, None], probs, 0.0), axis=1) * dx
        return p_pos - p_neg

    def smoothed_asymmetry_curve(self, feat_indices, smoothing_sigma):
        """Same smoother, same width, same padding as the WNM path."""
        raw = self.signed_arc_asymmetry(feat_indices)
        return jnp.stack([gaussian_curve_smoother(row, smoothing_sigma) for row in raw])

    def pooled_circular_sd(self, bin_weights):
        """Circular SD of each bin's pooled distribution.

        The same estimator the mixture computes analytically, reached the way
        this family has always reached it: mix the per-column densities on the
        grid, then take the first moment of the mixture. Kept as an integral
        rather than converted, because these are the numbers the deployed plots
        were produced with.

        Args:
            bin_weights: ``(n_surfaces, n_bins, n_feat_diff)`` mixture weights.

        Returns:
            ``(n_surfaces, n_bins)`` circular SDs in degrees, NaN for empty bins.
        """
        weights = jnp.asarray(bin_weights)
        if weights.shape[-1] != self.log_surfaces.shape[2]:
            raise ValueError(
                f"bin weights span {weights.shape[-1]} feature columns but the surfaces have "
                f"{self.log_surfaces.shape[2]}")

        grid = mu1_grid()
        probabilities = jnp.exp(self.log_surfaces)
        mixtures = jnp.einsum('smf,sbf->smb', probabilities, weights)

        angles = jnp.radians(grid)
        mass = periodic_integral(mixtures, axis=1)
        cosine = periodic_integral(mixtures * jnp.cos(angles)[None, :, None], axis=1)
        sine = periodic_integral(mixtures * jnp.sin(angles)[None, :, None], axis=1)

        # The clamp is the deployed one, verbatim: [1e-10, 1 - 1e-10], not a
        # tidier [1e-12, 1]. It is the *lower* bound that carries the behaviour:
        # a surface whose first moment cancels in float32 gives r near zero, and
        # 1e-10 against 1e-12 moved the answer by 37 degrees. The upper bound is
        # inert -- 1 - 1e-10 is exactly 1.0 in float32 -- which is a separate
        # quirk, recorded as decision 6 in OPEN_DECISIONS.md rather than fixed
        # here. No golden fixture goes near either bound, so nothing in the
        # recorded reference would have caught the change.
        resultant = jnp.sqrt(cosine ** 2 + sine ** 2) / jnp.where(mass > 0, mass, jnp.nan)
        safe = jnp.minimum(jnp.maximum(resultant, 1e-10), 1.0 - 1e-10)
        return jnp.degrees(jnp.sqrt(-2 * jnp.log(safe)))


def mixture_plot_curves(predictor, params_by_row, feat_grid, bin_weights=None,
                        sd_motor_by_row=None, emp_density_weights_sd=20.0,
                        density_smoothing_sigma=None, feature_operators=None,
                        density_bandwidths=None):
    """The four curve families the subject plots draw, for a mixture fit.

    The surface backend derives these by integrating its 180-row grid; the
    mixture has a closed form for each, so this computes them directly rather
    than materialising a surface and then integrating it back down. That is not
    an optimisation: a component narrower than the 2-degree reporting cell is
    mis-massed by the grid, and those are exactly the fits this surrogate exists
    to represent.

    Deliberately additive. The surface path is untouched by this function, so no
    plotted number on that side can move because of it.

    Args:
        predictor: a ``WrappedMixturePredictor``.
        params_by_row: ``(n_rows, 3)`` of ``[sd_feat1, sd_feat2, sd_spat]`` --
            one row per fitted condition-and-optimizer the plot will draw.
        feat_grid: feature differences to evaluate, in model degrees.
        bin_weights: ``(n_rows, n_bins, n_feat)`` for the pooled SD panel, or
            ``None`` to skip it.
        sd_motor_by_row: per-row motor SD, or ``None`` for no motor noise. Motor
            noise is per fitted row because it is a fitted parameter. These
            override the predictor's own stored SD, zero included -- a row of 0
            means no motor noise even when the predictor carries some.
        emp_density_weights_sd, density_smoothing_sigma: **must be the settings
            the fit ran under**, not this function's defaults. The asymmetry
            curve is smoothed here exactly as the density objective smooths its
            target, so a fit run at one sigma and plotted at another shows a
            curve the fit never optimised, with nothing in the plot to say so.
            The caller holds the fit record; this function cannot check it.
        feature_operators: optional ``(n_rows, n_feat, n_feat)`` observed-design
            operators. When supplied, bias and asymmetry are exactly the
            recovery-selected fitted curves rather than legacy grid smoothing.
        density_bandwidths: pooled-SJ bias KDE SD per row. Required together
            with ``feature_operators`` for the matched density curve.

    Returns:
        ``{"bias", "asymmetry", "sd"}`` each ``(n_rows, n_feat)``, plus
        ``"pooled_sd"`` ``(n_rows, n_bins)`` when weights are given.
    """
    params_by_row = np.asarray(params_by_row, dtype=np.float64)
    feat_grid = jnp.asarray(feat_grid, dtype=jnp.float32)
    n_rows = len(params_by_row)
    motors = (np.zeros(n_rows) if sd_motor_by_row is None
              else np.asarray(sd_motor_by_row, dtype=np.float64))
    if not np.all(np.isfinite(motors)) or np.any(motors < 0):
        raise ValueError(
            f"motor SDs must be finite and non-negative, got {list(motors)}: motor noise "
            "enters as a variance, so a negative SD draws exactly its positive twin's curve "
            "while the plot's record says otherwise.")
    if len(motors) != n_rows:
        raise ValueError(
            f"{len(motors)} motor SDs for {n_rows} parameter rows; they are paired "
            "positionally, so a mismatch draws one fit's curve at another's motor noise.")
    if (feature_operators is None) != (density_bandwidths is None):
        raise ValueError("feature_operators and density_bandwidths must be supplied together")
    if feature_operators is not None:
        feature_operators = np.asarray(feature_operators, dtype=np.float32)
        density_bandwidths = np.asarray(density_bandwidths, dtype=np.float32)
        if feature_operators.shape != (n_rows, len(feat_grid), len(feat_grid)):
            raise ValueError(
                "feature_operators must have shape "
                f"{(n_rows, len(feat_grid), len(feat_grid))}, got {feature_operators.shape}")
        if density_bandwidths.shape != (n_rows,):
            raise ValueError(
                f"density_bandwidths must have shape {(n_rows,)}, got {density_bandwidths.shape}")

    from shared.config import config

    smoothing = (float(emp_density_weights_sd) / config.feat_diff_step
                 if density_smoothing_sigma is None else float(density_smoothing_sigma))

    bias, asymmetry, circular_sd, pooled = [], [], [], []
    for index in range(n_rows):
        sd_feat1, sd_feat2, sd_spat = params_by_row[index]
        rows = jnp.stack([
            jnp.full(feat_grid.shape, float(sd_feat1), jnp.float32),
            jnp.full(feat_grid.shape, float(sd_feat2), jnp.float32),
            jnp.full(feat_grid.shape, float(sd_spat), jnp.float32),
            feat_grid], axis=-1)
        motor = float(motors[index])

        # Validated once per row, then skipped for the three calls that follow on
        # the identical array. Without this the whole helper ran unvalidated: a
        # negative sd_feat returned a NaN curve and a mis-ordered feature grid
        # returned plausible numbers, both of which a plot renders without
        # complaint. The four families share `rows`, so one check covers them.
        mean, resultant = predictor.mean_and_resultant(rows, validate=True, sd_motor=motor)
        if feature_operators is None:
            bias.append(np.asarray(mean))
            asymmetry.append(np.asarray(gaussian_curve_smoother(
                predictor.signed_arc_asymmetry(
                    rows, validate=False, sd_motor=motor), smoothing)))
        else:
            operator = jnp.asarray(feature_operators[index])
            radians = jnp.radians(mean)
            bias.append(np.asarray(jnp.degrees(jnp.arctan2(
                operator @ (resultant * jnp.sin(radians)),
                operator @ (resultant * jnp.cos(radians))))))
            kde_motor = jnp.hypot(motor, density_bandwidths[index])
            asymmetry.append(np.asarray(operator @ predictor.signed_arc_asymmetry(
                rows, validate=False, sd_motor=kde_motor)))
        circular_sd.append(np.asarray(
            predictor.circular_sd(rows, validate=False, sd_motor=motor)))
        if bin_weights is not None:
            pooled.append(np.asarray(predictor.pooled_circular_sd(
                rows, jnp.asarray(bin_weights[index]), validate=False, sd_motor=motor)))

    bundle = {"bias": np.stack(bias), "asymmetry": np.stack(asymmetry),
              "sd": np.stack(circular_sd)}
    if bin_weights is not None:
        bundle["pooled_sd"] = np.stack(pooled)
    return bundle


def predictor_from_surrogate(loaded, sd_motor: float = 0.0) -> BiasPredictor:
    """Build a predictor from :func:`shared.surrogate.load_surrogate`'s result."""
    from shared import surrogate

    if loaded.family == surrogate.FAMILY_WNM:
        return WrappedMixturePredictor(
            model=loaded.payload["model"], variables=loaded.payload["variables"],
            n_samples=loaded.n_samples, artifact=loaded.path.name, meta=loaded.meta,
            sd_motor=sd_motor)
    raise NotImplementedError(
        "The surface backend's predictions come from the optimizer's compiled batching; "
        "build a SurfacePredictor from surfaces it produced rather than from a checkpoint.")
