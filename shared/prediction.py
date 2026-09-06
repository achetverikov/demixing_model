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

import jax.numpy as jnp
import numpy as np

from shared.mu1_axis import mu1_grid, sign_masks, mu1_cell_width
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
        if sd_motor is None:
            # A concrete zero means "no motor noise" and is skipped entirely, so
            # the no-motor case stays bit-identical to a predictor built without
            # one rather than being convolved with a zero-width kernel.
            if not self.sd_motor:
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
        """
        edges = self._default_edges() if edges is None else jnp.asarray(edges)
        dist = self.distribution(params, validate, sd_motor)
        weights = jnp.exp(dist['log_pi'])

        def mass(lo, hi):
            per_component = self._wm.wrapped_normal_interval_probability(
                dist['mu'], dist['sigma'], lo, hi, self.arc_wraps)
            return jnp.sum(weights * per_component, axis=-1)

        return jnp.stack([mass(float(edges[i]), float(edges[i + 1]))
                          for i in range(len(edges) - 1)], axis=-1)

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

    @staticmethod
    def _default_edges():
        """Cell edges of the production reporting grid, from its centres."""
        centres = np.asarray(mu1_grid())
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
