"""Conditional wrapped-normal mixture density for Demixing Model bias.

Represents ``p(b | sd_feat1, sd_feat2, sd_ident, feat_diff)`` directly as

    q(b|x) = sum_k pi_k(x) * WN(b; mu_k(x), sigma_k(x))

with (pi, mu, sigma) produced by a small MLP.  ``b`` is the ``mu1_bias`` value
returned by ``jax_fit_main.simulate_dual_component_bias_distribution``; the
circle is 360 degrees wide and densities are **per degree**, the same
convention the production surface network uses
(``mirror_aware_model.normalize_to_density_flexible``).

There is no bias grid, no feat_diff grid and no KDE anywhere in this module.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

PERIOD = 360.0

# Fixed featurisation constants so a checkpoint is self-contained and does not
# depend on the training set's empirical moments.  They bracket the simulated
# design domain (sd 5-200, feat_diff 2-180) from `design.py`.
SD_LOG_LO = float(jnp.log(5.0))
SD_LOG_HI = float(jnp.log(200.0))
FEAT_DIFF_LO = 0.0
FEAT_DIFF_HI = 180.0


def wrap_deg(x):
    """Wrap angles in degrees onto the half-open circle [-180, 180)."""
    return jnp.mod(x + 180.0, PERIOD) - 180.0


def featurise(params):
    """Map raw parameters to normalised network inputs.

    Args:
        params: ``(..., 4)`` array of ``[sd_feat1, sd_feat2, sd_ident, feat_diff]``.

    Returns:
        ``(..., 6)`` array.  The three SDs enter as log values rescaled to
        roughly [-1, 1] because their geometry is multiplicative; ``feat_diff``
        enters both linearly and as ``cos``/``sin`` so the network can express
        the near-periodic structure of the feature separation.
    """
    params = jnp.asarray(params)
    log_sd = jnp.log(params[..., :3])
    log_sd = 2.0 * (log_sd - SD_LOG_LO) / (SD_LOG_HI - SD_LOG_LO) - 1.0
    d = params[..., 3]
    d_scaled = 2.0 * (d - FEAT_DIFF_LO) / (FEAT_DIFF_HI - FEAT_DIFF_LO) - 1.0
    d_rad = jnp.radians(d)
    return jnp.stack([log_sd[..., 0], log_sd[..., 1], log_sd[..., 2],
                      d_scaled, jnp.cos(d_rad), jnp.sin(d_rad)], axis=-1)


def mirror_params(params):
    """Swap ``sd_feat1`` and ``sd_feat2``.

    The simulator's two components are exchangeable: the component-2 bias at
    ``(sd_feat1, sd_feat2)`` is distributed exactly like the component-1 bias at
    ``(sd_feat2, sd_feat1)``.  The model always predicts the component-1 bias, so
    this map — not an architectural constraint — carries the symmetry.
    """
    params = jnp.asarray(params)
    return jnp.stack([params[..., 1], params[..., 0],
                      params[..., 2], params[..., 3]], axis=-1)


# ---------------------------------------------------------------------------
# Wrapped-normal likelihood
# ---------------------------------------------------------------------------

#: Above this sigma (degrees) the spatial wrap sum stops converging in a fixed
#: window and the Fourier series takes over; below it the spatial sum is exact
#: and the Fourier series would need too many harmonics.  The two forms agree to
#: better than float32 across a wide band around the switch, so the exact value
#: is not delicate — 60 deg sits comfortably in that band for ``n_wraps=4`` /
#: ``n_harmonics=8`` (see ``tests/test_wrapped_mixture.py``).
SIGMA_SWITCH = 60.0


def wrapped_normal_logpdf(b, mu, sigma, n_wraps: int = 4,
                          n_harmonics: int = 8, sigma_switch: float = SIGMA_SWITCH):
    """Log density (per degree) of a wrapped normal on the 360-degree circle.

    Normalised for *any* reachable sigma by switching representation at
    ``sigma_switch``:

    * ``sigma <= sigma_switch``: the spatial wrap sum
      ``sum_{j=-J..J} N(b + 360 j; mu, sigma)`` via ``logsumexp``.  Its truncated
      tail falls off as ``exp(-((2J+1)*180)^2 / 2 sigma^2)``, below float32
      resolution for ``n_wraps=4`` up to well past ``sigma_switch``.
    * ``sigma >= sigma_switch``: the Fourier series
      ``WN = (1 + 2 sum_{n=1..N} exp(-(n s)^2/2) cos(n (b-mu))) / 360`` (``s`` in
      radians), which converges *fastest* for large sigma — exactly where the
      spatial sum, in a fixed window, would leak mass and stop integrating to 1.

    Each branch evaluates its true sigma only inside its valid region (the other
    branch sees sigma clamped to ``sigma_switch``), so both stay finite with clean
    gradients and the selected value is continuous across the switch.

    Broadcasting: ``b``, ``mu`` and ``sigma`` must be mutually broadcastable.
    """
    d = wrap_deg(b - mu)

    sigma_lo = jnp.minimum(sigma, sigma_switch)
    shifts = jnp.arange(-n_wraps, n_wraps + 1, dtype=d.dtype) * PERIOD
    z = (d[..., None] + shifts) / sigma_lo[..., None]
    logp = -0.5 * z ** 2 - jnp.log(sigma_lo[..., None]) - 0.5 * jnp.log(2.0 * jnp.pi)
    logp_spatial = jax.scipy.special.logsumexp(logp, axis=-1)

    sigma_hi = jnp.maximum(sigma, sigma_switch)
    s_rad = jnp.radians(sigma_hi)
    dphi = jnp.radians(d)
    n = jnp.arange(1, n_harmonics + 1, dtype=d.dtype)
    terms = jnp.exp(-0.5 * (n * s_rad[..., None]) ** 2) * jnp.cos(n * dphi[..., None])
    dens = jnp.clip((1.0 + 2.0 * terms.sum(axis=-1)) / PERIOD, 1e-30, None)
    logp_fourier = jnp.log(dens)

    return jnp.where(sigma >= sigma_switch, logp_fourier, logp_spatial)


def mixture_logpdf(b, dist, n_wraps: int = 4):
    """Log density (per degree) of the mixture at ``b``.

    Args:
        b: ``(N,)`` bias values in degrees (any representative angle).
        dist: dict with ``log_pi``, ``mu``, ``sigma`` of shape ``(N, K)``.
    """
    comp = wrapped_normal_logpdf(b[..., None], dist['mu'], dist['sigma'], n_wraps)
    return jax.scipy.special.logsumexp(dist['log_pi'] + comp, axis=-1)


def mixture_logpdf_samples(samples, dist, n_wraps: int = 4):
    """Per-case mixture log density: ``(N, S)`` samples against ``(N, K)`` params.

    Unlike :func:`mixture_logpdf` (one bias per row) and
    :func:`mixture_logpdf_grid` (one shared grid for all rows), each of the ``N``
    cases here has its *own* block of ``S`` samples evaluated against its *own*
    mixture.  This is what lets the reference evaluation score 100k raw outcomes
    per case without repeating the parameters across every sample or building a
    ``cases x grid x samples`` tensor.
    """
    comp = wrapped_normal_logpdf(
        samples[..., None], dist['mu'][:, None, :], dist['sigma'][:, None, :], n_wraps)
    return jax.scipy.special.logsumexp(dist['log_pi'][:, None, :] + comp, axis=-1)


def mixture_logpdf_grid(grid, dist, n_wraps: int = 4):
    """Evaluate the mixture on a shared grid: ``(N, K)`` params -> ``(N, G)``.

    Grids are for evaluation and plotting only; the model itself never uses one.
    """
    comp = wrapped_normal_logpdf(
        grid[None, :, None], dist['mu'][:, None, :], dist['sigma'][:, None, :], n_wraps
    )
    return jax.scipy.special.logsumexp(dist['log_pi'][:, None, :] + comp, axis=-1)


def add_motor_noise(dist, sd_motor):
    """Convolve the mixture with an independent wrapped normal of SD ``sd_motor``.

    The wrapped normal is closed under circular convolution, so this is exact
    rather than an approximation: variances add component-wise and the weights
    and means are unchanged.  No FFT and no density grid is involved.
    """
    sd_motor = jnp.asarray(sd_motor)
    return {**dist,
            'sigma': jnp.sqrt(dist['sigma'] ** 2 + sd_motor[..., None] ** 2)}


def circular_moment(dist, n: int = 1):
    """Analytic ``E[exp(i n b)]`` of the mixture, as a complex array ``(N,)``.

    Uses the closed form ``E[e^{i n b}] = sum_k pi_k e^{i n mu_k} e^{-n^2 s_k^2/2}``
    with ``s_k`` in radians; no quadrature.
    """
    mu_rad = jnp.radians(dist['mu'])
    s_rad = jnp.radians(dist['sigma'])
    weights = jnp.exp(dist['log_pi']) * jnp.exp(-0.5 * (n * s_rad) ** 2)
    return jnp.sum(weights * jnp.exp(1j * n * mu_rad), axis=-1)


def mean_and_resultant(dist):
    """Mean direction (degrees) and resultant length of the mixture."""
    m1 = circular_moment(dist, 1)
    return jnp.degrees(jnp.angle(m1)), jnp.abs(m1)


def circular_sd(dist):
    """Circular SD in degrees, ``sqrt(-2 ln R)``, from the analytic first moment."""
    r = jnp.clip(jnp.abs(circular_moment(dist, 1)), 1e-12, 1.0)
    return jnp.degrees(jnp.sqrt(-2.0 * jnp.log(r)))


def wrapped_normal_interval_probability(mu, sigma, lo: float, hi: float,
                                        n_wraps: int = 8):
    """Probability that a wrapped normal lies in the open arc ``(lo, hi)``.

    Boundaries have zero mass for this continuous family.  Summing unwrapped
    Gaussian CDF differences avoids the severe sign-mass error produced when a
    narrow peak at zero is integrated on the legacy 2-degree reporting grid.
    """
    shifts = jnp.arange(-n_wraps, n_wraps + 1, dtype=jnp.asarray(mu).dtype) * PERIOD
    upper = (hi + shifts - mu[..., None]) / sigma[..., None]
    lower = (lo + shifts - mu[..., None]) / sigma[..., None]
    return jnp.sum(jax.scipy.special.ndtr(upper) - jax.scipy.special.ndtr(lower), axis=-1)


def density_asymmetry(dist, n_wraps: int = 8):
    """Analytic mixture ``P(0 < b < 180) - P(-180 < b < 0)``."""
    positive = wrapped_normal_interval_probability(
        dist['mu'], dist['sigma'], 0.0, 180.0, n_wraps)
    negative = wrapped_normal_interval_probability(
        dist['mu'], dist['sigma'], -180.0, 0.0, n_wraps)
    return jnp.sum(jnp.exp(dist['log_pi']) * (positive - negative), axis=-1)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def _spread_mean_init(n_components: int):
    """Bias init placing the K circular means evenly around the circle.

    Also keeps the ``(cos, sin)`` pair away from the origin at initialisation,
    where ``arctan2`` has no defined gradient.
    """
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_components, endpoint=False)

    def init(key, shape, dtype=jnp.float32):
        del key
        return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)]).astype(dtype)

    return init


class ConditionalWrappedMixture(nn.Module):
    """MLP mapping ``[sd_feat1, sd_feat2, sd_ident, feat_diff]`` to mixture parameters.

    Attributes:
        n_components: number of wrapped-normal components ``K``.
        hidden_dims: MLP widths.
        min_scale: additive floor on ``sigma`` in degrees; keeps the likelihood
            finite when many simulated biases coincide.  It is a numerical guard,
            not a smoothing prior, and is far below the 2-degree resolution of the
            production bias grid.
    """

    n_components: int = 8
    hidden_dims: Sequence[int] = (64, 128, 128)
    min_scale: float = 0.25

    @nn.compact
    def __call__(self, params):
        h = featurise(params)
        for dim in self.hidden_dims:
            h = nn.gelu(nn.Dense(dim)(h))

        k = self.n_components
        log_pi = nn.log_softmax(nn.Dense(k)(h), axis=-1)

        # Circular means via an unconstrained (cos, sin) pair: periodic by
        # construction, so no wrapping discontinuity in the parameterisation.
        cs = nn.Dense(2 * k, bias_init=_spread_mean_init(k),
                      kernel_init=nn.initializers.zeros)(h)
        mu = jnp.degrees(jnp.arctan2(cs[..., k:], cs[..., :k]))

        raw_scale = nn.Dense(k, bias_init=nn.initializers.constant(2.0))(h)
        sigma = nn.softplus(raw_scale) + self.min_scale

        return {'log_pi': log_pi, 'mu': mu, 'sigma': sigma}


def save_model(path, variables, model: ConditionalWrappedMixture, meta: dict):
    """Write a self-contained checkpoint: weights, architecture and provenance."""
    import pickle
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'variables': variables,
                     'model_config': {'n_components': model.n_components,
                                      'hidden_dims': tuple(model.hidden_dims),
                                      'min_scale': model.min_scale},
                     'meta': meta}, f)


def load_model(path):
    """Inverse of :func:`save_model`; returns ``(model, variables, meta)``."""
    import pickle

    with open(path, 'rb') as f:
        blob = pickle.load(f)
    return (ConditionalWrappedMixture(**blob['model_config']),
            blob['variables'], blob['meta'])


def nll(model, variables, params, bias, n_wraps: int = 4):
    """Mean negative log likelihood of ``bias`` under ``q(.|params)``.

    ``params`` is ``(N, 4)`` and ``bias`` is ``(N,)``: one simulated EM outcome
    per row.  This is the only training objective — no histogram, KDE, bias grid
    or surface is constructed.
    """
    dist = model.apply(variables, params)
    return -jnp.mean(mixture_logpdf(bias, dist, n_wraps))
