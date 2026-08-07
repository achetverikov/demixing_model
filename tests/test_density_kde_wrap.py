"""The empirical bias KDE must close around the circle.

The error axis is periodic, so a kernel centred near +max_diss belongs partly just
past -max_diss. Truncating instead both loses mass and mis-signs it. On
orientation/colour data every error sits near 0 and the effect is nil, which is why
this went unnoticed; motion-direction data carry a mode of 180-degree-off reversals
sitting exactly on the boundary.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from shared.utils import _compute_empirical_density_asymmetry_core, KDE_WRAPS

CIRC_SPACE = 360.0
GRID = np.arange(2.0, 181.0, 2.0)


def _asymmetry(fd, bias, n_images):
    """Reference implementation with an explicit number of periodic images."""
    max_diss = CIRC_SPACE / 2
    n = 180
    dx = 2 * max_diss / n
    br = -max_diss + dx * np.arange(n)
    bw = 0.9 * min(np.std(bias),
                   (np.percentile(bias, 75) - np.percentile(bias, 25)) / 1.34) * len(bias) ** -0.2
    w = np.exp(-0.5 * ((GRID[:, None] - fd[None, :]) / 20.0) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    diff = br[:, None] - bias[None, :]
    k = sum(np.exp(-0.5 * ((diff + i * CIRC_SPACE) / bw) ** 2)
            for i in range(-n_images, n_images + 1))
    dens = (w @ (k / (bw * np.sqrt(2 * np.pi))).T)
    pos = dens[:, br > 0].sum(axis=1) * dx
    neg = dens[:, (br < 0) & (br > -max_diss)].sum(axis=1) * dx
    return pos - neg, (dens.sum(axis=1) * dx)


def _under_test(fd, bias):
    return np.asarray(_compute_empirical_density_asymmetry_core(
        jnp.asarray(fd), jnp.asarray(bias), jnp.asarray(GRID),
        weights_sd=20.0, circ_space=CIRC_SPACE)[1])


def _sample(rng, n=3000, p_reversal=0.0):
    fd = rng.uniform(2, 180, n)
    bias = rng.normal(6 * np.sin(np.pi * fd / 180), 15, n)
    rev = rng.random(n) < p_reversal
    bias[rev] = rng.normal(180.0, 20, rev.sum())
    return fd, ((bias + 180) % 360) - 180


def test_no_op_when_no_mass_near_the_antipode():
    """Orientation/colour-like data: wrapping must not perturb the existing target."""
    fd, bias = _sample(np.random.default_rng(0), p_reversal=0.0)
    truncated, _ = _asymmetry(fd, bias, n_images=0)
    assert np.max(np.abs(_under_test(fd, bias) - truncated)) < 1e-5


@pytest.mark.parametrize("p_reversal", [0.05, 0.15])
def test_reversal_mode_changes_the_target(p_reversal):
    """With mass on the boundary the wrapped and truncated targets must diverge."""
    fd, bias = _sample(np.random.default_rng(1), p_reversal=p_reversal)
    truncated, _ = _asymmetry(fd, bias, n_images=0)
    got = _under_test(fd, bias)
    assert np.max(np.abs(got - truncated)) > 1e-4


def test_matches_a_many_image_reference():
    """KDE_WRAPS images must already be converged: adding more changes nothing."""
    fd, bias = _sample(np.random.default_rng(2), p_reversal=0.15)
    reference, _ = _asymmetry(fd, bias, n_images=KDE_WRAPS + 3)
    assert np.max(np.abs(_under_test(fd, bias) - reference)) < 1e-5


def test_wrapping_conserves_mass():
    """Truncation leaks mass off the support; wrapping keeps the total at 1.

    The signed integration is still unnormalized, so this is what the leak would
    otherwise silently rescale (MODEL_PIPELINE_FOR_AGENTS.md D.11).
    """
    fd, bias = _sample(np.random.default_rng(3), p_reversal=0.20)
    _, mass_truncated = _asymmetry(fd, bias, n_images=0)
    _, mass_wrapped = _asymmetry(fd, bias, n_images=KDE_WRAPS)
    assert mass_truncated.min() < 0.999
    assert np.allclose(mass_wrapped, 1.0, atol=1e-6)
