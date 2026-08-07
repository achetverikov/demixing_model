"""The bias KDE must not be asked to resolve detail its grid cannot represent.

The empirical density target evaluates a Gaussian kernel on a fixed 2-model-degree
bias grid. A bandwidth much below the cell width puts the kernel BETWEEN grid
points, and the rectangle rule stops approximating the density it is meant to be:
the answer starts depending on where trials happen to fall within a cell rather
than on the data.

That is not hypothetical. Sheather-Jones returns bw = 0.09 model degrees on real
moors cells (bw/dx = 0.046), where the production target was wrong by 278% of curve
range against a 40x-finer reference. Silverman reached bw/dx = 0.21 on the same
data, so the exposure predates SJ; SJ made it worse.

The fix is to widen the distribution rather than the kernel: the asymmetry is a
signed MASS difference and scaling every error by one positive factor preserves
sign, so the estimand is unchanged while the bandwidth scales out of the aliasing
regime. Measured against the bandwidth the data actually support, that lands within
0.91% of it on real moors cells (max 1.94%), where flooring the bandwidth instead
leaves 4.53% median and 38.8% max.

The oracles here are independent of the implementation: a probability density
integrates to exactly 1 over the circle, and the coarse-grid target must track the
same estimator evaluated on a much finer grid.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from shared.utils import _compute_empirical_density_asymmetry_core, silverman_bandwidth

CIRC = 360.0
DX = 2.0                      # model-degree bias cell width at circ_space=360
GRID = np.arange(2.0, 181.0, 2.0)


def _sampled_mass(bw, offset, period=CIRC, dx=DX):
    """Rectangle-rule mass of one wrapped-Gaussian kernel. Analytic truth is 1.0.

    Deliberately written from the definition rather than by calling the estimator,
    so it cannot inherit the estimator's error.
    """
    grid = -period / 2 + dx * np.arange(int(period / dx))
    d = grid - offset
    k = sum(np.exp(-0.5 * ((d + i * period) / bw) ** 2) for i in (-2, -1, 0, 1, 2))
    return float((k / (bw * np.sqrt(2 * np.pi))).sum() * dx)


@pytest.mark.parametrize("ratio,tol", [(0.5, 0.02), (0.75, 1e-3), (1.0, 1e-6)])
def test_mass_is_conserved_at_and_above_the_floor(ratio, tol):
    """At bw >= dx/2 the sampled kernel carries its analytic mass, for any offset."""
    for offset in (0.0, DX / 4, DX / 2):
        assert _sampled_mass(ratio * DX, offset) == pytest.approx(1.0, abs=tol)


def test_mass_collapses_below_the_floor():
    """Why the floor exists: at bw = dx/20 a trial contributes ~8x its mass or none,
    purely according to where in the cell it sits."""
    centred = _sampled_mass(DX / 20, 0.0)
    half_cell = _sampled_mass(DX / 20, DX / 2)
    assert centred > 5.0, centred
    assert half_cell < 0.01, half_cell


def _target(fd, bias, kernel_bw):
    return np.asarray(_compute_empirical_density_asymmetry_core(
        jnp.asarray(fd), jnp.asarray(bias), jnp.asarray(GRID),
        weights_sd=20.0, circ_space=CIRC, kernel_bw=kernel_bw)[1])


def _sample(rng, n=800, sd=6.0):
    fd = rng.uniform(2, 180, n)
    bias = rng.normal(6 * np.sin(np.pi * fd / 180), sd, n)
    return fd, ((bias + 180) % 360) - 180


def _fine_reference(fd, bias, bw, refine=40):
    """The same estimator on a 40x finer bias grid, written from the definition.

    The coarse grid's discretization error is the difference. This does not call
    the implementation under test, so it cannot inherit its error.
    """
    n = 180 * refine
    dx = CIRC / n
    br = -CIRC / 2 + dx * np.arange(n)
    w = np.exp(-0.5 * ((GRID[:, None] - fd[None, :]) / 20.0) ** 2)
    w /= w.sum(axis=1, keepdims=True)
    d = br[:, None] - bias[None, :]
    k = sum(np.exp(-0.5 * ((d + i * CIRC) / bw) ** 2) for i in (-1, 0, 1))
    dens = w @ (k / (bw * np.sqrt(2 * np.pi))).T
    pos = br > 0
    neg = (br < 0) & (br > -CIRC / 2)
    return (dens[:, pos].sum(axis=1) - dens[:, neg].sum(axis=1)) * dx


def test_sub_cell_bandwidth_is_rescaled_not_honoured_naively():
    """A sub-cell request must not be evaluated as-is on the coarse grid.

    It should land near the same estimator at a resolution that can represent it,
    NOT near the aliased coarse evaluation.
    """
    fd, bias = _sample(np.random.default_rng(0))
    got = _target(fd, bias, 0.09)
    supported = _fine_reference(fd, bias, 0.09)
    assert np.max(np.abs(got - supported)) < 0.10 * np.ptp(supported)


def test_scaling_preserves_the_estimand():
    """Scaling is sign-preserving, so the signed mass difference must not move.

    Feeding pre-scaled data with a correspondingly scaled bandwidth has to give the
    same curve as letting the estimator do the scaling itself.
    """
    fd, bias = _sample(np.random.default_rng(4), sd=0.4)   # narrow enough to trigger it
    direct = _target(fd, bias, None)
    prescaled = _target(fd, bias * 8.0, None)
    np.testing.assert_allclose(direct, prescaled, atol=0.02 * np.ptp(direct))


def test_wrap_headroom_is_respected():
    """Scaling treats a circular axis as linear, so it must not push mass to the wrap."""
    rng = np.random.default_rng(5)
    fd = rng.uniform(2, 180, 600)
    bias = np.concatenate([rng.normal(0, 0.3, 570), rng.normal(179.0, 0.5, 30)])
    rng.shuffle(bias)
    out = _target(fd, ((bias + 180) % 360) - 180, None)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-6, "mass difference cannot exceed total mass"


def test_coarse_target_tracks_a_finer_grid_at_the_floor():
    """With the floor in force the 2-degree grid still represents the density.

    Residual is ~a few % and does NOT go to zero with bandwidth: the signed sum is
    over half-lines, whose endpoints break the spectral accuracy the rectangle rule
    enjoys on a full periodic integral. Measured on real moors cells: max error
    against this reference is 18.8% of curve range with the floor, 278% without.
    """
    fd, bias = _sample(np.random.default_rng(2))
    got = _target(fd, bias, DX / 2)
    ref = _fine_reference(fd, bias, DX / 2)
    assert np.max(np.abs(got - ref)) < 0.05 * np.ptp(ref)
