"""Flag-dispatch invariants for the surface-generation EM fit.

`jax_generate_and_fit` used to declare `algorithm`, `fix_weights`, and
`diagonal_covariance` but ignore them (it always ran a diagonal, circular,
free-but-floored-weights EM). That silently produced `fullcov`-labeled artifacts that were
really diagonal and made `fix_weights=True` a no-op. These tests pin that the returned
weights / covariance actually reflect the requested mode.
"""
import sys
from pathlib import Path

import pytest

SC = Path(__file__).resolve().parents[1] / "surface_computation"
if str(SC) not in sys.path:
    sys.path.insert(0, str(SC))

import jax
import jax.numpy as jnp

import jax_fit_functions as jf
from jax_fit_functions import ResCol

# Two well-separated clusters (feat_diff = spat_diff = 80°, sd = 10°) so EM recovers
# the mixture weights cleanly and any weight movement is data-driven, not noise.
_MU1 = jnp.array([-40.0, 40.0])
_MU2 = jnp.array([-40.0, 40.0])
_SD1 = jnp.array([10.0, 10.0])
_SD2 = jnp.array([10.0, 10.0])


def _fit(weights_gen, fix_weights, diagonal_covariance=True, algorithm="EM",
         seed=0, n_samples=500):
    key = jax.random.PRNGKey(seed)
    return jf.jax_generate_and_fit(
        key, _MU1, _MU2, _SD1, _SD2,
        weights=weights_gen, n_samples=n_samples,
        algorithm=algorithm, fix_weights=fix_weights,
        diagonal_covariance=diagonal_covariance,
    )


def test_fix_weights_holds_equal():
    """fix_weights=True must pin the fitted mixture weights at 0.5/0.5 on 80/20 data."""
    res = _fit(weights_gen=0.8, fix_weights=True)
    w = res[:, ResCol.weight]
    assert jnp.allclose(w, 0.5, atol=1e-6), f"fixed weights should be 0.5/0.5, got {w}"


def test_free_weights_track_data():
    """fix_weights=False must let the weights follow the ~80/20 data away from 0.5."""
    res = _fit(weights_gen=0.8, fix_weights=False)
    w = res[:, ResCol.weight]
    assert float(jnp.max(w)) > 0.65, f"free weights should track 80/20 data, got {w}"


def test_diagonal_reports_zero_correlation():
    """A diagonal fit must report r_est == 0 (no cross-dimension correlation)."""
    res = _fit(weights_gen=0.5, fix_weights=False, diagonal_covariance=True)
    r = res[:, ResCol.r_est]
    assert jnp.allclose(r, 0.0), f"diagonal fit must report r_est=0, got {r}"


def test_full_covariance_not_silently_diagonal():
    """Requesting full covariance must not silently return a diagonal fit."""
    with pytest.raises(NotImplementedError):
        _fit(weights_gen=0.5, fix_weights=False, diagonal_covariance=False, n_samples=200)


def test_unimplemented_algorithm_raises():
    """A non-EM algorithm must fail loudly rather than silently running EM."""
    with pytest.raises(NotImplementedError):
        _fit(weights_gen=0.5, fix_weights=False, algorithm="VBEM", n_samples=200)
