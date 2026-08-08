"""Equivariance tests for the periodic mu1 decoder operations.

Each invariant is also exercised against the edge-based operation it replaces.
That A/B check keeps the test from passing vacuously on a seam-free fixture.

Comparison tolerance
--------------------
The equivariances below are exact in real arithmetic but not in float32. Rolling
the input changes the order in which XLA accumulates each convolution sum, and
floating-point addition is not associative, so the last bits can differ. Measured
on these fixtures: three of the four decoder geometries agree bitwise, while
``rows=16`` differs in 35 of 1792 elements by at most 5.96e-08 -- below one ULP
(float32 spacing at that magnitude is 1.19e-07). Asserting exact equality
therefore made the test depend on tiling that happens to line up, and it failed
on that one geometry for no scientific reason.

`SEAM_TOLERANCE` is the tolerance the plan left open as a coding-time decision
(CIRCULARITY_FIX_PLAN_temp.md Phase B: "the comparison tolerance still ha[s] to
be written down"). It is set at 1e-6, three orders of magnitude below the defect
a genuine seam error produces: the faulty controls in this file assert their
edge-padded counterparts break equivariance by more than 1e-3, because a padding
error changes seam rows by O(1), not by an ULP. So the loosening cannot hide the
class of bug these tests exist to catch, and
`test_the_tolerance_separates_rounding_from_seam_errors` pins that gap.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

#: Absolute and relative tolerance for the equivariance comparisons. See the
#: module docstring: rounding lands at ~6e-08, a real seam error at >1e-03.
SEAM_TOLERANCE = 1e-6

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neural_network_optimization"))

from mirror_aware_model import (CircularMu1ConvTranspose, MirrorAwareMu1Predictor,
                                periodic_linear_resize_mu1)
from shared.mu1_axis import legacy_mu1_axis


@pytest.mark.parametrize(
    "rows,kernel,stride",
    [(8, (4, 4), (2, 2)),
     (16, (4, 4), (2, 2)),
     (32, (4, 4), (2, 2)),
     (64, (3, 3), (1, 1))],
)
def test_each_decoder_layer_is_circular_shift_equivariant(rows, kernel, stride):
    """A roll of k input rows becomes a roll of stride*k output rows."""
    x = jax.random.normal(jax.random.PRNGKey(rows), (1, rows, 7, 3))
    layer = CircularMu1ConvTranspose(
        features=4, kernel_size=kernel, strides=stride, use_bias=False)
    variables = layer.init(jax.random.PRNGKey(rows + 1), x)

    output = layer.apply(variables, x)
    shifted = layer.apply(variables, jnp.roll(x, 1, axis=1))
    np.testing.assert_allclose(
        np.asarray(shifted), np.asarray(jnp.roll(output, stride[0], axis=1)),
        rtol=SEAM_TOLERANCE, atol=SEAM_TOLERANCE)

    # Explicit faulty control: SAME padding treats the two angular edges as
    # boundaries, so it cannot commute with a roll across the seam.
    faulty = nn.ConvTranspose(
        features=4, kernel_size=kernel, strides=stride,
        padding="SAME", use_bias=False)
    faulty_vars = {"params": variables["params"]["ConvTranspose_0"]}
    faulty_output = faulty.apply(faulty_vars, x)
    faulty_shifted = faulty.apply(faulty_vars, jnp.roll(x, 1, axis=1))
    # The crop has the same coordinate alignment as SAME padding: only the two
    # rows that were missing cross-seam contributions are changed.
    np.testing.assert_allclose(
        np.asarray(output[:, 1:-1]), np.asarray(faulty_output[:, 1:-1]),
        rtol=1e-6, atol=1e-6)
    defect = jnp.max(jnp.abs(
        faulty_shifted - jnp.roll(faulty_output, stride[0], axis=1)))
    assert float(defect) > 1e-3


def test_periodic_64_to_180_resize_has_gcd_shift_equivariance():
    """At gcd(64, 180)=4, 16 input rows correspond to 45 output rows."""
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 5, 3))
    output = periodic_linear_resize_mu1(x, 180)
    shifted = periodic_linear_resize_mu1(jnp.roll(x, 16, axis=1), 180)
    np.testing.assert_allclose(
        np.asarray(shifted), np.asarray(jnp.roll(output, 45, axis=1)),
        rtol=SEAM_TOLERANCE, atol=SEAM_TOLERANCE)

    # Explicit faulty control: jax.image.resize is anchored at the two edges.
    faulty_shape = (x.shape[0], 180, x.shape[2], x.shape[3])
    faulty_output = jax.image.resize(x, faulty_shape, method="linear")
    faulty_shifted = jax.image.resize(
        jnp.roll(x, 16, axis=1), faulty_shape, method="linear")
    defect = jnp.max(jnp.abs(
        faulty_shifted - jnp.roll(faulty_output, 45, axis=1)))
    assert float(defect) > 1e-3


def test_periodic_resize_wraps_the_interpolation_stencil():
    x = jnp.arange(8, dtype=jnp.float32).reshape(1, 8, 1, 1)
    output = periodic_linear_resize_mu1(x, 16)

    # The final output lies halfway between rows 7 and 0, not at an edge-clamped
    # extrapolation. This directly checks the stencil used at the seam.
    assert float(output[0, -1, 0, 0]) == pytest.approx(3.5)


def test_model_keeps_legacy_parameter_layout_and_forward_path():
    """The architecture fix must not make old checkpoint trees unloadable."""
    model = MirrorAwareMu1Predictor()
    x = jnp.array([[30.0, 40.0, 50.0]])
    variables = model.init(jax.random.PRNGKey(7), x)

    assert set(variables["params"]) == {
        "Dense_0", "Dense_1", "Dense_2", "Dense_3",
        "ConvTranspose_0", "ConvTranspose_1",
        "ConvTranspose_2", "ConvTranspose_3",
    }
    assert model.apply(variables, x).shape == (1, 180, 90)
    with legacy_mu1_axis():
        assert model.apply(variables, x).shape == (1, 181, 90)


def test_model_supports_128_native_mu1_rows():
    model = MirrorAwareMu1Predictor(native_mu1_rows=128)
    x = jnp.array([[20.0, 20.0, 40.0]])
    variables = model.init(jax.random.PRNGKey(8), x)

    assert variables['params']['Dense_3']['kernel'].shape[-1] == 4096
    assert model.apply(variables, x).shape == (1, 180, 90)


def test_model_can_train_at_native_feature_resolution():
    model = MirrorAwareMu1Predictor(output_feat_cols=128)
    x = jnp.array([[20.0, 20.0, 40.0]])
    variables = model.init(jax.random.PRNGKey(9), x)

    assert model.apply(variables, x).shape == (1, 180, 128)


def test_the_tolerance_separates_rounding_from_seam_errors():
    """SEAM_TOLERANCE must sit far below the defect a real seam error causes.

    Without this, loosening the equivariance assertions above could drift until
    they no longer catch what they exist for. The gap is measured, not assumed:
    the edge-padded control's equivariance defect against the circular layer's
    rounding noise, on the geometry where that noise is largest.
    """
    rows, kernel, stride = 16, (4, 4), (2, 2)
    x = jax.random.normal(jax.random.PRNGKey(rows), (1, rows, 7, 3))
    layer = CircularMu1ConvTranspose(
        features=4, kernel_size=kernel, strides=stride, use_bias=False)
    variables = layer.init(jax.random.PRNGKey(rows + 1), x)

    rounding = float(jnp.max(jnp.abs(
        layer.apply(variables, jnp.roll(x, 1, axis=1))
        - jnp.roll(layer.apply(variables, x), stride[0], axis=1))))

    faulty = nn.ConvTranspose(features=4, kernel_size=kernel, strides=stride,
                              padding="SAME", use_bias=False)
    faulty_vars = {"params": variables["params"]["ConvTranspose_0"]}
    seam_error = float(jnp.max(jnp.abs(
        faulty.apply(faulty_vars, jnp.roll(x, 1, axis=1))
        - jnp.roll(faulty.apply(faulty_vars, x), stride[0], axis=1))))

    assert rounding < SEAM_TOLERANCE < seam_error
    assert seam_error / max(rounding, np.spacing(np.float32(1.0))) > 1e3, (
        "a real seam error must be orders of magnitude larger than rounding, "
        "or the tolerance is not separating the two")
