"""The density objective's three computations must agree to the last few bits.

`density_objective` defines `1 - CCC` twice -- elementwise, and from precomputed
statistics with one matrix product -- because the two search backends reach it by
different routes. The hierarchical path additionally computes it a third time, in
`_compute_curve_losses(loss_type="ccc")`, inside the JIT.

Drift between them is the failure mode this file exists for, and it is invisible
by construction: each form keeps returning plausible losses, and only the ranking
of candidates changes. So the batched form is checked against the reference on
adversarial inputs (near-constant, huge offsets, tiny amplitudes), not just easy
ones -- the batched form's arithmetic differs, and that is exactly where an
algebraically-equal rearrangement stops being numerically equal.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jax.numpy as jnp

import density_objective as do
from grid_based_multi_condition_optimizer_jax_loops import _compute_curve_losses

N_POINTS = 90


def batched(predicted, target):
    """Score via the cache-backed route: precomputed stats + one GEMM."""
    centered_pred, pred_mean = do.centered_curves(predicted)
    centered_target, target_mean, target_var = do.target_statistics(target)
    return do.ccc_loss_batched(
        centered_pred @ centered_target,
        pred_mean,
        jnp.var(jnp.asarray(predicted), axis=1),
        target_mean, target_var, N_POINTS,
    )


def curve_families(rng):
    """Curves that stress the arithmetic, not just typical ones."""
    grid = np.linspace(2, 180, N_POINTS)
    base = 0.05 * np.sin(np.pi * grid / 180.0) - 0.01
    return {
        "typical": base + rng.normal(0, 0.004, N_POINTS),
        "ten_x_too_small": 0.1 * base,
        "hundred_x_too_large": 100 * base,
        "large_offset": base + 50.0,          # mean^2 >> var: cancellation risk
        "tiny_amplitude": 1e-6 * base,
        "near_constant": np.full(N_POINTS, 0.03) + rng.normal(0, 1e-5, N_POINTS),
        "anticorrelated": -base,
        "pure_noise": rng.normal(0, 0.05, N_POINTS),
    }


@pytest.mark.parametrize("pred_name", list(curve_families(np.random.default_rng(0))))
# tiny_amplitude is absent as a TARGET on purpose: its variance is below
# DEGENERATE_TARGET_EPS, so check_targets_fittable refuses it and it can never
# reach the scorer. Comparing the two forms there would be comparing two
# behaviours of the guard, not of the objective.
@pytest.mark.parametrize("target_name", ["typical", "large_offset", "anticorrelated"])
def test_batched_matches_reference(pred_name, target_name):
    rng = np.random.default_rng(4)
    curves = curve_families(rng)
    predicted = curves[pred_name][None, :]
    target = curves[target_name]

    reference = float(do.ccc_loss(predicted, target)[0])
    # float32 throughout (jax x64 is off), so ~1e-7 relative is the floor.
    assert float(batched(predicted, target)[0]) == pytest.approx(reference, rel=1e-5, abs=1e-6)


def test_batched_matches_reference_over_a_whole_slab():
    """The real usage: many candidate curves, one target, one matrix product."""
    rng = np.random.default_rng(7)
    grid = np.linspace(2, 180, N_POINTS)
    target = 0.05 * np.sin(np.pi * grid / 180.0)
    predicted = np.stack([
        amp * np.sin(np.pi * grid / 180.0 + phase) + offset
        for amp in (0.001, 0.05, 5.0)
        for phase in (0.0, 1.0, 3.0)
        for offset in (-2.0, 0.0, 0.02)
    ])
    np.testing.assert_allclose(np.asarray(batched(predicted, target)),
                               np.asarray(do.ccc_loss(predicted, target)),
                               rtol=1e-5, atol=1e-6)


def test_reference_matches_the_hierarchical_jit_path():
    """`_compute_curve_losses` is what the hierarchical backend actually calls;
    if it and `density_objective` disagree, the two backends disagree."""
    rng = np.random.default_rng(11)
    curves = curve_families(rng)
    target = curves["typical"]
    predicted = np.stack(list(curves.values()))
    targets = np.repeat(target[None, :], predicted.shape[0], axis=0)

    jit_path = np.asarray(_compute_curve_losses(
        jnp.asarray(predicted), jnp.asarray(targets), loss_type="ccc", is_angular=False))
    np.testing.assert_allclose(np.asarray(do.ccc_loss(predicted, target)),
                               jit_path, rtol=1e-5, atol=1e-6)


def test_identity_with_mse_over_D_computed_independently():
    """An oracle outside all three implementations: 1 - CCC = MSE/D."""
    rng = np.random.default_rng(13)
    grid = np.linspace(2, 180, N_POINTS)
    target = 0.05 * np.sin(np.pi * grid / 180.0)
    predicted = 0.6 * target + rng.normal(0, 0.005, N_POINTS)

    mse = np.mean((predicted - target) ** 2)
    D = predicted.var() + target.var() + (predicted.mean() - target.mean()) ** 2
    assert float(do.ccc_loss(predicted, target)) == pytest.approx(mse / D, rel=1e-6)


def test_both_sides_must_be_centered_not_just_the_target():
    """Regression: centering only the target is algebraically identical and
    numerically wrong. On a curve with a large mean relative to its variation it
    returned -0.0268 in float32 for a loss whose true value is 0.0 -- negative,
    which 1 - CCC cannot be for identical curves."""
    grid = np.linspace(2, 180, N_POINTS)
    offset_curve = 0.05 * np.sin(np.pi * grid / 180.0) + 50.0
    predicted = offset_curve[None, :]

    centered_target, target_mean, target_var = do.target_statistics(offset_curve)
    target_only = float(do.ccc_loss_batched(
        jnp.asarray(predicted) @ centered_target,
        jnp.mean(jnp.asarray(predicted), axis=1),
        jnp.var(jnp.asarray(predicted), axis=1),
        target_mean, target_var, N_POINTS)[0])
    assert abs(target_only) > 1e-3, "fixture no longer exercises the cancellation"

    assert float(batched(predicted, offset_curve)[0]) == pytest.approx(0.0, abs=1e-6)


def test_ccc_components_factorise_the_loss():
    rng = np.random.default_rng(19)
    grid = np.linspace(2, 180, N_POINTS)
    target = 0.05 * np.sin(np.pi * grid / 180.0)
    for scale in (0.1, 1.0, 4.0):
        predicted = scale * target + rng.normal(0, 0.003, N_POINTS)
        parts = do.ccc_components(predicted, target)
        assert parts['ccc'] == pytest.approx(parts['r'] * parts['C_b'], rel=1e-9)
        assert parts['ccc'] == pytest.approx(1 - float(do.ccc_loss(predicted, target)), rel=1e-6)


def test_the_guard_and_the_refusal_share_one_threshold():
    """A guard set looser than the refusal would leave curves that pass the
    refusal and then have their denominator silently altered."""
    constant = np.full(N_POINTS, 0.03)
    assert np.var(constant) < do.DEGENERATE_TARGET_EPS
    # Anything the refusal admits has D >= var_target >= eps, so the guard is inert.
    admitted = constant + np.linspace(0, 1e-3, N_POINTS)
    assert np.var(admitted) > do.DEGENERATE_TARGET_EPS
    assert float(do.ccc_loss(admitted, admitted)) == pytest.approx(0.0, abs=1e-12)


def test_check_targets_fittable_names_every_offender():
    targets = np.stack([np.linspace(0, 1, N_POINTS),
                        np.full(N_POINTS, 0.2),
                        np.full(N_POINTS, -0.7)])
    with pytest.raises(ValueError) as excinfo:
        do.check_targets_fittable(targets, ["good", "flat_a", "flat_b"], "density")
    message = str(excinfo.value)
    assert "flat_a" in message and "flat_b" in message
    assert "2 of 3 conditions" in message
    do.check_targets_fittable(targets[:1], ["good"], "density")
