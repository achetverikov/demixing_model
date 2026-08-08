"""Invariants for the circular distributional NN training objective."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neural_network_optimization"))

from loss_functions import (circular_energy_loss, combined_probabilistic_loss,
                            circular_moment_trajectory_loss,
                            density_asymmetry_loss,
                            discrete_kl_divergence_loss, expectation_loss,
                            feature_probability_gradient_loss,
                            hellinger_loss, loss_profile,
                            probability_curvature_regularization)
from shared.mu1_axis import mu1_grid, mu1_size


def _impulse(index, feat_points=3):
    logits = jnp.full((1, mu1_size(), feat_points), -12.0)
    return logits.at[:, index, :].set(12.0)


def test_all_terms_are_zero_at_the_target():
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, mu1_size(), 5))
    total, components = combined_probabilistic_loss(logits, logits)

    assert set(components) == {
        "kl", "energy", "expectation", "asymmetry", "total"}
    assert float(jnp.abs(total)) < 1e-6
    assert float(jnp.abs(sum(components[name] for name in components if name != "total")
                         - components["total"])) < 1e-7


def test_loss_is_invariant_to_arbitrary_column_offsets():
    pred = jax.random.normal(jax.random.PRNGKey(1), (2, mu1_size(), 4))
    target = jax.random.normal(jax.random.PRNGKey(2), (2, mu1_size(), 4))
    pred_offsets = jnp.array([[[2.0, -4.0, 9.0, 1.0]],
                              [[-3.0, 8.0, 2.0, -5.0]]])
    target_offsets = -2 * pred_offsets

    original, _ = combined_probabilistic_loss(pred, target)
    shifted, _ = combined_probabilistic_loss(
        pred + pred_offsets, target + target_offsets)
    assert float(shifted) == pytest.approx(float(original), rel=1e-6)

    # Explicit faulty control: the removed raw-log MSE changes under offsets
    # even though every represented probability distribution is unchanged.
    raw_mse = jnp.mean((pred - target) ** 2)
    shifted_raw_mse = jnp.mean(
        ((pred + pred_offsets) - (target + target_offsets)) ** 2)
    assert not np.isclose(float(raw_mse), float(shifted_raw_mse))


def test_circular_moment_treats_the_seam_as_two_degrees_not_358():
    target = _impulse(mu1_size() - 1)  # +178 degrees
    prediction = _impulse(0)           # -180 degrees
    fixed = float(expectation_loss(prediction, target))

    grid = mu1_grid()[None, :, None]
    pred_prob = jax.nn.softmax(prediction, axis=1)
    target_prob = jax.nn.softmax(target, axis=1)
    faulty_linear = float(jnp.mean(
        jnp.sum((pred_prob - target_prob) * grid, axis=1) ** 2) / 4.0)

    assert fixed < 1e-3
    assert faulty_linear > 30_000


def test_circular_energy_uses_geodesic_seam_distance():
    target = _impulse(mu1_size() - 1)
    prediction = _impulse(0)
    # Energy divergence between point masses is 2*d; d=2 degrees, normalized
    # by 360 degrees.
    assert float(circular_energy_loss(prediction, target)) == pytest.approx(
        4 / 360, rel=1e-5)


def test_asymmetry_directly_detects_mass_crossing_zero():
    target = _impulse(89)      # -2 degrees
    prediction = _impulse(91)  # +2 degrees
    assert float(density_asymmetry_loss(prediction, target)) == pytest.approx(
        1.0, rel=1e-5)


def test_kl_identifies_an_energy_blind_spot():
    angles = jnp.radians(mu1_grid())
    target = jnp.zeros((1, mu1_size(), 3))
    prediction = jnp.log(1.0 + 0.5 * jnp.cos(2 * angles))[None, :, None]
    prediction = jnp.repeat(prediction, 3, axis=2)

    # Geodesic angular energy is proper but not strictly proper: an even
    # harmonic is a theoretical null direction (float32 leaves a tiny residue).
    assert abs(float(circular_energy_loss(prediction, target))) < 2e-4
    assert float(discrete_kl_divergence_loss(prediction, target)) > 0.05


def test_forward_kl_keeps_a_gradient_when_js_would_saturate():
    target = _impulse(0, feat_points=1)
    prediction = _impulse(90, feat_points=1)
    _, kl_gradient = jax.value_and_grad(discrete_kl_divergence_loss)(
        prediction, target)

    def js_loss(logits):
        pred_prob = jax.nn.softmax(logits, axis=1)
        target_prob = jax.nn.softmax(target, axis=1)
        mixture = 0.5 * (pred_prob + target_prob)
        return 0.5 * jnp.mean(jnp.sum(
            target_prob * (jnp.log(target_prob) - jnp.log(mixture)) +
            pred_prob * (jnp.log(pred_prob) - jnp.log(mixture)), axis=1))

    _, js_gradient = jax.value_and_grad(js_loss)(prediction)
    assert float(jnp.linalg.norm(kl_gradient)) > 1.0
    assert float(jnp.linalg.norm(js_gradient)) < 1e-6


def test_hellinger_is_bounded_and_offset_invariant():
    target = _impulse(0, feat_points=2)
    prediction = _impulse(90, feat_points=2)
    distance = float(hellinger_loss(prediction, target))
    shifted = float(hellinger_loss(prediction + 7.0, target - 3.0))
    assert distance == pytest.approx(1.0, abs=2e-5)
    assert shifted == pytest.approx(distance, abs=1e-6)


def test_probability_curvature_wraps_and_is_zero_for_uniform_density():
    uniform = jnp.zeros((1, mu1_size(), 4))
    assert float(probability_curvature_regularization(uniform)) == pytest.approx(0.0)

    peak = _impulse(0, feat_points=4)
    shifted = jnp.roll(peak, 37, axis=1)
    assert float(probability_curvature_regularization(peak)) == pytest.approx(
        float(probability_curvature_regularization(shifted)), rel=1e-6)


def test_cross_column_losses_detect_a_jagged_trajectory():
    target = jnp.zeros((1, mu1_size(), 7))
    prediction = target.at[:, 90, 1::2].set(3.0)

    assert float(circular_moment_trajectory_loss(prediction, target)) > 0
    assert float(feature_probability_gradient_loss(prediction, target)) > 0
    assert float(circular_moment_trajectory_loss(target, target)) == pytest.approx(0)
    assert float(feature_probability_gradient_loss(target, target)) == pytest.approx(0)


def test_loss_profiles_keep_regularizers_explicit():
    assert loss_profile('kl')['energy_weight'] == 0.0
    assert loss_profile('circular_log_smooth')['log_smoothness_weight'] == 0.1
    assert loss_profile('circular_curvature_1k')['curvature_weight'] == 1e3
    assert loss_profile('circular_curvature_10k')['curvature_weight'] == 1e4
    assert loss_profile('circular_curvature_100k')['curvature_weight'] == 1e5
    assert loss_profile('circular_trajectory')['trajectory_weight'] == 1.0
    assert loss_profile('circular_feature_gradient')['feature_gradient_weight'] == 1.0
    with pytest.raises(ValueError, match='Unknown loss profile'):
        loss_profile('not-a-profile')


def test_inactive_profile_terms_are_not_returned_for_logging():
    logits = jnp.zeros((1, mu1_size(), 3))
    _, components = combined_probabilistic_loss(
        logits, logits, **loss_profile('kl'))
    assert set(components) == {'kl', 'total'}
