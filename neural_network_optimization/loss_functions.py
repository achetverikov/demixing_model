"""Loss functions for neural network surface optimization."""

import jax
import jax.numpy as jnp
from shared.mu1_axis import assert_mu1_axis, mu1_grid, sign_masks


def _discrete_log_probabilities(logits):
    """Normalize each mu1 column to a discrete probability distribution."""
    assert_mu1_axis(logits.shape[1], name="NN loss input")
    log_prob = jax.nn.log_softmax(logits, axis=1)
    return log_prob, jnp.exp(log_prob)


def discrete_kl_divergence_loss(pred_logits, target_logits):
    """Forward KL(target || prediction), averaged over batch and feat_diff."""
    pred_log_prob, _ = _discrete_log_probabilities(pred_logits)
    target_log_prob, target_prob = _discrete_log_probabilities(target_logits)
    return jnp.mean(jnp.sum(
        target_prob * (target_log_prob - pred_log_prob), axis=1))


def kl_divergence_loss(pred_log_density, target_log_density):
    """Compute KL divergence from target to predicted log-probability density.

    Sums the KL integrand over the mu1_error axis and averages over batch and
    feat_diff dimensions.  The inputs are continuous densities (1/dx scale), so
    this sum equals the discrete-bin KL divided by dx — a constant factor, but
    one that inflates this term relative to the fixed-weight MSE/smoothness
    terms it is combined with in combined_probabilistic_loss (see
    MODEL_PIPELINE_FOR_AGENTS.md S5.3b).

    Args:
        pred_log_density: Predicted log-density with shape (batch, mu1_error, feat_diff).
        target_log_density: Target log-density of the same shape.

    Returns:
        Scalar KL divergence loss.
    """
    # KL divergence: sum over mu1_error (axis=1), mean over batch and feat_diff
    kl_integrand = jnp.exp(target_log_density) * (target_log_density - pred_log_density)
    kl_div = jnp.sum(kl_integrand, axis=1)  # Sum over mu1_error dimension (axis=1)

    return jnp.mean(kl_div)  # Mean over batch and feat_diff


def expectation_loss(pred_logits, target_logits):
    """Match first circular-moment vectors, continuously across the seam.

    Comparing the sine/cosine vectors avoids both the ±180 discontinuity of a
    linear mean and the undefined angle of a nearly uniform distribution.  The
    squared vector distance is divided by its maximum, four, so the loss lies
    in [0, 1].
    """
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    angles = jnp.radians(mu1_grid())[None, :, None]
    prob_delta = pred_prob - target_prob
    delta_cos = jnp.sum(prob_delta * jnp.cos(angles), axis=1)
    delta_sin = jnp.sum(prob_delta * jnp.sin(angles), axis=1)
    return jnp.mean(delta_cos ** 2 + delta_sin ** 2) / 4.0


def density_asymmetry_loss(pred_logits, target_logits):
    """Match the signed probability-mass functional used downstream."""
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    positive, negative = sign_masks()
    signs = positive.astype(pred_prob.dtype) - negative.astype(pred_prob.dtype)
    delta = jnp.sum(
        (pred_prob - target_prob) * signs[None, :, None], axis=1)
    return jnp.mean(delta ** 2) / 4.0


def circular_moment_trajectory_loss(pred_logits, target_logits):
    """Match feature-axis curvature of the circular-moment trajectory."""
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    angles = jnp.radians(mu1_grid())[None, :, None]

    def moments(prob):
        return jnp.stack([
            jnp.sum(prob * jnp.cos(angles), axis=1),
            jnp.sum(prob * jnp.sin(angles), axis=1),
        ], axis=1)

    error = moments(pred_prob) - moments(target_prob)
    curvature = error[:, :, 2:] - 2 * error[:, :, 1:-1] + error[:, :, :-2]
    return jnp.mean(curvature ** 2)


def feature_probability_gradient_loss(pred_logits, target_logits):
    """Match adjacent-column probability changes across feature difference."""
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    gradient_error = (jnp.diff(pred_prob, axis=2) -
                      jnp.diff(target_prob, axis=2))
    return jnp.mean(jnp.sum(gradient_error ** 2, axis=1))


def circular_energy_loss(pred_logits, target_logits):
    """Circular CRPS/energy divergence between target and prediction.

    Geodesic distance makes nearby mass across the seam nearby in the loss.
    Division by 360 bounds the divergence by one.  This score is proper but not
    strictly proper on the circle, so it complements rather than replaces the
    strictly identifying forward-KL term.
    """
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    grid = mu1_grid()
    difference = jnp.abs(grid[:, None] - grid[None, :])
    distance = jnp.minimum(difference, 360.0 - difference)
    cross = jnp.einsum("bnf,nm,bmf->bf", pred_prob, distance, target_prob)
    pred_self = jnp.einsum("bnf,nm,bmf->bf", pred_prob, distance, pred_prob)
    target_self = jnp.einsum("bnf,nm,bmf->bf", target_prob, distance, target_prob)
    return jnp.mean(2 * cross - pred_self - target_self) / 360.0


def hellinger_loss(pred_logits, target_logits):
    """Squared Hellinger distance, averaged over surfaces and feat_diff."""
    _, pred_prob = _discrete_log_probabilities(pred_logits)
    _, target_prob = _discrete_log_probabilities(target_logits)
    affinity = jnp.sum(jnp.sqrt(pred_prob * target_prob), axis=1)
    return jnp.mean(1.0 - affinity)


def probability_curvature_regularization(pred_logits):
    """Penalize grid-scale probability wiggles without flattening broad slopes.

    The mu1 second difference wraps around the circular seam; feat_diff remains
    bounded.  Dividing by eight bounds the sum of the two mean-square terms by
    one.  This is deliberately a prediction-only regularizer: its purpose in an
    ablation is to suppress Monte Carlo noise in a target, not to reproduce the
    target's noisy derivatives.
    """
    _, prob = _discrete_log_probabilities(pred_logits)
    curvature_mu1 = (jnp.roll(prob, -1, axis=1) - 2 * prob +
                     jnp.roll(prob, 1, axis=1))
    curvature_feat = prob[:, :, 2:] - 2 * prob[:, :, 1:-1] + prob[:, :, :-2]
    return (jnp.mean(curvature_mu1 ** 2) +
            jnp.mean(curvature_feat ** 2)) / 8.0


def smoothness_regularization(log_density):
    """Penalize rapid changes in the log-density surface along both spatial axes.

    Args:
        log_density: Log-density array with shape (batch, mu1_error, feat_diff).

    Returns:
        Scalar regularization loss (mean squared finite differences along both axes).
    """
    # Gradients along both spatial dimensions.  mu1_error is circular, so the
    # wrap-around step (last row -> first row) is a real adjacent-row step and
    # gets the same weight as any other; jnp.diff would give it zero weight.
    # feat_diff stays non-periodic — it is a bounded interval, not a circle.
    grad_mu1 = log_density - jnp.roll(log_density, 1, axis=1)  # Along mu1_error
    grad_feat = jnp.diff(log_density, axis=2)  # Along feat_diff

    return jnp.mean(grad_mu1 ** 2) + jnp.mean(grad_feat ** 2)


def surface_gradient_matching_loss(pred_log_density, target_log_density):
    """Match finite-difference gradients within each surface (mu1_bias × feat_diff axes).

    Penalises spurious local gradients (noise) in the predicted surface that are
    absent from the smooth KDE target.

    Args:
        pred_log_density:   (batch, mu1_bias, feat_diff)
        target_log_density: (batch, mu1_bias, feat_diff)
    """
    pred_g1 = jnp.diff(pred_log_density, axis=1)
    tgt_g1  = jnp.diff(target_log_density, axis=1)
    pred_g2 = jnp.diff(pred_log_density, axis=2)
    tgt_g2  = jnp.diff(target_log_density, axis=2)
    return (jnp.mean((pred_g1 - tgt_g1) ** 2) +
            jnp.mean((pred_g2 - tgt_g2) ** 2))


def combined_probabilistic_loss(pred_log_probs, target_log_probs,
                                kl_weight=1.0, energy_weight=1.0,
                                expectation_weight=1.0,
                                asymmetry_weight=1.0,
                                hellinger_weight=0.0,
                                log_smoothness_weight=0.0,
                                curvature_weight=0.0,
                                trajectory_weight=0.0,
                                feature_gradient_weight=0.0):
    """Distributional circular loss with downstream-functional auxiliaries.

    Args:
        pred_log_probs: Predicted log-probabilities with shape (batch, mu1_error, feat_diff).
        target_log_probs: Target log-probabilities of the same shape.
        kl_weight: Weight for forward KL(target || prediction).
        energy_weight: Weight for normalized circular energy divergence.
        expectation_weight: Weight for normalized circular-moment discrepancy.
        asymmetry_weight: Weight for normalized density-asymmetry discrepancy.
        hellinger_weight: Weight for squared Hellinger distance.
        log_smoothness_weight: Weight for the legacy first-difference penalty.
        curvature_weight: Weight for probability-space second differences.
        trajectory_weight: Weight for circular-moment curvature matching.
        feature_gradient_weight: Weight for probability-gradient matching along
            feature difference.

    Returns:
        Tuple of (total_loss, breakdown). The breakdown contains ``total`` and
        only terms whose profile weight is nonzero, so inactive diagnostics do
        not consume training time.
    """
    total_loss = jnp.zeros((), dtype=pred_log_probs.dtype)
    components = {}

    # Profile weights are fixed for a complete run, so these Python branches
    # resolve before JIT tracing and inactive objectives add no GPU work.
    if kl_weight:
        components['kl'] = discrete_kl_divergence_loss(
            pred_log_probs, target_log_probs)
        total_loss += kl_weight * components['kl']
    if energy_weight:
        components['energy'] = circular_energy_loss(
            pred_log_probs, target_log_probs)
        total_loss += energy_weight * components['energy']
    if expectation_weight:
        components['expectation'] = expectation_loss(
            pred_log_probs, target_log_probs)
        total_loss += expectation_weight * components['expectation']
    if asymmetry_weight:
        components['asymmetry'] = density_asymmetry_loss(
            pred_log_probs, target_log_probs)
        total_loss += asymmetry_weight * components['asymmetry']
    if hellinger_weight:
        components['hellinger'] = hellinger_loss(pred_log_probs, target_log_probs)
        total_loss += hellinger_weight * components['hellinger']
    if log_smoothness_weight:
        components['log_smoothness'] = smoothness_regularization(pred_log_probs)
        total_loss += log_smoothness_weight * components['log_smoothness']
    if curvature_weight:
        components['curvature'] = probability_curvature_regularization(pred_log_probs)
        total_loss += curvature_weight * components['curvature']
    if trajectory_weight:
        components['trajectory'] = circular_moment_trajectory_loss(
            pred_log_probs, target_log_probs)
        total_loss += trajectory_weight * components['trajectory']
    if feature_gradient_weight:
        components['feature_gradient'] = feature_probability_gradient_loss(
            pred_log_probs, target_log_probs)
        total_loss += feature_gradient_weight * components['feature_gradient']

    components['total'] = total_loss
    return total_loss, components


LOSS_PROFILES = {
    'kl': dict(kl_weight=1.0, energy_weight=0.0,
               expectation_weight=0.0, asymmetry_weight=0.0),
    'kl_energy': dict(kl_weight=1.0, energy_weight=1.0,
                      expectation_weight=0.0, asymmetry_weight=0.0),
    'circular': dict(kl_weight=1.0, energy_weight=1.0,
                     expectation_weight=1.0, asymmetry_weight=1.0),
    'circular_hellinger': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, hellinger_weight=1.0),
    'circular_log_smooth': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, log_smoothness_weight=0.1),
    'circular_curvature_1k': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, curvature_weight=1e3),
    'circular_curvature_10k': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, curvature_weight=1e4),
    'circular_curvature_100k': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, curvature_weight=1e5),
    'circular_trajectory': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, trajectory_weight=1.0),
    'circular_feature_gradient': dict(
        kl_weight=1.0, energy_weight=1.0, expectation_weight=1.0,
        asymmetry_weight=1.0, feature_gradient_weight=1.0),
}


def loss_profile(name):
    """Return an independent copy of a named objective-ablation profile."""
    try:
        return dict(LOSS_PROFILES[name])
    except KeyError as exc:
        raise ValueError(
            f"Unknown loss profile {name!r}; choose from {sorted(LOSS_PROFILES)}"
        ) from exc
