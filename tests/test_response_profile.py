"""Unit constrained updates and fitting contract: fixed means, ordered cues, bounded SDs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from surface_computation import likelihood_em as em, response_profile as profile

pytestmark = pytest.mark.gpu


def test_fixed_mean_variance_and_cue_pooling():
    theta = tuple(jnp.array(v) for v in ([.5, .5], [-1., 1.], [-.5, .5], [1., 1.], [1., 1.]))
    count = jnp.array([3., 1.])
    # Nonzero response shifts must not be subtracted from fixed-mean variance.
    moments = tuple(jnp.array(v) for v in ([1., 2.], [.01, 16.], [2., -2.], [4., 4.]))
    updated = profile.update(theta, count, moments, 4, (False, False), False,
                             identifying_sd=1., sd_bounds=jnp.array([.2, 3.]))
    np.testing.assert_array_equal(updated[1], theta[1])
    np.testing.assert_allclose(updated[2], [.75, .75])
    np.testing.assert_allclose(updated[3], [.2, 3.])
    np.testing.assert_array_equal(updated[4], [1., 1.])


def test_profile_matches_independent_constrained_optimization():
    # Small deterministic overlapping fixture catches fixed-mean drift and label swaps.
    with jax.enable_x64():
        x = jnp.array([[-1.2, -.8], [.5, -.6], [1.7, -.4], [-.7, .2], [.2, .5], [1., 1.1]])
        grid = jnp.array([[-.5, .5], [.5, -.5]])
        bounds = jnp.array([.2, 3.])
        fitted = jax.vmap(lambda means: profile.fit(x, means, 1., bounds, first_cap=2))(grid)
        pi, means, identifying, sd, identifying_sd, ll, diagnostics = jax.device_get(fitted)
        np.testing.assert_array_equal(means, grid)
        np.testing.assert_array_equal(identifying_sd, np.ones((2, 2)))
        assert np.all(identifying[:, 0] <= identifying[:, 1])
        assert np.all((sd >= .2) & (sd <= 3.))
        assert diagnostics['converged'].all()
        for i, pair in enumerate(np.asarray(grid)):
            def objective(par):
                z = par[0] + np.array([-.5, .5]) * par[1]
                logp = (norm.logpdf(np.asarray(x)[:, :1], pair, np.exp(par[3:]))
                        + norm.logpdf(np.asarray(x)[:, 1:], z, 1.)
                        + np.log([par[2], 1 - par[2]]))
                return -logsumexp(logp, axis=1).sum()
            best = [identifying[i].mean(), np.diff(identifying[i])[0], pi[i, 0], *np.log(sd[i])]
            assert abs(objective(best) + ll[i]) < 1e-8
            starts = [best] + [[0., gap, w, np.log(s1), np.log(s2)]
                              for gap in (.1, 2.) for w in (.25, .75)
                              for s1, s2 in ((.2, 3.), (3., .2), (1., 1.))]
            reference = min(minimize(objective, start, method='L-BFGS-B',
                bounds=[(None, None), (0., None), (1e-8, 1-1e-8), (np.log(.2), np.log(3.)),
                        (np.log(.2), np.log(3.))], options={'ftol': 1e-12}).fun for start in starts)
            assert abs(ll[i] + reference) < 1e-5


def test_free_response_fit_labels_by_identifying_means():
    with jax.enable_x64():
        x = jnp.array([[1.8, -2.], [2., -2.], [2.2, -1.8],
                       [-2., 2.], [-1.8, 2.], [-2.2, 1.8]])
        fitted = profile.fit_mle(jax.random.key(32), x, .4, jnp.array([.1, 3.]), n_init=8)
        pi, means, identifying, sd, _, ll, diagnostics = jax.device_get(fitted)
        assert diagnostics['converged']
        np.testing.assert_allclose(means, [2., -2.], atol=1e-5)
        assert identifying[0] < identifying[1]
        np.testing.assert_allclose(sd, np.sqrt(.08 / 3), atol=1e-5)
        expected = logsumexp(norm.logpdf(np.asarray(x)[:, :1], means, sd)
                            + norm.logpdf(np.asarray(x)[:, 1:], identifying, .4)
                            + np.log(pi), axis=1).sum()
        assert abs(expected - ll) < 1e-8
        np.testing.assert_array_equal(diagnostics['parameters'], np.stack(fitted[:5]))
