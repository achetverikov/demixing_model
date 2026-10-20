"""Compiled integration: linear EM agrees with Rmixmod and GaussianMixtures.jl.

Common-start one-step and converged fits catch weighted-moment, SD/variance,
accidental wrapping and stopping errors independently of JAX's own equations.
Two seeded fixtures exercise separated and overlapping soft assignments; the
production SD/weight bounds remain inactive throughout the compared fits.
"""
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from surface_computation import likelihood_em as em

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]
REFERENCES = Path(__file__).parent / 'references'


def _fixtures():
    rng = np.random.default_rng(312)
    for name, means in [('separated', [[350., -25.], [420., 25.]]),
                        ('overlap', [[360., -8.], [385., 10.]])]:
        scales = np.array([[12., 12.], [18., 14.]])
        means = np.array(means)
        x = np.concatenate([rng.normal(m, s, (n, 2))
                            for m, s, n in zip(means, scales, [72, 48])])
        initial = np.column_stack(([.55, .45], means + [[-5., -4.], [5., 4.]], scales*1.3))
        yield name, x, initial


def _ordered(theta):
    return theta[np.argsort(theta[:, 2])]


def _likelihood(x, theta):
    terms = norm.logpdf(x[:, None, :], theta[None, :, 1:3], theta[None, :, 3:5]).sum(-1)
    return logsumexp(terms + np.log(theta[:, 0]), axis=1).sum()


@pytest.fixture(scope='module')
def reference_fits(tmp_path_factory):
    folder = tmp_path_factory.mktemp('linear_gmm')
    fixtures = dict((name, (x, theta)) for name, x, theta in _fixtures())
    for name, (x, theta) in fixtures.items():
        np.savetxt(folder / f'{name}_data.csv', x, delimiter=',')
        np.savetxt(folder / f'{name}_initial.csv', theta, delimiter=',')
    julia = shutil.which('julia') or str(Path.home() / '.local/bin/julia')
    environment = dict(os.environ, JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    environment.setdefault('JULIA_DEPOT_PATH', '/tmp/demixing_julia_depot')
    commands = [
        ['Rscript', str(REFERENCES / 'linear_gmm.R'), str(folder), *fixtures],
        [julia, '--startup-file=no', f'--project={REFERENCES / "julia"}',
         str(REFERENCES / 'linear_gmm.jl'), str(folder), *fixtures],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Warning' not in result.stderr and 'WARNING' not in result.stderr, result.stderr
    return folder, fixtures


@pytest.mark.parametrize('name', ['separated', 'overlap'])
def test_linear_em_matches_r_and_julia(reference_fits, name):
    folder, fixtures = reference_fits
    x, initial = fixtures[name]
    theta = tuple(jnp.asarray(v) for v in initial.T)
    xx = jnp.asarray(x)
    basis = em.prepare(xx, (False, False))
    state = em.initialize(xx, theta, basis, (False, False))
    advance = jax.jit(partial(em.advance, wrap_axes=(False, False)))
    one = advance(xx, state, basis, 1)
    fitted = advance(xx, state, basis, 5000)
    assert bool(em.converged(fitted))
    for stage, actual in [('one', one), ('converged', fitted)]:
        actual_theta = _ordered(np.stack(actual.theta, axis=1))
        for language in ['r', 'julia']:
            result = np.loadtxt(folder / f'{name}_{language}_{stage}.csv', delimiter=',')
            expected = _ordered(result[:, :5])
            # All compared fits are unconstrained interior solutions.
            assert expected[:, 0].min() > .11
            assert expected[:, 3:].min() > 2.
            # One update tests float32 arithmetic. At convergence, production's
            # per-step stopping tolerance allows a small residual parameter tail.
            # Scale means by SD so translating the data cannot weaken the check.
            scale = np.column_stack((np.ones(2), expected[:, 3:5], expected[:, 3:5]))
            np.testing.assert_allclose((actual_theta-expected)/scale, 0, rtol=0,
                                       atol=2e-5 if stage=='one' else 2e-4)
            np.testing.assert_allclose(actual_theta[:, 0], expected[:, 0], rtol=0,
                                       atol=1e-6 if stage=='one' else 1e-4)
            np.testing.assert_allclose(result[:, 5], _likelihood(x, expected), rtol=0, atol=1e-8)
            np.testing.assert_allclose(actual.ll, result[0, 5], rtol=0, atol=1e-3)
            if stage == 'converged':
                extra = np.loadtxt(folder / f'{name}_{language}_extra.csv', delimiter=',')
                np.testing.assert_allclose(_ordered(extra[:, :5]), expected, atol=2e-5, rtol=0)
                np.testing.assert_allclose(extra[:, 5], result[:, 5], atol=1e-8, rtol=0)
    if name == 'overlap':
        logp = norm.logpdf(x[:, None, :], initial[None, :, 1:3], initial[None, :, 3:5]).sum(-1)
        logp += np.log(initial[:, 0])
        posterior = np.exp(logp-logsumexp(logp, axis=1)[:, None])
        assert np.count_nonzero((posterior[:, 0] > .2) & (posterior[:, 0] < .8)) >= 10
