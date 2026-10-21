"""GPU backend contracts: FFI layouts, cached continuation and full-start guards.

These compare the independent CUDA implementation with the authoritative JAX EM
on small deterministic datasets; they catch stream/buffer, packing, selection
and routing bugs that the JAX-only mathematical tests cannot exercise.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surface_computation import cuda_em, em_batch, likelihood_em as em
from surface_computation import wnm_simulation as simulation

pytestmark = pytest.mark.gpu


def evidence(n):
    t = jnp.linspace(-3., 3., n // 2)
    one = jnp.concatenate((jnp.column_stack((-60. + t, -30. + t)),
                           jnp.column_stack((60. + t, 30. - t))))
    return jnp.stack((one, one + jnp.array([15., 0.]), one * .8))


@pytest.mark.parametrize('n', cuda_em.SAMPLE_COUNTS)
def test_cuda_starts_and_cached_continuation_match_jax(n):
    x = evidence(n)
    keys = jax.random.split(jax.random.key(17), len(x))
    theta = cuda_em.initial_parameters(keys, x, n_init=5)
    expected = jax.vmap(partial(em.initial_parameters, n_init=5))(keys, x)[1]
    np.testing.assert_array_equal(theta, jnp.stack(expected, axis=2))
    # A one-update all-start prefix forces the selected continuation path.
    raw, compact = cuda_em._run(x, theta, first_cap=1, max_iter=1000, occupancy=n <= 100)
    actual = cuda_em.result_from_compact(compact)
    basis, state = em_batch.initialize(keys, x, n_init=5)
    state = em_batch.advance_all(x, basis, state, 1, n_init=5)
    reference = em_batch.finish(x, basis, state, n_init=5, max_iter=1000)
    np.testing.assert_allclose(actual[5], reference[5], atol=.002, rtol=1e-6)
    # Compare item means separately after the intentional IDF ordering.
    def ordered(result):
        p = np.asarray(result[-1]['parameters'])
        return np.take_along_axis(p[:, 1], np.argsort(p[:, 2], axis=1), axis=1)
    np.testing.assert_allclose(ordered(actual), ordered(reference), atol=.02)
    assert raw.shape == (3, 5, 26)
    em_batch.check_results(actual)
    np.testing.assert_array_equal(actual[-1]['converged'], reference[-1]['converged'])


def test_cuda_retains_225_ties_and_nonfinite_guards():
    x = jnp.zeros((3, 4, 2))
    theta = jnp.tile(jnp.array([[.5, .5], [0., 0.], [0., 0.], [1., 1.], [1., 1.]])[None, None],
                     (3, 225, 1, 1))
    _, compact = cuda_em._run(x, theta, first_cap=2, max_iter=2, occupancy=True)
    result = cuda_em.result_from_compact(compact)
    assert (np.asarray(result[-1]['best_tie_count']) == 225).all()
    assert (np.asarray(compact[:, 26]) == 0).all()  # first-index tie selection
    em_batch.check_results(result)
    # The public array boundary can receive bad observations; no starts vanish.
    _, compact = cuda_em._run(x.at[1, 0, 0].set(jnp.nan), theta,
                             first_cap=2, max_iter=2, occupancy=True)
    result = cuda_em.result_from_compact(compact)
    assert int(result[-1]['nonfinite_starts'][1]) == 225
    with pytest.raises(em_batch.EMValidationError, match='225 nonfinite starts'):
        em_batch.check_results(result)


def test_cuda_broad_component_update_matches_jax():
    # Keep both feature SDs on the Fourier branch; narrow separated fixtures
    # alone could miss a broken broad-component moment calculation.
    angle = jnp.linspace(-179., 179., 500)
    x = jnp.column_stack((angle, jnp.where(jnp.arange(500) % 2, -30., 30.)))[None]
    theta = jnp.array([[[[.5, .5], [-100., 80.], [-30., 30.], [80., 140.], [10., 20.]]]])
    raw, _ = cuda_em._run(x, theta, first_cap=1, max_iter=1, occupancy=False)
    initial = tuple(theta[0, 0, k] for k in range(5))
    basis = em.prepare(x[0])
    reference = jax.jit(em.advance)(x[0], em.initialize(x[0], initial, basis), basis, 1)
    np.testing.assert_allclose(raw[0, 0, :10].reshape(5, 2), jnp.stack(reference.theta), atol=.003, rtol=4e-6)
    np.testing.assert_allclose(raw[0, 0, 10], reference.ll, atol=.003, rtol=1e-6)


def test_cuda_refuses_caps_and_unsupported_model_options():
    x = evidence(20)
    keys = jax.random.split(jax.random.key(17), len(x))
    result = em_batch.fit_batch(keys, x, backend='cuda', n_init=5, max_iter=1)
    with pytest.raises(em_batch.EMValidationError, match='iteration cap'):
        em_batch.check_results(result)
    for options in (dict(fix_weights=True), dict(wrap_axes=(False, False)),
                    dict(rtol=.01), dict(stages=(50, 5000))):
        with pytest.raises(ValueError, match='CUDA EM'):
            em_batch.fit_batch(None, None, backend='cuda', **options)


def test_cuda_simulation_uses_native_keys_and_item_readout():
    design = jnp.array([[10., 20., 30., 60.], [20., 10., 30., 100.]])
    key = jax.random.key(31)
    bias, diagnostic = simulation.simulate(key, design, n_simulations=2, n_samples=4,
        block_rows=2, em_backend='cuda', return_diagnostics=True)
    keys = simulation.simulation_keys(key, 2)
    x, fit_keys = simulation.generate_block(keys, design, 2, 4)
    direct = cuda_em.fit_batch(fit_keys, x)
    np.testing.assert_array_equal(bias, simulation.bias_readout(direct, design, 2))
    np.testing.assert_array_equal(diagnostic['parameters'], direct[-1]['parameters'].reshape(2, 2, 5, 2))
