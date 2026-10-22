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

from surface_computation import cuda_em, em_batch, likelihood_em as em, structured_starts
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
    expected = np.asarray(jnp.stack(expected, axis=2))
    # Keys, weights and sampled means are exact; eager/JIT scale arithmetic can
    # differ by a few float32 ULPs without changing the initialization law.
    np.testing.assert_array_equal(theta[:, :, :3], expected[:, :, :3])
    np.testing.assert_array_max_ulp(np.asarray(theta[:, :, 3:]), expected[:, :, 3:], maxulp=3)
    # A one-update all-start prefix forces the selected continuation path.
    raw, compact = cuda_em._run(x, theta, first_cap=1, max_iter=1000, occupancy=True)
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
    with pytest.raises(em_batch.EMValidationError, match='1 selected fits failed'):
        em_batch.check_results(result)
    # Backend guard contract: finite initial likelihood, overflowing moments,
    # then a nonfinite first map must stop and be reported rather than iterated.
    huge = theta[:1].at[:, :, 3, :].set(1.e20)
    raw, compact = cuda_em._run(x[:1], huge, first_cap=15, max_iter=15, occupancy=True)
    np.testing.assert_array_equal(raw[:, :, 21], 1.)
    result = cuda_em.result_from_compact(compact)
    assert (np.asarray(result[-1]['nonfinite_starts']) == 225).all()
    with pytest.raises(em_batch.EMValidationError, match='1 selected fits failed'):
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


def test_cuda_squarem_stabilization_preserves_second_iterate_ascent():
    # Backend contract: catch a disabled acceleration route or a proposal accepted
    # before stabilization/without its F^2 safeguard. The JAX suite owns EM math.
    angle = jnp.linspace(-170., 170., 20)
    x = jnp.column_stack((angle, 20*jnp.sin(jnp.arange(20)) + .03*angle))[None]
    theta = jnp.array([[[[.35, .65], [-80., 80.], [-5., 5.], [90., 110.], [20., 25.]],
                         [[.6, .4], [130., -130.], [7., -7.], [100., 80.], [25., 20.]]]])
    # The first cycle keeps F^2 exactly at iteration12; the next cycle can
    # extrapolate from there, with iteration14 as its ordinary fallback.
    second, _ = cuda_em._run(x, theta, first_cap=14, max_iter=14,
                            occupancy=True, accelerate=False)
    ordinary, _ = cuda_em._run(x, theta, first_cap=15, max_iter=15,
                              occupancy=True, accelerate=False)
    accelerated, _ = cuda_em._run(x, theta, first_cap=15, max_iter=15, occupancy=True)
    accelerated = np.asarray(accelerated)
    assert not np.array_equal(accelerated[:, :, :10], ordinary[:, :, :10])
    assert (accelerated[:, :, 10] >= np.asarray(second[:, :, 10])).all()
    assert np.isfinite(accelerated).all()
    p = accelerated[:, :, :10]
    assert (p[:, :, :2] >= .1-1e-7).all() and (p[:, :, :2] <= .9+1e-7).all()
    np.testing.assert_allclose(p[:, :, :2].sum(-1), 1., atol=1e-7)
    assert (p[:, :, 2:4] >= -180.).all() and (p[:, :, 2:4] < 180.).all()
    assert (p[:, :, 6:] >= 1.).all()


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


def test_cuda_simulation_dispatches_complete_n500_evaluators(monkeypatch):
    # Routing contract: mixed blocks and unknown generating SDs must keep the
    # full evaluator. Numerical density and optimizer tests live below this API.
    routes = []
    def fitted(keys, x, **options):
        routes.append(options['cuda_single_image'])
        return ({'converged':jnp.ones(len(x),dtype=bool),
                 'winner_decreases':jnp.zeros(len(x),dtype=int)},)
    monkeypatch.setattr(em_batch, 'fit_batch', fitted)
    monkeypatch.setattr(simulation, 'bias_readout',
        lambda result, block, count: jnp.zeros((len(block), count, 2)))
    design = np.array([[10., 100., 30., 60.], [40., 80., 30., 60.],
                       [50., 100., 30., 60.], [10., 100., 30., 60.]])
    simulation.simulate(jax.random.key(31), design, n_simulations=1,
                        n_samples=500, block_rows=2, em_backend='cuda')
    assert routes == [True, False]
    routes.clear()
    simulation.simulate(jax.random.key(31), design[:1], n_simulations=1,
                        n_samples=20, em_backend='cuda')
    assert routes == [False]


def stub_fit(converged, decreases=None):
    decreases = np.zeros(len(converged), int) if decreases is None else decreases
    return (jnp.zeros(len(converged)),)*5+(jnp.zeros(len(converged)),
        {'converged':jnp.asarray(converged),'winner_decreases':jnp.asarray(decreases)})


@pytest.mark.parametrize('trials,expected_calls', [(4,3),(4097,2)])
def test_cuda_simulation_checks_bounded_groups_before_return(monkeypatch,trials,expected_calls):
    # Simulator contract: a systematic failure in a queued block cannot escape a
    # deferred check; large blocks must still be checked individually. No optimizer needed.
    calls,repairs=[],[]
    def fitted(keys,x,**options):
        calls.append(len(x))
        return stub_fit(np.full(len(x),len(calls)!=2))
    def refit(x,theta,**options):
        repairs.append(options['accelerate'])
        return stub_fit(np.zeros(len(x),bool))
    monkeypatch.setattr(em_batch,'fit_batch',fitted)
    monkeypatch.setattr(cuda_em,'start_parameters',lambda keys,x,**options:jnp.zeros((len(x),1,5,2)))
    monkeypatch.setattr(cuda_em,'fit_from_starts',refit)
    monkeypatch.setattr(simulation,'bias_readout',
        lambda result,block,count:jnp.zeros((len(block),count,2)))
    design=jnp.array([[10.,20.,30.,60.]]*3)
    with pytest.raises(em_batch.EMValidationError,match=f'{trials} selected fits still failed'):
        simulation.simulate(jax.random.key(31),design,n_simulations=trials,
                            n_samples=4,em_backend='cuda')
    assert len(calls)==expected_calls and repairs==[False]


def test_cuda_simulation_repairs_then_masks_isolated_failed_winners(monkeypatch):
    # Simulator contract: a failed winner is refitted with ordinary EM on its own
    # data and its batch's own starts (indexed, never recomputed); a successful
    # refit replaces it, and a failed one becomes a recorded NaN outcome instead
    # of refusing the whole tile.
    failed = np.array([False, True, True, False])
    def fitted(keys,x,**options):
        result = stub_fit(~failed, np.array([0,0,2,0]))
        return tuple(v+1. for v in result[:6])+result[6:]
    refits,banks=[],[]
    def bank(keys,x,**options):
        banks.append((np.asarray(jax.random.key_data(keys)),options['starts']))
        return jnp.broadcast_to(jnp.arange(len(x),dtype=jnp.float32)[:,None,None,None],(len(x),1,5,2))
    def refit(x,theta,**options):
        refits.append((np.asarray(theta[:,0,0,0]),np.asarray(x),options['accelerate']))
        return stub_fit(np.array([True,False]))
    monkeypatch.setattr(em_batch,'fit_batch',fitted)
    monkeypatch.setattr(cuda_em,'start_parameters',bank)
    monkeypatch.setattr(cuda_em,'fit_from_starts',refit)
    monkeypatch.setattr(simulation,'bias_readout',
        lambda result,block,count:jnp.broadcast_to(result[5].reshape(len(block),count,1),(len(block),count,2)))
    key=jax.random.key(31);design=jnp.array([[10.,20.,30.,60.]]*2)
    bias,diagnostic = simulation.simulate(key,design,n_simulations=2,n_samples=4,
        block_rows=2,em_backend='cuda',return_diagnostics=True,em_starts='structured')
    x,fit_keys = simulation.generate_block(simulation.simulation_keys(key,2),design,2,4)
    (rows,data,accelerate), = refits
    (keys,starts), = banks
    np.testing.assert_array_equal(keys,np.asarray(jax.random.key_data(fit_keys)))
    assert starts=='structured'
    np.testing.assert_array_equal(rows,[1.,2.])
    np.testing.assert_array_equal(data,np.asarray(x[1:3]))
    assert accelerate is False
    np.testing.assert_array_equal(diagnostic['repaired'],[[False,True],[False,False]])
    np.testing.assert_array_equal(diagnostic['masked'],[[False,False],[True,False]])
    np.testing.assert_array_equal(bias[...,0],[[1.,0.],[np.nan,1.]])
    np.testing.assert_array_equal(diagnostic['winner_decreases'],[[0,0],[2,0]])


def test_structured_starts_route_through_simulation_and_fit_the_bank():
    # Routing contract: the simulator must fit the deterministic bank (not random
    # starts) for every dataset, and the JAX backend must refuse it explicitly.
    design = jnp.array([[10., 20., 30., 60.], [20., 10., 30., 100.]])
    key = jax.random.key(31)
    bias, diagnostic = simulation.simulate(key, design, n_simulations=2, n_samples=20,
        block_rows=2, em_backend='cuda', return_diagnostics=True, em_starts='structured')
    x, fit_keys = simulation.generate_block(simulation.simulation_keys(key, 2), design, 2, 20)
    _, compact = cuda_em._run(x, structured_starts.batch(x), first_cap=em.FIRST_CAP,
                              max_iter=em.MAX_ITER, occupancy=True)
    direct = cuda_em.result_from_compact(compact)
    np.testing.assert_array_equal(bias, simulation.bias_readout(direct, design, 2))
    assert (np.asarray(compact[:, 26]) < structured_starts.count(20)).all()
    with pytest.raises(ValueError, match='require the CUDA'):
        simulation.simulate(key, design, n_simulations=1, n_samples=20, em_starts='structured')


def test_cuda_single_image_library_preserves_complete_start_states():
    # FFI contract: compiling the gate must keep broad fallback and continuation
    # reachable. This catches a wrong build identity or a narrow-only library.
    x = evidence(500)
    keys = jax.random.split(jax.random.key(17), len(x))
    theta = cuda_em.initial_parameters(keys, x, n_init=5)
    outputs = [cuda_em._run(x, theta, first_cap=1, max_iter=1000,
               occupancy=True, single_image=gate) for gate in (False, True)]
    for baseline, gated in zip(*outputs):
        np.testing.assert_array_equal(np.asarray(baseline).view(np.uint32),
                                      np.asarray(gated).view(np.uint32))
