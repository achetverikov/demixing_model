"""Unit EM invariants and maintained batching/persistence contracts.

These catch inconsistent circular M-steps, fractional-count shrinkage, lost
continuation state, cross-row key changes and accepting capped training samples.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surface_computation import likelihood_em as em, em_batch
from surface_computation import wnm_simulation as sim

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('fix_weights', [False, True])
def test_fractional_count_update_is_likelihood_monotone(fix_weights):
    x = jnp.array([[0.,0.],[1.,1.],[2.,2.]])
    theta = tuple(jnp.array(v) for v in ([.9,.1],[0.,8.],[0.,8.],[2.,2.],[2.,2.]))
    basis = em.prepare(x)
    ll,count,moments = em.stats(x,theta,basis)
    assert 0 < count[1] < 1
    updated = em.update(theta,count,moments,len(x),fix_weights=fix_weights)
    # Means use the actual fractional posterior count rather than flooring it1.
    expected = theta[2]+moments[2]
    np.testing.assert_allclose(updated[2],expected,atol=1e-6)
    assert float(em.stats(x,updated,basis)[0]) >= float(ll)-1e-5
    assert np.min(np.asarray(updated[3:])) >= 1.
    np.testing.assert_allclose(updated[0],theta[0] if fix_weights else [.9,.1],atol=1e-7)


def test_selected_continuation_preserves_cached_state():
    x = jnp.column_stack((jnp.linspace(-40.,40.,16),jnp.arange(16.)*2-10))
    theta = tuple(jnp.array(v) for v in ([.5,.5],[-15.,15.],[-5.,5.],[50.,60.],[30.,30.]))
    basis = em.prepare(x)
    initial = em.initialize(x,theta,basis)
    advance = jax.jit(em.advance)
    short = advance(x,initial,basis,2)
    expected = advance(x,initial,basis,1000)
    states = jax.tree.map(lambda v:v[None] if hasattr(v,'shape') else jnp.array([v]),short)
    actual = jax.jit(partial(em.finish,max_iter=1000))(x,basis,states)
    for a,b in zip(actual[:5],expected.theta):
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-4)
    np.testing.assert_allclose(actual[5],expected.ll,atol=1e-3)
    assert int(actual[-1]['iterations']) == int(expected.iteration)


def test_packed_continuation_freezes_completed_starts_and_discards_padding():
    x = jnp.stack((jnp.zeros((16,2)),jnp.column_stack((jnp.linspace(-40.,40.,16),jnp.arange(16.)*2-10))))
    keys = jax.random.split(jax.random.key(17),2)
    basis,state = em_batch.initialize(keys,x,n_init=4)
    state = em_batch.advance_all(x,basis,state,5,n_init=4)
    active = np.flatnonzero(np.asarray(em_batch.active_indices(state)))
    frozen = np.asarray(em.converged(state))
    assert frozen.any() and len(active)
    padded = np.pad(active,(0,16-len(active)),mode='edge')
    result = em_batch.advance_compact(x,basis,state,jnp.asarray(padded),len(active),100,n_init=4)
    for before,after in zip(jax.tree.leaves(state),jax.tree.leaves(result)):
        np.testing.assert_array_equal(np.asarray(before)[frozen],np.asarray(after)[frozen])


@pytest.mark.parametrize('n', [4,10,20,100,500])
def test_native_and_staged_datasets_agree_at_all_sample_counts(n):
    keys = jax.random.split(jax.random.key(17),2)
    design = jnp.array([[10.,20.,30.,60.],[30.,60.,30.,120.]])
    x,fit_keys = sim.generate_block(keys,design,2,n)
    options = dict(n_init=4,max_iter=1000,fix_weights=True)
    native = em_batch.fit_batch(fit_keys,x,**options)
    # N20 also exercises several serial continuation chunks, including padding.
    staged = em_batch.fit_batch(fit_keys,x,stages=(5,50,1000),
                               continuation_batch=3 if n==20 else None,**options)
    np.testing.assert_allclose(native[5],staged[5],atol=.002,rtol=1e-6)
    np.testing.assert_array_equal(native[-1]['converged'],staged[-1]['converged'])
    np.testing.assert_allclose(native[0],.5,atol=0)
    np.testing.assert_allclose(staged[0],.5,atol=0)
    np.testing.assert_allclose(sim.bias_readout(native,design,2),sim.bias_readout(staged,design,2),atol=.005)


def test_multirow_keys_observations_and_native_readout_are_preserved():
    design = jnp.array([[10.,20.,30.,60.],[10.,60.,30.,120.]])
    keys = sim.simulation_keys(jax.random.key(17),2,row_offset=13,simulation_offset=50)
    x,fit = sim.generate_block(keys,design,2,4)
    for row,point in enumerate(design):
        for trial,k in enumerate(jax.random.split(keys[row],2)):
            gen,expected_fit = jax.random.split(k)
            expected = sim.jfm.jf.jax_jax_mixture(gen,jnp.array([-.5,.5])*point[3],
                jnp.array([-.5,.5])*sim.IDF_DIFF,point[:2],jnp.repeat(point[2],2),n_samples=4)
            # XLA may fuse scalar and row-vmapped draws differently by one ULP.
            np.testing.assert_allclose(x[row*2+trial],expected,rtol=2e-7,atol=2e-5)
            np.testing.assert_array_equal(jax.random.key_data(fit[row*2+trial]),jax.random.key_data(expected_fit))
    fitted = em_batch.fit_batch(fit,x,fix_weights=True)
    actual = sim.bias_readout(fitted,design,2)
    for row,point in enumerate(design):
        expected = sim.jfm.simulate_dual_component_bias_distribution(keys[row],*point,sim.IDF_DIFF,
            n_simulations=2,n_samples=4,fix_weights=True,return_diagnostics=True)
        np.testing.assert_allclose(actual[row],expected[0],atol=.002)
        assert np.asarray(expected[-1]['converged']).all()


def test_checked_simulation_refuses_systematic_capped_fits_and_unused_options():
    # More capped winners than the masking allowance must refuse the call.
    count = sim.MASKED_ALLOWANCE[0]+1
    with pytest.raises(RuntimeError,match=f'{count} selected fits still failed after repair'):
        sim.simulate(jax.random.key(17),jnp.array([[10.,60.,30.,60.]]),
            n_simulations=count,n_samples=20,em_max_iter=1)
    with pytest.raises(ValueError,match='requires stages'):
        em_batch.fit_batch(None,None,continuation_batch=32)


def test_checked_results_count_losing_start_failures_but_refuse_failed_winners():
    diagnostic = dict(converged=jnp.array([True]),nonfinite_starts=jnp.array([1]),
                      material_decreases=jnp.array([3]),winner_decreases=jnp.array([0]))
    em_batch.check_results((diagnostic,))
    with pytest.raises(RuntimeError,match='1 selected fits decreased likelihood'):
        em_batch.check_results((dict(diagnostic,winner_decreases=jnp.array([1])),))


def test_selected_tie_reports_the_actual_capped_winner():
    x=jnp.array([[0.,0.],[2.,2.]])
    theta=tuple(jnp.array(v) for v in ([.5,.5],[-1.,1.],[-1.,1.],[2.,2.],[2.,2.]))
    basis=em.prepare(x)
    state=em.initialize(x,theta,basis)
    states=jax.tree.map(lambda v:jnp.repeat(jnp.asarray(v)[None],2,axis=0),state)
    states=states._replace(iteration=jnp.array([5,3]),ll=jnp.array([-10.,-10.]),
        relative_change=jnp.array([1e-3,1e-8]),parameter_change=jnp.array([1e-3,1e-8]))
    result=em.finish(x,basis,states,max_iter=5)
    assert not bool(result[-1]['converged'])
    assert bool(result[-1]['any_best_converged'])
    assert int(result[-1]['best_tie_count'])==2


def test_winner_is_reselected_after_roundoff_changes_order(monkeypatch):
    x=jnp.array([[0.,0.],[2.,2.]])
    theta=tuple(jnp.array(v) for v in ([.5,.5],[-1.,1.],[-1.,1.],[2.,2.],[2.,2.]))
    basis=em.prepare(x)
    state=em.initialize(x,theta,basis)
    states=jax.tree.map(lambda v:jnp.repeat(jnp.asarray(v)[None],2,axis=0),state)
    changed=list(states.theta);changed[1]=changed[1].at[1,0].set(-2.)
    states=states._replace(theta=tuple(changed),iteration=jnp.array([1,1]),ll=jnp.array([-10.,-10.000001]))
    def extend(x,s,basis,max_iter,*args):
        decrement=jnp.where(s.theta[1][0]==-1.,2e-6,0.)
        return s._replace(iteration=max_iter,ll=s.ll-decrement,relative_change=0.,parameter_change=0.)
    monkeypatch.setattr(em,'advance',extend)
    result=em.finish(x,basis,states,max_iter=3)
    assert float(result[1][0])==-2.
    assert bool(result[-1]['converged'])
