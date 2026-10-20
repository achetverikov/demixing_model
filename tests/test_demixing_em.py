"""Unit regressions for premature stopping in the EM fitters.

Flag-dispatch tests do not check that EM continues to a finite stopping statistic.
This small deterministic fixture exercises the loop itself, without simulation.
"""
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from surface_computation.jax_fit_functions import train_em_jax


@pytest.fixture
def observed():
    return jnp.column_stack((jnp.linspace(-40., 40., 16),
        jnp.array([-18., 3., -10., 12., -15., 0., 8., -4.,
                   16., -8., 4., 20., -2., 14., 7., 24.])))


@pytest.mark.gpu
@pytest.mark.parametrize('fix_weights', [False, True])
def test_em_continues_until_finite_convergence(fix_weights, observed):
    options = dict(n_init=4, fix_weights=fix_weights, return_diagnostics=True)
    first = train_em_jax(jax.random.key(17), observed, max_iter=1, **options)
    fitted = train_em_jax(jax.random.key(17), observed, **options)
    assert int(first[-1]['iterations']) == 1
    assert np.isfinite(first[-1]['relative_change'])
    assert not bool(first[-1]['converged'])
    assert int(fitted[-1]['iterations']) > 1
    assert np.isfinite(fitted[-1]['relative_change'])
    assert bool(fitted[-1]['converged'])
    assert float(fitted[-2]) > float(first[-2])
    assert all(np.isfinite(value).all() for value in fitted[:-1])


@pytest.mark.gpu
@pytest.mark.parametrize('module_name, function_name', [
    ('surface_computation.jax_fit_functions', 'train_em_jax_Lk_B'),
    ('surface_computation.jax_fit_functions', 'train_em_jax_diag_common'),
    ('scripts.fit_shared_uev_jax_LkBk_circular_old_init', 'train_em_jax_circular_old_init'),
    ('scripts.fit_shared_uev_jax_LkBk_main_init', 'train_em_jax_old_init'),
])
def test_em_variants_improve_beyond_first_update(module_name, function_name, observed):
    fit = jax.jit(getattr(import_module(module_name), function_name),
                  static_argnames=('n_init', 'max_iter'))
    key = jax.random.key(17)
    first = fit(key, observed, n_init=4, max_iter=1)
    fitted = fit(key, observed, n_init=4)
    assert len(fitted) == 6
    assert all(np.isfinite(value).all() for value in fitted)
    assert float(fitted[-1]) > float(first[-1]) + 1e-3
