"""Gaussian response profiles and free-mean fits using the shared likelihood EM engine.

Input columns are response and identifying feature, as in likelihood_em.
Profiles hold response means fixed and constrain identifying means left <= right.
Free-mean fits infer labels afterward. Both use a known common identifying SD
and separate response SDs within bounds.
"""
from functools import partial

import jax
import jax.numpy as jnp

from surface_computation import likelihood_em as em


def update(theta, count, moments, n, wrap_axes, fix_weights, *, identifying_sd, sd_bounds, fixed_response=True):
    """Exact M-step for fixed or free response means and bounded response SDs."""
    proposed = jnp.where(count > 0, theta[2] + moments[2], theta[2])
    if fixed_response:
        pooled = jnp.sum(count * proposed) / n
        identifying = jnp.where(proposed[0] > proposed[1], pooled, proposed)
        response, variance = theta[1], moments[1]
    else:
        identifying = proposed
        response = jnp.where(count > 0, theta[1] + moments[0], theta[1])
        variance = moments[1] - moments[0]**2
    response_sd = jnp.where(count > 0, jnp.sqrt(jnp.maximum(variance, 0)), theta[3])
    response_sd = jnp.clip(response_sd, *sd_bounds)
    p = jnp.clip(count[0] / n, 1e-8, 1 - 1e-8)
    weights = theta[0] if fix_weights else jnp.array([p, 1 - p])
    return weights, response, identifying, response_sd, jnp.full(2, identifying_sd)


def initial_parameters(x, response_means, identifying_sd, sd_bounds):
    """45 deterministic starts span cue separation, weights and response SDs."""
    centre = x[:, 1].mean()
    spread = jnp.maximum(x[:, 1].std(ddof=1), identifying_sd)
    pooled_sd = jnp.clip(x[:, 0].std(ddof=1), *sd_bounds)
    lo, hi = sd_bounds
    allocations = jnp.array([[pooled_sd, pooled_sd], [lo, lo], [lo, hi], [hi, lo], [hi, hi]])
    separation, weight, allocation = jnp.meshgrid(
        jnp.array([.1, 1., 2.]), jnp.array([.25, .5, .75]), jnp.arange(5), indexing='ij')
    gap = separation.ravel() * spread / 2
    weight = weight.ravel()
    return (jnp.stack((weight, 1 - weight), axis=-1),
            jnp.broadcast_to(response_means, (len(gap), 2)),
            jnp.stack((centre - gap, centre + gap), axis=-1),
            allocations[allocation.ravel()], jnp.full((len(gap), 2), identifying_sd))


@partial(jax.jit, static_argnames=('first_cap', 'max_iter', 'rtol', 'parameter_tol'))
def fit(x, response_means, identifying_sd, sd_bounds, *, first_cap=1000,
        max_iter=25000, rtol=1e-9, parameter_tol=1e-6):
    """Fit one fixed response-mean pair; vmap this over a grid for GPU batching.

    Shares densities, EM state, convergence diagnostics, and continuation with
    likelihood_em. Only the M-step differs. Use float64 for precise profiles.
    """
    theta = initial_parameters(x, response_means, identifying_sd, sd_bounds)
    return _fit_starts(x, theta, identifying_sd, sd_bounds, first_cap, max_iter, rtol, parameter_tol)


def _fit_starts(x, theta, identifying_sd, sd_bounds, first_cap, max_iter, rtol,
                parameter_tol, fixed_response=True):
    basis = (None, None)
    update_fn = partial(update, identifying_sd=identifying_sd, sd_bounds=sd_bounds,
                        fixed_response=fixed_response)
    state = jax.vmap(partial(em.initialize, wrap_axes=(False, False)),
                     in_axes=(None, 0, None))(x, theta, basis)
    advance = partial(em.advance, wrap_axes=(False, False), update_fn=update_fn,
                      rtol=rtol, parameter_tol=parameter_tol)
    state = jax.vmap(advance, in_axes=(None, 0, None, None))(
        x, state, basis, min(first_cap, max_iter))
    return em.finish(x, basis, state, max_iter, rtol, parameter_tol,
                     wrap_axes=(False, False), update_fn=update_fn)


@partial(jax.jit, static_argnames=('n_init', 'first_cap', 'max_iter', 'rtol', 'parameter_tol'))
def fit_mle(key, x, identifying_sd, sd_bounds, *, n_init=em.N_INIT, first_cap=1000,
            max_iter=25000, rtol=1e-9, parameter_tol=1e-6):
    """Fit both response means freely, then label components by fitted location.

    Uses the shared random starts plus one narrow-component start at each observed
    point. No generating means or true sample labels enter the fit.
    """
    _, theta = em.initial_parameters(key, x, n_init, wrap_axes=(False, False))
    theta = (*theta[:3], jnp.clip(theta[3], *sd_bounds), jnp.full((n_init, 2), identifying_sd))
    n = len(x)
    narrow = (jnp.broadcast_to(jnp.array([1 / n, 1 - 1 / n]), (n, 2)),
              jnp.stack((x[:, 0], jnp.full(n, x[:, 0].mean())), axis=-1),
              jnp.stack((x[:, 1], jnp.full(n, x[:, 1].mean())), axis=-1),
              jnp.broadcast_to(jnp.array([sd_bounds[0], jnp.clip(x[:, 0].std(), *sd_bounds)]), (n, 2)),
              jnp.full((n, 2), identifying_sd))
    theta = tuple(jnp.concatenate((a, b)) for a, b in zip(theta, narrow))
    result = _fit_starts(x, theta, identifying_sd, sd_bounds, first_cap, max_iter,
                         rtol, parameter_tol, fixed_response=False)
    order = jnp.argsort(result[2])
    parameters = tuple(value[order] for value in result[:5])
    diagnostics = dict(result[-1], parameters=jnp.stack(parameters))
    return (*parameters, result[5], diagnostics)
