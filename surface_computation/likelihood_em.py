"""Likelihood EM for two diagonal components with optional wrapped coordinates.

Wrapped coordinates augment each soft component assignment with a posterior
winding number. Cached conditional moments give a likelihood-consistent M-step;
the circular sine/cosine moment estimator is not used. SD>=1 and pi>=.1 are
explicit model constraints. All starts receive up to 5000 updates, followed by
exact-state continuation of a capped selected winner up to the requested limit.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from shared import wnm
from surface_computation.em_identity import ESTIMATOR_ID, DIAGNOSTIC_DTYPES

N_INIT = 225
RTOL = 1e-6
PARAMETER_TOL = 1e-5
FIRST_CAP = 5000
MAX_ITER = 25000
N_HARMONICS = 6
DIAGNOSTIC_FIELDS = tuple(DIAGNOSTIC_DTYPES)


class State(NamedTuple):
    iteration: object
    theta: tuple
    ll: object
    count: object
    moments: tuple
    relative_change: object
    parameter_change: object
    worst_change: object
    material_decreases: object


def initial_parameters(key, x, n_init=N_INIT, num_comp=2, wrap_axes=(True, False)):
    key, idx_key = jax.random.split(key)
    means = x[jax.random.randint(idx_key, (n_init, num_comp), 0, len(x))]
    key, scale_key = jax.random.split(key)
    sd = jax.random.uniform(scale_key, (n_init, num_comp, 2),
        minval=x.std(0)[None, None, :]*.3, maxval=x.std(0)[None, None, :]*2.)
    means = tuple(wnm.wrap_deg(means[..., k]) if circular else means[..., k]
                  for k, circular in enumerate(wrap_axes))
    return key, (jnp.full((n_init, num_comp), 1./num_comp), *means,
                 jnp.maximum(sd[..., 0], 1.), jnp.maximum(sd[..., 1], 1.))


def prepare(x, wrap_axes=(True, False)):
    n = jnp.arange(1, N_HARMONICS+1, dtype=x.dtype)
    return tuple((jnp.cos(jnp.radians(x[:, k, None])*n),
                  jnp.sin(jnp.radians(x[:, k, None])*n)) if circular else None
                 for k, circular in enumerate(wrap_axes))


def wrapped_stats(x, mean, sd, basis):
    residual = wnm.wrap_deg(x[:, None]-mean)
    narrow = jnp.minimum(sd, wnm.SIGMA_SWITCH)
    z = residual[:, :, None]+jnp.arange(-1, 2, dtype=x.dtype)*wnm.PERIOD
    logw = -.5*(z/narrow[None, :, None])**2
    logsum = jsp.special.logsumexp(logw, axis=-1)
    q = jnp.exp(logw-logsum[:, :, None])
    spatial = (logsum-jnp.log(narrow)-.5*jnp.log(2*jnp.pi),
               jnp.sum(q*z, axis=-1), jnp.sum(q*z*z, axis=-1))

    broad = jnp.maximum(sd, wnm.SIGMA_SWITCH)
    n = jnp.arange(1, N_HARMONICS+1, dtype=x.dtype)
    omega = jnp.pi/180
    cm, sm = jnp.cos(mean[:, None]*omega*n), jnp.sin(mean[:, None]*omega*n)
    cx, sx = basis
    cd = cx[:, None, :]*cm[None, :, :]+sx[:, None, :]*sm[None, :, :]
    sine = sx[:, None, :]*cm[None, :, :]-cx[:, None, :]*sm[None, :, :]
    a = jnp.exp(-.5*(broad[:, None]*omega*n)**2)
    denominator = 1+2*jnp.sum(a[None, :, :]*cd, axis=-1)
    fourier = (jnp.log(denominator)-jnp.log(wnm.PERIOD),
        2*broad**2*omega*jnp.sum(n*a[None, :, :]*sine, axis=-1)/denominator,
        broad**2-2*broad**4*omega**2*jnp.sum(n**2*a[None, :, :]*cd, axis=-1)/denominator)
    return tuple(jnp.where(sd >= wnm.SIGMA_SWITCH, f, s) for f, s in zip(fourier, spatial))


def stats(x, theta, basis, wrap_axes=(True, False)):
    pi, mf, mi, sf, si = theta
    axes = []
    for k, mean, sd in [(0, mf, sf), (1, mi, si)]:
        delta = x[:, k, None]-mean
        axes.append(wrapped_stats(x[:, k], mean, sd, basis[k]) if wrap_axes[k] else
                    (jsp.stats.norm.logpdf(x[:, k, None], mean, sd), delta, delta**2))
    lf, ef, e2f = axes[0]
    li, ei, e2i = axes[1]
    logp = lf+li+jnp.log(pi)
    norm = jsp.special.logsumexp(logp, axis=1)
    r = jnp.exp(logp-norm[:, None])
    count = r.sum(0)
    denom = jnp.where(count > 0, count, 1.)
    return norm.sum(), count, tuple(jnp.sum(r*v, axis=0)/denom for v in (ef, e2f, ei, e2i))


def update(theta, count, moments, n, wrap_axes=(True, False), fix_weights=False):
    means, scales = [], []
    for k, circular in enumerate(wrap_axes):
        mean, sd = theta[k+1], theta[k+3]
        shift, second = moments[2*k:2*k+2]
        proposed = mean+shift
        if circular:
            proposed = wnm.wrap_deg(proposed)
        means.append(jnp.where(count > 0, proposed, mean))
        scales.append(jnp.where(count > 0, jnp.sqrt(jnp.maximum(second-shift**2, 1.)), sd))
    # Exact constrained maximizer for two components, including fractional Nk.
    p = jnp.clip(count[0]/n, .1, .9)
    pi = theta[0] if fix_weights else jnp.array([p, 1-p])
    return pi, *means, *scales


def parameter_change(before, after, wrap_axes=(True, False)):
    changes = [jnp.abs(after[0]-before[0]).max()]
    for k, circular in enumerate(wrap_axes):
        delta = after[k+1]-before[k+1]
        if circular:
            delta = wnm.wrap_deg(delta)
            roundoff = 4*jnp.finfo(delta.dtype).eps*wnm.PERIOD
        else:
            roundoff = 2*jnp.finfo(delta.dtype).eps*(jnp.abs(before[k+1])+jnp.abs(after[k+1]))
        scale = before[k+3]
        changes.extend([(jnp.maximum(jnp.abs(delta)-roundoff, 0)/scale).max(),
                        jnp.abs((after[k+3]-scale)/scale).max()])
    return jnp.stack(changes).max()


def initialize(x, theta, basis, wrap_axes=(True, False)):
    ll, count, moments = stats(x, theta, basis, wrap_axes)
    return State(0, theta, ll, count, moments, jnp.inf, jnp.inf, 0., 0)


def converged(state, rtol=RTOL, parameter_tol=PARAMETER_TOL):
    finite = jnp.isfinite(state.ll) & jnp.all(jnp.stack([jnp.isfinite(v).all(-1) for v in state.theta]), axis=0)
    return finite & (state.relative_change <= rtol) & (state.parameter_change <= parameter_tol)


def unfinished(state, limit, rtol=RTOL, parameter_tol=PARAMETER_TOL):
    return (state.iteration < limit) & jnp.isfinite(state.ll) & ~converged(state, rtol, parameter_tol)


def advance(x, initial, basis, max_iter, rtol=RTOL, parameter_tol=PARAMETER_TOL,
            wrap_axes=(True, False), fix_weights=False):
    def iteration(s):
        theta = update(s.theta, s.count, s.moments, len(x), wrap_axes, fix_weights)
        ll, count, moments = stats(x, theta, basis, wrap_axes)
        delta = ll-s.ll
        return State(s.iteration+1, theta, ll, count, moments,
            jnp.abs(delta)/(1e-12+jnp.abs(s.ll)), parameter_change(s.theta, theta, wrap_axes),
            jnp.minimum(s.worst_change, delta),
            s.material_decreases+(delta < -8*jnp.finfo(ll.dtype).eps*(1+jnp.abs(s.ll))))
    return jax.lax.while_loop(lambda s: unfinished(s, max_iter, rtol, parameter_tol), iteration, initial)


def select(state):
    best = jnp.argmax(jnp.where(jnp.isfinite(state.ll), state.ll, -jnp.inf))
    return best, jax.tree.map(lambda v: v[best], state)


def finish(x, basis, state, max_iter=MAX_ITER, rtol=RTOL,
           parameter_tol=PARAMETER_TOL, wrap_axes=(True, False), fix_weights=False):
    def extend(states):
        best, selected = select(states)
        selected = advance(x, selected, basis, max_iter, rtol, parameter_tol, wrap_axes, fix_weights)
        return jax.tree.map(lambda old, new: old.at[best].set(new), states, selected)
    # Usually at most one extension. Re-select if floating-point changes make
    # another capped start the winner; never replace it with a worse converged fit.
    state = jax.lax.while_loop(lambda s: unfinished(select(s)[1], max_iter, rtol, parameter_tol), extend, state)
    best, selected = select(state)
    ok = converged(selected, rtol, parameter_tol)
    base_ok = converged(state, rtol, parameter_tol)
    tied = jnp.isfinite(state.ll) & (state.ll == state.ll[best])
    diagnostics = dict(iterations=selected.iteration, relative_change=selected.relative_change,
        parameter_change=selected.parameter_change, converged=ok,
        any_best_converged=ok | jnp.any(tied & base_ok), best_tie_count=tied.sum(),
        capped_starts=jnp.sum(jnp.isfinite(state.ll) & ~base_ok),
        continued_winner=selected.iteration > FIRST_CAP,
        nonfinite_starts=jnp.sum(~jnp.isfinite(state.ll)),
        material_decreases=jnp.sum(state.material_decreases),
        worst_change=jnp.minimum(state.worst_change.min(), selected.worst_change),
        log_likelihood=selected.ll, parameters=jnp.stack(selected.theta))
    return selected.theta+(selected.ll, diagnostics)


@partial(jax.jit, static_argnames=('n_init', 'rtol', 'parameter_tol', 'max_iter', 'wrap_axes', 'fix_weights'))
def fit(key, x, n_init=N_INIT, rtol=RTOL, parameter_tol=PARAMETER_TOL,
        max_iter=MAX_ITER, wrap_axes=(True, False), fix_weights=False):
    if n_init < 1 or max_iter < 1:
        raise ValueError('n_init and max_iter must be positive')
    _, theta = initial_parameters(key, x, n_init, wrap_axes=wrap_axes)
    basis = prepare(x, wrap_axes)
    state = jax.vmap(partial(initialize, wrap_axes=wrap_axes), in_axes=(None, 0, None))(x, theta, basis)
    state = jax.vmap(partial(advance, rtol=rtol, parameter_tol=parameter_tol,
        wrap_axes=wrap_axes, fix_weights=fix_weights), in_axes=(None, 0, None, None))(
        x, state, basis, min(FIRST_CAP, max_iter))
    return finish(x, basis, state, max_iter, rtol, parameter_tol, wrap_axes, fix_weights)
