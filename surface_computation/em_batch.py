"""Batch independent datasets and pool unfinished starts across parameter rows.

Data and Fourier bases are stored once per dataset. Only small cached EM states
are flattened across starts; continuation gathers observations by dataset index.
Native fitting remains available because packing can cost more than it saves.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from surface_computation import likelihood_em as em

OPTIONS = ('n_init', 'rtol', 'parameter_tol', 'max_iter', 'wrap_axes', 'fix_weights')


class EMValidationError(RuntimeError):
    """A deterministic fitting failure that must not be retried as storage I/O."""


@partial(jax.jit, static_argnames=OPTIONS)
def native(keys, x, n_init=em.N_INIT, rtol=em.RTOL, parameter_tol=em.PARAMETER_TOL,
           max_iter=em.MAX_ITER, wrap_axes=(True, False), fix_weights=False):
    return jax.vmap(partial(em.fit, n_init=n_init, rtol=rtol, parameter_tol=parameter_tol,
        max_iter=max_iter, wrap_axes=wrap_axes, fix_weights=fix_weights))(keys, x)


@partial(jax.jit, static_argnames=('n_init', 'wrap_axes'))
def initialize(keys, x, n_init=em.N_INIT, wrap_axes=(True, False)):
    theta = jax.vmap(partial(em.initial_parameters, n_init=n_init, wrap_axes=wrap_axes))(keys, x)[1]
    basis = jax.vmap(partial(em.prepare, wrap_axes=wrap_axes))(x)
    one = jax.vmap(partial(em.initialize, wrap_axes=wrap_axes), in_axes=(None, 0, None))
    state = jax.vmap(one)(x, theta, basis)
    return basis, jax.tree.map(lambda v: v.reshape((-1,)+v.shape[2:]), state)


@partial(jax.jit, static_argnames=('n_init', 'rtol', 'parameter_tol', 'wrap_axes', 'fix_weights'))
def advance_all(x, basis, state, limit, n_init=em.N_INIT, rtol=em.RTOL,
                parameter_tol=em.PARAMETER_TOL, wrap_axes=(True, False), fix_weights=False):
    shaped = jax.tree.map(lambda v: v.reshape((len(x), n_init)+v.shape[1:]), state)
    one = jax.vmap(partial(em.advance, max_iter=limit, rtol=rtol, parameter_tol=parameter_tol,
        wrap_axes=wrap_axes, fix_weights=fix_weights), in_axes=(None, 0, None))
    advanced = jax.vmap(one)(x, shaped, basis)
    return jax.tree.map(lambda v: v.reshape((-1,)+v.shape[2:]), advanced)


@partial(jax.jit, static_argnames=('n_init', 'rtol', 'parameter_tol', 'wrap_axes', 'fix_weights'))
def advance_compact(x, basis, state, indices, valid_count, limit, n_init=em.N_INIT,
                    rtol=em.RTOL, parameter_tol=em.PARAMETER_TOL,
                    wrap_axes=(True, False), fix_weights=False):
    gathered = jax.tree.map(lambda v: v[indices], state)
    dataset = indices//n_init
    result = jax.vmap(partial(em.advance, max_iter=limit, rtol=rtol, parameter_tol=parameter_tol,
        wrap_axes=wrap_axes, fix_weights=fix_weights))(
            x[dataset], gathered, jax.tree.map(lambda v: v[dataset], basis))
    destinations = jnp.where(jnp.arange(len(indices)) < valid_count, indices, len(state.iteration))
    return jax.tree.map(lambda old, new: old.at[destinations].set(new, mode='drop'), state, result)


@partial(jax.jit, static_argnames=OPTIONS)
def finish(x, basis, state, n_init=em.N_INIT, rtol=em.RTOL,
           parameter_tol=em.PARAMETER_TOL, max_iter=em.MAX_ITER,
           wrap_axes=(True, False), fix_weights=False):
    shaped = jax.tree.map(lambda v: v.reshape((len(x), n_init)+v.shape[1:]), state)
    return jax.vmap(partial(em.finish, max_iter=max_iter, rtol=rtol,
        parameter_tol=parameter_tol, wrap_axes=wrap_axes, fix_weights=fix_weights))(x, basis, shaped)


@partial(jax.jit, static_argnames=('rtol', 'parameter_tol'))
def active_indices(state, rtol=em.RTOL, parameter_tol=em.PARAMETER_TOL):
    return em.unfinished(state, em.FIRST_CAP, rtol, parameter_tol)


def fit_batch(keys, x, *, stages=None, continuation_batch=None,
              n_init=em.N_INIT, rtol=em.RTOL, parameter_tol=em.PARAMETER_TOL,
              max_iter=em.MAX_ITER, wrap_axes=(True, False), fix_weights=False,
              backend='jax', cuda_single_image=False):
    """Return batched six-value fits and diagnostics; stages are cumulative caps.

    ``stages=None`` uses the fully compiled native loop. Staged execution pools
    all unfinished starts across every dataset/parameter row in this call.
    ``continuation_batch=None`` uses one power-of-two bucket for the active set;
    an explicit cap bounds continuation memory at the cost of serial calls.
    """
    if backend == 'cuda':
        if stages is not None or continuation_batch is not None:
            raise ValueError('CUDA EM uses a persistent work queue; omit stages and continuation_batch')
        if fix_weights or wrap_axes != (True, False) or rtol != em.RTOL or parameter_tol != em.PARAMETER_TOL:
            raise ValueError('CUDA EM requires free weights, wrapped feature, linear IDF and default tolerances')
        from surface_computation import cuda_em
        return cuda_em.fit_batch(keys, x, n_init=n_init, max_iter=max_iter,
                                 single_image=cuda_single_image)
    if backend != 'jax':
        raise ValueError('EM backend must be jax or cuda')
    options = dict(n_init=n_init, rtol=rtol, parameter_tol=parameter_tol,
                   max_iter=max_iter, wrap_axes=wrap_axes, fix_weights=fix_weights)
    if stages is None:
        if continuation_batch is not None:
            raise ValueError('continuation_batch requires stages')
        return native(keys, x, **options)
    cap = min(em.FIRST_CAP, max_iter)
    if not stages or stages[-1] != cap or stages[0] < 1 or any(a >= b for a, b in zip(stages, stages[1:])):
        raise ValueError('Stages must strictly increase and end at min(5000, max_iter)')
    if continuation_batch is not None and continuation_batch < 1:
        raise ValueError('continuation_batch must be positive')
    if n_init < 1 or max_iter < 1:
        raise ValueError('n_init and max_iter must be positive')
    basis, state = initialize(keys, x, n_init, wrap_axes)
    loop_options = {k: v for k, v in options.items() if k != 'max_iter'}
    state = advance_all(x, basis, state, stages[0], **loop_options)
    for limit in stages[1:]:
        indices = np.flatnonzero(np.asarray(active_indices(state, rtol, parameter_tol)))
        if not len(indices):
            break
        size = max(8, 1 << (len(indices)-1).bit_length())
        if continuation_batch is not None:
            size = min(size, continuation_batch)
        for start in range(0, len(indices), size):
            ids = indices[start:start+size]
            padded = np.pad(ids, (0, size-len(ids)), mode='edge')
            state = advance_compact(x, basis, state, jnp.asarray(padded), len(ids), limit, **loop_options)
    return finish(x, basis, state, **options)


@jax.jit
def validation_counts(diagnostic):
    return jnp.array([jnp.sum(~diagnostic['converged']),
                      jnp.sum(diagnostic['nonfinite_starts']),
                      jnp.sum(diagnostic['material_decreases'])])


def check_results(result):
    """Refuse failed/capped winners or numerical failures before writing samples."""
    failures, nonfinite, decreases = map(int, np.asarray(validation_counts(result[-1])))
    if failures or nonfinite or decreases:
        raise EMValidationError(f'EM validation failed: {failures} selected fits failed or reached the iteration cap; '
            f'{nonfinite} nonfinite starts; {decreases} material likelihood decreases; no samples saved')


def summarize(diagnostic):
    """Reduce assembled host diagnostics without uploading a corpus to the GPU."""
    names=('converged','capped_starts','nonfinite_starts','material_decreases','continued_winner','iterations')
    d=jax.device_get({k:diagnostic[k] for k in names})
    pairs={'selected_cap_hits':~np.asarray(d['converged']), 'capped_losing_starts':d['capped_starts'],
           'nonfinite_starts':d['nonfinite_starts'], 'material_decreases':d['material_decreases'],
           'continued_winners':d['continued_winner']}
    return dict({k:int(np.sum(v,dtype=np.int64)) for k,v in pairs.items()},
                maximum_iterations=int(np.max(d['iterations'])))
