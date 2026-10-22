"""Deterministic data-derived EM starts: partitions and local neighbourhoods of each dataset.

Each start is a moment initialization from a hard two-way split of the observations:
single points, decile cuts on the identifying coordinate, quintile cuts on the feature
coordinate in two circular charts, and small local neighbourhoods, plus one even split.
This gives ``min(N, 20) + 34`` starts (54 for N >= 20), replacing the 225 random starts.
They do not certify a global maximum. All M-steps remain the production likelihood EM.

Provenance: ported unchanged (without its unused pooled-scale variant) from
``$DEMIXING_ARTIFACT_ROOT/performance_search_20261021/astra/structured_init.py``, the
"structured54" bank validated in ``density_tolerance_20261022``,
``density_trajectory_20261022`` and ``narrow_interval_check_20261022``.
"""
import jax
import jax.numpy as jnp

from shared import wnm

#: Datasets per device call; bounds the transient (starts, N, N) neighbourhood arrays.
CHUNK = 128


def count(n):
    return min(n, 20)+34


def seeds(x, wrap_axes=(True, False)):
    """Start parameters ``(pi, mu_feat, mu_idf, sd_feat, sd_idf)``, each ``(count(N), 2)``."""
    n = len(x)
    identity_order = jnp.argsort(x[:, 1])
    feature_order = jnp.argsort(x[:, 0])
    if n <= 20:
        singles = jnp.arange(n)
    else:
        positions = jnp.linspace(0, n-1, 10).astype(jnp.int32)
        singles = jnp.concatenate((identity_order[positions], feature_order[positions]))
    masks = [jnp.arange(n)[None, :] == singles[:, None]]

    # All nine decile cuts on the identity coordinate.
    rank = jnp.argsort(identity_order)
    sizes = jnp.clip(jnp.rint(n*jnp.arange(1, 10)/10).astype(jnp.int32), 1, n-1)
    masks.append(rank[None, :] < sizes[:, None])

    # Feature cuts in two circular charts cover clusters straddling +/-180.
    sizes = jnp.clip(jnp.rint(n*jnp.arange(1, 5)/5).astype(jnp.int32), 1, n-1)
    for phase in (0., 90.):
        values = wnm.wrap_deg(x[:, 0]-phase) if wrap_axes[0] else x[:, 0]
        rank = jnp.argsort(jnp.argsort(values))
        masks.append(rank[None, :] < sizes[:, None])

    # Compact local neighborhoods uncover small narrow components. Four centers
    # from each coordinate ordering avoid depending on an arbitrary data order.
    positions = jnp.linspace(0, n-1, 4).astype(jnp.int32)
    centers = jnp.concatenate((identity_order[positions], feature_order[positions]))
    delta = x[None, :, :] - x[centers, None, :]
    if wrap_axes[0]:
        delta = delta.at[:, :, 0].set(wnm.wrap_deg(delta[:, :, 0]))
    if wrap_axes[1]:
        delta = delta.at[:, :, 1].set(wnm.wrap_deg(delta[:, :, 1]))
    distance = jnp.sum((delta/jnp.maximum(x.std(axis=0), 1.))**2, axis=-1)
    rank = jnp.argsort(jnp.argsort(distance, axis=1), axis=1)
    for size in (min(2, n-1), min(max(2, round(n*.1)), n-1)):
        masks.append(rank < size)
    first = jnp.concatenate(masks, axis=0).astype(x.dtype)
    r = jnp.stack((first, 1-first), axis=-1)
    r = jnp.concatenate((r, jnp.full((1, n, 2), .5, dtype=x.dtype)), axis=0)
    return from_responsibilities(x, r, wrap_axes)


def from_responsibilities(x, r, wrap_axes=(True, False)):
    """Moment initialization from per-start responsibilities ``r`` of shape (starts, N, 2)."""
    n = len(x)
    nk = r.sum(axis=1)
    p = jnp.clip(nk[:, 0]/n, .1, .9)
    means, scales = [], []
    for axis, circular in enumerate(wrap_axes):
        values = x[:, axis]
        if circular:
            angle = jnp.radians(values)
            mean = jnp.degrees(jnp.arctan2(jnp.sum(r*jnp.sin(angle)[None, :, None], axis=1),
                                         jnp.sum(r*jnp.cos(angle)[None, :, None], axis=1)))
            residual = wnm.wrap_deg(values[None, :, None]-mean[:, None, :])
        else:
            mean = jnp.sum(r*values[None, :, None], axis=1)/nk
            residual = values[None, :, None]-mean[:, None, :]
        means.append(mean)
        scales.append(jnp.sqrt(jnp.maximum(jnp.sum(r*residual**2, axis=1)/nk, 1.)))
    pi = jnp.stack((p, 1-p), axis=-1)
    return (pi, *means, *scales)


@jax.jit
def _chunk(x):
    return jax.vmap(lambda one: jnp.stack(seeds(one), axis=1))(x)


def batch(x):
    """CUDA start layout ``(datasets, count(N), 5, 2)`` for observations ``(datasets, N, 2)``."""
    return jnp.concatenate([_chunk(x[a:a+CHUNK]) for a in range(0, len(x), CHUNK)])
