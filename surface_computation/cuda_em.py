"""Persistent SQUAREM-accelerated likelihood EM through JAX's CUDA FFI.

The optional backend requires nvcc and supports the maintained two-component,
wrapped-feature/linear-identifying model at N=4,10,20,100,500. Sources are built
once per compiler/device/source identity under /tmp; importing does no GPU work.
JAX owns and reuses all buffers, including the work queue and full start states.
"""
import ctypes
from functools import cache, partial
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from surface_computation import likelihood_em as em

SAMPLE_COUNTS = (4, 10, 20, 100, 500)
SOURCE_NAMES = ('cuda/likelihood_em.cu', 'cuda/likelihood_em.cuh')


@cache
def _load(architecture):
    compiler = shutil.which('nvcc')
    if compiler is None:
        raise RuntimeError('The CUDA EM backend requires nvcc on PATH')
    root = Path(__file__).resolve().parent
    identity = subprocess.check_output([compiler, '--version']) + architecture.encode() + version('jaxlib').encode() + Path(__file__).read_bytes()
    for name in SOURCE_NAMES:
        identity += (root / name).read_bytes()
    token = hashlib.sha256(identity).hexdigest()
    target = 'dm_likelihood_em_' + token
    folder = Path('/tmp/demixing_cuda_em') / token
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / 'likelihood_em.so'
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix='dm_cuda_build_', dir='/tmp') as scratch:
            output = Path(scratch) / library.name
            command = [compiler, '-O3', '-std=c++17', '-shared', '-Xcompiler', '-fPIC',
                       '-arch=' + architecture, '-I' + jax.ffi.include_dir(),
                       str(root / SOURCE_NAMES[0]), '-o', str(output)]
            process = subprocess.run(command, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError('CUDA EM compilation failed:\n' + process.stderr)
            os.replace(output, library)
    handle = ctypes.CDLL(str(library))
    jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(handle.dm_likelihood_em), platform='CUDA')
    return target, handle


def _run(x, theta, *, first_cap, max_iter, occupancy, accelerate=True):
    """Return raw states/diagnostics; accelerate=False retains ordinary EM for checks."""
    device, = x.devices()
    if device.platform != 'gpu':
        raise ValueError('The CUDA EM backend requires an NVIDIA GPU')
    if x.dtype != jnp.float32 or theta.dtype != jnp.float32:
        raise ValueError('The CUDA EM backend requires float32 arrays')
    batch, n, axes = x.shape
    if n not in SAMPLE_COUNTS or axes != 2 or theta.shape[0] != batch or theta.shape[2:] != (5, 2):
        raise ValueError('CUDA EM requires x=(batch,N,2), theta=(batch,starts,5,2), N in ' + str(SAMPLE_COUNTS))
    if batch < 1 or theta.shape[1] < 1 or not 0 < first_cap <= max_iter <= em.MAX_ITER:
        raise ValueError('CUDA EM requires nonempty inputs and 0 < first_cap <= max_iter <= 25000')
    if theta.devices() != x.devices():
        raise ValueError('CUDA EM inputs must be on the same device')
    capability = str(device.compute_capability).replace('.', '')
    target, _ = _load('sm_' + capability)
    shapes = (jax.ShapeDtypeStruct((batch, theta.shape[1], 26), jnp.float32),
              jax.ShapeDtypeStruct((batch, 33), jnp.float32),
              jax.ShapeDtypeStruct((1,), jnp.int32))
    return jax.ffi.ffi_call(target, shapes)(x, theta, first_cap=np.int64(first_cap),
        max_iter=np.int64(max_iter), occupancy=np.int64(occupancy),
        accelerate=np.int64(accelerate))[:2]


def build_identity():
    """Fingerprint the compiled implementation for generation/resume metadata."""
    device = jax.devices('gpu')[0]
    target, _ = _load('sm_' + str(device.compute_capability).replace('.', ''))
    return target


@partial(jax.jit, static_argnames=('n_init',))
def initial_parameters(keys, x, n_init=em.N_INIT):
    """Generate the unchanged starts without discarded JAX EM statistics/bases."""
    theta = jax.vmap(partial(em.initial_parameters, n_init=n_init))(keys, x)[1]
    return jnp.stack(theta, axis=2)


@jax.jit
def result_from_compact(c):
    parameters = c[:, :10].reshape((-1, 5, 2))
    ok = jnp.isfinite(c[:, 10]) & jnp.isfinite(c[:, :10]).all(axis=1)
    ok &= (c[:, 22] <= em.RTOL) & (c[:, 23] <= em.PARAMETER_TOL)
    diagnostic = dict(iterations=c[:, 21].astype(jnp.int32), relative_change=c[:, 22],
        parameter_change=c[:, 23], converged=ok, any_best_converged=ok | (c[:, 32] > 0),
        best_tie_count=c[:, 27].astype(jnp.int32), capped_starts=c[:, 28].astype(jnp.int32),
        continued_winner=c[:, 21] > em.FIRST_CAP, nonfinite_starts=c[:, 29].astype(jnp.int32),
        material_decreases=c[:, 30].astype(jnp.int32), worst_change=c[:, 31],
        log_likelihood=c[:, 10], parameters=parameters)
    return tuple(parameters[:, k] for k in range(5)) + (c[:, 10], diagnostic)


def fit_batch(keys, x, *, n_init=em.N_INIT, max_iter=em.MAX_ITER):
    """Keep every start, continue/reselect capped winners, and return EM diagnostics."""
    if n_init < 1 or not 0 < max_iter <= em.MAX_ITER:
        raise ValueError('n_init must be positive and max_iter must be in [1,25000]')
    theta = initial_parameters(keys, x, n_init)
    _, compact = _run(x, theta, first_cap=min(em.FIRST_CAP, max_iter),
                      max_iter=max_iter, occupancy=True)
    return result_from_compact(compact)
