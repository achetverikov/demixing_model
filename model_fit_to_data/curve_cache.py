"""On-disk cache of density-asymmetry curves, and its identity.

The exhaustive backend scans a lattice of precomputed curves instead of calling
the surrogate. That is only sound if the cache provably matches the run reading
it: a cache built from a different checkpoint, a different feat_diff grid, or
different density-target settings produces curves that are still perfectly
plausible and simply wrong. Cache invalidation is the main operational hazard of
the whole approach, so identity is checked, not assumed.

Three mechanisms, none of which replaces the others:

  **cache key** -- a digest of everything that changes the curves. Two builds
  that differ in any of it get different directories, so they cannot overwrite
  each other and a reader cannot pick up the wrong one.
  **manifest** -- the axis order and the row-to-parameter mapping, written out
  explicitly rather than left implicit in the array shape. A wrong flattening
  silently misattributes every curve, and no checksum would catch it because the
  bytes are intact.
  **per-array checksums + a completion marker** -- written last and atomically,
  so a build killed partway through is not mistaken for a finished one.

Layout on disk (one directory per cache key)::

    manifest.json          identity, axes, checksums
    COMPLETE               written last; absent means "do not read this"
    curves.npy             (n_spat, n_pairs, n_points) float32, CENTERED
    means.npy              (n_spat, n_pairs) float32
    variances.npy          (n_spat, n_pairs) float32
    sd_spat.npy            (n_spat,) float64
    feat_pairs.npy         (n_pairs, 2) float64

Curves are stored **centered**, with their means alongside, because the scorer
needs a centered dot product on both sides -- see
``density_objective.ccc_loss_batched``. Storing raw curves and centering at scan
time would repeat the work per candidate per condition; storing them centered
costs the same bytes and moves it to build time. ``curves.npy`` is memory-mapped
and its slab axis is outermost, so reading one ``sd_spat`` is one contiguous read.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import density_objective
    from exhaustive_density import CurveSource
except ModuleNotFoundError:  # imported as `model_fit_to_data.<module>`
    from model_fit_to_data import density_objective
    from model_fit_to_data.exhaustive_density import CurveSource

#: Bump when the on-disk layout changes in any way a reader must notice: array
#: names, dtypes, axis order, or what the stored values mean (raw vs centered).
#: It enters the cache key, so a bump orphans every existing cache rather than
#: letting an old one be read under new assumptions.
CACHE_FORMAT = 1

MANIFEST_NAME = "manifest.json"
COMPLETION_MARKER = "COMPLETE"
_ARRAYS = ("curves", "means", "variances", "sd_spat", "feat_pairs")


def compute_cache_key(
    *,
    checkpoint_sha256: str,
    sd_spat_grid: Tuple[float, float, float],
    feat_grid: Tuple[float, float, float],
    feat_diff_range: Tuple[int, int],
    feat_diff_step: int,
    mu1_bias_range: Tuple[int, int],
    mu1_bias_step: int,
    emp_density_weights_sd: float,
    density_smoothing_sigma: Optional[float],
    encoding: Dict[str, Any],
) -> str:
    """Digest of everything that changes a cached curve's value.

    Args:
        sd_spat_grid, feat_grid: ``(low, high, step)`` of the parameter lattice.
        feat_diff_range, feat_diff_step: the curve's own x axis.
        mu1_bias_range, mu1_bias_step: the bias axis the asymmetry is integrated
            over -- not visible in the stored curve, but it determines its value,
            and it is the axis the circularity fix changed.
        emp_density_weights_sd, density_smoothing_sigma: density-target settings;
            these are applied when the curve is built, so a cache is only valid
            for a fit using the same ones.
        encoding: ``{"kind": "exact"|"pca", ...}``. Rank, dtype, centering and
            seed belong here for a compressed build. Exact and PCA builds must
            not share a directory: they hold different numbers.

    Returns:
        A 16-character hex digest, used as the cache directory name.
    """
    payload = {
        "cache_format": CACHE_FORMAT,
        "checkpoint_sha256": checkpoint_sha256,
        "sd_spat_grid": [float(v) for v in sd_spat_grid],
        "feat_grid": [float(v) for v in feat_grid],
        "feat_diff_range": [int(v) for v in feat_diff_range],
        "feat_diff_step": int(feat_diff_step),
        "mu1_bias_range": [int(v) for v in mu1_bias_range],
        "mu1_bias_step": int(mu1_bias_step),
        "emp_density_weights_sd": float(emp_density_weights_sd),
        "density_smoothing_sigma": (None if density_smoothing_sigma is None
                                    else float(density_smoothing_sigma)),
        "encoding": encoding,
        # The objective is NOT in the key: the cache holds curves, not losses, so
        # changing how they are scored does not invalidate them.
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def default_cache_key(*, checkpoint_path: os.PathLike | str, low: float, high: float,
                      step: float, emp_density_weights_sd: float,
                      density_smoothing_sigma: Optional[float]) -> str:
    """The cache key for a standard exact build.

    One definition, called by both the builder and the fitter: if they computed
    it separately the fitter could look, forever, for a key the builder never
    writes -- and "cache absent" is indistinguishable from "cache built with
    different settings" at the directory level.
    """
    try:
        from run_fingerprint import file_sha256
    except ModuleNotFoundError:
        from model_fit_to_data.run_fingerprint import file_sha256
    from shared.config import config

    return compute_cache_key(
        checkpoint_sha256=file_sha256(checkpoint_path),
        sd_spat_grid=(low, high, step),
        feat_grid=(low, high, step),
        feat_diff_range=config.feat_diff_range,
        feat_diff_step=config.feat_diff_step,
        mu1_bias_range=config.mu1_bias_range,
        mu1_bias_step=config.mu1_bias_step,
        emp_density_weights_sd=emp_density_weights_sd,
        density_smoothing_sigma=density_smoothing_sigma,
        encoding={"kind": "exact", "dtype": "float32"},
    )


def build_curve_lattice(optimizer, sd_spat_values, feat_pairs, *,
                        chunk_size: int = 4096, verbosity: int = 1) -> np.ndarray:
    """Generate the curve lattice from the surrogate, one ``sd_spat`` slab at a time.

    Slab-at-a-time so peak memory is one slab of surfaces rather than the whole
    lattice: at production size a slab is ~2.4 GB of surfaces, and the full
    lattice of surfaces would be ~470 GB. The curves that survive are three
    orders of magnitude smaller, which is the entire point of the cache.

    NN parameter order is ``[sd_feat1, sd_feat2, sd_spat]`` -- the order
    ``fit_hierarchical_grid`` feeds the surrogate. Getting it wrong here would
    misattribute every curve while leaving every checksum valid, which is why
    ``tests/test_curve_cache_matches_live_model.py`` checks an asymmetric pair.
    """
    import jax.numpy as _jnp
    try:
        from grid_based_multi_condition_optimizer_jax_loops import (
            generate_nn_density_asymmetry_batch, predict_nn,
        )
    except ModuleNotFoundError:
        from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import (
            generate_nn_density_asymmetry_batch, predict_nn,
        )
    from shared.config import config

    import time as _time
    feat_pairs = np.asarray(feat_pairs, dtype=float)
    sd_spat_values = np.asarray(sd_spat_values, dtype=float)
    n_points = len(config.create_grid('feat_diff'))
    curves = np.empty((len(sd_spat_values), len(feat_pairs), n_points), dtype=np.float32)

    for spat_index, sd_spat in enumerate(sd_spat_values):
        started = _time.time()
        for start in range(0, len(feat_pairs), chunk_size):
            chunk = feat_pairs[start:start + chunk_size]
            nn_params = _jnp.column_stack([
                _jnp.asarray(chunk[:, 0]),                     # sd_feat1
                _jnp.asarray(chunk[:, 1]),                     # sd_feat2
                _jnp.full(len(chunk), float(sd_spat)),         # sd_spat
            ])
            surfaces = predict_nn(optimizer, nn_params)
            asymmetry = generate_nn_density_asymmetry_batch(
                surfaces,
                weights_sd=optimizer.emp_density_weights_sd,
                smoothing_sigma=optimizer.density_smoothing_sigma,
            )
            curves[spat_index, start:start + len(chunk)] = np.asarray(asymmetry, dtype=np.float32)
        if verbosity > 0:
            elapsed = _time.time() - started
            remaining = elapsed * (len(sd_spat_values) - spat_index - 1)
            print(f"  sd_spat {sd_spat:6.1f}  ({spat_index + 1}/{len(sd_spat_values)})  "
                  f"{elapsed:5.1f}s  ETA {remaining / 60:.1f}m", flush=True)
    return curves


def lattice(low: float, high: float, step: float) -> np.ndarray:
    """Inclusive parameter lattice. One definition, so the builder and the key agree."""
    n = int(round((high - low) / step)) + 1
    return low + step * np.arange(n, dtype=float)


def feat_pair_lattice(values: np.ndarray) -> np.ndarray:
    """``(n_pairs, 2)`` enumeration of feature pairs: sd_feat1 outer, sd_feat2 inner.

    This order is what the exhaustive backend's tie policy refers to (smallest
    sd_feat1, then smallest sd_feat2), so it is fixed here and written into the
    manifest rather than reconstructed by a reader.
    """
    f1, f2 = np.meshgrid(values, values, indexing="ij")
    return np.column_stack([f1.reshape(-1), f2.reshape(-1)])


def _sha256_file(path: Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CacheIncompleteError(RuntimeError):
    """Raised when a cache directory exists but was never finished."""


class CacheCorruptError(RuntimeError):
    """Raised when a cache's contents do not match its manifest."""


def write_cache(out_dir: os.PathLike | str, *, cache_key: str, sd_spat_values,
                feat_pairs, curves, manifest_extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write a complete cache directory. Centering and statistics happen here.

    The completion marker is written **last**, after every array and the
    manifest: a build killed in between leaves a directory that reads as
    unfinished rather than as a short cache.

    Args:
        curves: ``(n_spat, n_pairs, n_points)`` RAW curves. Stored centered.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = np.asarray(curves, dtype=np.float32)
    sd_spat_values = np.asarray(sd_spat_values, dtype=float)
    feat_pairs = np.asarray(feat_pairs, dtype=float)

    if curves.ndim != 3 or curves.shape[:2] != (len(sd_spat_values), len(feat_pairs)):
        raise ValueError(
            f"curves must be (n_spat, n_pairs, n_points); got {curves.shape} against "
            f"{len(sd_spat_values)} sd_spat values and {len(feat_pairs)} pairs"
        )

    means = curves.mean(axis=2, dtype=np.float64)
    centered = (curves - means[:, :, None]).astype(np.float32)
    variances = (centered.astype(np.float64) ** 2).mean(axis=2)

    arrays = {
        "curves": centered,
        "means": means.astype(np.float32),
        "variances": variances.astype(np.float32),
        "sd_spat": sd_spat_values,
        "feat_pairs": feat_pairs,
    }
    for name, array in arrays.items():
        np.save(out_dir / f"{name}.npy", array)

    manifest = {
        "cache_format": CACHE_FORMAT,
        "cache_key": cache_key,
        "n_spat": int(len(sd_spat_values)),
        "n_pairs": int(len(feat_pairs)),
        "n_points": int(curves.shape[2]),
        # Explicit, because a wrong flattening misattributes every curve and no
        # checksum can see it: the bytes are intact, they just mean something else.
        "axis_order": ["sd_spat", "feat_pair", "feat_diff"],
        "feat_pair_enumeration": "sd_feat1 outer, sd_feat2 inner, both ascending",
        "curves_are_centered": True,
        "checksums": {name: _sha256_file(out_dir / f"{name}.npy") for name in _ARRAYS},
    }
    manifest.update(manifest_extra or {})

    tmp = out_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, out_dir / MANIFEST_NAME)

    marker = out_dir / COMPLETION_MARKER
    tmp_marker = out_dir / f".{COMPLETION_MARKER}.tmp"
    tmp_marker.write_text(manifest["cache_key"])
    os.replace(tmp_marker, marker)
    return out_dir


def read_manifest(cache_dir: os.PathLike | str, *, verify: bool = False) -> Dict[str, Any]:
    """Load and validate a cache manifest.

    Args:
        verify: also re-hash every array and compare. Off by default because it
            costs a full read of a multi-GB cache; on for `--verify` builds and
            whenever a cache has moved between machines.

    Raises:
        CacheIncompleteError: the completion marker is absent.
        CacheCorruptError: manifest missing, malformed, or checksums disagree.
    """
    cache_dir = Path(cache_dir)
    if not (cache_dir / COMPLETION_MARKER).exists():
        raise CacheIncompleteError(
            f"{cache_dir} has no {COMPLETION_MARKER} marker: the build never finished, or is "
            "still running. Do not read it — rebuild, or wait for the writer."
        )
    try:
        manifest = json.loads((cache_dir / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheCorruptError(f"{cache_dir}: cannot read {MANIFEST_NAME}: {exc}") from exc

    if manifest.get("cache_format") != CACHE_FORMAT:
        raise CacheCorruptError(
            f"{cache_dir} was written in cache format {manifest.get('cache_format')}, but this "
            f"code reads format {CACHE_FORMAT}. Rebuild it: the layout or the meaning of the "
            "stored values changed."
        )
    if not manifest.get("curves_are_centered"):
        raise CacheCorruptError(
            f"{cache_dir} stores raw curves; the scorer requires centered ones (see "
            "density_objective.ccc_loss_batched). Rebuild."
        )
    if verify:
        for name, expected in manifest["checksums"].items():
            actual = _sha256_file(cache_dir / f"{name}.npy")
            if actual != expected:
                raise CacheCorruptError(
                    f"{cache_dir}/{name}.npy checksum {actual} != manifest {expected}. "
                    "The cache is damaged; rebuild it."
                )
    return manifest


class CachedCurveSource(CurveSource):
    """A `CurveSource` backed by a cache directory, read through a memmap.

    The slab axis is outermost, so ``slab(i)`` is one contiguous read and the
    whole cache never has to be resident.
    """

    def __init__(self, cache_dir: os.PathLike | str, *, verify: bool = False):
        self.cache_dir = Path(cache_dir)
        self.manifest = read_manifest(self.cache_dir, verify=verify)
        self.n_points = int(self.manifest["n_points"])
        self.sd_spat_values = np.load(self.cache_dir / "sd_spat.npy")
        self.feat_pairs = np.load(self.cache_dir / "feat_pairs.npy")
        self._curves = np.load(self.cache_dir / "curves.npy", mmap_mode="r")
        self._means = np.load(self.cache_dir / "means.npy")
        self._variances = np.load(self.cache_dir / "variances.npy")

        expected = (self.manifest["n_spat"], self.manifest["n_pairs"], self.n_points)
        if self._curves.shape != expected:
            raise CacheCorruptError(
                f"{self.cache_dir}: curves.npy has shape {self._curves.shape}, manifest says "
                f"{expected}. The row-to-parameter mapping cannot be trusted."
            )

    def slab(self, spat_index: int):
        return (np.asarray(self._curves[spat_index]),
                self._means[spat_index],
                self._variances[spat_index])


def cache_dir_for(out_root: os.PathLike | str, cache_key: str) -> Path:
    return Path(out_root) / f"curve_cache_{cache_key}"


def open_or_build(out_root: os.PathLike | str, cache_key: str, build_fn,
                  *, verify: bool = False, lock_ttl_seconds: int = 10800):
    """Return a `CachedCurveSource`, building the cache first if it is absent.

    Build-on-demand needs **single-writer locking, not just the completion
    marker**. The marker tells a reader the cache is finished; it does nothing to
    stop two concurrent fits from both finding the cache absent and building it
    into the same directory, interleaving their writes. That matters here
    regardless of how the pipeline is run, because a shared ``--curve-cache``
    path invites exactly that.

    A process that loses the race waits for the marker rather than building a
    second copy.

    Args:
        build_fn: called with the target directory; must produce a complete cache
            there (i.e. end in `write_cache`).
    """
    from surface_computation.lock_backend import make_lock_backend

    cache_dir = cache_dir_for(out_root, cache_key)
    if (cache_dir / COMPLETION_MARKER).exists():
        return CachedCurveSource(cache_dir, verify=verify)

    Path(out_root).mkdir(parents=True, exist_ok=True)
    backend = make_lock_backend("file", lock_dir=Path(out_root))
    owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    with backend.held(f"curve_cache_{cache_key}", owner, ttl_seconds=lock_ttl_seconds) as acquired:
        if acquired:
            # Re-check under the lock: another process may have finished between
            # our first check and acquiring it.
            if not (cache_dir / COMPLETION_MARKER).exists():
                build_fn(cache_dir)
            return CachedCurveSource(cache_dir, verify=verify)

    raise CacheIncompleteError(
        f"Another process is building {cache_dir}. Wait for it to finish and re-run, or point "
        "--curve-cache somewhere else. (Two processes building into one directory would "
        "interleave their writes.)"
    )
