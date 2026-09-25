"""Load raw EM bias samples from an existing production sample corpus.

``$DEMIXING_ARTIFACT_ROOT/sim_samples_10k_100samples_circular_em_fullcov_free_weights``
holds ~66k filenames on the 5-degree ``(sd_feat1, sd_feat2, sd_ident)`` grid
(5-200 in each dimension).  **Most of them are processed stubs, not raw
samples:** once a combination has been reduced to a KDE surface the raw outcomes
are dropped and the file is rewritten as a ~400-byte stub (a ``'stub'`` marker
and empty ``mu1_samples``/``mu2_samples`` arrays).  Only ~2096 files (~6.5 MB
each, covering ~1605 of the 64000 grid triples plus their ``_r0``/``_r1`` mirror
runs) still carry the raw per-simulation biases — 90 ``feat_diff`` rows x 10,000
EM outcomes x 2 components — which are exactly what this prototype wants.

Because stubs outnumber usable files ~30:1, :func:`list_files` filters by size
*before* any ``n_files`` subsample, so a request for 800 files returns 800
usable files rather than ~24 survivors of a random draw dominated by stubs.

Two further caveats, both handled by the callers rather than hidden here:

* The usable files are still **on a grid** in all four inputs (5 degrees in the
  SDs, 2 degrees in ``feat_diff``).  Fine for training, but they cannot validate
  off-grid generalisation; ``generate_training_data.py --validation`` simulates
  fresh continuous parameter combinations for that.
* Despite the ``fullcov`` in the folder name these fits are diagonal — the flag
  was a no-op when they were produced (see the repo's flag-dispatch fix), and
  ``jax_generate_and_fit`` now refuses non-diagonal covariance outright.  They
  are free-weights, ``n_samples=100`` runs.

Nothing here reads or produces a KDE surface: the usable ``.pkl.gz`` files store
the samples themselves.
"""

from __future__ import annotations

import gzip
import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from shared.config import artifact_root, config

DEFAULT_CORPUS = "sim_samples_10k_100samples_circular_em_fullcov_free_weights"

_FILE_RE = re.compile(
    r"samples_sf1_(?P<sf1>[\d.]+)_sf2_(?P<sf2>[\d.]+)_sp_(?P<sp>[\d.]+)"
    r"(?:_r(?P<run>\d))?_[a-f0-9]{8}\.pkl\.gz$"
)


#: Size floor separating usable raw sample files from processed stubs.  Raw
#: files are ~6.5 MB (90 x 10000 x 2 float16); stubs are a few hundred bytes.
#: The gap is four orders of magnitude, so the exact threshold is not delicate.
USABLE_MIN_BYTES = 100_000


def corpus_dir(name: str = DEFAULT_CORPUS) -> Path:
    return artifact_root() / name


def list_files(directory: Path, min_bytes: int = USABLE_MIN_BYTES
               ) -> List[Tuple[Path, Tuple[float, float, float]]]:
    """Return ``(path, (sd_feat1, sd_feat2, sd_ident))`` for every *usable* sample file.

    Files below ``min_bytes`` are dropped: in the production corpus the vast
    majority of entries are ~400-byte stubs whose raw samples were removed after
    KDE processing, and selecting a subsample before excluding them leaves almost
    nothing usable.  The size check reads only ``scandir`` metadata, so it costs
    no file opens; :func:`load_file` still guards against a truncated survivor.

    Parameters come from the filename, not from ``config``: the corpus extends
    down to 5, below ``config.param_range_low``.
    """
    out = []
    with os.scandir(directory) as entries:
        for e in entries:
            m = _FILE_RE.match(e.name)
            if m and e.stat().st_size >= min_bytes:
                out.append((Path(e.path),
                            (float(m['sf1']), float(m['sf2']), float(m['sp']))))
    return sorted(out, key=lambda t: t[1])


def feat_diff_grid(n_rows: int) -> np.ndarray:
    """The feat_diff values matching a stored array's row count.

    The current format has one row per ``config`` feat_diff value; the oldest
    files carry an extra leading ``feat_diff = 0`` row.
    """
    grid = np.asarray(config.create_grid('feat_diff'))
    if n_rows == len(grid):
        return grid
    if n_rows == len(grid) + 1:
        return np.concatenate([[0.0], grid])
    raise ValueError(f"{n_rows} sample rows do not match the {len(grid)}-point "
                     "feat_diff grid")


def load_file(path: Path, params: Tuple[float, float, float],
              max_sims: Optional[int] = None, rng=None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Read one sample file.

    Returns ``(design, bias)`` with shapes ``(R, 4)`` and ``(R, S, 2)`` — the
    same layout ``sim_interface.simulate`` produces — where ``R`` is the number
    of ``feat_diff`` values kept and ``S <= 10000`` simulations per value.
    """
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
    bias = np.asarray(data['mu1_samples'], dtype=np.float32)
    if bias.ndim != 3 or bias.shape[0] == 0:
        raise ValueError(f"{path.name} holds no usable samples (shape {bias.shape})")

    fd = feat_diff_grid(bias.shape[0])
    keep = fd > 0
    bias, fd = bias[keep], fd[keep]

    if max_sims is not None and max_sims < bias.shape[1]:
        idx = (rng or np.random.default_rng(0)).choice(
            bias.shape[1], size=max_sims, replace=False)
        bias = bias[:, idx]

    sf1, sf2, sp = params
    design = np.column_stack([
        np.full(fd.shape, sf1), np.full(fd.shape, sf2),
        np.full(fd.shape, sp), fd,
    ]).astype(np.float32)
    return design, bias


def load_corpus(directory: Path, n_files: Optional[int] = None,
                max_sims: Optional[int] = 1000, seed: int = 0,
                exclude_params: Optional[Sequence[Tuple[float, float, float]]] = None,
                progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load a random subset of the corpus into one ``(design, bias)`` pair.

    Memory scales as ``n_files * 90 * max_sims * 2 * 4`` bytes, so the defaults
    are deliberately modest; raise ``max_sims`` toward 10,000 once the pipeline
    is known to work.

    ``exclude_params`` drops whole ``(sd_feat1, sd_feat2, sd_ident)`` triples —
    use it to hold out grid points for validation.
    """
    files = list_files(directory)
    if exclude_params:
        excluded = {tuple(map(float, p)) for p in exclude_params}
        files = [f for f in files if f[1] not in excluded]
    rng = np.random.default_rng(seed)
    if n_files is not None and n_files < len(files):
        files = [files[i] for i in rng.choice(len(files), n_files, replace=False)]

    designs, biases = [], []
    for i, (path, params) in enumerate(files):
        try:
            d, b = load_file(path, params, max_sims=max_sims, rng=rng)
        except (ValueError, EOFError, gzip.BadGzipFile, pickle.PickleError) as e:
            print(f"  skipping {path.name}: {e}")
            continue
        designs.append(d)
        biases.append(b)
        if progress and (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(files)} files", flush=True)
    if not designs:
        raise RuntimeError(f"no usable sample files under {directory}")
    return np.concatenate(designs, axis=0), np.concatenate(biases, axis=0)
