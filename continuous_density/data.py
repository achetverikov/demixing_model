"""Minibatch sampling over raw simulator outcomes.

Keeps the compact ``(design, bias)`` layout — ``(M, 4)`` parameters and
``(M, S, 2)`` per-simulation biases — and materialises training rows only per
batch, so the mirror augmentation costs no memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class SampleStore:
    """Raw EM outcomes with on-the-fly mirror augmentation.

    Each simulation supplies two observations: the component-1 bias at
    ``(sd_feat1, sd_feat2, sd_ident, feat_diff)`` and the component-2 bias, which
    by the exchangeability of the two components is a draw from the same
    conditional density at the mirrored parameter vector.
    """

    def __init__(self, design, bias):
        self.design = np.asarray(design, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        if self.bias.shape[0] != self.design.shape[0] or self.bias.ndim != 3:
            raise ValueError(f"bias {self.bias.shape} does not match design "
                             f"{self.design.shape}")
        self.mirrored = self.design[:, [1, 0, 2, 3]]
        self.finite = np.isfinite(self.bias)

    @property
    def n_rows(self) -> int:
        return self.design.shape[0]

    @property
    def n_observations(self) -> int:
        return int(self.finite.sum())

    def param_key(self) -> np.ndarray:
        """Row-wise ``(sd_feat1, sd_feat2, sd_ident)`` triple, for reporting."""
        return self.design[:, :3]

    def canonical_param_key(self) -> np.ndarray:
        """Mirror-invariant ``(min(sd_feat1, sd_feat2), max(...), sd_ident)`` triple.

        The mirror augmentation turns a row at ``(sd1, sd2, sp)`` into a training
        observation at ``(sd2, sd1, sp)`` as well, so ``(sd1, sd2, sp)`` and
        ``(sd2, sd1, sp)`` are the *same* group as far as leakage is concerned.
        Splitting on this canonical key is what keeps a mirrored twin from landing
        in validation while its original trains (see :func:`split_by_params`).
        """
        sd = self.design[:, :2]
        return np.column_stack([sd.min(axis=1), sd.max(axis=1), self.design[:, 2]])

    def subset(self, row_idx) -> "SampleStore":
        return SampleStore(self.design[row_idx], self.bias[row_idx])

    def batch(self, rng: np.random.Generator, batch_size: int
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``(x, b, weight)``; ``weight`` is 0 for non-finite EM outcomes."""
        m = rng.integers(0, self.bias.shape[0], batch_size)
        j = rng.integers(0, self.bias.shape[1], batch_size)
        c = rng.integers(0, 2, batch_size)
        x = np.where(c[:, None] == 0, self.design[m], self.mirrored[m])
        b = self.bias[m, j, c]
        w = self.finite[m, j, c].astype(np.float32)
        return x, np.nan_to_num(b).astype(np.float32), w

    def all_rows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every finite observation as flat ``(x, b)`` arrays. Use on small stores."""
        x = np.concatenate([
            np.repeat(self.design, self.bias.shape[1], axis=0),
            np.repeat(self.mirrored, self.bias.shape[1], axis=0)])
        b = np.concatenate([self.bias[:, :, 0].ravel(), self.bias[:, :, 1].ravel()])
        keep = np.isfinite(b)
        return x[keep], b[keep]


def split_by_params(store: SampleStore, val_fraction: float = 0.1, seed: int = 0
                    ) -> Tuple[SampleStore, SampleStore]:
    """Split rows by mirror-invariant parameter group, never by simulation.

    Holding out whole parameter combinations is the only split that measures what
    matters here — generalisation of the parameter-to-density map.  Splitting
    simulations instead would leak the target distribution into validation.

    The grouping key is :meth:`SampleStore.canonical_param_key`, which is
    invariant under swapping ``sd_feat1`` and ``sd_feat2``.  Grouping on the raw
    triple instead would let the mirror augmentation of a training row (which is
    an observation at the swapped parameter vector) reproduce a held-out
    validation combination — the exact leak this function must not allow.
    """
    keys = store.canonical_param_key()
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    if len(uniq) < 2:
        raise ValueError('at least two mirror-distinct parameter groups are required')
    rng = np.random.default_rng(seed)
    n_val = min(len(uniq) - 1, max(1, int(round(val_fraction * len(uniq)))))
    val_keys = set(rng.choice(len(uniq), n_val, replace=False).tolist())
    is_val = np.isin(inverse.ravel(), list(val_keys))
    return store.subset(~is_val), store.subset(is_val)


def load_npz(path) -> Tuple[SampleStore, dict]:
    """Load a file written by ``generate_training_data.py``."""
    blob = np.load(Path(path), allow_pickle=True)
    meta = json.loads(str(blob['meta']))
    if 'strata' in blob and blob['strata'].size:
        meta['strata'] = [str(s) for s in blob['strata']]
    return SampleStore(blob['design'], blob['bias']), meta


def load_source(source: str, corpus_files: Optional[int] = None,
                corpus_sims: Optional[int] = 1000, seed: int = 0,
                progress: bool = False) -> Tuple[SampleStore, dict]:
    """Load training data from either an ``.npz`` file or a sample corpus directory."""
    path = Path(source)
    if path.suffix == '.npz':
        return load_npz(path)

    from continuous_density import existing_samples
    directory = path if path.is_dir() else existing_samples.corpus_dir(source)
    design, bias = existing_samples.load_corpus(
        directory, n_files=corpus_files, max_sims=corpus_sims, seed=seed,
        progress=progress)
    return SampleStore(design, bias), {'source': str(directory),
                                       'corpus_files': corpus_files,
                                       'corpus_sims': corpus_sims,
                                       'n_samples': 100,
                                       'on_grid': True}


def model_n_samples(meta: dict) -> Optional[int]:
    """Observer sample count recorded in a density checkpoint's training metadata."""
    value = meta.get('source_meta', {}).get('n_samples')
    return int(value) if value is not None else None


def require_matching_n_samples(model_meta: dict, reference_meta: dict) -> Optional[int]:
    """Reject comparisons between distinct 20/100-observation estimands."""
    model_n = model_n_samples(model_meta)
    ref_n = reference_meta.get('n_samples')
    ref_n = int(ref_n) if ref_n is not None else None
    if model_n is not None and ref_n is not None and model_n != ref_n:
        raise ValueError(f'density model was trained for n_samples={model_n}, but '
                         f'the reference uses n_samples={ref_n}')
    return model_n if model_n is not None else ref_n
