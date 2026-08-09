"""Continuous, space-filling parameter designs for the density prototype.

Deliberately *not* restricted to the production 5/10-degree grid: the point of
the prototype is that the emulator accepts arbitrary continuous parameters.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import qmc

#: Domain covered by the pretrained production surface network.
SD_BOUNDS = (5.0, 200.0)
FEAT_DIFF_BOUNDS = (2.0, 180.0)

PARAM_NAMES = ('sd_feat1', 'sd_feat2', 'sd_ident', 'feat_diff')

# Grid used by the unequal-encoding-variability figures.  Only canonical
# ``sd_feat1 <= sd_feat2`` pairs are simulated because each run already returns
# both component biases.
UEV_SD_FEAT = (10.0, 20.0, 30.0, 60.0)
UEV_SPAT_DPRIME = (0.5, 1.0, 2.0)
UEV_SPAT_DIFF = 40.0


def sobol_design(n_points: int, seed: int = 0, sd_scale: str = 'log',
                 sd_bounds: Tuple[float, float] = SD_BOUNDS,
                 feat_diff_bounds: Tuple[float, float] = FEAT_DIFF_BOUNDS):
    """Scrambled Sobol design over ``[sd_feat1, sd_feat2, sd_ident, feat_diff]``.

    Args:
        sd_scale: ``'log'`` spreads the SDs uniformly in log space (the scale
            geometry the network sees), ``'linear'`` matches the production grid's
            uniform spacing.
    """
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random(n_points)
    lo, hi = sd_bounds
    if sd_scale == 'log':
        sd = np.exp(np.log(lo) + u[:, :3] * (np.log(hi) - np.log(lo)))
    elif sd_scale == 'linear':
        sd = lo + u[:, :3] * (hi - lo)
    else:
        raise ValueError(f"sd_scale must be 'log' or 'linear', got {sd_scale!r}")
    d_lo, d_hi = feat_diff_bounds
    feat_diff = d_lo + u[:, 3] * (d_hi - d_lo)
    return np.column_stack([sd, feat_diff]).astype(np.float32)


def low_dprime_augmentation_design(n_points: int, seed: int = 0):
    """Off-grid training design emphasizing poorly separated items.

    With the experiment's 40-degree spatial separation, ``sd_ident=50..140``
    corresponds to spatial d-prime about ``0.8..0.29``. Half the rows target
    unequal feature noise (ratio ``1.5..8``), one quarter target similar noise
    over the full SD domain, and one quarter use independent full-domain SDs.
    The ordinary component mirror augmentation supplies both input orderings.
    """
    u = qmc.Sobol(d=5, scramble=True, seed=seed).random(n_points)
    log_sd = lambda z, lo, hi: np.exp(np.log(lo) + z * (np.log(hi) - np.log(lo)))
    a = log_sd(u[:, 1], *SD_BOUNDS)
    b = log_sd(u[:, 2], *SD_BOUNDS)

    unequal = u[:, 0] < .5
    similar = (u[:, 0] >= .5) & (u[:, 0] < .75)
    low = log_sd(u[:, 1], 5., 40.)
    ratio = log_sd(u[:, 2], 1.5, 8.)
    a[unequal], b[unequal] = low[unequal], np.minimum(
        low[unequal] * ratio[unequal], SD_BOUNDS[1])
    base = log_sd(u[:, 1], *SD_BOUNDS)
    similar_ratio = log_sd(u[:, 2], .8, 1.25)
    a[similar], b[similar] = base[similar], np.clip(
        base[similar] * similar_ratio[similar], *SD_BOUNDS)
    sd_low, sd_high = np.minimum(a, b), np.maximum(a, b)
    sd_ident = log_sd(u[:, 3], 50., 140.)
    feat_diff = FEAT_DIFF_BOUNDS[0] + u[:, 4] * np.diff(FEAT_DIFF_BOUNDS)[0]
    return np.column_stack([sd_low, sd_high, sd_ident, feat_diff]).astype(np.float32)


# ---------------------------------------------------------------------------
# Stratified off-grid validation design
# ---------------------------------------------------------------------------

#: Named strata from the task's validation checklist.  Each entry gives sampling
#: ranges for (sd_feat1, sd_feat2, sd_ident, feat_diff); ``ratio`` strata draw
#: sd_feat2 as a multiple of sd_feat1 instead of independently.
_STRATA: Dict[str, dict] = {
    'low_noise':        dict(sd1=(5, 25), sd2=(5, 25), sp=(5, 60), d=(2, 180)),
    'high_noise':       dict(sd1=(90, 200), sd2=(90, 200), sp=(60, 200), d=(2, 180)),
    'similar_sd_feat':  dict(sd1=(5, 200), ratio=(0.9, 1.1), sp=(5, 200), d=(2, 180)),
    'unequal_sd_feat':  dict(sd1=(5, 60), ratio=(3.0, 12.0), sp=(5, 200), d=(2, 180)),
    'low_sd_ident':     dict(sd1=(5, 200), sd2=(5, 200), sp=(5, 20), d=(2, 180)),
    'high_sd_ident':    dict(sd1=(5, 200), sd2=(5, 200), sp=(120, 200), d=(2, 180)),
    'small_feat_diff':  dict(sd1=(5, 200), sd2=(5, 200), sp=(5, 200), d=(2, 20)),
    'mid_feat_diff':    dict(sd1=(5, 200), sd2=(5, 200), sp=(5, 200), d=(40, 100)),
    'large_feat_diff':  dict(sd1=(5, 200), sd2=(5, 200), sp=(5, 200), d=(140, 180)),
}


def _draw(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    def log_u(lo, hi, size):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size))

    sd1 = log_u(*spec['sd1'], n)
    if 'ratio' in spec:
        sd2 = np.clip(sd1 * log_u(*spec['ratio'], n), *SD_BOUNDS)
    else:
        sd2 = log_u(*spec['sd2'], n)
    sp = log_u(*spec['sp'], n)
    d = rng.uniform(*spec['d'], n)
    return np.column_stack([sd1, sd2, sp, d]).astype(np.float32)


def validation_design(per_stratum: int = 12, seed: int = 20260808
                      ) -> Tuple[np.ndarray, List[str]]:
    """Stratified continuous validation design that is disjoint from any grid.

    Attraction / repulsion / multimodality cannot be specified a priori, so they
    are not strata: they are *labelled after the fact* from the reference
    simulations by ``label_cases``.

    Returns ``(design, stratum_names)``.
    """
    rng = np.random.default_rng(seed)
    blocks, names = [], []
    for name, spec in _STRATA.items():
        blocks.append(_draw(rng, spec, per_stratum))
        names.extend([name] * per_stratum)
    return np.concatenate(blocks, axis=0), names


def circular_kde(bias_samples, grid, kappa: float = 40.0,
                 chunk: int = 5000) -> np.ndarray:
    """Reporting-only circular KDE via linear deposition and FFT convolution.

    ``bias_samples`` is ``(M, S)`` in degrees; non-finite EM failures are
    excluded. Samples are deposited linearly onto the uniform periodic grid and
    convolved with a sampled von-Mises kernel. This is first-order accurate in
    the reporting-grid spacing and costs ``O(S + G log G)`` per case instead of
    materialising or evaluating a ``G x S`` kernel matrix. ``chunk`` is retained
    for API compatibility and ignored.
    """
    del chunk
    b = np.asarray(bias_samples, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError('grid must be a one-dimensional periodic grid')
    dx = float(grid[1] - grid[0])
    if not np.allclose(np.diff(grid), dx) or not np.isclose(dx * grid.size, 360.0):
        raise ValueError('grid must be uniform and cover one 360-degree period')
    M, _ = b.shape
    kernel = np.exp(kappa * (np.cos(np.radians(np.arange(grid.size) * dx)) - 1.0))
    kernel_fft = np.fft.rfft(kernel)
    dens = np.empty((M, grid.size))
    for i in range(M):
        samples = b[i, np.isfinite(b[i])]
        if not len(samples):
            dens[i] = np.nan
            continue
        pos = np.mod(samples - grid[0], 360.0) / dx
        lo = np.floor(pos).astype(np.int64) % grid.size
        frac = pos - np.floor(pos)
        hist = np.zeros(grid.size)
        np.add.at(hist, lo, 1.0 - frac)
        np.add.at(hist, (lo + 1) % grid.size, frac)
        smooth = np.fft.irfft(np.fft.rfft(hist) * kernel_fft, n=grid.size)
        dens[i] = smooth / (smooth.sum() * dx)
    return dens


# Difficult fixed-SD regimes established by the earlier mu1 surrogate
# experiments.  Their archived 100k outcomes survive only as KDE surfaces, so
# these anchors are perturbed off-grid and freshly re-simulated here.
_STRESS_TRIPLES = (
    ('peaked', 5, 55, 5), ('peaked', 5, 190, 5), ('peaked', 10, 25, 5),
    ('flat', 190, 195, 110), ('flat', 190, 195, 200), ('flat', 190, 195, 10),
    ('seam', 15, 20, 200), ('seam', 15, 20, 170), ('seam', 30, 185, 200),
    ('asymmetric', 30, 30, 200), ('asymmetric', 30, 30, 170),
    ('asymmetric', 5, 55, 200), ('multimodal', 10, 20, 175),
    ('multimodal', 10, 20, 135), ('multimodal', 15, 45, 200),
)


def stress_trajectory_design(n_curves: int = 15, points_per_curve: int = 24,
                             seed: int = 20260808
                             ) -> Tuple[np.ndarray, List[str]]:
    """Off-grid difficult SD triples crossed with bounded feat-diff trajectories.

    This complements :func:`validation_design`: the latter tests scattered 4-D
    generalisation, whereas this design measures shape and second differences as
    ``feat_diff`` changes at fixed SDs.  Uniform feature spacing retains the exact
    2/180-degree boundaries and permits an interpretable residual second-
    difference metric; intermediate values are off the production 2-degree grid
    for the usual trajectory sizes.
    """
    if not 1 <= n_curves <= len(_STRESS_TRIPLES):
        raise ValueError(f'n_curves must be in [1, {len(_STRESS_TRIPLES)}]')
    if points_per_curve < 3:
        raise ValueError('points_per_curve must be at least 3')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(_STRESS_TRIPLES), n_curves, replace=False)
    feat = np.linspace(*FEAT_DIFF_BOUNDS, points_per_curve)
    rows, labels = [], []
    for index in chosen:
        scenario, *triple = _STRESS_TRIPLES[index]
        triple = np.asarray(triple, dtype=np.float64)
        # Deterministic sub-degree perturbations make every SD triple off-grid.
        delta = np.array([0.37, -0.61, 0.43])
        delta *= np.where(triple >= SD_BOUNDS[1] - 1, -1.0, 1.0)
        sd = np.clip(triple + delta, *SD_BOUNDS)
        rows.append(np.column_stack([np.repeat(sd[None, :], len(feat), axis=0), feat]))
        labels.extend([f'trajectory_{scenario}_{index + 1:02d}'] * len(feat))
    return np.concatenate(rows).astype(np.float32), labels


def uev_design(feature_step: float = 2.0) -> Tuple[np.ndarray, List[str]]:
    """Canonical grid for raw-vs-density unequal-variability bias curves.

    Spatial discriminability follows the experimental definition
    ``dprime = spatial separation / sd_ident`` with a 40-degree separation, so
    the three d-prime levels map to ``sd_ident = 80, 40, 20`` degrees.  The
    feature grid includes both trained-domain endpoints.
    """
    if feature_step <= 0:
        raise ValueError('feature_step must be positive')
    lo, hi = FEAT_DIFF_BOUNDS
    n_steps = (hi - lo) / feature_step
    if not np.isclose(n_steps, round(n_steps)):
        raise ValueError('feature_step must divide the 2-to-180 degree interval')
    feat = np.linspace(lo, hi, int(round(n_steps)) + 1)
    rows, labels = [], []
    for i, sd1 in enumerate(UEV_SD_FEAT):
        for sd2 in UEV_SD_FEAT[i:]:
            for dprime in UEV_SPAT_DPRIME:
                sd_ident = UEV_SPAT_DIFF / dprime
                rows.append(np.column_stack([
                    np.full(len(feat), sd1), np.full(len(feat), sd2),
                    np.full(len(feat), sd_ident), feat]))
                labels.extend([f'uev_dprime_{dprime:g}'] * len(feat))
    return np.concatenate(rows).astype(np.float32), labels


def circular_modes(density, grid, min_mass: float = 0.05) -> Dict[str, np.ndarray]:
    """Modes of one circular density, as basin count / locations / masses.

    A mode is a local maximum; its *mass* is the integral of the density over the
    basin bounded by the two neighbouring local minima (the arcs between
    successive minima tile the circle, so the masses sum to 1).  Modes whose
    basin mass is below ``min_mass`` are dropped as noise.  Returned modes are
    sorted by mass, descending, so ``locations[0]``/``masses[0]`` is the primary
    mode and ``[1:]`` are the secondary modes.
    """
    p = np.asarray(density, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    tot = p.sum()
    empty = dict(n_modes=0, locations=np.zeros(0), masses=np.zeros(0))
    if not np.isfinite(tot) or tot <= 0:
        return empty
    p = p / tot
    left, right = np.roll(p, 1), np.roll(p, -1)
    is_min = (p <= left) & (p <= right) & ((p < left) | (p < right))
    mins = np.flatnonzero(is_min)
    if mins.size == 0:  # perfectly flat or single monotone basin: one mode
        loc = int(np.argmax(p))
        return dict(n_modes=1, locations=grid[[loc]], masses=np.array([1.0]))
    G = p.size
    locs, masses = [], []
    for i in range(mins.size):
        a, b = mins[i], mins[(i + 1) % mins.size]
        idx = (np.arange(a + 1, b + 1) if a < b
               else np.concatenate([np.arange(a + 1, G), np.arange(0, b + 1)]))
        if idx.size == 0:
            continue
        masses.append(p[idx].sum())
        locs.append(idx[np.argmax(p[idx])])
    masses, locs = np.asarray(masses), np.asarray(locs)
    keep = masses >= min_mass
    masses, locs = masses[keep], locs[keep]
    order = np.argsort(masses)[::-1]
    return dict(n_modes=int(masses.size), locations=grid[locs[order]],
                masses=masses[order])


def secondary_mode_summary(mode_dicts) -> Dict[str, np.ndarray]:
    """Compact per-case secondary-mode arrays from a list of :func:`circular_modes`.

    Returns ``n_modes``, ``secondary_loc`` (degrees; ``nan`` if unimodal) and
    ``secondary_mass`` (basin mass of the largest secondary mode; 0 if unimodal).
    """
    n = np.array([m['n_modes'] for m in mode_dicts])
    loc = np.array([m['locations'][1] if m['n_modes'] > 1 else np.nan
                    for m in mode_dicts])
    mass = np.array([m['masses'][1] if m['n_modes'] > 1 else 0.0
                     for m in mode_dicts])
    return {'n_modes': n, 'secondary_loc': loc, 'secondary_mass': mass}


def label_cases(bias_samples, n_modes_grid: int = 360, kappa: float = 40.0,
                min_mode_mass: float = 0.05, chunk: int = 5000
                ) -> Dict[str, np.ndarray]:
    """Describe each reference distribution: direction, dispersion, modality.

    Args:
        bias_samples: ``(M, S)`` raw reference biases in degrees.  Non-finite EM
            outcomes are excluded from every quantity.

    Returns dict with ``mean_bias`` (degrees), ``resultant``, ``n_modes``,
    ``secondary_loc``/``secondary_mass`` and a boolean ``multimodal`` flag.
    ``mean_bias > 0`` is attraction under the project's sign convention, ``< 0``
    repulsion.

    The circular kernel smoother here exists only to *describe* modes for
    reporting; it is never used as a target or in any loss.
    """
    b = np.asarray(bias_samples, dtype=np.float64)
    finite = np.isfinite(b)
    z = np.where(finite, np.exp(1j * np.radians(np.where(finite, b, 0.0))), 0.0)
    count = finite.sum(axis=1)
    m1 = z.sum(axis=1) / np.maximum(count, 1)
    m1 = np.where(count > 0, m1, np.nan + 1j * np.nan)

    grid = np.linspace(-180.0, 180.0, n_modes_grid, endpoint=False)
    dens = circular_kde(b, grid, kappa=kappa, chunk=chunk)
    modes = [circular_modes(d, grid, min_mode_mass) for d in dens]
    summary = secondary_mode_summary(modes)

    return {
        'mean_bias': np.degrees(np.angle(m1)),
        'resultant': np.abs(m1),
        'n_modes': summary['n_modes'],
        'secondary_loc': summary['secondary_loc'],
        'secondary_mass': summary['secondary_mass'],
        'multimodal': summary['n_modes'] > 1,
    }
