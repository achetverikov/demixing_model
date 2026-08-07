"""The single supported source of truth for the circular ``mu1_bias`` axis.

The mu1_bias axis is a **circle**, not an interval: -180° and +180° are the same
angle.  It is represented half-open — ``-180, -178, …, 178`` (180 points, step
2°) — so every physical angle occupies exactly one cell and the period is
``mu1_bias_range[1] - mu1_bias_range[0] = 360°``.

Historically the grid was built inclusively (181 points, both ±180), and a long
tail of call sites *reconstructed* the axis by hand — ``linspace(min, max,
shape[1])``, ``(max-min)/(n-1)``, ``trapezoid(x=grid)``, ``clip``-based binning,
``grid > 0`` sign masks.  Every one of those is correct at 181 points and
*silently* wrong at 180: no crash, just slightly wrong numbers.  This module
exists so that a defect becomes "someone did not use the accessor" — greppable
and finite — rather than "someone reconstructed the axis wrongly".

Use these helpers.  Do not rebuild the axis from an array's row count, and do
not derive the step as ``(grid[-1] - grid[0]) / n`` (that gives 358/180 = 1.989
on a half-open grid — wrong in a new way).
"""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from shared.config import config

#: Number of rows the legacy (inclusive, dual-endpoint) representation had.
LEGACY_MU1_GRID_SIZE = 181

#: Value stored in new checkpoints' metadata to declare the periodic axis.
GRID_CONVENTION = "periodic180"


@contextlib.contextmanager
def legacy_mu1_axis():
    """Temporarily restore the pre-fix inclusive 181-point mu1 axis.

    The *only* legitimate use is running a legacy checkpoint at the row count it
    was trained on, so that its output can be **trimmed** (never interpolated)
    back onto the periodic grid.  Do not use it to write data.
    """
    previous = config.mu1_inclusive_legacy
    config.mu1_inclusive_legacy = True
    try:
        yield
    finally:
        config.mu1_inclusive_legacy = previous


def mu1_period() -> float:
    """Angular period of the mu1_bias axis in degrees (360.0)."""
    lo, hi = config.mu1_bias_range
    return float(hi - lo)


def mu1_step() -> float:
    """Spacing between adjacent mu1_bias grid points, in degrees."""
    return float(config.mu1_bias_step)


def mu1_size() -> int:
    """Number of mu1_bias grid points (180 under the periodic representation)."""
    return int(config.mu1_bias_grid_size)


def mu1_grid() -> jnp.ndarray:
    """The mu1_bias grid: half-open, ``-180 … 178``, step 2°."""
    return config.create_grid('mu1_bias')


def mu1_grid_np() -> np.ndarray:
    """The mu1_bias grid as plain NumPy.

    ``mu1_grid()`` builds a ``jnp.arange``, which becomes a *tracer* when called
    inside a jit-traced function — unusable for the concrete comparisons the
    assertions below make. This twin is always concrete.
    """
    lo, hi = config.mu1_bias_range
    return np.arange(lo, hi + (mu1_step() if config.mu1_inclusive_legacy else 0.0),
                     mu1_step(), dtype=float)


def mu1_cell_width() -> float:
    """Width of one mu1_bias cell, ``period / n``.

    Equals :func:`mu1_step` for the standard grid; kept distinct because the
    *periodic* cell width is ``period / n`` and not the inclusive-grid
    ``(max - min) / (n - 1)``.
    """
    return mu1_period() / mu1_size()


def mu1_antipode() -> float:
    """The sign-ambiguous row opposite 0° (-180.0)."""
    return -mu1_period() / 2.0


def assert_mu1_axis(grid_or_n, *, name: str = "mu1_bias axis") -> None:
    """Assert an axis really is *the* mu1_bias axis — values, not just length.

    A row-count check passes on a 180-point grid holding *wrong values* (e.g. an
    inclusive ``linspace`` with a 2.0112 step), which is exactly the silent
    failure this migration guards against.  Accepts either an array of grid
    values or an integer row count; an integer can only be checked for size, so
    pass the values whenever they exist.
    """
    expected = mu1_grid_np()

    if isinstance(grid_or_n, (int, np.integer)):
        n = int(grid_or_n)
        if n != expected.size:
            raise ValueError(
                f"{name}: {n} rows, expected {expected.size}. "
                f"{_legacy_hint(n)}")
        return

    grid = np.asarray(grid_or_n, dtype=float).reshape(-1)
    if grid.size != expected.size:
        raise ValueError(
            f"{name}: {grid.size} points, expected {expected.size}. "
            f"{_legacy_hint(grid.size)}")
    if not np.allclose(grid, expected, atol=1e-6):
        raise ValueError(
            f"{name}: grid values differ from config "
            f"(first={grid[0]}, last={grid[-1]}, step={grid[1] - grid[0]}; "
            f"expected first={expected[0]}, last={expected[-1]}, "
            f"step={expected[1] - expected[0]})")


def _legacy_hint(n: int) -> str:
    if n == LEGACY_MU1_GRID_SIZE:
        return ("This is the legacy dual-endpoint (±180) representation; run "
                "cloud/migrate_surfaces_to_periodic_mu1.py to convert stored "
                "surfaces.")
    return ""


def guard_surface_mu1_axis(surface_obj, source: str = "surface") -> None:
    """Raise unless a loaded surface is on the periodic mu1 axis.

    A **guard, not a shim**: legacy 181-row surfaces are migrated once on disk
    (``cloud/migrate_surfaces_to_periodic_mu1.py``), never converted at load
    time, so anything still carrying the old axis is an error rather than
    something to silently reinterpret — and, critically, never something to
    silently *skip*.
    """
    grid = getattr(surface_obj, "mu1_bias_grid", None)
    if grid is not None:
        assert_mu1_axis(grid, name=f"{source}: embedded mu1_bias_grid")
    rows = getattr(surface_obj, "mu1_comp1_surface", None)
    if rows is not None:
        assert_mu1_axis(rows.shape[0], name=f"{source}: mu1_comp1_surface rows")


def sign_masks(grid: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Boolean (positive, negative) masks for a signed split of the mu1 axis.

    Both 0° and the antipode (-180°) are excluded from *both* masks: they are
    the two angles that are not on one side or the other.  Counting the antipode
    as negative — the naive result of dropping the old duplicate row — injects a
    spurious asymmetry at exactly the row that carries mass for broad
    ``sd_feat``.
    """
    if grid is None:
        grid = mu1_grid()
    grid = jnp.asarray(grid)
    half = mu1_period() / 2.0
    ambiguous = jnp.isclose(jnp.abs(grid), half, atol=1e-6)
    positive = (grid > 0) & ~ambiguous
    negative = (grid < 0) & ~ambiguous
    return positive, negative


def periodic_integral(values: jnp.ndarray, axis: int = 0,
                      cell_width: Optional[float] = None) -> jnp.ndarray:
    """Integrate over the mu1_bias axis on a periodic, uniformly-spaced grid.

    On a circle the correct quadrature is a plain sum times the cell width.
    ``jnp.trapezoid(x=grid)`` half-weights the two ends and spans only
    ``grid[-1] - grid[0]`` = 358° of the 360° period.
    """
    if cell_width is None:
        cell_width = mu1_cell_width()
    return jnp.sum(values, axis=axis) * cell_width


def bin_indices(bias_deg, n: Optional[int] = None) -> jnp.ndarray:
    """Map bias angles (degrees) to mu1_bias grid indices, wrapping not clipping.

    ``+180`` and ``-180`` are the same angle and land in the same bin; angles in
    ``(179, 180)`` wrap to bin 0 rather than being clipped onto the last row.
    """
    if n is None:
        n = mu1_size()
    lo = config.mu1_bias_range[0]
    step = mu1_step()
    idx = jnp.round((jnp.asarray(bias_deg) - lo) / step).astype(jnp.int32)
    return jnp.mod(idx, n)


def bin_indices_np(bias_deg, n: Optional[int] = None) -> np.ndarray:
    """NumPy twin of :func:`bin_indices` for the non-JAX post-processing paths."""
    if n is None:
        n = mu1_size()
    lo = config.mu1_bias_range[0]
    step = mu1_step()
    idx = np.round((np.asarray(bias_deg, dtype=np.float32) - np.float32(lo))
                   / np.float32(step)).astype(np.int64)
    return np.mod(idx, n)


def is_legacy_mu1_length(n: int) -> bool:
    """True if ``n`` is the legacy dual-endpoint row count."""
    return int(n) == LEGACY_MU1_GRID_SIZE


def trim_legacy_rows(rows: jnp.ndarray, axis: int = 0,
                     *, check_endpoints: bool = True,
                     atol: float = 0.0, name: str = "surface") -> jnp.ndarray:
    """Drop the duplicated ``+180`` row from a legacy 181-row array.

    This is a **trim**, never a resize: the retained rows are bit-preserved, so
    their ratios are unchanged (that is exactly what distinguishes a trim from an
    interpolation).  Renormalisation, where required, is the caller's job.

    Raises if the two endpoint rows are not identical — they are the same angle,
    so any difference means the array is not what this migration assumes.
    """
    arr = jnp.asarray(rows)
    n = arr.shape[axis]
    if n == mu1_size():
        return arr
    if not is_legacy_mu1_length(n):
        raise ValueError(
            f"{name}: {n} rows on the mu1_bias axis; expected {mu1_size()} "
            f"(periodic) or {LEGACY_MU1_GRID_SIZE} (legacy).")

    first = jnp.take(arr, jnp.array([0]), axis=axis)
    last = jnp.take(arr, jnp.array([n - 1]), axis=axis)
    if check_endpoints:
        max_diff = float(jnp.max(jnp.abs(first - last)))
        if not (max_diff <= atol):
            raise ValueError(
                f"{name}: legacy endpoint rows -180 and +180 differ by "
                f"{max_diff:g} (> {atol:g}). They are the same angle; refusing "
                f"to trim data that does not satisfy the identity.")
    return jnp.take(arr, jnp.arange(n - 1), axis=axis)


def trim_legacy_grid(grid) -> np.ndarray:
    """Trim an embedded legacy 181-value mu1 grid to the periodic 180 values.

    Surface pickles carry their own ``mu1_bias_grid``, and
    ``Surface.__post_init__`` validates array shapes against *that* grid — so it
    must be trimmed in lockstep with the rows.
    """
    arr = np.asarray(grid)
    if arr.size == mu1_size():
        return arr
    if not is_legacy_mu1_length(arr.size):
        raise ValueError(
            f"embedded mu1_bias_grid has {arr.size} points; expected "
            f"{mu1_size()} or {LEGACY_MU1_GRID_SIZE}.")
    return arr[:-1]
