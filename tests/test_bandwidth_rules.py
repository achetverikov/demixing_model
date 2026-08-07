"""The Sheather-Jones bandwidth must reproduce R's `stats::bw.SJ`.

scipy has no SJ, so `sheather_jones_bandwidth` is a port. R's own documentation
recommends SJ over the Silverman rule it ships as the default ("nrd0 has remained
the default for historical and compatibility reasons, rather than as a general
recommendation, where e.g. \"SJ\" would rather fit", ?density), so the port has to
be faithful rather than merely SJ-flavoured.

Expected values below were produced by R 4.6.0 and are pinned to 10 decimals. The
tolerance is R's own uniroot tolerance (tol = 0.1 * lower = 0.01 * hmax), which is
the limit of what R itself resolves; agreement within it is exact agreement. The
port was additionally checked against bw.SJ on all 452 real subject x condition
cells in example_data: median relative difference 0.000000%, p95 0.106%, max 0.589%.
"""
import pathlib

import numpy as np
import pytest

from shared.utils import (KDE_BW_FLOOR, sheather_jones_bandwidth,
                          silverman_bandwidth)

def _load_reference():
    """The exact samples R saw, vendored alongside the answers R gave."""
    path = pathlib.Path(__file__).parent / "data" / "bw_sj_reference.tsv"
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("sample\t"):
                continue
            name, sj, nrd0, values = line.rstrip("\n").split("\t")
            out[name] = np.array([float(v) for v in values.split()])
            out[f"{name}_sj"] = float(sj)
            out[f"{name}_nrd0"] = float(nrd0)
    return out


_REFERENCE = _load_reference()


def _r_sample(name):
    return _REFERENCE[name]


@pytest.mark.parametrize("name", [
    "x1",   # n=200, unimodal
    "x2",   # n=460, bimodal + heavy tail
    "x3",   # n=1200, exercises R's binned path
])
def test_matches_r_bw_sj(name):
    x = _r_sample(name)
    expected_sj = _REFERENCE[f"{name}_sj"]
    expected_nrd0 = _REFERENCE[f"{name}_nrd0"]
    got = sheather_jones_bandwidth(x)
    # R's own root tolerance is 0.01 * hmax; scale it to a relative bound.
    assert got == pytest.approx(expected_sj, rel=0.01), f"{name}: {got} vs R {expected_sj}"
    # Silverman is exercised on the same samples so the two rules cannot silently
    # converge on one implementation.
    assert float(silverman_bandwidth(x)) == pytest.approx(expected_nrd0, rel=1e-6)


def test_sj_and_silverman_actually_differ():
    """Guard against the rule switch quietly becoming a no-op."""
    x = _r_sample("x1")
    assert abs(sheather_jones_bandwidth(x) / float(silverman_bandwidth(x)) - 1) > 0.05


def test_degenerate_sample_falls_back_not_raises():
    """A fitting pipeline must not die on one degenerate cell; R raises here."""
    x = np.full(50, 3.0)
    with pytest.warns(RuntimeWarning, match="Sheather-Jones"):
        bw = sheather_jones_bandwidth(x)
    # Falls back to Silverman, which for identical values lands on the floor.
    # Compared loosely because the jnp path is float32, so the floor comes back as
    # 9.99999997e-07 rather than exactly 1e-6.
    assert np.isfinite(bw) and bw > 0
    assert bw == pytest.approx(KDE_BW_FLOOR, rel=1e-5)


def test_silverman_floor_guards_divide_by_zero():
    """D.11: identical values drive Silverman to 0 and the kernel divides by it."""
    assert float(silverman_bandwidth(np.full(20, 1.0))) == pytest.approx(KDE_BW_FLOOR)


def test_sj_is_positive_and_finite_on_real_shapes():
    rng = np.random.default_rng(3)
    for n in (30, 200, 900):
        x = rng.normal(0, 15, n)
        bw = sheather_jones_bandwidth(x)
        assert np.isfinite(bw) and 0 < bw < np.ptp(x)
