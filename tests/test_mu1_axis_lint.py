"""Lint: nobody may reconstruct the mu1_bias axis by hand.

The accessor in ``shared/mu1_axis.py`` is only "unfakeable" if something
enforces it.  Six audit rounds of the circularity fix each found sites the
previous round had missed (1 -> 5 -> 6 -> 12 -> 19+ -> 23+), and the count never
converged, because the failure mode is invisible: code that keeps running and
returns slightly wrong numbers.  Enumerating the inventory by inspection does
not work; this lint is the mechanical closure criterion in its place.

A new violation is a **new** entry in ALLOWED below.  Adding one is a decision,
not a formality: read what the pattern means first.
"""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SEARCH_DIRS = [
    "shared", "model_fit_to_data", "neural_network_optimization",
    "surface_computation", "surface_simulator_for_predictions",
    "surface_browser", "scripts", "cloud",
]

# (name, regex, why it is dangerous)
PATTERNS = [
    ("inclusive_mu1_arange",
     re.compile(r"arange\(\s*-180[^)]*180\s*\+"),
     "an inclusive arange over the mu1 range duplicates ±180 (histogram EDGES "
     "are the one legitimate exception — n+1 edges for n bins)"),
    ("mu1_linspace",
     re.compile(r"linspace\(\s*-?\s*(180|max_diss|mu1_error_min|mu1_bias_range)"),
     "reconstructing the angular axis with linspace gives an inclusive grid "
     "(2.0112° step, phantom +180 sample)"),
    ("inclusive_step_formula",
     re.compile(r"/\s*\(\s*n_mu1_points\s*-\s*1\s*\)"),
     "(max-min)/(n-1) is the inclusive-grid cell width; the periodic one is "
     "period/n (use mu1_cell_width())"),
    ("mu1_trapezoid",
     re.compile(r"trapezoid\([^)]*x\s*=\s*mu1_bias_grid"),
     "trapezoid over the mu1 axis spans 358° of the 360° period and "
     "half-weights the ends (use periodic_integral())"),
    ("mu1_sign_split",
     re.compile(r"mu1_bias_grid\s*[<>]\s*0"),
     "a bare sign mask counts the antipode (-180) as negative (use sign_masks())"),
    ("mu1_bin_clip",
     re.compile(r"clip\([^)]*(bias_indices|bias_bin)"),
     "clipping the circular bias axis sends (179, 180) onto the last row "
     "instead of wrapping to bin 0 (use bin_indices())"),
    # Only in mu1/surface context: 181 is also a legitimate inclusive bound on
    # unrelated bounded axes (e.g. arange(0, 181, 5) over feat distances).
    ("literal_181",
     re.compile(r"(?<![\w.])181(?![\w.])"),
     "a hardcoded legacy row count"),
]

LITERAL_181_CONTEXT = re.compile(r"mu1|bias|surface|shape|grid", re.IGNORECASE)

# Deliberate, reviewed exceptions: path suffix -> pattern names allowed in it.
ALLOWED = {
    # Defines the axis and the legacy constant; talks about both conventions.
    "shared/mu1_axis.py": {"literal_181", "mu1_linspace", "inclusive_mu1_arange",
                           "mu1_sign_split", "inclusive_step_formula",
                           "mu1_trapezoid", "mu1_bin_clip"},
    # Legacy-checkpoint route: names the legacy row count on purpose.
    "shared/utils.py": {"literal_181"},
    "shared/config.py": {"literal_181"},
    # The legacy branch of normalize_to_density_flexible reproduces the old
    # (max-min)/(n-1) width on purpose, so a legacy checkpoint's forward pass is
    # bit-exact before its output is trimmed.
    "neural_network_optimization/mirror_aware_model.py": {"inclusive_step_formula"},
    # Migration script: its whole job is the legacy representation.
    "cloud/migrate_surfaces_to_periodic_mu1.py": {"literal_181"},
    # Histogram EDGES, not grid points: the closing +180 edge is correct there.
    "shared/averaging.py": {"inclusive_mu1_arange"},
    # mu2_bias is a linear, bounded axis — inclusive quadrature is right for it.
    "surface_simulator_for_predictions/surface_simulator.py": set(),
    "surface_browser/core/data_manager.py": {"literal_181"},
}


def _iter_sources():
    for directory in SEARCH_DIRS:
        root = REPO / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            yield path


def _violations():
    found = []
    for path in _iter_sources():
        rel = path.relative_to(REPO).as_posix()
        allowed = ALLOWED.get(rel, set())
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            for name, regex, why in PATTERNS:
                if name in allowed:
                    continue
                if name == "literal_181":
                    # Docstrings count here: a stale shape in a docstring is how
                    # the next person learns the wrong convention.
                    if not (regex.search(line) and LITERAL_181_CONTEXT.search(line)):
                        continue
                elif not regex.search(code):
                    continue
                found.append((rel, lineno, name, why, line.strip()))
    return found


def test_no_hand_rolled_mu1_axis_arithmetic():
    found = _violations()
    if found:
        report = "\n".join(
            f"  {rel}:{lineno}  [{name}] {why}\n      {line}"
            for rel, lineno, name, why, line in found)
        pytest.fail(
            "The mu1_bias axis must come from shared/mu1_axis.py, never be "
            "reconstructed:\n" + report)


if __name__ == "__main__":
    for row in _violations():
        print("%s:%d [%s] %s" % row[:4])
    sys.exit(1 if _violations() else 0)
