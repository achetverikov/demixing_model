"""The curve-level CRPS objectives predict on the observed design, like their targets.

`target_d` is built from `q_emp`, the design-weighted empirical mixture at each
feature-grid point. The energy score is proper against the target it is handed,
so its minimum over predictions sits at `p_f = q_emp_f`. A prediction evaluated
at one bare grid coordinate is a different object and cannot reach that minimum
for any parameter value -- the objective's argmin is displaced, not merely its
scale. That is what these tests pin.

They run through `score_condition`, the branch the fitter minimises and the same
branch `evaluate_parameter_losses` exports, rather than through the energy helper
with already-pooled arrays handed to it. The helper was always correct; pinning
it is what let the operator gap survive.

The conditional distribution is supplied analytically and varies with
dissimilarity. That is deliberate: it makes the prediction *identical* on both
sides of every comparison here, so a difference can only come from the operator.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "model_fit_to_data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

pytestmark = pytest.mark.integration
CBD_ROOT = ROOT.parent / "contextual_biases_database"
CBD_SRC = CBD_ROOT / "src"
if not CBD_SRC.exists():
    pytest.skip(
        "requires sibling contextual_biases_database checkout",
        allow_module_level=True,
    )
if str(CBD_SRC) not in sys.path:
    sys.path.insert(0, str(CBD_SRC))

import wnm_scoring as S  # noqa: E402
from compiled_bundle import (CompiledWNMGroup,  # noqa: E402
                             fitting_targets_from_compiled,
                             load_compiled_wnm_bundle)
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score)

from contextual_biases_database.objectives import (  # noqa: E402
    BIAS_GRID_DEG, FEATURE_GRID_DEG, circular_distance_matrix,
    compile_cell_targets)

BUNDLE = (ROOT.parent / "contextual_biases_database" / "data" / "bundles"
          / "fischer" / "fischer_signed_v1")

D_CIRC = jnp.asarray(circular_distance_matrix())

#: The DM path is float32 by construction (`condition_rows`, the compiled
#: operators). The NumPy references here are float64, so agreement is pinned at
#: single precision. That is still four orders of magnitude tighter than the
#: operator gaps these tests detect: 3.15 units on the counterexample, 1.89 on
#: the Fischer design.
FLOAT32 = 1e-4


# ---------------------------------------------------------------------------
# An analytic conditional, shared by every comparison below
# ---------------------------------------------------------------------------

def _cell_mass(coordinates, sd, amplitude=25.0):
    """Wrapped-normal mass per bias cell, with a mean that varies with dissimilarity.

    Exactly the law the audit counterexample used. The variation matters: a
    conditional that ignored its coordinate would make pooling a no-op, and every
    assertion here would pass against either operator.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    mean = amplitude * np.sin(coordinates * np.pi / 90.0)[:, None]
    bias = BIAS_GRID_DEG[None, :]
    return sum(ndtr((bias + 1.0 + shift - mean) / sd)
               - ndtr((bias - 1.0 + shift - mean) / sd)
               for shift in (-360.0, 0.0, 360.0))


class AnalyticConditional:
    """The minimum of the predictor protocol `score_condition` actually calls.

    A stub rather than the surrogate, because the point of these tests is that
    *the same* conditional gives the same pooled score on both sides. Two
    surrogate evaluations would differ by their own roundoff and could not
    isolate the operator.
    """

    def __init__(self, sd=5.0, amplitude=25.0):
        self.sd = sd
        self.amplitude = amplitude
        self.sd_motor = 0.0

    def cell_probabilities(self, params, edges=None, validate=True, sd_motor=None):
        # params rows are [sd_feat1, sd_feat2, sd_spat, feat_diff]; only the
        # coordinate is used, so the score is a pure function of the design.
        return jnp.asarray(
            _cell_mass(np.asarray(params)[:, 3], self.sd, self.amplitude))


def _score(method, targets, condition_index, predictor=None):
    """The production dispatch, with nothing about pooling passed in."""
    return float(S.score_condition(
        method, predictor or AnalyticConditional(), targets, condition_index,
        20.0, 20.0, 20.0, curve_losses=_compute_curve_losses,
        ccc_or_combined_kwargs={"corr_weight": 0.25},
        energy_score=bwcrps_energy_score, d_circ_matrix=D_CIRC,
        feat_diff_grid=jnp.asarray(FEATURE_GRID_DEG),
        emp_density_weights_sd=20.0))


def _reference(prob_by_coordinate, operator, target_d, weights, pool):
    """Independent NumPy energy score; `pool` selects the operator under test.

    Rows of `prob_by_coordinate` are coordinates when pooling and feature-grid
    points when not, which is exactly the substitution under test.
    """
    distance = np.asarray(circular_distance_matrix())
    predicted = (np.asarray(operator) @ np.asarray(prob_by_coordinate)) if pool \
        else np.asarray(prob_by_coordinate)
    cross = np.sum(predicted * np.asarray(target_d), axis=1)
    self_energy = np.sum(predicted * (predicted @ distance), axis=1)
    return float(np.average(2.0 * cross - self_energy,
                            weights=np.asarray(weights)))


# ---------------------------------------------------------------------------
# The audit counterexample, through the production dispatch
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def counterexample():
    """The audit's five-dissimilarity design, compiled by the shared compiler."""
    coordinates = np.repeat([10.0, 30.0, 60.0, 120.0, 170.0], 20)
    bias = 25.0 * np.sin(coordinates * np.pi / 90.0)
    return coordinates, compile_cell_targets(coordinates, bias, 5.0)


def _targets_from_cell(compiled, bandwidth=5.0, repeats=1):
    """Wrap compiled cell products in the production adapter, `repeats` conditions."""
    unique = compiled["prediction_coordinates_deg"]
    capacity = len(unique) * repeats
    operator = np.zeros((repeats, len(FEATURE_GRID_DEG), capacity), dtype=np.float32)
    coordinates = np.tile(unique, repeats).astype(np.float32)
    condition_index = np.repeat(np.arange(repeats, dtype=np.int32), len(unique))
    for index in range(repeats):
        span = slice(index * len(unique), (index + 1) * len(unique))
        operator[index, :, span] = compiled["feature_operator"]
    stack = lambda key: np.stack([compiled[key]] * repeats)  # noqa: E731
    group = CompiledWNMGroup(
        fit_group_id="counterexample", fit_group_values={},
        analysis_cell_ids=tuple(f"cell{index}" for index in range(repeats)),
        analysis_cell_values=({},) * repeats,
        analysis_cell_index=np.arange(repeats, dtype=np.int32),
        prediction_coordinates_deg=coordinates,
        prediction_condition_index=condition_index,
        feature_operator=operator, coordinate_count=capacity,
        prediction_capacity=capacity, row_id=np.arange(1),
        coordinate_model_deg=np.zeros(1), bias_model_deg=np.zeros(1),
        trial_condition_index=np.zeros(1, dtype=np.int32))
    shared = {
        "feature_grid_model_deg": FEATURE_GRID_DEG,
        "smoothed_bias_deg": stack("smoothed_bias_deg"),
        "effective_support": stack("effective_support"),
        "density_asymmetry": stack("density_asymmetry"),
        "bwcrps_target_distance": stack("bwcrps_target_distance"),
        "bwcrps_support_weight": stack("bwcrps_support_weight"),
        "bwcrps_bias_weight": stack("bwcrps_bias_weight"),
        "density_bandwidth_model_deg": np.full(repeats, bandwidth),
    }
    return fitting_targets_from_compiled(group, shared)


def test_the_audited_counterexample_now_scores_pooled(counterexample):
    """The circular-weight contract keeps the audited pooling gap.

    Both values use the migrated circular empirical weights and are computed
    from the same conditional and targets, so their gap is the pooling operator.
    """
    coordinates, compiled = counterexample
    targets = _targets_from_cell(compiled)

    assert _score("bias_weighted_crps", targets, 0) == pytest.approx(6.431261, rel=FLOAT32)
    assert _reference(_cell_mass(FEATURE_GRID_DEG, 5.0), compiled["feature_operator"],
                      compiled["bwcrps_target_distance"], compiled["bwcrps_bias_weight"],
                      pool=False) == pytest.approx(9.625543, rel=FLOAT32)


def test_a_model_that_reproduces_the_bias_law_cannot_reach_the_floor_unpooled(counterexample):
    """The property the number expresses: the unpooled route is not minimisable here.

    `q_emp` scoring itself is the floor of the pooled criterion. The pooled
    prediction of the exact data-generating law gets close to it; the unpooled
    prediction of the same law is further away than the floor is from either.
    """
    coordinates, compiled = counterexample
    floor = _reference(compiled["bwcrps_empirical_distribution"],
                       compiled["feature_operator"], compiled["bwcrps_target_distance"],
                       compiled["bwcrps_bias_weight"], pool=False)
    targets = _targets_from_cell(compiled)
    pooled = _score("bias_weighted_crps", targets, 0)
    unpooled = _reference(_cell_mass(FEATURE_GRID_DEG, 5.0), compiled["feature_operator"],
                          compiled["bwcrps_target_distance"],
                          compiled["bwcrps_bias_weight"], pool=False)
    assert floor == pytest.approx(4.720259, rel=FLOAT32)
    assert floor < pooled < unpooled
    assert unpooled - pooled > pooled - floor


def test_each_condition_pools_with_its_own_operator(counterexample):
    """Packed multi-condition operators: padding columns must stay inert.

    A condition's operator is zero outside its own coordinate block, so a second
    condition sharing the same data must score identically. Indexing the packed
    coordinate axis with the wrong offset gives a plausible, wrong number.
    """
    _coordinates, compiled = counterexample
    targets = _targets_from_cell(compiled, repeats=2)
    first = _score("bias_weighted_crps", targets, 0)
    second = _score("bias_weighted_crps", targets, 1)
    # FLOAT32, not exact equality: the two conditions reduce over different
    # columns of the same packed operator, so they agree only to the precision
    # the path runs in, and that precision depends on the process-wide x64 flag.
    # A wrong coordinate offset would move this by whole units, not by 1e-6.
    assert first == pytest.approx(second, rel=FLOAT32)
    assert first == pytest.approx(6.431261, rel=FLOAT32)


def test_balanced_crps_pools_as_well(counterexample):
    """It reads the same pooled `target_d`; only the weights differ."""
    _coordinates, compiled = counterexample
    targets = _targets_from_cell(compiled)
    got = _score("balanced_crps", targets, 0)
    assert got == pytest.approx(
        _reference(_cell_mass(compiled["prediction_coordinates_deg"], 5.0),
                   compiled["feature_operator"], compiled["bwcrps_target_distance"],
                   compiled["bwcrps_support_weight"], pool=True), rel=FLOAT32)


# ---------------------------------------------------------------------------
# A real compiled production design
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production():
    if not BUNDLE.exists():
        pytest.skip(f"no compiled bundle at {BUNDLE}")
    groups, shared, _assignments, _manifest = load_compiled_wnm_bundle(BUNDLE)
    group = groups[0]
    return group, shared, fitting_targets_from_compiled(group, shared)


def test_production_design_matches_an_independent_pooled_reference(production):
    """Dense real design, repeated coordinates, boundary features, packed capacity."""
    group, shared, targets = production
    predictor = AnalyticConditional(sd=15.0)
    cell = int(np.asarray(group.analysis_cell_index)[0])
    expected = _reference(_cell_mass(np.asarray(group.prediction_coordinates_deg), 15.0),
                          np.asarray(group.feature_operator)[0],
                          np.asarray(shared["bwcrps_target_distance"])[cell],
                          np.asarray(shared["bwcrps_bias_weight"])[cell], pool=True)
    got = _score("bias_weighted_crps", targets, 0, predictor)
    assert got == pytest.approx(expected, rel=FLOAT32)


def test_the_gap_does_not_close_on_a_dense_production_design(production):
    """Density of coordinates is not a defence: the kernel still truncates at the
    ends of the coordinate range, and curvature still biases the interior."""
    group, shared, targets = production
    cell = int(np.asarray(group.analysis_cell_index)[0])
    unpooled = _reference(_cell_mass(FEATURE_GRID_DEG, 15.0), None,
                          np.asarray(shared["bwcrps_target_distance"])[cell],
                          np.asarray(shared["bwcrps_bias_weight"])[cell], pool=False)
    pooled = _score("bias_weighted_crps", targets, 0, AnalyticConditional(sd=15.0))
    assert unpooled - pooled > 1.0    # measured 1.889 on this cell


def test_the_per_trial_crps_is_deliberately_not_pooled(production):
    """`crps` is a genuine per-trial energy score at each trial's own location.

    Pooling it would silently replace a trial-summed objective with a curve
    criterion. Zeroing the design operator must leave it untouched and must break
    the curve-level objectives.
    """
    group, shared, targets = production
    trials = (jnp.asarray(group.coordinate_model_deg), jnp.asarray(group.bias_model_deg))
    kwargs = dict(curve_losses=_compute_curve_losses,
                  ccc_or_combined_kwargs={"corr_weight": 0.25},
                  energy_score=bwcrps_energy_score, d_circ_matrix=D_CIRC,
                  feat_diff_grid=jnp.asarray(FEATURE_GRID_DEG),
                  emp_density_weights_sd=20.0)
    predictor = AnalyticConditional(sd=15.0)
    flattened = type(targets)(**{
        **{field: getattr(targets, field) for field in targets.__dataclass_fields__},
        "feature_operator": jnp.zeros_like(targets.feature_operator)})

    with_operator = float(S.score_condition("crps", predictor, targets, 0, 20.0, 20.0,
                                            20.0, trials=trials, **kwargs))
    without = float(S.score_condition("crps", predictor, flattened, 0, 20.0, 20.0,
                                      20.0, trials=trials, **kwargs))
    assert with_operator == pytest.approx(without, rel=1e-12)

    curve = float(S.score_condition("bias_weighted_crps", predictor, targets, 0,
                                    20.0, 20.0, 20.0, **kwargs))
    zeroed = float(S.score_condition("bias_weighted_crps", predictor, flattened, 0,
                                     20.0, 20.0, 20.0, **kwargs))
    assert not np.isclose(curve, zeroed, rtol=1e-6)
