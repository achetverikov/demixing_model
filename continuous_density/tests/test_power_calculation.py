from continuous_density import power_calculation as power


def test_required_n_increases_with_multiplicity_and_precision():
    single = power.required_n(0.00316, 100_000, 0.005, 1)
    trajectory = power.required_n(0.00316, 100_000, 0.005, 90)
    tighter = power.required_n(0.00316, 100_000, 0.002, 1)
    assert trajectory > single
    assert tighter > single


def test_covariance_inflation_removes_coherent_gain():
    independent = power.planning_table(90, 1.0)[0]
    correlated = power.planning_table(90, 90 ** 0.5)[0]
    assert correlated["coherent_n_with_50pct_error_headroom"] > 80 * independent[
        "coherent_n_with_50pct_error_headroom"]


def test_planning_table_contains_fixed_asymmetry_margin():
    row = next(item for item in power.planning_table()
               if item["metric"] == "density_asymmetry")
    assert row["pointwise_margin"] == 0.005
    assert row["coherent_margin"] == 0.002
