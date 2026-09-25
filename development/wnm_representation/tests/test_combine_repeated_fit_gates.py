import json

import pytest

from development.wnm_representation import combine_repeated_fit_gates as combine


def _write_case(root, estimate):
    case = root / "trajectory" / "component_1"
    case.mkdir(parents=True)
    core = {"candidates": [{"family": "maxent_fourier", "size": 8,
        "metrics": {"response_sd": {"coherent_estimate": estimate,
            "simultaneous_radius": 0.1}}}]}
    distribution = {"candidates": [{"family": "maxent_fourier", "size": 8,
        "excess_nll": {"estimate": [0.0], "simultaneous_upper": [0.0001]},
        "circular_wasserstein_deg": {
            "estimate": [0.1], "simultaneous_upper": [0.2]}}]}
    (case / "core_equivalence.json").write_text(json.dumps(core))
    (case / "distribution_gate.json").write_text(json.dumps(distribution))


def test_combined_gate_envelopes_independent_fit_intervals(tmp_path):
    primary, repeat = tmp_path / "primary", tmp_path / "repeat"
    _write_case(primary, 0.2)
    _write_case(repeat, -0.2)
    frame = combine.combine(primary, repeat)
    response = frame[frame.metric == "response_sd"].iloc[0]
    assert response.combined_lower == pytest.approx(-0.3)
    assert response.combined_upper == pytest.approx(0.3)
    assert not bool(response.combined_pass)
    assert not frame.candidate_all_metrics_pass.any()
