"""Contracts for the frozen first actual-DM recovery panel."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_fit_to_data"))

import single_condition_recovery as panel  # noqa: E402


def test_the_script_entry_point_runs_outside_the_repository(tmp_path):
    output = tmp_path / "protocol"
    subprocess.run([
        sys.executable, str(ROOT / "model_fit_to_data/single_condition_recovery.py"),
        "--out", str(output)], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert (output / "protocol.json").exists()
    assert (output / "single_condition_design.npz").exists()


def test_the_protocol_freezes_the_agreed_single_condition_design(tmp_path):
    panel.main(["--out", str(tmp_path)])

    design, labels = panel.design_rows()
    assert design.shape == (12 * 90, 4)
    assert set(labels) == {case[0] for case in panel.CASES}
    assert np.array_equal(np.unique(design[:, 3]), np.arange(2, 181, 2))

    manifest = panel.protocol()
    assert manifest["n_samples"] == 100
    assert manifest["model_period_degrees"] == 360.0
    assert manifest["global_orientation_degrees"] == 0.0
    assert manifest["sd_motor_degrees"] == 0.0
    assert manifest["trial_counts"] == [180, 450, 900]
    assert manifest["response_seeds"] == [0, 1, 2, 3, 4]
    assert manifest["trials_per_feature_difference"] == {
        "180": 2, "450": 5, "900": 10}
    assert manifest["n_datasets"] == 180
    assert sum(case["split"] == "development" for case in manifest["cases"]) == 8
    assert sum(case["split"] == "held_out" for case in manifest["cases"]) == 4
    assert [(case["name"], case["split"], case["sd_feat1"], case["sd_feat2"],
             case["sd_spat"]) for case in manifest["cases"]] == [
        ("narrow_1", "development", 5.0, 10.0, 15.0),
        ("narrow_2", "held_out", 7.5, 15.0, 30.0),
        ("narrow_3", "development", 10.0, 20.0, 60.0),
        ("ordinary_1", "development", 10.0, 30.0, 60.0),
        ("ordinary_2", "held_out", 15.0, 45.0, 15.0),
        ("ordinary_3", "development", 20.0, 50.0, 30.0),
        ("reversed_1", "development", 30.0, 10.0, 60.0),
        ("reversed_2", "held_out", 45.0, 15.0, 15.0),
        ("reversed_3", "development", 50.0, 20.0, 30.0),
        ("broad_1", "development", 60.0, 80.0, 30.0),
        ("broad_2", "held_out", 80.0, 120.0, 60.0),
        ("broad_3", "development", 120.0, 160.0, 15.0),
    ]
    written = json.loads((tmp_path / "protocol.json").read_text())
    assert written["design_sha256"] == manifest["design_sha256"]
    assert set(written["code_sha256"]) == {
        str(path.relative_to(panel.ROOT)) for path in panel.CODE_PATHS}


def test_the_generation_entry_point_runs_all_five_frozen_seeds(
        tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(panel.subprocess, "run",
                        lambda command, **kwargs: calls.append((command, kwargs)))

    panel.main(["--out", str(tmp_path), "--generate"])

    assert len(calls) == 5
    assert [call[0][call[0].index("--seed") + 1] for call in calls] == [
        "0", "1", "2", "3", "4"]
    assert all(call[1] == {"cwd": panel.ROOT, "check": True} for call in calls)


def test_trial_counts_are_nested_within_each_dissimilarity(tmp_path):
    design, _ = panel.design_rows()
    bias = np.empty((len(design), panel.RESPONSES_PER_DIFFERENCE, 2),
                    dtype=np.float32)
    bias[:, :, 0] = np.arange(panel.RESPONSES_PER_DIFFERENCE)
    bias[:, :, 1] = -1
    raw_path = tmp_path / "raw.npz"
    np.savez(raw_path, design=design, bias=bias)

    small = panel.load_dataset(raw_path, "narrow_1", 180)
    medium = panel.load_dataset(raw_path, "narrow_1", 450)
    large = panel.load_dataset(raw_path, "narrow_1", 900)

    assert [len(small), len(medium), len(large)] == [180, 450, 900]
    assert np.array_equal(small[small[:, 0] == 2, 1], [0, 1])
    assert np.array_equal(medium[medium[:, 0] == 2, 1], np.arange(5))
    assert np.array_equal(large[large[:, 0] == 2, 1], np.arange(10))
