"""Maintained integration: the actual trainer output must reach production loading.

A two-step tiny network catches checkpoint/provenance incompatibility; no fit of
behavioral data is needed. Reuse the one trained checkpoint for refusal checks.
"""
import json
import pickle
import sys

import numpy as np
import pytest

from shared import surrogate
from surrogate_training.wnm import train, package_artifact


def test_train_package_load(tmp_path, monkeypatch):
    design = np.array([[10+i, 30+i, 20+i, 5+10*i] for i in range(8)], dtype=np.float32)
    source = tmp_path / "source.npz"
    np.savez(source, design=design, bias=np.zeros((8, 4, 2), dtype=np.float32),
             meta=json.dumps({"n_samples": 20}))
    checkpoint = tmp_path / "train.pkl"
    monkeypatch.setattr(sys, "argv", ["train", "--source", str(source), "--out", str(checkpoint),
        "--steps", "2", "--warmup", "0", "--eval-every", "1", "--selection-every", "1",
        "--eval-batches", "1", "--batch-size", "4", "--components", "2", "--hidden", "4"])
    train.main()
    with pytest.raises(ValueError, match="training checkpoint"):
        surrogate.load_surrogate(checkpoint_path=checkpoint, n_samples=20)
    artifact = tmp_path / "custom.pkl"
    def package(count):
        monkeypatch.setattr(sys, "argv", ["package", "--fit", str(checkpoint),
            "--out", str(artifact), "--n-samples", str(count)])
        package_artifact.main()
    with pytest.raises(SystemExit, match="n_samples"):
        package(100)
    assert not artifact.exists()
    package(20)
    loaded = surrogate.load_surrogate(checkpoint_path=artifact, n_samples=20)
    assert loaded is not None
    original = pickle.loads(checkpoint.read_bytes())
    packaged = pickle.loads(artifact.read_bytes())
    assert packaged["meta"]["supported_domain"] == original["meta"]["supported_domain"]
    assert packaged["meta"]["selected_step"] in (1, 2)
    assert packaged["meta"]["checkpoint_selection"] == "held_out_parameter_groups"
