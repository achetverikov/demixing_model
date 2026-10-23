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
from surface_computation import simulation_corpus as storage
from surrogate_training.wnm import train, package_artifact


@pytest.mark.parametrize("n_samples", [20, 500])
def test_train_package_load(tmp_path, monkeypatch, n_samples):
    design = np.array([[10+i, 30+i, 20+i, 5+10*i] for i in range(8)], dtype=np.float32)
    source = tmp_path / "source.npz"
    np.savez(source, design=design, bias=np.zeros((8, 4, 2), dtype=np.float32),
             meta=json.dumps({"n_samples": n_samples}))
    checkpoint = tmp_path / "train.pkl"
    monkeypatch.setattr(sys, "argv", ["train", "--source", str(source), "--out", str(checkpoint),
        "--steps", "2", "--warmup", "0", "--eval-every", "1", "--selection-every", "1",
        "--eval-batches", "1", "--batch-size", "4", "--components", "2", "--hidden", "4"])
    train.main()
    with pytest.raises(ValueError, match="training checkpoint"):
        surrogate.load_surrogate(checkpoint_path=checkpoint, n_samples=n_samples)
    artifact = tmp_path / "custom.pkl"
    def package(count):
        monkeypatch.setattr(sys, "argv", ["package", "--fit", str(checkpoint),
            "--out", str(artifact), "--n-samples", str(count)])
        package_artifact.main()
    with pytest.raises(SystemExit, match="n_samples"):
        package(100)
    assert not artifact.exists()
    package(n_samples)
    loaded = surrogate.load_surrogate(checkpoint_path=artifact, n_samples=n_samples)
    assert loaded is not None
    assert loaded.n_samples == n_samples
    original = pickle.loads(checkpoint.read_bytes())
    packaged = pickle.loads(artifact.read_bytes())
    assert packaged["meta"]["supported_domain"] == original["meta"]["supported_domain"]
    assert packaged["meta"]["selected_step"] in (1, 2)
    assert packaged["meta"]["checkpoint_selection"] == "held_out_parameter_groups"


def test_histogram_corpus_trains_and_packages(tmp_path, monkeypatch, capsys):
    # Entry-point contract: a histogram corpus selects the per-bin objective,
    # reaches production loading, and refuses raw-only options before training.
    import hashlib
    design = np.array([[10+i, 30+i, 20+i, 5+10*i] for i in range(8)], dtype=np.float32)
    signature = dict(storage_format=storage.HISTOGRAM_FORMAT, histogram_bins=36, n_rows=8,
        n_simulations=4, shard_rows=8, simulation_chunk=4, n_samples=20, seed=0,
        design_sha256=hashlib.sha256(design.tobytes()).hexdigest(),
        strata_sha256=hashlib.sha256(json.dumps(None).encode()).hexdigest())
    source = tmp_path / "histograms"
    histogram = np.zeros((8, 1, 2, 36), np.int32)
    histogram[..., 18] = 3
    histogram[..., 20] = 1
    with storage.writer(source, signature, design, None) as corpus:
        corpus.write_tile(corpus.coordinates(0, 0), dict(histogram=histogram,
            counts=np.zeros((8, 1, len(storage.HISTOGRAM_COUNTS)), np.int64),
            max_iterations=np.ones((8, 1), np.int32)))
        corpus.finish({"n_samples": 20})
    checkpoint = tmp_path / "train.pkl"
    common = ["train", "--source", str(source), "--out", str(checkpoint), "--steps", "2",
              "--warmup", "0", "--eval-every", "1", "--selection-every", "1", "--eval-batches", "1",
              "--histogram-batch", "4", "--components", "2", "--hidden", "4"]
    monkeypatch.setattr(sys, "argv", common + ["--grouped-outcomes", "2"])
    with pytest.raises(SystemExit):
        train.main()
    assert not checkpoint.exists()
    raw_source = tmp_path / "raw.npz"
    np.savez(raw_source, design=design, bias=np.zeros((8, 4, 2), dtype=np.float32), meta=json.dumps({"n_samples": 20}))
    monkeypatch.setattr(sys, "argv", ["train", "--source", str(raw_source), "--augmentation", str(source),
                                      "--out", str(checkpoint), "--steps", "1"])
    with pytest.raises(SystemExit):
        train.main()
    monkeypatch.setattr(sys, "argv", ["train", "--source", str(raw_source), "--selection-reference", str(source),
                                      "--out", str(checkpoint), "--steps", "1"])
    with pytest.raises(SystemExit):
        train.main()
    # A raw selection reference sharing the held-out groups must be refused.
    reference = tmp_path / "reference.npz"
    np.savez(reference, design=design, bias=np.zeros((8, 4, 2), dtype=np.float32), strata=["t"] * 8,
             meta=json.dumps({"n_samples": 20}))
    monkeypatch.setattr(sys, "argv", ["train", "--source", str(raw_source), "--selection-reference", str(reference),
                                      "--holdout-fraction", "0.25", "--out", str(checkpoint), "--steps", "1"])
    with pytest.raises(SystemExit):
        train.main()
    assert "shares parameter groups" in capsys.readouterr().err
    assert not checkpoint.exists()
    monkeypatch.setattr(sys, "argv", common + ["--holdout-fraction", "0.25", "--checkpoint-metric", "final"])
    train.main()
    # A different test set must not be fine-tuned from a model that may have seen it.
    tuned = tmp_path / "tuned.pkl"
    monkeypatch.setattr(sys, "argv", [a if a != str(checkpoint) else str(tuned) for a in common]
                        + ["--holdout-fraction", "0.5", "--init-model", str(checkpoint)])
    with pytest.raises(SystemExit):
        train.main()
    assert not tuned.exists()
    meta = pickle.loads(checkpoint.read_bytes())["meta"]
    assert meta["training_objective"] == "histogram_bin_cross_entropy" and meta["histogram_bins"] == 36
    assert meta["holdout"]["rows"] == 2 and meta["selected_step"] == 2 and meta["checkpoint_selection"] == "final_step"
    assert len(meta["holdout"]["groups_sha256"]) == 64
    artifact = tmp_path / "custom.pkl"
    monkeypatch.setattr(sys, "argv", ["package", "--fit", str(checkpoint), "--out", str(artifact),
                                      "--n-samples", "20"])
    package_artifact.main()
    assert surrogate.load_surrogate(checkpoint_path=artifact, n_samples=20).n_samples == 20
