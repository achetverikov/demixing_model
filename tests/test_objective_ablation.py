"""Checks for the high-resolution objective-ablation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'neural_network_optimization'))

from objective_ablation import _column_metrics, _surface_scores
from shared.mu1_axis import mu1_size


class _Surface:
    pass


def _log_impulse(index):
    logits = np.full((mu1_size(), 90), -20.0)
    logits[index] = 20.0
    return logits


def test_reference_metrics_are_zero_for_an_identical_surface():
    target = _log_impulse(17)
    metrics = _column_metrics(target, target)
    for values in metrics.values():
        assert np.max(np.abs(values)) < 1e-10


def test_reference_energy_uses_the_circular_seam():
    target = _log_impulse(mu1_size() - 1)
    near = _column_metrics(_log_impulse(0), target)['energy'].mean()
    far = _column_metrics(_log_impulse(mu1_size() // 2), target)['energy'].mean()
    assert near == pytest.approx(4 / 360)
    assert far > near * 50


def test_selector_distinguishes_peaked_and_flat_surfaces():
    peaked = _Surface()
    peaked.mu1_comp1_surface = _log_impulse(20)
    peaked.mu1_comp2_surface = _log_impulse(160)
    flat = _Surface()
    flat.mu1_comp1_surface = np.zeros((mu1_size(), 90))
    flat.mu1_comp2_surface = np.zeros((mu1_size(), 90))

    peaked_scores = _surface_scores(peaked)
    flat_scores = _surface_scores(flat)
    assert peaked_scores['peaked'] > flat_scores['peaked']
    assert flat_scores['flat'] > peaked_scores['flat']
