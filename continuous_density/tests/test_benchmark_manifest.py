import numpy as np
import pytest

from continuous_density import benchmark_manifest as bm


def test_canonical_keys_are_mirror_invariant():
    design = np.array([[10, 60, 20, 2], [60, 10, 20, 4], [30, 40, 50, 2]])
    assert bm.canonical_trajectory_keys(design) == {(10.0, 60.0, 20.0),
                                                    (30.0, 40.0, 50.0)}


def test_manifest_rejects_selection_confirmation_overlap():
    manifest = {"corpora": [
        {"role": "selection", "trajectory_keys": [[10, 60, 20]]},
        {"role": "confirmation", "trajectory_keys": [[10, 60, 20]]},
    ]}
    with pytest.raises(ValueError, match="overlap"):
        bm.validate_manifest(manifest)


def test_manifest_allows_labeled_selection_precision_increment():
    manifest = {"corpora": [
        {"role": "selection", "trajectory_keys": [[10, 60, 20]]},
        {"role": "selection", "trajectory_keys": [[10, 60, 20]],
         "precision_increment": True},
    ]}
    bm.validate_manifest(manifest)


def test_confirmation_cannot_drive_adaptation():
    for operation in ("fit", "select", "adapt", "debug"):
        with pytest.raises(PermissionError):
            bm.assert_operation_allowed("confirmation", operation)
    bm.assert_operation_allowed("confirmation", "evaluate")
    bm.assert_operation_allowed("selection", "adapt")
