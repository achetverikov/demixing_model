import numpy as np

from continuous_density import local_representation_benchmark as benchmark


def test_trajectory_records_preserve_file_order(tmp_path):
    design = np.array([[10, 20, 30, 2], [10, 20, 30, 180],
                       [40, 50, 60, 2], [40, 50, 60, 180]], dtype=np.float32)
    path = tmp_path / "reference.npz"
    np.savez(path, design=design, bias=np.zeros((4, 2, 2)),
             strata=np.array(["first", "first", "second", "second"]))
    records = benchmark.trajectory_records(path)
    assert [label for label, _ in records] == ["first", "second"]
    np.testing.assert_array_equal(records[1][1], [40, 50, 60])
