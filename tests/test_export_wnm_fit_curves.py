import pickle
from types import SimpleNamespace

import numpy as np

from model_fit_to_data import export_wnm_fit_curves as export_module


def test_direct_export_uses_stored_matched_operator_and_writes_plot(tmp_path, monkeypatch):
    results_dir = tmp_path / "fit"
    output_dir = tmp_path / "curves"
    results_dir.mkdir()
    checkpoint = tmp_path / "wnm.pkl"
    checkpoint.write_bytes(b"checkpoint")
    operator = np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float32)
    results = {
        "condition one": {
            "n_trials": 2,
            "angle_scale_to_model": 1.0,
            "empirical_curves": {
                "feature_operator": operator,
                "density_bandwidth": 7.5,
                "target_bias_curve": np.array([1.0, 2.0]),
                "matched_density_target": np.array([0.1, 0.2]),
                "prediction_coordinates": np.array([1.0, 2.0]),
            },
            "density_fitted_params": np.array([10.0, 20.0, 30.0, 0.0]),
            "density_loss": 0.5,
            "bundle_identity": {"bundle_id": "bundle-test"},
            "density_eval_density_loss": 0.5,
            "density_eval_smoothed_exp_loss": 1.0,
            "density_eval_likelihood_loss": 2.0,
            "density_eval_bias_weighted_crps_loss": 3.0,
        }
    }
    with (results_dir / "extended_fit_results.pkl").open("wb") as handle:
        pickle.dump(results, handle)

    monkeypatch.setattr(export_module, "read_fingerprint_sidecar", lambda _: {
        "digest": "run-digest",
        "payload": {
            "surrogate_family": "wnm", "checkpoint_sha256": "digest",
            "density_curve_spec": {
                "emp_density_weights_sd": 15.0,
                "density_smoothing_sigma": 6.0,
            },
            "continuous_spec": {"matmul_precision": "highest"},
        },
    })
    monkeypatch.setattr(export_module, "file_sha256", lambda _: "digest")
    monkeypatch.setattr(export_module.surrogate, "load_surrogate", lambda **_: object())
    identity = {"dm_version": "wnm_k12_20samples", "surrogate_family": "wnm"}
    predictor = SimpleNamespace(identity=lambda: SimpleNamespace(as_dict=lambda: identity))
    monkeypatch.setattr(export_module, "predictor_from_surrogate", lambda _: predictor)
    monkeypatch.setattr(export_module.config, "create_grid",
                        lambda _: np.array([2.0, 4.0], dtype=np.float32))

    def fake_curves(_predictor, params, feat_grid, **kwargs):
        np.testing.assert_array_equal(kwargs["feature_operators"], operator[None, :, :])
        np.testing.assert_array_equal(kwargs["density_bandwidths"], [7.5])
        np.testing.assert_array_equal(kwargs["operator_feature_coordinates"], [1.0, 2.0])
        assert kwargs["emp_density_weights_sd"] == 15.0
        assert kwargs["density_smoothing_sigma"] == 6.0
        assert params.shape == (1, 3)
        assert feat_grid.shape == (2,)
        return {
            "bias": np.array([[3.0, 4.0]]),
            "asymmetry": np.array([[0.3, 0.4]]),
            "sd": np.array([[5.0, 6.0]]),
        }

    monkeypatch.setattr(export_module, "mixture_plot_curves", fake_curves)
    frame = export_module.export_curves(
        results_dir, checkpoint, output_dir, methods=("density",))

    assert len(frame) == 2
    assert frame["bias_deg"].tolist() == [3.0, 4.0]
    assert frame["dm_version"].unique().tolist() == ["wnm_k12_20samples"]
    assert (output_dir / "fitted_curves.csv").exists()
    parameters = export_module.pd.read_csv(output_dir / "fitted_parameters.csv")
    assert parameters.loc[0, "density_loss"] == 0.5
    assert parameters.loc[0, "eval_likelihood_loss"] == 2.0
    assert parameters.loc[0, "bundle_id"] == "bundle-test"
    assert frame.loc[0, "bundle_id"] == "bundle-test"
    assert (output_dir / "condition_one.png").exists()
    assert "direct analytic WNM" in (output_dir / "manifest.json").read_text()


def test_compiled_likelihood_export_uses_stored_trial_coordinates(monkeypatch):
    result = {
        "data_df": np.array([[4.0, -2.0], [8.0, 6.0]], dtype=np.float32),
        "ordered_row_ids": np.array(["row-a", "row-b"]),
        "angle_scale_to_model": 2.0,
        "circ_space": 180.0,
        "fit_group_id": "fit-a",
        "analysis_cell_values": {
            "experiment_id": "exp", "subject_id": "1", "condition_id": "cond"},
        "bundle_identity": {
            "bundle_id": "bundle-a", "canonical_trial_sha256": "canonical",
            "analysis_spec_sha256": "spec", "population": "signed_bias",
            "ordered_row_id_sha256": "rows", "empirical_targets_sha256": "targets"},
        "density_fitted_params": np.array([10.0, 20.0, 30.0, 0.0]),
        "density_eval_likelihood_loss": 3.0,
    }

    def fake_log_density(_predictor, sd1, sd2, sd_spat, feature, bias, sd_motor):
        assert (sd1, sd2, sd_spat, sd_motor) == (10.0, 20.0, 30.0, 0.0)
        np.testing.assert_array_equal(feature, [4.0, 8.0])
        np.testing.assert_array_equal(bias, [-2.0, 6.0])
        return np.array([-1.0, -2.0])

    monkeypatch.setattr(export_module, "trial_log_density", fake_log_density)
    likelihood, checks = export_module.compiled_trial_likelihoods(
        {"cell-a": result}, object(), {"dm_version": "wnm"}, ("density",))

    assert likelihood["row_id"].tolist() == ["row-a", "row-b"]
    assert likelihood["dissimilarity_deg"].tolist() == [2.0, 4.0]
    assert likelihood["bias_toward_context_deg"].tolist() == [-1.0, 3.0]
    np.testing.assert_allclose(likelihood["loglik_mass"],
                               [-1.0 + np.log(2.0), -2.0 + np.log(2.0)])
    np.testing.assert_allclose(likelihood["loglik_density_deg"],
                               [-1.0 + np.log(2.0), -2.0 + np.log(2.0)])
    assert likelihood["include_common_eval"].all()
    assert checks.loc[0, "abs_diff"] == 0.0
