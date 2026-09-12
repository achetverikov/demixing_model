"""Standard, resumable WNM parameter-recovery protocol.

The protocol deliberately has one surrogate family and one optimizer path.  A
dataset is generated once for each case/sample-size/replicate cell, then every
requested objective fits those identical trials from the same deterministic
multistart design.  Fits are canonically rescored under all supported objectives
and exported with parameter and dissimilarity-stratified curve tables.

Usage::

    JAX_PLATFORMS=cpu python model_fit_to_data/standardized_recovery.py \
        --protocol model_fit_to_data/recovery_protocol.example.json \
        --out /tmp/wnm_recovery
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "model_fit_to_data"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import recovery as R  # noqa: E402
import wnm_scoring as S  # noqa: E402
from continuous_fit import ContinuousEngine  # noqa: E402
from continuous_optimizer import OPTIMIZER_VERSION, condition_parameter_layout  # noqa: E402
from create_unified_subject_plots import (  # noqa: E402
    compute_empirical_sd_curve, compute_feat_bin_weights)
from density_objective import degenerate_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score, compute_bwcrps_condition_targets,
    compute_target_bias_curve_core)
from shared import surrogate  # noqa: E402
from shared.prediction import mixture_plot_curves, predictor_from_surrogate  # noqa: E402


SCHEMA = "wnm_recovery_protocol/1"
TARGET_DEFAULTS = {
    "emp_density_weights_sd": 20.0,
    "density_smoothing_sigma": None,
    "density_bandwidth_rule": "sj",
    "density_bandwidth_mode": "pooled",
    "corr_weight": 0.25,
}
CODE_PATHS = (
    "model_fit_to_data/standardized_recovery.py",
    "model_fit_to_data/standardized_recovery_plots.py",
    "model_fit_to_data/recovery.py",
    "model_fit_to_data/continuous_fit.py",
    "model_fit_to_data/continuous_optimizer.py",
    "model_fit_to_data/wnm_scoring.py",
    "model_fit_to_data/fitting_targets.py",
    "model_fit_to_data/density_objective.py",
    "model_fit_to_data/grid_based_multi_condition_optimizer_jax_loops.py",
    "model_fit_to_data/create_unified_subject_plots.py",
    "shared/prediction.py",
    "shared/surrogate.py",
    "shared/config.py",
    "shared/utils.py",
)


def _canonical_digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def code_digest() -> tuple[str, dict[str, str]]:
    """Digest every source file that can change a standardized recovery result."""
    files = {}
    for relative in CODE_PATHS:
        path = ROOT / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _canonical_digest(files), files


def normalize_protocol(raw: dict) -> dict:
    """Fill stable defaults and reject protocols outside the WNM-only contract."""
    protocol = dict(raw)
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA!r}")
    surrogate_spec = dict(protocol.get("surrogate", {}))
    family = surrogate_spec.get("family", surrogate.FAMILY_WNM)
    if family != surrogate.FAMILY_WNM:
        raise ValueError(
            "standardized recovery supports only the new WNM predictor; "
            f"got surrogate family {family!r}")
    surrogate_spec["family"] = family
    if "n_samples" not in surrogate_spec:
        raise ValueError("protocol surrogate must declare n_samples")
    protocol["surrogate"] = surrogate_spec

    objectives = tuple(protocol.get("objectives", S.SUPPORTED_METHODS))
    unknown = set(objectives) - set(S.SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported objectives: {sorted(unknown)}")
    if not objectives or len(set(objectives)) != len(objectives):
        raise ValueError("objectives must be a non-empty list without duplicates")
    protocol["objectives"] = list(objectives)
    protocol["target"] = TARGET_DEFAULTS | dict(protocol.get("target", {}))
    dataset_source = {"kind": "wnm_closed_loop"} | dict(
        protocol.get("dataset_source", {}))
    if dataset_source["kind"] not in ("wnm_closed_loop", "npz"):
        raise ValueError("dataset_source.kind must be 'wnm_closed_loop' or 'npz'")
    if dataset_source["kind"] == "npz" and "directory" not in dataset_source:
        raise ValueError("an npz dataset source must declare its directory")
    dataset_source.setdefault(
        "pattern", "{case}__n{trial_count}__r{replicate}.npz")
    protocol["dataset_source"] = dataset_source
    protocol.setdefault("n_starts", 64)
    protocol.setdefault("seed", 0)
    protocol.setdefault("n_replicates", 1)
    protocol.setdefault("trial_counts", [400])
    protocol.setdefault("fit_motor", False)
    if int(protocol["n_starts"]) < 1 or int(protocol["n_replicates"]) < 1:
        raise ValueError("n_starts and n_replicates must be positive")
    if not protocol["trial_counts"] or any(int(count) < 2
                                            for count in protocol["trial_counts"]):
        raise ValueError("trial_counts must contain positive counts of at least two")
    if len(set(map(int, protocol["trial_counts"]))) != len(protocol["trial_counts"]):
        raise ValueError("trial_counts must not contain duplicates")
    if not protocol.get("cases"):
        raise ValueError("protocol must contain at least one generating case")

    names = [case["name"] for case in protocol["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"case name {name!r} is not safe for artifact filenames")
    if protocol["fit_motor"] and set(objectives) & set(S.MEAN_ONLY_METHODS):
        raise ValueError(
            "expectation and smoothed_exp cannot identify a fitted motor SD; "
            "remove them or set fit_motor=false")
    if not protocol["fit_motor"] and set(objectives) & set(S.MEAN_ONLY_METHODS):
        noisy = [case["name"] for case in protocol["cases"]
                 if float(case.get("sd_motor", 0.0)) != 0.0]
        if noisy:
            raise ValueError(
                "expectation and smoothed_exp are invariant to fixed motor noise; "
                f"remove them for motor-noise cases {noisy}")
    return protocol


def recovery_cases(protocol: dict, trial_count: int) -> list[R.RecoveryCase]:
    return [R.RecoveryCase(
        name=spec["name"],
        condition_feature_sds=[tuple(pair) for pair in spec["condition_feature_sds"]],
        sd_spat=float(spec["sd_spat"]), sd_motor=float(spec.get("sd_motor", 0.0)),
        n_trials_per_condition=int(trial_count),
        feat_diff_values=spec.get("feat_diff_values"))
        for spec in protocol["cases"]]


def dataset_digest(datasets: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in datasets.items():
        array = np.ascontiguousarray(values, dtype=np.float32)
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _dataset_path(out: Path, dataset_id: str) -> Path:
    return out / "datasets" / f"{dataset_id}.npz"


def _save_dataset(path: Path, datasets: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **datasets)
    os.replace(temporary, path)


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def _source_dataset(protocol, adapter, case, case_index, trial_count, replicate):
    """Generate WNM trials or read simulator-produced trials in the same contract."""
    source = protocol["dataset_source"]
    if source["kind"] == "wnm_closed_loop":
        rng = np.random.default_rng(
            [int(protocol["seed"]), case_index, int(trial_count), replicate])
        return R.generate_case_data(adapter.predictor, case, rng)
    path = Path(source["directory"]) / source["pattern"].format(
        case=case.name, trial_count=trial_count, replicate=replicate)
    if not path.exists():
        raise FileNotFoundError(
            f"no simulator dataset for {case.name}, n={trial_count}, r={replicate}: {path}")
    datasets = _load_dataset(path)
    if tuple(datasets) != tuple(f"c{index}" for index in range(case.n_conditions)):
        raise ValueError(
            f"{path} contains conditions {tuple(datasets)}, expected "
            f"{tuple(f'c{index}' for index in range(case.n_conditions))}")
    return datasets


def _validate_datasets(datasets, case):
    expected = tuple(f"c{index}" for index in range(case.n_conditions))
    if tuple(datasets) != expected:
        raise ValueError(f"dataset conditions are {tuple(datasets)}, expected {expected}")
    for name, values in datasets.items():
        array = np.asarray(values)
        if array.shape != (case.n_trials_per_condition, 2):
            raise ValueError(
                f"{name} has shape {array.shape}, expected "
                f"{(case.n_trials_per_condition, 2)} of [feature_difference, bias]")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite trials")


def _checkpoint(path: Path, expected: dict) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"stale checkpoint {path}: {key} is {record.get(key)!r}, expected {value!r}")
    return record


class WNMRecoveryAdapter:
    """One adapter exposing every supported objective through production APIs."""

    def __init__(self, protocol: dict):
        spec = protocol["surrogate"]
        checkpoint = (None if spec.get("checkpoint") is None
                      else Path(spec["checkpoint"]))
        loaded = surrogate.load_surrogate(
            family=surrogate.FAMILY_WNM, n_samples=int(spec["n_samples"]),
            checkpoint_path=checkpoint)
        self.predictor = predictor_from_surrogate(loaded)
        self.checkpoint = loaded.path
        self.identity = self.predictor.identity().as_dict()
        self.target_spec = protocol["target"]
        self.n_starts = int(protocol["n_starts"])
        self.fit_motor = bool(protocol["fit_motor"])

    def engine(self, datasets: dict[str, np.ndarray], fit_seed: int) -> ContinuousEngine:
        target = self.target_spec
        engine = ContinuousEngine(
            self.predictor, curve_losses=_compute_curve_losses,
            energy_score=bwcrps_energy_score, degenerate_targets=degenerate_targets,
            bwcrps_condition_targets=compute_bwcrps_condition_targets,
            target_bias_curve_core=compute_target_bias_curve_core,
            emp_density_weights_sd=target["emp_density_weights_sd"],
            density_smoothing_sigma=target["density_smoothing_sigma"],
            density_bandwidth_rule=target["density_bandwidth_rule"],
            density_bandwidth_mode=target["density_bandwidth_mode"],
            corr_weight=target["corr_weight"], skip_motor_noise=not self.fit_motor,
            n_starts=self.n_starts, seed=fit_seed)
        engine.update_dataset({name: jnp.asarray(values)
                               for name, values in datasets.items()})
        return engine

    @staticmethod
    def _trials(datasets):
        return [(jnp.asarray(values[:, 0]), jnp.asarray(values[:, 1]))
                for values in datasets.values()]

    def score(self, objective, engine, datasets, parameters, case):
        predictor = self.predictor
        if not self.fit_motor and case.sd_motor:
            predictor = predictor.with_motor_noise(case.sd_motor)
        return float(S.score_all_conditions(
            objective, predictor, engine.targets, jnp.asarray(parameters),
            curve_losses=_compute_curve_losses, energy_score=bwcrps_energy_score,
            d_circ_matrix=engine.D_circ_matrix,
            feat_diff_grid=engine.feat_diff_grid,
            emp_density_weights_sd=self.target_spec["emp_density_weights_sd"],
            density_smoothing_sigma=self.target_spec["density_smoothing_sigma"],
            corr_weight=self.target_spec["corr_weight"],
            condition_trials=self._trials(datasets), fit_motor=self.fit_motor))

    def fit(self, objective, engine, datasets, case, replicate):
        started = time.time()
        fit = engine.fit(
            objective, sd_motor=0.0 if self.fit_motor else case.sd_motor,
            verbosity=0)
        runtime = time.time() - started
        recovered = np.asarray(
            [fit["condition_results"][name][parameter]
             for name in datasets for parameter in ("sd_feat1", "sd_feat2")]
            + [fit["shared_params"]["sd_spat"]]
            + ([fit["shared_params"]["sd_motor"]] if self.fit_motor else []))
        truth = case.truth_vector(fit_motor=self.fit_motor)
        names = tuple(condition_parameter_layout(
            case.n_conditions, fit_motor=self.fit_motor))
        scores = {method: self.score(method, engine, datasets, recovered, case)
                  for method in S.SUPPORTED_METHODS}
        result = R.RecoveryResult(
            case=case.name, replicate=replicate, truth=truth, recovered=recovered,
            names=names, loss_at_truth=self.score(objective, engine, datasets, truth, case),
            loss_at_fit=scores[objective], loss_spread=float(fit["loss_spread"]),
            n_starts=int(fit["n_starts"]), n_converged=int(fit["n_converged"]),
            at_bound=tuple(fit["at_bound"]), runtime_seconds=runtime,
            start_losses=list(fit["start_losses"]))
        row = result.row() | {
            "objective": objective, "optimizer_loss": float(fit["best_loss"]),
            "canonical_score": scores[objective],
            "optimizer_rescore_delta": scores[objective] - float(fit["best_loss"]),
        }
        row.update({f"score_{method}": value for method, value in scores.items()})
        return result, row


def curve_rows(adapter, engine, datasets, case, result, context):
    """True/recovered/empirical curves without pooling over dissimilarity."""
    feat_grid = np.asarray(engine.feat_diff_grid)
    motors = ([case.sd_motor, result.recovered[-1]] if adapter.fit_motor
              else [case.sd_motor, case.sd_motor])
    rows = []
    for index, (condition, values) in enumerate(datasets.items()):
        truth_params = [*case.condition_feature_sds[index], case.sd_spat]
        recovered_params = [*result.recovered[2 * index:2 * index + 2],
                            result.recovered[2 * case.n_conditions]]
        parameters = np.asarray([truth_params, recovered_params])
        operators = np.repeat(
            np.asarray(engine.targets.feature_operator[index])[None, :, :], 2, axis=0)
        bandwidths = np.repeat(engine.targets.density_bandwidth[index], 2)
        bin_weights = compute_feat_bin_weights(values[:, 0], feat_grid)
        curves = mixture_plot_curves(
            adapter.predictor, parameters, feat_grid,
            bin_weights=np.repeat(bin_weights[None, :, :], 2, axis=0),
            sd_motor_by_row=motors,
            emp_density_weights_sd=adapter.target_spec["emp_density_weights_sd"],
            density_smoothing_sigma=adapter.target_spec["density_smoothing_sigma"],
            feature_operators=operators, density_bandwidths=bandwidths)
        legacy = mixture_plot_curves(
            adapter.predictor, parameters, feat_grid, sd_motor_by_row=motors,
            emp_density_weights_sd=adapter.target_spec["emp_density_weights_sd"],
            density_smoothing_sigma=adapter.target_spec["density_smoothing_sigma"])

        def add(metric, dissimilarity, source_values, counts=None):
            for source, curve in source_values.items():
                for point, value in enumerate(curve):
                    row = context | {
                        "condition": condition, "metric": metric, "source": source,
                        "dissimilarity": float(dissimilarity[point]),
                        "value": (float(value) if np.isfinite(value) else None),
                        "bin_n_trials": None,
                    }
                    if counts is not None:
                        row["bin_n_trials"] = int(counts[point])
                    rows.append(row)

        add("bias", feat_grid, {
            "true": curves["bias"][0], "recovered": curves["bias"][1],
            "empirical": np.asarray(engine.targets.target_bias_curve[index])})
        add("density", feat_grid, {
            "true": curves["asymmetry"][0], "recovered": curves["asymmetry"][1],
            "empirical": np.asarray(engine.targets.matched_density_target[index])})
        add("density_legacy", feat_grid, {
            "true": legacy["asymmetry"][0], "recovered": legacy["asymmetry"][1],
            "empirical": np.asarray(engine.targets.target_density[index])})
        centers, empirical_sd, counts = compute_empirical_sd_curve(
            values[:, 0], values[:, 1])
        add("circular_sd", centers, {
            "true": curves["pooled_sd"][0], "recovered": curves["pooled_sd"][1],
            "empirical": empirical_sd}, counts)
    return rows


def _fit_checkpoint_record(result, row, expected):
    return expected | {
        "row": row, "truth": result.truth.tolist(),
        "recovered": result.recovered.tolist(), "names": list(result.names)}


def _result_from_checkpoint(record):
    row = record["row"]
    return R.RecoveryResult(
        case=row["case"], replicate=int(row["replicate"]),
        truth=np.asarray(record["truth"]), recovered=np.asarray(record["recovered"]),
        names=tuple(record["names"]), loss_at_truth=float(row["loss_at_truth"]),
        loss_at_fit=float(row["loss_at_fit"]), loss_spread=float(row["loss_spread"]),
        n_starts=int(row["n_starts"]), n_converged=int(row["n_converged"]),
        at_bound=tuple(filter(None, str(row["at_bound"]).split(","))),
        runtime_seconds=float(row["runtime_seconds"]),
        start_losses=[float(value) for value in str(row["start_losses"]).split(",")
                      if value])


def _aggregate(out: Path, manifest: dict) -> None:
    cells = [json.loads(path.read_text()) for path in sorted((out / "cells").glob("*.json"))]
    run_rows, parameter_rows, curves = [], [], []
    for cell in cells:
        row = cell["fit"]["row"]
        run_rows.append(row)
        for name, truth, recovered in zip(
                cell["fit"]["names"], cell["fit"]["truth"], cell["fit"]["recovered"]):
            parameter_rows.append({
                key: row[key] for key in
                ("case", "trial_count", "replicate", "objective", "dataset_digest")
            } | {"parameter": name, "true": truth, "recovered": recovered,
                 "signed_error": recovered - truth,
                 "log_ratio": float(np.log(recovered / truth))})
        curves.extend(cell["curves"])
    runs = pd.DataFrame(run_rows)
    parameters = pd.DataFrame(parameter_rows)
    curve_frame = pd.DataFrame(curves)
    _atomic_csv(out / "recovery_runs.csv", runs)
    _atomic_csv(out / "recovery_parameters.csv", parameters)
    _atomic_csv(out / "recovery_curves.csv", curve_frame)
    grouped = parameters.groupby(
        ["objective", "trial_count", "parameter"], as_index=False)
    parameter_summary = grouped.agg(
        n=("log_ratio", "size"), median_signed_error=("signed_error", "median"),
        median_log_ratio=("log_ratio", "median"),
        rmse_log_ratio=("log_ratio", lambda values: float(np.sqrt(np.mean(values ** 2)))))
    _atomic_csv(out / "recovery_parameter_summary.csv", parameter_summary)
    _atomic_json(out / "summary.json", {
        "schema": "wnm_recovery_summary/1",
        "protocol_digest": manifest["protocol_digest"],
        "code_digest": manifest["code_digest"],
        "completed_cells": len(cells),
        "expected_cells": manifest["expected_cells"],
    })
    from standardized_recovery_plots import plot_artifact

    plot_artifact(out)


def run(protocol_path: Path, out: Path) -> None:
    protocol = normalize_protocol(json.loads(protocol_path.read_text()))
    protocol_digest = _canonical_digest(protocol)
    source_digest, source_files = code_digest()
    adapter = WNMRecoveryAdapter(protocol)
    manifest = {
        "schema": "wnm_recovery_manifest/1",
        "protocol": protocol, "protocol_digest": protocol_digest,
        "code_digest": source_digest, "code_files": source_files,
        "surrogate_identity": adapter.identity,
        "checkpoint": str(adapter.checkpoint),
        "checkpoint_sha256": surrogate.file_digest(adapter.checkpoint),
        "optimizer": OPTIMIZER_VERSION,
        "objective_adapters": {
            method: {"optimizer": OPTIMIZER_VERSION,
                     "scorer": "wnm_scoring.score_all_conditions"}
            for method in S.SUPPORTED_METHODS},
        "expected_cells": (len(protocol["cases"]) * len(protocol["trial_counts"])
                           * int(protocol["n_replicates"])
                           * len(protocol["objectives"])),
    }
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("datasets", "fits", "cells"):
        (out / directory).mkdir(exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("protocol_digest", "code_digest", "checkpoint_sha256"):
            if existing.get(key) != manifest[key]:
                raise ValueError(
                    f"cannot resume {out}: manifest {key} changed from "
                    f"{existing.get(key)!r} to {manifest[key]!r}")
    else:
        _atomic_json(manifest_path, manifest)

    ordinal = 0
    for trial_count in protocol["trial_counts"]:
        for case_index, case in enumerate(recovery_cases(protocol, int(trial_count))):
            for replicate in range(int(protocol["n_replicates"])):
                dataset_id = f"{case.name}__n{trial_count}__r{replicate}"
                data_path = _dataset_path(out, dataset_id)
                generated = _source_dataset(
                    protocol, adapter, case, case_index, int(trial_count), replicate)
                _validate_datasets(generated, case)
                if data_path.exists():
                    datasets = _load_dataset(data_path)
                    if dataset_digest(datasets) != dataset_digest(generated):
                        raise ValueError(
                            f"stored dataset {data_path} does not match its protocol seed")
                else:
                    datasets = generated
                    _save_dataset(data_path, datasets)
                data_digest = dataset_digest(datasets)
                fit_seed = int(protocol["seed"]) + ordinal
                engine = adapter.engine(datasets, fit_seed)
                for objective in protocol["objectives"]:
                    cell_id = f"{dataset_id}__{objective}"
                    expected = {
                        "protocol_digest": protocol_digest, "code_digest": source_digest,
                        "checkpoint_sha256": manifest["checkpoint_sha256"],
                        "dataset_digest": data_digest, "cell_id": cell_id,
                        "objective": objective,
                    }
                    cell_path = out / "cells" / f"{cell_id}.json"
                    if _checkpoint(cell_path, expected) is not None:
                        print(f"skip complete {cell_id}", flush=True)
                        continue
                    fit_path = out / "fits" / f"{cell_id}.json"
                    fit_record = _checkpoint(fit_path, expected)
                    if fit_record is None:
                        result, row = adapter.fit(
                            objective, engine, datasets, case, replicate)
                        row |= {"trial_count": int(trial_count), "fit_seed": fit_seed,
                                "dataset_digest": data_digest,
                                "surrogate_family": surrogate.FAMILY_WNM,
                                "surrogate_artifact":
                                    adapter.identity["surrogate_artifact"]}
                        fit_record = _fit_checkpoint_record(result, row, expected)
                        _atomic_json(fit_path, fit_record)
                    else:
                        result = _result_from_checkpoint(fit_record)
                    context = {
                        "case": case.name, "trial_count": int(trial_count),
                        "replicate": replicate, "objective": objective,
                        "dataset_digest": data_digest,
                    }
                    curves = curve_rows(
                        adapter, engine, datasets, case, result, context)
                    _atomic_json(cell_path, expected | {"fit": fit_record, "curves": curves})
                    print(f"complete {cell_id}: {result.diagnosis}", flush=True)
                ordinal += 1
    _aggregate(out, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.protocol, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
