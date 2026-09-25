#!/usr/bin/env python3
"""Freeze and read the first actual-DM recovery panel.

The expensive simulation is delegated to
``surface_computation/generate_wnm_training_data.py``. This module owns only the
agreed single-condition design and the conversion from its raw simulator output
to nested behavioural datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import surrogate
from surface_computation.wnm_design import SIM_SPAT_DIFF


FEATURE_DIFFERENCES = np.arange(2.0, 181.0, 2.0, dtype=np.float32)
TRIAL_COUNTS = (180, 450, 900)
RESPONSE_SEEDS = (0, 1, 2, 3, 4)
N_SAMPLES = 100
GLOBAL_ORIENTATION = 0.0
RESPONSES_PER_DIFFERENCE = 10
RESPONSE_COMPONENT = 0
CODE_PATHS = (
    ROOT / "surface_computation/generate_wnm_training_data.py",
    ROOT / "surface_computation/wnm_simulation.py",
    ROOT / "surface_computation/jax_fit_main.py",
    ROOT / "surface_computation/jax_fit_functions.py",
)
BASELINE_CODE_PATHS = (
    Path(__file__),
    ROOT / "model_fit_to_data/fit_model_to_data.py",
    ROOT / "model_fit_to_data/grid_based_multi_condition_optimizer_jax_loops.py",
    ROOT / "model_fit_to_data/curve_cache.py",
    ROOT / "model_fit_to_data/exhaustive_density.py",
    ROOT / "model_fit_to_data/fitting_targets.py",
    ROOT / "model_fit_to_data/run_fingerprint.py",
    ROOT / "shared/utils.py",
)
BASELINE_METHODS = ("likelihood", "bias_weighted_crps", "smoothed_exp", "density")

# One held-out tuple per regime; all other tuples are search-development cases.
CASES = (
    ("narrow_1", "narrow", "development", 5.0, 10.0, 15.0),
    ("narrow_2", "narrow", "held_out", 7.5, 15.0, 30.0),
    ("narrow_3", "narrow", "development", 10.0, 20.0, 60.0),
    ("ordinary_1", "ordinary_asymmetric", "development", 10.0, 30.0, 60.0),
    ("ordinary_2", "ordinary_asymmetric", "held_out", 15.0, 45.0, 15.0),
    ("ordinary_3", "ordinary_asymmetric", "development", 20.0, 50.0, 30.0),
    ("reversed_1", "reversed_asymmetric", "development", 30.0, 10.0, 60.0),
    ("reversed_2", "reversed_asymmetric", "held_out", 45.0, 15.0, 15.0),
    ("reversed_3", "reversed_asymmetric", "development", 50.0, 20.0, 30.0),
    ("broad_1", "broad_weak", "development", 60.0, 80.0, 30.0),
    ("broad_2", "broad_weak", "held_out", 80.0, 120.0, 60.0),
    ("broad_3", "broad_weak", "development", 120.0, 160.0, 15.0),
)


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def design_rows():
    """Return simulator rows and their case labels in their frozen order."""
    rows = []
    labels = []
    for name, _, _, sd_feat1, sd_feat2, sd_spat in CASES:
        rows.extend((sd_feat1, sd_feat2, sd_spat, float(difference))
                    for difference in FEATURE_DIFFERENCES)
        labels.extend([name] * len(FEATURE_DIFFERENCES))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels)


def protocol():
    """The complete data-generation design, suitable for JSON serialization."""
    design, _ = design_rows()
    checkpoints = {
        "wnm": ROOT / "pretrained/wnm_k12_100samples.pkl",
        "surface_nn": ROOT / "pretrained/model_epoch1500_10ktrain_100samples.pkl",
    }
    return {
        "protocol": "single_condition_actual_dm_recovery_v1",
        "n_samples": N_SAMPLES,
        "model_period_degrees": 360.0,
        "global_orientation_degrees": GLOBAL_ORIENTATION,
        "sd_motor_degrees": 0.0,
        "feature_differences_degrees": FEATURE_DIFFERENCES.tolist(),
        "trial_counts": list(TRIAL_COUNTS),
        "trials_per_feature_difference": {
            str(count): count // len(FEATURE_DIFFERENCES) for count in TRIAL_COUNTS},
        "response_seeds": list(RESPONSE_SEEDS),
        "responses_per_feature_difference": RESPONSES_PER_DIFFERENCE,
        "nested_trial_counts": True,
        "response_component": RESPONSE_COMPONENT,
        "stimulus_geometry": {
            "feature_centres": "[-feature_difference / 2, +feature_difference / 2]",
            "spatial_difference_degrees": SIM_SPAT_DIFF,
        },
        "simulator": {
            "function": "jax_fit_main.simulate_dual_component_bias_distribution",
            "algorithm": "EM",
            "diagonal_covariance": True,
            "fix_weights": False,
        },
        "cases": [
            {"name": name, "regime": regime, "split": split,
             "sd_feat1": sd_feat1, "sd_feat2": sd_feat2, "sd_spat": sd_spat}
            for name, regime, split, sd_feat1, sd_feat2, sd_spat in CASES
        ],
        "n_datasets": len(CASES) * len(RESPONSE_SEEDS) * len(TRIAL_COUNTS),
        "design_sha256": _array_digest(design),
        "fit_artifacts": {
            family: {"path": str(path.relative_to(ROOT)),
                     "sha256": surrogate.file_digest(path)}
            for family, path in checkpoints.items()
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): surrogate.file_digest(path)
            for path in CODE_PATHS
        },
    }


def load_dataset(raw_path: Path, case_name: str, trial_count: int) -> np.ndarray:
    """Read one nested ``[feature_difference, bias]`` behavioural dataset."""
    if trial_count not in TRIAL_COUNTS:
        raise ValueError(f"trial_count must be one of {TRIAL_COUNTS}, got {trial_count}")
    names = [case[0] for case in CASES]
    if case_name not in names:
        raise ValueError(f"unknown recovery case {case_name!r}")

    expected_design, _ = design_rows()
    with np.load(raw_path) as raw:
        design = np.asarray(raw["design"])
        bias = np.asarray(raw["bias"])
    expected_shape = (len(expected_design), RESPONSES_PER_DIFFERENCE, 2)
    if not np.array_equal(design, expected_design) or bias.shape != expected_shape:
        raise ValueError(
            f"{raw_path} does not match the frozen design: design {design.shape}, "
            f"bias {bias.shape}, expected {expected_design.shape} and {expected_shape}")
    if not np.all(np.isfinite(bias)):
        raise ValueError(f"{raw_path} contains non-finite simulator responses")

    start = names.index(case_name) * len(FEATURE_DIFFERENCES)
    stop = start + len(FEATURE_DIFFERENCES)
    per_difference = trial_count // len(FEATURE_DIFFERENCES)
    selected = bias[start:stop, :per_difference, RESPONSE_COMPONENT]
    return np.stack([
        np.repeat(FEATURE_DIFFERENCES, per_difference), selected.reshape(-1)
    ], axis=-1).astype(np.float32)


def _expected_manifest():
    manifest = protocol()
    manifest["design_file"] = "single_condition_design.npz"
    manifest["raw_files"] = {
        str(seed): f"observer_seed_{seed}.npz" for seed in RESPONSE_SEEDS}
    return manifest


def prepare_surface_baseline(output_dir: Path):
    """Write the common public-fitter input and its frozen baseline identity."""
    manifest_path = output_dir / "protocol.json"
    if json.loads(manifest_path.read_text()) != _expected_manifest():
        raise ValueError(f"generated data do not match the current protocol: {manifest_path}")

    frames = []
    raw_digests = {}
    for seed in RESPONSE_SEEDS:
        raw_path = output_dir / f"observer_seed_{seed}.npz"
        raw_digests[str(seed)] = surrogate.file_digest(raw_path)
        for case_name, regime, split, sd_feat1, sd_feat2, sd_spat in CASES:
            for trial_count in TRIAL_COUNTS:
                data = load_dataset(raw_path, case_name, trial_count)
                frames.append(pd.DataFrame({
                    "expName": "single_condition_n100",
                    "subject": f"{case_name}_seed{seed}_n{trial_count}",
                    "condition": "c0",
                    "abs_td_dist": data[:, 0],
                    "bias_to_distr_corr": data[:, 1],
                    "is_outlier": 0,
                    "case": case_name,
                    "regime": regime,
                    "split": split,
                    "response_seed": seed,
                    "n_trials": trial_count,
                    "true_sd_feat1": sd_feat1,
                    "true_sd_feat2": sd_feat2,
                    "true_sd_spat": sd_spat,
                }))
    frame = pd.concat(frames, ignore_index=True)
    input_path = output_dir / "surface_baseline_input.csv"
    temporary = input_path.with_suffix(".csv.partial")
    frame.to_csv(temporary, index=False)
    if input_path.exists():
        if surrogate.file_digest(temporary) != surrogate.file_digest(input_path):
            raise ValueError(f"existing baseline input does not match: {input_path}")
        temporary.unlink()
    else:
        temporary.rename(input_path)

    baseline = {
        "protocol": "single_condition_surface_baseline_v1",
        "source_protocol_sha256": surrogate.file_digest(manifest_path),
        "raw_file_sha256": raw_digests,
        "input_file": input_path.name,
        "input_sha256": surrogate.file_digest(input_path),
        "n_rows": len(frame),
        "n_subjects": frame["subject"].nunique(),
        "checkpoint": protocol()["fit_artifacts"]["surface_nn"],
        "methods": list(BASELINE_METHODS),
        "search": {
            "density": "exhaustive_1_degree_curve_cache",
            "likelihood": "deployed_hierarchical",
            "bias_weighted_crps": "deployed_hierarchical",
            "smoothed_exp": "deployed_hierarchical",
        },
        "circ_space": 360,
        "skip_motor_noise": True,
        "code_sha256": {
            str(path.relative_to(ROOT)): surrogate.file_digest(path)
            for path in BASELINE_CODE_PATHS
        },
        "held_out_policy": "fit now; do not inspect until WNM search settings are frozen",
    }
    baseline_path = output_dir / "surface_baseline_manifest.json"
    if baseline_path.exists() and json.loads(baseline_path.read_text()) != baseline:
        raise ValueError(
            f"existing baseline manifest does not match current inputs: {baseline_path}")
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    return input_path, baseline_path


def surface_baseline_commands(output_dir: Path, curve_cache_root: Path):
    """Public fitter commands implementing the currently deployed surface policy."""
    input_path = output_dir / "surface_baseline_input.csv"
    checkpoint = ROOT / "pretrained/model_epoch1500_10ktrain_100samples.pkl"
    common = [
        sys.executable, "model_fit_to_data/fit_model_to_data.py",
        "--data-path", str(input_path), "--checkpoint-path", str(checkpoint),
        "--circ-space", "360",
    ]
    return [
        common + [
            "--output-dir", str(output_dir / "surface_hierarchical"),
            "--include-methods", "likelihood", "bias_weighted_crps", "smoothed_exp",
            "--search", "hierarchical",
        ],
        common + [
            "--output-dir", str(output_dir / "surface_density_exhaustive"),
            "--include-methods", "density", "--search", "exhaustive",
            "--curve-cache", str(curve_cache_root), "--curve-cache-step", "1.0",
        ],
    ]


def surface_baseline_environment():
    """The import path used by the deployed external fitting pipeline."""
    paths = [str(ROOT), str(ROOT / "neural_network_optimization")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generate", action="store_true",
                        help="run the existing resumable DM simulator after freezing")
    parser.add_argument("--surface-baseline", action="store_true",
                        help="prepare the public-fitter input and run the deployed surface fit")
    parser.add_argument("--curve-cache-root", type=Path,
                        help="required with --surface-baseline for deployed density search")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    design, labels = design_rows()
    design_path = args.out / "single_condition_design.npz"
    if design_path.exists():
        with np.load(design_path) as existing:
            if (not np.array_equal(existing["design"], design)
                    or not np.array_equal(existing["strata"], labels)):
                raise ValueError(f"existing frozen design does not match: {design_path}")
    else:
        np.savez(design_path, design=design, strata=labels)

    manifest = _expected_manifest()
    manifest_path = args.out / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            f"existing frozen protocol does not match current code: {manifest_path}. "
            "Use a new output directory rather than mixing simulation versions.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {design_path} and {manifest_path}")
    commands = []
    for seed in RESPONSE_SEEDS:
        commands.append([
            sys.executable, "-m", "surface_computation.generate_wnm_training_data",
            "--design-file", str(design_path),
            "--n-simulations", str(RESPONSES_PER_DIFFERENCE),
            "--n-samples", str(N_SAMPLES), "--seed", str(seed),
            "--shard-rows", "90", "--block-rows", "90",
            "--simulation-chunk", "10", "--resume",
            "--out", str(args.out / f"observer_seed_{seed}.npz"),
        ])
    if args.generate:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
    elif not args.surface_baseline:
        print("generate each response seed with:")
        for command in commands:
            print("  " + " ".join(command))
    if args.surface_baseline:
        if args.curve_cache_root is None:
            parser.error("--surface-baseline requires --curve-cache-root")
        input_path, baseline_path = prepare_surface_baseline(args.out)
        print(f"wrote {input_path} and {baseline_path}")
        for command in surface_baseline_commands(args.out, args.curve_cache_root):
            subprocess.run(command, cwd=ROOT, check=True,
                           env=surface_baseline_environment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
