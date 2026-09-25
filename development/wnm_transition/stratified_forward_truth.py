#!/usr/bin/env python3
"""Prepare and summarize a diagnostic CSH2026 actual-GMM fidelity panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shared import surrogate


BANDS = (
    (0.0, 18.0, "[0,18)"),
    (18.0, 60.0, "[18,60)"),
    (60.0, 120.0, "[60,120)"),
    (120.0, 180.0, "[120,180]"),
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns, path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def build_case_table(comparison_dir: Path, objective: str) -> pd.DataFrame:
    """Recover the same-parameter surface-to-WNM loss gap for every condition."""
    comparison_dir = Path(comparison_dir)
    params_path = comparison_dir / "paired_parameters.csv"
    scores_path = comparison_dir / "common_wnm_cross_scores.csv"
    manifest_path = comparison_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "paired_wnm_surface_comparison":
        raise ValueError(f"unexpected comparison manifest format in {manifest_path}")

    params = pd.read_csv(params_path)
    scores = pd.read_csv(scores_path)
    _require_columns(params, ["condition", "family", "fitted_objective",
                              "sd_feat1", "sd_feat2", "sd_spat", "sd_motor"], params_path)
    _require_columns(scores, ["condition", "family", "fitted_objective",
                              "scoring_objective", "common_wnm_loss"], scores_path)
    params = params[(params.family == "surface_nn") &
                    (params.fitted_objective == objective)].copy()
    scores = scores[(scores.family == "surface_nn") &
                    (scores.fitted_objective == objective) &
                    (scores.scoring_objective == objective)][
                        ["condition", "common_wnm_loss"]].copy()
    if params.condition.duplicated().any() or scores.condition.duplicated().any():
        raise ValueError("surface parameter/score rows are not unique by condition")
    cases = params.merge(scores, on="condition", validate="one_to_one")

    surface_path = Path(manifest["surface_results"])
    with surface_path.open("rb") as handle:
        surface_results = pickle.load(handle)
    missing = sorted(set(cases.condition) - set(surface_results))
    if missing:
        raise ValueError(f"surface results lack conditions: {', '.join(missing[:5])}")
    cases["surface_native_loss"] = [
        float(np.asarray(surface_results[key][f"{objective}_loss"]))
        for key in cases.condition
    ]
    cases = cases.rename(columns={"common_wnm_loss": "same_parameter_wnm_loss"})
    cases["forward_loss_gap"] = (
        cases.same_parameter_wnm_loss - cases.surface_native_loss)
    cases["abs_forward_loss_gap"] = cases.forward_loss_gap.abs()
    numeric = ["sd_feat1", "sd_feat2", "sd_spat", "sd_motor",
               "surface_native_loss", "same_parameter_wnm_loss", "forward_loss_gap"]
    if not np.isfinite(cases[numeric].to_numpy()).all():
        raise ValueError("case table contains non-finite parameters or losses")
    return cases.sort_values("condition").reset_index(drop=True)


def select_cases(cases: pd.DataFrame, n_per_stratum: int) -> pd.DataFrame:
    """Choose disjoint high, median-nearest, and low absolute-gap cases."""
    if n_per_stratum < 1 or len(cases) < 3 * n_per_stratum:
        raise ValueError("need at least three disjoint non-empty gap strata")
    ordered = cases.sort_values(["abs_forward_loss_gap", "condition"])
    low = ordered.head(n_per_stratum)
    high = ordered.tail(n_per_stratum).sort_values(
        ["abs_forward_loss_gap", "condition"], ascending=[False, True])
    used = set(low.index) | set(high.index)
    middle = ordered.loc[~ordered.index.isin(used)].copy()
    target = float(cases.abs_forward_loss_gap.median())
    middle["median_distance"] = (middle.abs_forward_loss_gap - target).abs()
    medium = middle.sort_values(["median_distance", "condition"]).head(n_per_stratum)
    selected = []
    for label, frame in (("high", high), ("medium", medium), ("low", low)):
        part = frame.drop(columns="median_distance", errors="ignore").copy()
        part.insert(0, "gap_stratum", label)
        selected.append(part)
    out = pd.concat(selected, ignore_index=True)
    out["absolute_gap_percentile"] = out.abs_forward_loss_gap.map(
        lambda value: float((cases.abs_forward_loss_gap <= value).mean()))
    return out


def prepare_panel(args) -> None:
    cases = build_case_table(args.comparison_dir, args.objective)
    selected = select_cases(cases, args.n_per_stratum)
    if not np.allclose(selected.sd_motor, 0.0, rtol=0, atol=1e-8):
        raise ValueError("actual-GMM panel currently requires zero-motor-noise fits")
    grid = np.arange(args.feat_start, args.feat_stop + args.feat_step / 2,
                     args.feat_step, dtype=np.float32)
    if len(grid) < 2 or grid[-1] > args.feat_stop + 1e-6:
        raise ValueError("invalid feature-dissimilarity grid")

    design_parts, labels, rows = [], [], []
    for case_index, row in selected.iterrows():
        design = np.column_stack([
            np.full(len(grid), row.sd_feat1), np.full(len(grid), row.sd_feat2),
            np.full(len(grid), row.sd_spat), grid,
        ]).astype(np.float32)
        label = f"gap_{row.gap_stratum}::{row.condition}"
        design_parts.append(design)
        labels.extend([label] * len(grid))
        for feat_diff in grid:
            rows.append({
                "design_row": len(rows), "case_index": case_index,
                "gap_stratum": row.gap_stratum, "condition": row.condition,
                "feat_diff": feat_diff,
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    design_path = args.out_dir / "design.npz"
    np.savez(design_path, design=np.concatenate(design_parts),
             strata=np.asarray(labels, dtype=object))
    selected.to_csv(args.out_dir / "selected_cases.csv", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "design_rows.csv", index=False)

    comparison_manifest = json.loads(
        (args.comparison_dir / "manifest.json").read_text())
    wnm_checkpoint = Path(comparison_manifest["checkpoint"]).resolve()
    surface_results = Path(comparison_manifest["surface_results"]).resolve()
    surface_checkpoint = surrogate.checkpoint_for_run(surface_results)
    raw_reference = args.out_dir / "actual_gmm_reference.npz"
    comparison_output = args.out_dir / "raw_truth_comparison.csv"
    # Preserve the virtualenv entry point. Resolving its symlink produces
    # /usr/bin/python and a copied reproduction command then leaves the project env.
    python = Path(sys.executable).absolute()
    commands = [
        [str(python), "surface_computation/generate_wnm_training_data.py",
         "--design-file", str(design_path.resolve()), "--n-simulations",
         str(args.n_simulations), "--n-samples", str(args.n_samples),
         "--crn-within-trajectories", "--seed", str(args.seed),
         "--block-rows", "4", "--shard-rows", "4", "--simulation-chunk", "5000",
         "--resume", "--out", str(raw_reference.resolve())],
        [str(python), "development/wnm_transition/compare_existing_model.py",
         "--model", str(wnm_checkpoint), "--checkpoint", str(surface_checkpoint),
         "--reference", str(raw_reference.resolve()), "--out",
         str(comparison_output.resolve())],
        [str(python), "development/wnm_transition/stratified_forward_truth.py", "summarize",
         "--comparison", str(comparison_output.resolve()), "--out-dir",
         str(args.out_dir.resolve())],
    ]
    manifest = {
        "format": "stratified_actual_gmm_panel", "version": 1,
        "objective": args.objective, "gap_definition":
            "abs(common WNM loss at surface-fit parameters - surface native loss)",
        "n_per_stratum": args.n_per_stratum, "n_samples": args.n_samples,
        "n_simulations_per_dissimilarity": args.n_simulations, "seed": args.seed,
        "feature_grid": grid.tolist(), "dissimilarity_bands": [band[2] for band in BANDS],
        "sources": {
            "paired_parameters": _sha256(args.comparison_dir / "paired_parameters.csv"),
            "common_wnm_cross_scores": _sha256(
                args.comparison_dir / "common_wnm_cross_scores.csv"),
            "surface_results": _sha256(surface_results),
            "wnm_checkpoint": _sha256(wnm_checkpoint),
            "surface_checkpoint": _sha256(surface_checkpoint),
        },
        "commands": [shlex.join(command) for command in commands],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(selected[["gap_stratum", "condition", "forward_loss_gap",
                    "abs_forward_loss_gap"]].to_string(index=False))
    print("\nNext commands (run sequentially from the DM repository):")
    for command in commands:
        print(shlex.join(command))


def dissimilarity_band(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    labels = np.full(values.shape, "", dtype=object)
    for low, high, label in BANDS:
        upper = values <= high if high == 180.0 else values < high
        labels[(values >= low) & upper] = label
    if np.any(labels == ""):
        raise ValueError("feature dissimilarities fall outside the declared bands")
    return labels


def _metric_summary(frame: pd.DataFrame, group_columns) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        wnm_mean = group.pred_mean_bias_error.to_numpy()
        surface_mean = group.production_mean_bias_error.to_numpy()
        wnm_sd = (group.pred_circ_sd - group.ref_circ_sd).to_numpy()
        surface_sd = (group.production_circ_sd - group.ref_circ_sd).to_numpy()
        wnm_asym = (group.pred_density_asym - group.ref_density_asym).to_numpy()
        surface_asym = (
            group.production_density_asym - group.ref_density_asym).to_numpy()
        row = dict(zip(group_columns, keys))
        row.update({
            "n_cases": group.condition.nunique(), "n_grid_points": len(group),
            "wnm_mean_bias_rmse_deg": float(np.sqrt(np.mean(wnm_mean ** 2))),
            "surface_mean_bias_rmse_deg": float(np.sqrt(np.mean(surface_mean ** 2))),
            "wnm_circ_sd_rmse_deg": float(np.sqrt(np.mean(wnm_sd ** 2))),
            "surface_circ_sd_rmse_deg": float(np.sqrt(np.mean(surface_sd ** 2))),
            "wnm_asymmetry_rmse": float(np.sqrt(np.mean(wnm_asym ** 2))),
            "surface_asymmetry_rmse": float(np.sqrt(np.mean(surface_asym ** 2))),
            "wnm_mean_nll": float(group.nll_density_model.mean()),
            "surface_mean_nll": float(group.nll_production_interp.mean()),
            "wnm_mean_density_l1": float(group.l1_density_model.mean()),
            "surface_mean_density_l1": float(group.l1_production.mean()),
        })
        for metric in ("mean_bias_rmse_deg", "circ_sd_rmse_deg", "asymmetry_rmse",
                       "mean_nll", "mean_density_l1"):
            row[f"{metric}_surface_minus_wnm"] = row[f"surface_{metric}"] - row[f"wnm_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_panel(args) -> None:
    frame = pd.read_csv(args.comparison)
    required = [
        "stratum", "component", "feat_diff", "pred_mean_bias_error",
        "production_mean_bias_error", "ref_circ_sd", "pred_circ_sd",
        "production_circ_sd", "ref_density_asym", "pred_density_asym",
        "production_density_asym", "nll_density_model", "nll_production_interp",
        "l1_density_model", "l1_production",
    ]
    _require_columns(frame, required, args.comparison)
    labels = frame.stratum.str.extract(r"^gap_(high|medium|low)::(.+)$")
    if labels.isna().any().any():
        raise ValueError("comparison contains an unrecognized panel stratum label")
    frame[["gap_stratum", "condition"]] = labels
    frame["dissimilarity_band"] = dissimilarity_band(frame.feat_diff)
    numeric = list(set(required) - {"stratum"})
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("comparison contains non-finite fidelity values")

    by_case = _metric_summary(
        frame, ["gap_stratum", "condition", "component", "dissimilarity_band"])
    by_stratum = _metric_summary(
        frame, ["gap_stratum", "component", "dissimilarity_band"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_case.to_csv(args.out_dir / "fidelity_by_case_and_dissimilarity_band.csv", index=False)
    by_stratum.to_csv(
        args.out_dir / "fidelity_by_gap_stratum_and_dissimilarity_band.csv", index=False)
    evidence = {
        "format": "stratified_actual_gmm_fidelity_summary", "version": 1,
        "sampling_scope": (
            "diagnostic extreme-and-median gap cases; not a corpus-average sample"),
        "primary_component": 1,
        "positive_surface_minus_wnm_values_favor_wnm": True,
        "primary_band_rows": by_stratum[by_stratum.component == 1].to_dict("records"),
        "decision_status": "diagnostic_only",
    }
    (args.out_dir / "fidelity_summary.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    print(by_stratum.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="select cases and write the simulation design")
    prepare.add_argument("--comparison-dir", type=Path, required=True)
    prepare.add_argument("--objective", default="smoothed_exp")
    prepare.add_argument("--n-per-stratum", type=int, default=2)
    prepare.add_argument("--feat-start", type=float, default=2.0)
    prepare.add_argument("--feat-stop", type=float, default=180.0)
    prepare.add_argument("--feat-step", type=float, default=2.0)
    prepare.add_argument("--n-samples", type=int, default=20, choices=(20, 100))
    prepare.add_argument("--n-simulations", type=int, default=20_000)
    prepare.add_argument("--seed", type=int, default=271828)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.set_defaults(func=prepare_panel)
    summarize = subparsers.add_parser("summarize", help="summarize shared-truth fidelity by band")
    summarize.add_argument("--comparison", type=Path, required=True)
    summarize.add_argument("--out-dir", type=Path, required=True)
    summarize.set_defaults(func=summarize_panel)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
