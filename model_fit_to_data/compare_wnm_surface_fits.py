#!/usr/bin/env python3
"""Compare objective-matched WNM and surface-NN fits.

The curve-objective native losses are directly comparable because both fitters
use the selected observed-design operators. Every fitted parameter vector is
also rescored through the WNM predictor; that second comparison isolates the
parameters/search from the surface-versus-WNM forward representation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuous_fit import ContinuousEngine
from density_objective import degenerate_targets
from fit_model_to_data import DENSITY_CURVE_SPEC
from grid_based_multi_condition_optimizer_jax_loops import (
    _compute_curve_losses,
    bwcrps_energy_score,
    compute_bwcrps_condition_targets,
    compute_target_bias_curve_core,
)
from run_fingerprint import file_sha256, objective_versions_for, read_fingerprint_sidecar
from shared import surrogate
from shared.prediction import predictor_from_surrogate


OBJECTIVES = ("likelihood", "bias_weighted_crps", "density", "smoothed_exp")
FAMILIES = ("wnm", "surface_nn")


def _results_file(path: Path) -> Path:
    path = path.resolve()
    return path / "extended_fit_results.pkl" if path.is_dir() else path


def _load_results(path: Path) -> dict:
    source = _results_file(path)
    with source.open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{source} does not contain a non-empty results mapping")
    return result


def _validate_fingerprints(wnm_path: Path, surface_path: Path) -> dict:
    sidecars = {
        "wnm": read_fingerprint_sidecar(_results_file(wnm_path).parent),
        "surface_nn": read_fingerprint_sidecar(_results_file(surface_path).parent),
    }
    for family, sidecar in sidecars.items():
        if sidecar is None:
            raise ValueError(f"{family} results have no run fingerprint")
        versions = sidecar["payload"]["objective_versions"]
        expected = objective_versions_for(family, ("density", "smoothed_exp"))
        for objective, version in expected.items():
            if versions.get(objective) != version:
                raise ValueError(
                    f"{family} {objective} objective is {versions.get(objective)!r}; "
                    f"the paired comparison requires {version!r}")
    if (sidecars["wnm"]["payload"]["data_sha256"] !=
            sidecars["surface_nn"]["payload"]["data_sha256"]):
        raise ValueError("WNM and surface results were fitted to different prepared data")
    return sidecars


def _validate_pair(wnm: dict, surface: dict) -> tuple[str, ...]:
    conditions = tuple(wnm)
    if tuple(surface) != conditions:
        raise ValueError(
            "paired results have different condition order or identities: "
            f"WNM={conditions}, surface={tuple(surface)}"
        )
    for condition in conditions:
        left = np.asarray(wnm[condition]["data_df"])
        right = np.asarray(surface[condition]["data_df"])
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"prepared trials differ for paired condition {condition!r}")
        for objective in OBJECTIVES:
            key = f"{objective}_fitted_params"
            if key not in wnm[condition] or key not in surface[condition]:
                raise ValueError(f"paired results are missing {key!r} for {condition!r}")
    return conditions


def _engine(predictor, datasets: dict) -> ContinuousEngine:
    engine = ContinuousEngine(
        predictor,
        curve_losses=_compute_curve_losses,
        energy_score=bwcrps_energy_score,
        degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core,
        corr_weight=0.25,
        skip_motor_noise=True,
        **DENSITY_CURVE_SPEC,
    )
    engine.update_dataset(datasets)
    return engine


def _fit_groups(conditions: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Preserve fit order while grouping conditions by subject and experiment."""
    groups = {}
    for condition in conditions:
        if "#" not in condition:
            raise ValueError(f"condition key has no fit-group separator: {condition!r}")
        groups.setdefault(condition.rsplit("#", 1)[0], []).append(condition)
    return {group: tuple(members) for group, members in groups.items()}


def _boundary_hits(params: np.ndarray, family: str) -> str:
    feature_low = 2.5 if family == "wnm" else 5.0
    hits = []
    for index, row in enumerate(params):
        for axis, value, low in (("sd_feat1", row[0], feature_low),
                                 ("sd_feat2", row[1], feature_low)):
            if np.isclose(value, low, rtol=0, atol=2e-4):
                hits.append(f"{axis}_c{index}@low")
            elif np.isclose(value, 200.0, rtol=0, atol=2e-4):
                hits.append(f"{axis}_c{index}@high")
    spatial = params[0, 2]
    if np.isclose(spatial, 5.0, rtol=0, atol=2e-4):
        hits.append("sd_spat@low")
    elif np.isclose(spatial, 200.0, rtol=0, atol=2e-4):
        hits.append("sd_spat@high")
    return ",".join(hits)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(str(column)), *(len(row[i]) for row in rows))
              for i, column in enumerate(columns)]
    header = "| " + " | ".join(str(column).ljust(widths[i])
                                for i, column in enumerate(columns)) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[i])
                               for i, value in enumerate(row)) + " |"
            for row in rows]
    return "\n".join([header, rule, *body])


def compare(wnm_path: Path, surface_path: Path, checkpoint: Path,
            output_dir: Path) -> pd.DataFrame:
    """Write paired comparison tables, plot, and a compact Markdown report."""
    sidecars = _validate_fingerprints(wnm_path, surface_path)
    if file_sha256(checkpoint) != sidecars["wnm"]["payload"]["checkpoint_sha256"]:
        raise ValueError("common-scoring checkpoint does not match the fitted WNM")
    paired = {"wnm": _load_results(wnm_path),
              "surface_nn": _load_results(surface_path)}
    conditions = _validate_pair(paired["wnm"], paired["surface_nn"])
    all_datasets = {
        condition: jnp.asarray(np.asarray(paired["wnm"][condition]["data_df"]))
        for condition in conditions
    }
    fit_groups = _fit_groups(conditions)
    loaded = surrogate.load_surrogate(checkpoint_path=str(checkpoint.resolve()))
    predictor = predictor_from_surrogate(loaded)

    parameter_rows = []
    score_rows = []
    group_summary_rows = []
    for fit_group, group_conditions in fit_groups.items():
        datasets = {condition: all_datasets[condition] for condition in group_conditions}
        engine = _engine(predictor, datasets)
        for family in FAMILIES:
            results = paired[family]
            first = results[group_conditions[0]]
            for fitted_objective in OBJECTIVES:
                params = np.stack([
                    np.asarray(results[condition][f"{fitted_objective}_fitted_params"],
                               dtype=float)
                    for condition in group_conditions
                ])
                if not np.allclose(params[:, 2:], params[0, 2:], rtol=0, atol=2e-4):
                    raise ValueError(
                        f"{family} {fitted_objective} has inconsistent shared parameters "
                        f"within {fit_group!r}")
                for condition, values in zip(group_conditions, params):
                    parameter_rows.append({
                        "fit_group": fit_group,
                        "family": family,
                        "fitted_objective": fitted_objective,
                        "condition": condition,
                        "sd_feat1": values[0], "sd_feat2": values[1],
                        "sd_spat": values[2], "sd_motor": values[3],
                    })

                rescored = engine.evaluate(jnp.asarray(params), list(OBJECTIVES))
                for scoring_objective, losses in rescored.items():
                    for condition, loss in zip(group_conditions,
                                               np.asarray(losses, dtype=float)):
                        score_rows.append({
                            "fit_group": fit_group,
                            "family": family,
                            "fitted_objective": fitted_objective,
                            "scoring_objective": scoring_objective,
                            "condition": condition,
                            "common_wnm_loss": float(loss),
                        })

                diagonal = np.asarray(rescored[fitted_objective], dtype=float)
                native = np.asarray([
                    results[condition][f"{fitted_objective}_loss"]
                    for condition in group_conditions
                ], dtype=float)
                hits = _boundary_hits(params, family)
                group_summary_rows.append({
                    "fit_group": fit_group,
                    "family": family,
                    "fitted_objective": fitted_objective,
                    "native_loss_sum": float(native.sum()),
                    "common_wnm_loss_sum": float(diagonal.sum()),
                    "optimization_seconds": float(
                        first[f"{fitted_objective}_optimization_time"]),
                    "boundary_hits": hits,
                    "n_boundary_hits": len(hits.split(",")) if hits else 0,
                    "n_converged": (
                        first.get(f"{fitted_objective}_n_converged")
                        if family == "wnm" else np.nan
                    ),
                    "n_starts": (
                        first.get(f"{fitted_objective}_n_starts")
                        if family == "wnm" else np.nan
                    ),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = pd.DataFrame(parameter_rows)
    scores = pd.DataFrame(score_rows)
    group_summary = pd.DataFrame(group_summary_rows)
    summary = group_summary.groupby(
        ["family", "fitted_objective"], sort=False, as_index=False
    ).agg(
        native_loss_sum=("native_loss_sum", "sum"),
        common_wnm_loss_sum=("common_wnm_loss_sum", "sum"),
        optimization_seconds=("optimization_seconds", "sum"),
        n_boundary_hits=("n_boundary_hits", "sum"),
        n_converged=("n_converged", lambda values: values.sum(min_count=1)),
        n_starts=("n_starts", lambda values: values.sum(min_count=1)),
    )
    best = summary.groupby("fitted_objective")["common_wnm_loss_sum"].transform("min")
    summary["common_loss_minus_best"] = summary["common_wnm_loss_sum"] - best
    native_best = summary.groupby("fitted_objective")["native_loss_sum"].transform("min")
    summary["native_loss_minus_best"] = np.where(
        summary["fitted_objective"].isin(("density", "smoothed_exp")),
        summary["native_loss_sum"] - native_best, np.nan)
    n_trials = sum(len(all_datasets[condition]) for condition in conditions)
    n_parameters = 2 * len(conditions) + len(fit_groups)
    is_likelihood = summary["fitted_objective"] == "likelihood"
    summary["common_wnm_aic"] = np.where(
        is_likelihood, 2.0 * summary["common_wnm_loss_sum"] + 2 * n_parameters, np.nan)
    summary["common_wnm_bic"] = np.where(
        is_likelihood,
        2.0 * summary["common_wnm_loss_sum"] + n_parameters * np.log(n_trials),
        np.nan,
    )
    paired_groups = group_summary.pivot(
        index=["fit_group", "fitted_objective"], columns="family",
        values=["native_loss_sum", "common_wnm_loss_sum"]
    ).reset_index()
    paired_groups.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in paired_groups.columns
    ]
    paired_groups["native_loss_surface_minus_wnm"] = (
        paired_groups["native_loss_sum_surface_nn"]
        - paired_groups["native_loss_sum_wnm"])
    paired_groups["common_loss_surface_minus_wnm"] = (
        paired_groups["common_wnm_loss_sum_surface_nn"]
        - paired_groups["common_wnm_loss_sum_wnm"])

    objective_rows = []
    for objective, rows in paired_groups.groupby("fitted_objective", sort=False):
        native_delta = rows["native_loss_surface_minus_wnm"]
        common_delta = rows["common_loss_surface_minus_wnm"]
        native_comparable = objective in ("density", "smoothed_exp")
        objective_rows.append({
            "fitted_objective": objective,
            "n_fit_groups": len(rows),
            "native_surface_minus_wnm_mean": (
                float(native_delta.mean()) if native_comparable else np.nan),
            "native_surface_minus_wnm_median": (
                float(native_delta.median()) if native_comparable else np.nan),
            "native_wnm_wins": int((native_delta > 0).sum()) if native_comparable else np.nan,
            "native_surface_wins": int((native_delta < 0).sum()) if native_comparable else np.nan,
            "common_surface_minus_wnm_mean": float(common_delta.mean()),
            "common_wnm_wins": int((common_delta > 0).sum()),
            "common_surface_wins": int((common_delta < 0).sum()),
        })
    objective_summary = pd.DataFrame(objective_rows)
    parameters.to_csv(output_dir / "paired_parameters.csv", index=False)
    scores.to_csv(output_dir / "common_wnm_cross_scores.csv", index=False)
    group_summary.to_csv(output_dir / "paired_group_summary.csv", index=False)
    paired_groups.to_csv(output_dir / "paired_group_differences.csv", index=False)
    objective_summary.to_csv(output_dir / "paired_objective_summary.csv", index=False)
    summary.to_csv(output_dir / "paired_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for objective, axis in zip(OBJECTIVES, axes.flat):
        selected = summary[summary["fitted_objective"] == objective]
        bars = axis.bar(selected["family"], selected["common_loss_minus_best"],
                        color=["#28666e", "#c97b63"])
        raw = selected["common_wnm_loss_sum"].to_numpy()
        gap = selected["common_loss_minus_best"].to_numpy()
        pad = max(float(np.max(gap)) * 0.03, 1e-6)
        for bar, raw_loss in zip(bars, raw):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                      f"score {raw_loss:.4f}", ha="center", va="bottom", fontsize=8)
        axis.set_ylim(0, max(float(np.max(gap)) * 1.2, pad * 8))
        axis.set_title(objective.replace("_", " "))
        axis.set_ylabel("common WNM loss gap to best")
        axis.tick_params(axis="x", rotation=12)
    fig.suptitle("Paired real-data fits rescored with one analytic WNM scorer")
    fig.savefig(output_dir / "paired_common_scores.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for objective, axis in zip(("density", "smoothed_exp"), axes):
        delta = paired_groups.loc[
            paired_groups["fitted_objective"] == objective,
            "native_loss_surface_minus_wnm",
        ].sort_values().to_numpy()
        axis.axhline(0, color="black", linewidth=0.8)
        axis.scatter(np.arange(1, len(delta) + 1), delta, s=16, color="#28666e")
        axis.set_title(objective.replace("_", " "))
        axis.set_xlabel("fit group, sorted by difference")
        axis.set_ylabel("native surface − WNM loss")
    fig.suptitle("Objective-matched curve fits (positive values favor WNM)")
    fig.savefig(output_dir / "paired_curve_native_differences.png", dpi=180)
    plt.close(fig)

    display = summary.copy()
    for column in ("native_loss_sum", "native_loss_minus_best", "common_wnm_loss_sum",
                   "common_loss_minus_best",
                   "optimization_seconds", "common_wnm_aic", "common_wnm_bic"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    objective_display = objective_summary.copy()
    for column in ("native_surface_minus_wnm_mean",
                   "native_surface_minus_wnm_median",
                   "common_surface_minus_wnm_mean"):
        objective_display[column] = objective_display[column].map(
            lambda value: f"{value:.4f}")
    lines = [
        "# Paired representative real-data comparison",
        "",
        f"Fit groups: {len(fit_groups)}; conditions: {len(conditions)}; retained trials: "
        f"{n_trials:,}.",
        "",
        "For density and smoothed expectation, `native_loss_sum` is the direct "
        "model-family comparison: both fits use the same target, observed-design "
        "operator, and loss. `common_wnm_loss_sum` evaluates both parameter sets through "
        "the analytic WNM and therefore isolates parameter/search quality. Native "
        "likelihood and BWCRPS retain family-specific grid-versus-continuous conventions.",
        "",
        "Production fitting bounds are family-specific validated domains: WNM feature "
        "SD 2.5–200°, surface-NN feature SD 5–200°, and spatial SD 5–200° for both. "
        "Boundary hits are reported rather than hidden by narrowing the common box.",
        "Per-fit convergence counts and boundary details are in "
        "`paired_group_summary.csv`; the table below aggregates them.",
        "",
        _markdown_table(display),
        "",
        "Paired differences use surface minus WNM loss, so positive values favor WNM. "
        "Native differences are intentionally omitted for likelihood and BWCRPS because "
        "their family-specific evaluation conventions differ.",
        "",
        _markdown_table(objective_display),
        "",
        "The surface-NN rows are a common analytic rescore of its fitted parameter "
        "vectors, not a claim that the NN forward surface equals the WNM. Native losses "
        "remain in the table for that distinction.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "format": "paired_wnm_surface_comparison",
        "version": 1,
        "wnm_results": str(_results_file(wnm_path)),
        "surface_results": str(_results_file(surface_path)),
        "checkpoint": str(checkpoint.resolve()),
        "objectives": list(OBJECTIVES),
        "fit_groups": list(fit_groups),
        "conditions": list(conditions),
        "common_scorer": "wrapped_normal_mixture_selected_objectives",
        "likelihood_information_criteria": {
            "n_trials": n_trials,
            "n_free_parameters": n_parameters,
            "parameter_count": "two feature SDs per condition plus one spatial SD per fit group",
            "aic": "2 * common_wnm_nll + 2 * k",
            "bic": "2 * common_wnm_nll + k * log(n_trials)",
        },
        "bounds_policy": {
            "wnm_sd_feat": [2.5, 200.0],
            "surface_nn_sd_feat": [5.0, 200.0],
            "sd_spat": [5.0, 200.0],
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wnm-results", type=Path, required=True)
    parser.add_argument("--surface-results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.wnm_results, args.surface_results, args.checkpoint,
                      args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote paired comparison to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
