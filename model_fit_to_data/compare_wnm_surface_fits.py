#!/usr/bin/env python3
"""Compare paired WNM and surface-NN fits on one common analytic scorer.

Native fit losses are retained for auditing, but are not treated as comparable:
the surface checkpoint keeps the legacy density-curve semantics while the WNM
production path uses the selected observed-design operators.  Every fitted
parameter vector is therefore rescored here through the same WNM predictor and
the same empirical targets.  The resulting diagonal comparison answers whether
each backend's optimizer found parameters that score well under the selected
production objective; the full cross-score export makes trade-offs visible.
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


def _engine(checkpoint: Path, datasets: dict) -> ContinuousEngine:
    loaded = surrogate.load_surrogate(checkpoint_path=str(checkpoint.resolve()))
    engine = ContinuousEngine(
        predictor_from_surrogate(loaded),
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
    paired = {"wnm": _load_results(wnm_path),
              "surface_nn": _load_results(surface_path)}
    conditions = _validate_pair(paired["wnm"], paired["surface_nn"])
    datasets = {
        condition: jnp.asarray(np.asarray(paired["wnm"][condition]["data_df"]))
        for condition in conditions
    }
    engine = _engine(checkpoint, datasets)

    parameter_rows = []
    score_rows = []
    summary_rows = []
    for family in FAMILIES:
        results = paired[family]
        first = results[conditions[0]]
        for fitted_objective in OBJECTIVES:
            params = np.stack([
                np.asarray(results[condition][f"{fitted_objective}_fitted_params"], dtype=float)
                for condition in conditions
            ])
            for condition, values in zip(conditions, params):
                parameter_rows.append({
                    "family": family,
                    "fitted_objective": fitted_objective,
                    "condition": condition,
                    "sd_feat1": values[0], "sd_feat2": values[1],
                    "sd_spat": values[2], "sd_motor": values[3],
                })

            rescored = engine.evaluate(jnp.asarray(params), list(OBJECTIVES))
            for scoring_objective, losses in rescored.items():
                for condition, loss in zip(conditions, np.asarray(losses, dtype=float)):
                    score_rows.append({
                        "family": family,
                        "fitted_objective": fitted_objective,
                        "scoring_objective": scoring_objective,
                        "condition": condition,
                        "common_wnm_loss": float(loss),
                    })

            diagonal = np.asarray(rescored[fitted_objective], dtype=float)
            native = np.asarray([
                results[condition][f"{fitted_objective}_loss"]
                for condition in conditions
            ], dtype=float)
            summary_rows.append({
                "family": family,
                "fitted_objective": fitted_objective,
                "native_loss_sum": float(native.sum()),
                "common_wnm_loss_sum": float(diagonal.sum()),
                "optimization_seconds": float(
                    first[f"{fitted_objective}_optimization_time"]),
                "boundary_hits": _boundary_hits(params, family),
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
    summary = pd.DataFrame(summary_rows)
    best = summary.groupby("fitted_objective")["common_wnm_loss_sum"].transform("min")
    summary["common_loss_minus_best"] = summary["common_wnm_loss_sum"] - best
    n_trials = sum(len(datasets[condition]) for condition in conditions)
    n_parameters = 2 * len(conditions) + 1  # two feature SDs each + shared spatial SD
    is_likelihood = summary["fitted_objective"] == "likelihood"
    summary["common_wnm_aic"] = np.where(
        is_likelihood, 2.0 * summary["common_wnm_loss_sum"] + 2 * n_parameters, np.nan)
    summary["common_wnm_bic"] = np.where(
        is_likelihood,
        2.0 * summary["common_wnm_loss_sum"] + n_parameters * np.log(n_trials),
        np.nan,
    )
    parameters.to_csv(output_dir / "paired_parameters.csv", index=False)
    scores.to_csv(output_dir / "common_wnm_cross_scores.csv", index=False)
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

    display = summary.copy()
    for column in ("native_loss_sum", "common_wnm_loss_sum", "common_loss_minus_best",
                   "optimization_seconds", "common_wnm_aic", "common_wnm_bic"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    lines = [
        "# Paired representative real-data comparison",
        "",
        f"Conditions: {len(conditions)}; retained trials: "
        f"{n_trials:,}.",
        "",
        "The `common_wnm_loss_sum` column is the comparable result: both fitted "
        "parameter sets are evaluated by the same analytic WNM scorer and the selected "
        "observed-design objectives. `native_loss_sum` is retained only as an audit "
        "value because density and smoothed-mean semantics differ across the two fitters.",
        "",
        "Production fitting bounds are family-specific validated domains: WNM feature "
        "SD 2.5–200°, surface-NN feature SD 5–200°, and spatial SD 5–200° for both. "
        "Boundary hits are reported rather than hidden by narrowing the common box.",
        "",
        _markdown_table(display),
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
        "conditions": list(conditions),
        "common_scorer": "wrapped_normal_mixture_selected_objectives",
        "likelihood_information_criteria": {
            "n_trials": n_trials,
            "n_free_parameters": n_parameters,
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
