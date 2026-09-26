#!/usr/bin/env python3
"""Generate a model-grounded illustration of the current WNM pipeline.

The construction panels illustrate the operations rather than one training run.
The response density comes from the WNM simulator; the network tile and curves
use packaged-WNM predictions and current fit exports.
"""

import argparse
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from docs.generate_readme_figures import _first_list, prepare_fitting_data  # noqa: E402
from shared.mu1_axis import bin_indices_np, mu1_grid_np  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402
from shared.surrogate import load_surrogate  # noqa: E402
from surface_computation.wnm_simulation import simulate  # noqa: E402

plt.style.use("default")


NAVY = "#101f5b"
INK = "#172033"
MUTED = "#53627c"
CONSTRUCTION = "#9db2d9"
CONSTRUCTION_BG = "#f5f7fb"
GREEN = "#138a36"
GREEN_BG = "#fbfefb"
BLUE = "#2878e8"
ORANGE = "#f26b21"
GRID = "#d7dde7"


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _rounded_box(fig, bounds, edge, face="white", linewidth=1.5, radius=0.018):
    x, y, width, height = bounds
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure, facecolor=face, edgecolor=edge,
        linewidth=linewidth, clip_on=False, zorder=-2,
    )
    fig.patches.append(patch)
    return patch


def _arrow(fig, start, end, color=NAVY, linewidth=2.2, head_size=14):
    arrow = FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=head_size, linewidth=linewidth, color=color,
        clip_on=False, zorder=5,
    )
    fig.patches.append(arrow)


def _polyline(fig, points, color=NAVY, linewidth=2.2):
    """Add a clean connector without arrowheads in figure coordinates."""
    line = Line2D(
        *zip(*points), transform=fig.transFigure, color=color, linewidth=linewidth,
        solid_capstyle="round", solid_joinstyle="round", clip_on=False, zorder=4,
    )
    fig.lines.append(line)


def _response_surfaces(predictions):
    """Simulated training responses and a matching packaged-WNM prediction."""
    row = predictions.iloc[len(predictions) // 2]
    feature_grid = _first_list(predictions["feat_diff_grid"])
    params = {name: float(row[name]) for name in ("sd_feat1", "sd_feat2", "sd_spat")}
    n_samples = int(predictions["n_samples"].dropna().iloc[0])
    design = np.column_stack((
        *(np.full_like(feature_grid, params[name]) for name in params), feature_grid,
    ))
    predictor = predictor_from_surrogate(load_surrogate(n_samples=n_samples))
    mass = np.asarray(predictor.cell_probabilities(design))
    bias_grid = mu1_grid_np()
    responses = np.asarray(simulate(jax.random.PRNGKey(7), design,
                                    n_simulations=200, n_samples=n_samples,
                                    block_rows=4))[:, :, 0]
    counts = np.stack([np.bincount(bin_indices_np(row), minlength=len(bias_grid))
                       for row in responses])
    simulated = gaussian_filter(counts.astype(float), sigma=(1.5, 1.5),
                                mode=("nearest", "wrap"))
    simulated /= simulated.sum(axis=1, keepdims=True)
    return {"parameters": {**params, "n_samples": n_samples,
                           "random_seed": 7},
            "surface": simulated.T, "predicted_surface": mass.T,
            "feat_diff_grid": feature_grid,
            "mu1_bias_grid": bias_grid}


def _generate_evidence(surface_params, feature_difference=40.0, ident_difference=25.0):
    """Sample the model's two Gaussian evidence distributions with known sources."""
    rng = np.random.default_rng(int(surface_params.get("random_seed", 7)))
    total = int(surface_params.get("n_samples", 100))
    counts = (total // 2, total - total // 2)
    sf1 = float(surface_params["sd_feat1"])
    sf2 = float(surface_params["sd_feat2"])
    ident_sd = float(surface_params["sd_spat"])
    means = np.array([
        [-feature_difference / 2, -ident_difference / 2],
        [feature_difference / 2, ident_difference / 2],
    ])
    samples = [
        rng.multivariate_normal(means[0], np.diag([sf1**2, ident_sd**2]), counts[0]),
        rng.multivariate_normal(means[1], np.diag([sf2**2, ident_sd**2]), counts[1]),
    ]
    return means, samples


def _plot_simulation(ax, surface_params):
    means, samples = _generate_evidence(surface_params)
    for points, mean, color, label in zip(samples, means, (BLUE, ORANGE), ("S1", "S2")):
        ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.72, color=color,
                   edgecolor="none", rasterized=True)
        ax.scatter(*mean, s=55, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(label, mean, xytext=(5, 5), textcoords="offset points",
                    color=color, weight="bold", fontsize=7)
    ax.set_xlabel("feature evidence", fontsize=8, labelpad=2)
    ax.set_ylabel("identifiability evidence", fontsize=8, labelpad=2)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.5)


def _surface_density(record, field="surface"):
    density = np.asarray(record[field], dtype=float)
    feature_grid = np.asarray(record["feat_diff_grid"], dtype=float)
    bias_grid = np.asarray(record["mu1_bias_grid"], dtype=float)
    mask_bias = (bias_grid >= -60) & (bias_grid <= 60)
    mask_feature = feature_grid <= 140
    relative = density[np.ix_(mask_bias, mask_feature)]
    relative /= relative.max()
    return feature_grid[mask_feature], bias_grid[mask_bias], relative


def _plot_surface(ax, record):
    feature_grid, bias_grid, relative = _surface_density(record)
    x, y = np.meshgrid(feature_grid, bias_grid)
    ax.plot_surface(x, y, relative, cmap="viridis", rstride=3, cstride=4,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel("dissimilarity", fontsize=6, labelpad=-1)
    ax.set_ylabel("response bias", fontsize=6, labelpad=-1)
    ax.set_zlabel("relative density", fontsize=6, labelpad=-3)
    ax.set_xticks((0, 45, 90, 135))
    ax.set_yticks((-60, 0, 60))
    ax.set_zticks((0, 0.5, 1))
    ax.tick_params(labelsize=5, pad=-1)
    ax.view_init(elev=28, azim=-58)
    ax.set_box_aspect((1.25, 1, 0.65))


def _plot_network(ax, record):
    """Draw a deliberately schematic neural approximation."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    layer_x = (0.12, 0.40, 0.68)
    layer_y = (
        np.linspace(0.28, 0.72, 3),
        np.linspace(0.18, 0.82, 5),
        np.linspace(0.24, 0.76, 4),
    )
    for x0, ys0, x1, ys1 in zip(layer_x[:-1], layer_y[:-1], layer_x[1:], layer_y[1:]):
        for y0 in ys0:
            for y1 in ys1:
                ax.plot((x0, x1), (y0, y1), color="#aab4c8", linewidth=0.45, zorder=1)
    for index, (x, ys) in enumerate(zip(layer_x, layer_y)):
        color = ("#91b7e9", "#9a76dd", "#80c96b")[index]
        ax.scatter(np.full_like(ys, x), ys, s=52, color=color, edgecolor=NAVY,
                   linewidth=0.65, zorder=2)
    _, _, tile = _surface_density(record, "predicted_surface")
    ax.imshow(tile, extent=(0.79, 0.98, 0.33, 0.67), origin="lower",
              cmap="viridis", aspect="auto", interpolation="bilinear", zorder=2)
    for y in layer_y[-1]:
        ax.plot((layer_x[-1], 0.79), (y, 0.5), color="#aab4c8", linewidth=0.55)
    ax.add_patch(plt.Rectangle((0.79, 0.33), 0.19, 0.34, fill=False,
                               edgecolor=NAVY, linewidth=0.8, zorder=3))
    ax.text(0.12, 0.08, "model inputs", ha="center", fontsize=7, color=MUTED)
    ax.text(0.88, 0.08, "predicted surface", ha="center", fontsize=7, color=MUTED)


def _plot_predictions(ax, predictions):
    required = {"sd_feat1", "sd_feat2", "sd_spat", "mu1_expectation_curve", "feat_diff_grid"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"WNM prediction export lacks columns: {sorted(missing)}")
    feature_grid = _first_list(predictions["feat_diff_grid"])
    for (_, row), color in zip(predictions.iterrows(), ("#2c7bb6", "#f28e2b", "#b2182b")):
        ax.plot(feature_grid, np.asarray(row["mu1_expectation_curve"]),
                color=color, linewidth=1.8, label=f"{row['sd_spat']:g}°")
    ax.axhline(0, color="#7b8794", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xlim(feature_grid[0], feature_grid[-1])
    ax.set_xticks((2, 45, 90, 135, 180))
    ax.set_xlabel("item dissimilarity (°)", fontsize=7, labelpad=1)
    ax.set_ylabel("predicted bias (°)", fontsize=7.3, labelpad=2)
    ax.tick_params(labelsize=6.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.45, alpha=0.65)
    ax.legend(title="Identifiability noise", frameon=False,
              fontsize=6, title_fontsize=6, loc="lower right", handlelength=2.3)


def _plot_fit(ax, trials_path, curves_path, objective):
    empirical, fitted = prepare_fitting_data(trials_path, curves_path, objective)
    ax.axhline(0, color="#7b8794", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.fill_between(fitted["feat_diff"], fitted["bias"] - fitted["sem"],
                    fitted["bias"] + fitted["sem"], color=BLUE, alpha=0.16, linewidth=0)
    model_line, = ax.plot(fitted["feat_diff"], fitted["bias"], color=BLUE,
                          linewidth=1.8, label="Model fit")
    data_points = ax.errorbar(
        empirical["x"], empirical["bias"], yerr=empirical["sem"], fmt="o",
        color=NAVY, ecolor=BLUE, markersize=3, capsize=1.5, linewidth=0.7,
        label="Data",
    )
    ax.set(xlim=(0, 90), xlabel="target–distractor dissimilarity (°)",
           ylabel="bias toward distractor (°)")
    ax.tick_params(labelsize=6.5, length=2)
    ax.xaxis.label.set_size(7.3)
    ax.yaxis.label.set_size(7.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.45, alpha=0.65)
    ax.legend([data_points, model_line], ["Data", "Model fit"], frameon=False,
              fontsize=6, loc="upper right", handlelength=1.8)


def create_pipeline_figure(args):
    predictions = pd.read_parquet(_require(args.predictions, "WNM prediction export"))
    surface_record = _response_surfaces(predictions)
    _require(args.trials, "Prepared behavioral trials")
    _require(args.fit_curves, "Fitted-curve export")

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.5, 0.963, "Demixing Model pipeline", ha="center", va="top",
             fontsize=29, weight="bold", color=NAVY)
    fig.text(0.5, 0.912,
             "From simulated responses to predictions and fits of behavioral data",
             ha="center", va="top", fontsize=14, style="italic", color="#49619c")

    _rounded_box(fig, (0.018, 0.055, 0.445, 0.80), CONSTRUCTION, CONSTRUCTION_BG)
    _rounded_box(fig, (0.588, 0.055, 0.394, 0.80), GREEN, GREEN_BG)
    fig.text(0.24, 0.815, "Model construction", ha="center", fontsize=17,
             weight="bold", color=INK)
    fig.text(0.24, 0.783, "How the included model is created", ha="center",
             fontsize=10.5, color="#49619c")
    fig.text(0.785, 0.815, "Two ways to use the model", ha="center", fontsize=17,
             weight="bold", color=GREEN)
    fig.text(0.785, 0.783, "No simulations or neural-network training required", ha="center",
             fontsize=10.5, color=GREEN, style="italic")

    construction_boxes = (
        (0.030, 0.165, 0.122, 0.57),
        (0.168, 0.165, 0.122, 0.57),
        (0.306, 0.165, 0.144, 0.57),
    )
    for bounds in construction_boxes:
        _rounded_box(fig, bounds, CONSTRUCTION, "white", linewidth=1.1, radius=0.012)
    fig.text(0.091, 0.704, "1. Simulations", ha="center", fontsize=12, weight="bold", color=INK)
    fig.text(0.229, 0.704, "2. Response density", ha="center", fontsize=12, weight="bold", color=INK)
    fig.text(0.378, 0.704, "3. Surrogate training", ha="center", fontsize=10.5,
             weight="bold", color=INK)

    sim_ax = fig.add_axes((0.041, 0.365, 0.100, 0.24))
    _plot_simulation(sim_ax, surface_record["parameters"])
    surface_ax = fig.add_axes((0.177, 0.335, 0.106, 0.30), projection="3d")
    _plot_surface(surface_ax, surface_record)
    network_ax = fig.add_axes((0.329, 0.335, 0.098, 0.30))
    _plot_network(network_ax, surface_record)

    fig.text(0.091, 0.245,
             "simulate noisy evidence\nfrom two items",
             ha="center", va="center", fontsize=8.5, color=INK)
    fig.text(0.229, 0.245,
             "learn how response bias\nchanges with dissimilarity",
             ha="center", va="center", fontsize=8.5, color=INK)
    fig.text(0.378, 0.245,
             "train a wrapped-mixture\ndensity predictor",
             ha="center", va="center", fontsize=8.5, color=INK)

    checkpoint_bounds = (0.472, 0.31, 0.098, 0.27)
    _rounded_box(fig, checkpoint_bounds, NAVY, "white", linewidth=1.4, radius=0.014)
    fig.text(0.521, 0.505, "PRETRAINED WNM", ha="center", fontsize=8.5, color=NAVY, weight="bold")
    fig.text(0.521, 0.445, "Demixing\nModel", ha="center", va="center",
             fontsize=14, color=INK, weight="bold")
    fig.text(0.521, 0.370,
             "included with\nthe repository",
             ha="center", va="center", fontsize=7.5, color="#49619c", style="italic")

    top_box = (0.606, 0.468, 0.356, 0.278)
    bottom_box = (0.606, 0.126, 0.356, 0.278)
    _rounded_box(fig, top_box, GREEN, "white", linewidth=1.3, radius=0.014)
    _rounded_box(fig, bottom_box, BLUE, "white", linewidth=1.3, radius=0.014)
    fig.text(0.784, 0.716, "A. Generate theoretical predictions", ha="center", fontsize=14,
             color=GREEN, weight="bold")
    fig.text(0.784, 0.374, "B. Fit behavioral data", ha="center", fontsize=14,
             color=BLUE, weight="bold")
    fig.text(0.784, 0.686, "Primary use of the normative model", ha="center",
             fontsize=8, color=GREEN, style="italic")
    fig.text(0.784, 0.344, "Secondary use of the model", ha="center",
             fontsize=8, color=BLUE, style="italic")

    prediction_ax = fig.add_axes((0.628, 0.500, 0.218, 0.162))
    _plot_predictions(prediction_ax, predictions)
    fit_ax = fig.add_axes((0.628, 0.158, 0.218, 0.162))
    _plot_fit(fit_ax, args.trials, args.fit_curves, args.fit_objective)
    fig.text(0.865, 0.626, "Outputs", fontsize=9, weight="bold", color=GREEN)
    fig.text(0.865, 0.599,
             "• bias curves\n• response distributions\n• response variability\n\n"
             "+ toward non-target\n− away from non-target",
             fontsize=8, color=INK, va="top", linespacing=1.28)
    fig.text(0.865, 0.284, "Outputs", fontsize=9, weight="bold", color=BLUE)
    fig.text(0.865, 0.257, "• fitted parameters\n• fitted curves\n• model–data diagnostics",
             fontsize=8, color=INK, va="top", linespacing=1.35)

    _arrow(fig, (0.153, 0.45), (0.167, 0.45), linewidth=1.5, head_size=10)
    _arrow(fig, (0.291, 0.45), (0.305, 0.45), linewidth=1.5, head_size=10)
    _arrow(fig, (0.450, 0.45), (0.472, 0.45), linewidth=1.8, head_size=11)
    _polyline(fig, ((0.570, 0.445), (0.580, 0.445), (0.580, 0.654)))
    _polyline(fig, ((0.580, 0.445), (0.580, 0.326)))
    _arrow(fig, (0.580, 0.654), (0.608, 0.654), linewidth=2.2)
    _arrow(fig, (0.580, 0.326), (0.608, 0.326), linewidth=2.2)

    fig.text(0.5, 0.022,
             "Response density comes from WNM simulations; network output and curves use the packaged predictor.",
             ha="center", fontsize=8, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/prediction_example.parquet"))
    parser.add_argument("--trials", type=Path, default=Path("example_data/fischer_whitney_prepared.csv"))
    parser.add_argument(
        "--fit-curves", type=Path,
        default=Path("results/fischer_whitney_20samples_circular/csv_exports/fitted_curves.csv"),
    )
    parser.add_argument("--fit-objective", default="smoothed_exp")
    parser.add_argument("--output", type=Path, default=Path("docs/images/model_pipeline.png"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    create_pipeline_figure(parse_args())
