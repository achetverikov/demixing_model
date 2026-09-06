#!/usr/bin/env python3
"""Independent confirmation gate for local-density NLL and circular W1.

Candidate labels are selected by NLL on one half of the untouched outcomes.  All
reported intervals use the other half, avoiding selection on the confirmatory
numbers.  These intervals cover evaluation-reference uncertainty only; repeated
fits are still required for the complete Phase A gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from continuous_density import fourier_moments as fm
from continuous_density import local_fit_common as common
from continuous_density import maxent_fourier as mf
from continuous_density import periodic_spline as ps
from continuous_density import wrapped_mixture_model as wm


@dataclass(frozen=True)
class Candidate:
    label: str
    family: str
    size: int
    parameters: Path


def parse_candidate(value: str) -> Candidate:
    """Parse ``label:family:size:parameter-file`` from the command line."""
    fields = value.split(":", 3)
    if len(fields) != 4 or fields[1] not in {
            "wrapped_mixture", "maxent_fourier", "periodic_spline"}:
        raise argparse.ArgumentTypeError(
            "candidate must be label:{wrapped_mixture|maxent_fourier|periodic_spline}:size:path")
    return Candidate(fields[0], fields[1], int(fields[2]), Path(fields[3]))


def _wrapped_dist(candidate: Candidate, row: int | None = None) -> dict[str, np.ndarray]:
    blob = np.load(candidate.parameters)
    prefix = f"k{candidate.size}_selected_"
    output = {name: blob[prefix + name] for name in ("log_pi", "mu", "sigma")}
    return ({name: value[row:row + 1] for name, value in output.items()}
            if row is not None else output)


def candidate_logpdf(candidate: Candidate, samples: np.ndarray,
                     chunk: int = 2000, row: int | None = None) -> np.ndarray:
    """Evaluate a saved local candidate against row-matched raw outcomes."""
    x = np.asarray(samples, dtype=np.float64)
    output = np.empty_like(x)
    if candidate.family == "wrapped_mixture":
        dist = _wrapped_dist(candidate, row)
        for first in range(0, x.shape[1], chunk):
            block = x[:, first:first + chunk]
            output[:, first:first + chunk] = np.asarray(
                wm.mixture_logpdf_samples(jnp.asarray(block), dist))
        return output

    theta = np.load(candidate.parameters)[f"k{candidate.size}_theta"]
    if row is not None:
        theta = theta[row:row + 1]
    if theta.shape[0] != x.shape[0]:
        raise ValueError("candidate and reference trajectory have different row counts")
    density_class = (mf.MaxentFourierDensity if candidate.family == "maxent_fourier"
                     else ps.PeriodicSplineDensity)
    for row_index, coefficients in enumerate(theta):
        density = density_class(coefficients)
        for first in range(0, x.shape[1], chunk):
            output[row_index, first:first + chunk] = density.logpdf(
                x[row_index, first:first + chunk])
    return output


def candidate_mass(candidate: Candidate, row: int | None = None) -> np.ndarray:
    """Evaluate a saved local candidate on the common 0.5-degree audit grid."""
    if candidate.family == "wrapped_mixture":
        density = np.exp(np.asarray(wm.mixture_logpdf_grid(
            jnp.asarray(common.GRID), _wrapped_dist(candidate, row))))
    else:
        theta = np.load(candidate.parameters)[f"k{candidate.size}_theta"]
        if row is not None:
            theta = theta[row:row + 1]
        density_class = (mf.MaxentFourierDensity if candidate.family == "maxent_fourier"
                         else ps.PeriodicSplineDensity)
        density = np.stack([density_class(values).density_grid(common.GRID)
                            for values in theta])
    mass = density * common.CELL_WIDTH
    return mass / mass.sum(axis=1, keepdims=True)


def one_sided_mean_intervals(loss_difference: np.ndarray, alpha: float = 0.05
                             ) -> dict:
    """Normal CIs for row and equal-trajectory means of paired log-loss."""
    values = np.asarray(loss_difference, dtype=np.float64)
    estimate = values.mean(axis=1)
    se = values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    critical = float(norm.ppf(1.0 - alpha / len(values)))
    upper = estimate + critical * se
    coherent_se = float(np.sqrt(np.sum(se ** 2)) / len(se))
    coherent_estimate = float(estimate.mean())
    coherent_upper = coherent_estimate + float(norm.ppf(1.0 - alpha)) * coherent_se
    return {"estimate": estimate.tolist(), "standard_error": se.tolist(),
            "simultaneous_upper": upper.tolist(), "critical_value": critical,
            "coherent_estimate": coherent_estimate,
            "coherent_standard_error": coherent_se,
            "coherent_upper": coherent_upper}


def bootstrap_wasserstein(samples: np.ndarray, candidate_probability: np.ndarray,
                          n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Conservative one-sided bootstrap bounds for candidate-to-reference W1."""
    observed_mass = common.histogram_mass(samples)
    estimate = fm.circular_wasserstein_grid(
        observed_mass, candidate_probability, common.CELL_WIDTH)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(samples)))
    for row, probability in enumerate(observed_mass):
        counts = rng.multinomial(samples.shape[1], probability, size=n_boot)
        draws[:, row] = fm.circular_wasserstein_grid(
            counts, np.broadcast_to(candidate_probability[row], counts.shape),
            common.CELL_WIDTH)
    centered = draws - estimate
    radius = float(np.quantile(np.max(centered, axis=1), 1.0 - alpha))
    coherent_draws = draws.mean(axis=1)
    return {"estimate": estimate.tolist(),
            "simultaneous_upper": (estimate + radius).tolist(),
            "simultaneous_radius": radius,
            "coherent_estimate": float(estimate.mean()),
            "coherent_upper": float(np.quantile(coherent_draws, 1.0 - alpha))}


def split_half_wasserstein(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    values = fm.circular_wasserstein_grid(
        common.histogram_mass(a), common.histogram_mass(b), common.CELL_WIDTH)
    return {"mean_deg": float(values.mean()), "max_deg": float(values.max()),
            "median_deg": float(np.median(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate,
                        required=True)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--candidate-row", type=int,
                        help="evaluate one saved trajectory row (for flagship points)")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="Reproduce the original fit/test split")
    parser.add_argument("--confirmation-seed", type=int, default=1729)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    blob = np.load(args.reference, allow_pickle=True)
    _, samples = common.select_trajectory(
        blob["design"], blob["bias"], args.sd_feat1, args.sd_feat2,
        args.sd_ident, args.component)
    _, heldout = common.split_outcomes(samples, seed=args.split_seed)
    selection, confirmation = common.split_outcomes(
        heldout, seed=args.confirmation_seed)
    logpdf = {candidate.label: candidate_logpdf(
                  candidate, heldout, row=args.candidate_row)
              for candidate in args.candidate}
    half = heldout.shape[1] // 2
    # Reproduce the second permutation to align cached log densities with both halves.
    rng = np.random.default_rng(args.confirmation_seed)
    order = rng.permutation(heldout.shape[1])
    selection_index, confirmation_index = order[:half], order[half:2 * half]
    selection_nll = {label: float(-value[:, selection_index].mean())
                     for label, value in logpdf.items()}
    reference_label = min(selection_nll, key=selection_nll.get)
    reference_loss = -logpdf[reference_label][:, confirmation_index]

    candidates = []
    for index, candidate in enumerate(args.candidate):
        loss = -logpdf[candidate.label][:, confirmation_index]
        nll = one_sided_mean_intervals(loss - reference_loss)
        w1 = bootstrap_wasserstein(
            confirmation, candidate_mass(candidate, args.candidate_row), args.bootstrap,
            args.confirmation_seed + index)
        candidates.append({"label": candidate.label, "family": candidate.family,
                           "size": candidate.size,
                           "confirmation_nll": float(loss.mean()),
                           "excess_nll": nll,
                           "nll_pointwise_pass": max(nll["simultaneous_upper"]) < 0.002,
                           "nll_coherent_pass": nll["coherent_upper"] < 0.0005,
                           "circular_wasserstein_deg": w1,
                           "wasserstein_pointwise_pass":
                               max(w1["simultaneous_upper"]) < 0.50,
                           "wasserstein_coherent_pass": w1["coherent_upper"] < 0.25})
    output = {
        "selection_n_outcomes": int(selection.shape[1]),
        "confirmation_n_outcomes": int(confirmation.shape[1]),
        "candidate_row": args.candidate_row,
        "selection_nll": selection_nll, "selected_nll_reference": reference_label,
        "wasserstein_split_half_floor": split_half_wasserstein(selection, confirmation),
        "uncertainty_scope": "confirmation-reference Monte Carlo only",
        "candidates": candidates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
