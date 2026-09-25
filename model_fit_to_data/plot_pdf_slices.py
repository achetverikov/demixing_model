#!/usr/bin/env python3
"""
Plot PDF slices p(mu1_bias | feat_diff) comparing model predictions to empirical data.

Rows = subjects (those with the clearest bias signal / dissociation selected automatically).
Columns = feature difference values.

Run from the repo root:
    python model_fit_to_data/plot_pdf_slices.py \
        --results-path results/<dataset>/extended_fit_results.pkl \
        --output-dir results/<dataset>

The run fingerprint selects and verifies the surrogate that produced the fit,
and the saved results provide the study's circular period. Use --checkpoint-path
or --circ-space only as explicit overrides/validation when necessary.
"""

import argparse
import sys
from pathlib import Path

from shared import surrogate
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "neural_network_optimization"))
sys.path.insert(0, str(REPO_ROOT / "model_fit_to_data"))

import jax.numpy as jnp
from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
from model_fit_to_data.create_unified_subject_plots import (
    _resolve_plot_circ_space,
    create_pdf_slice_plots,
    load_extended_results,
)
from shared.prediction import predictor_from_surrogate


def main():
    parser = argparse.ArgumentParser(
        description="Create PDF slice plots from model-fit results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-path", required=True,
                        help="Path to extended_fit_results.pkl.")
    parser.add_argument("--n-samples", type=int, default=20, choices=[20, 100],
                        help="Observer evidence samples per trial. Used to validate an explicit "
                             "legacy checkpoint when no fingerprint identity is available.")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Optional surrogate artifact override. It must match the fit fingerprint.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output plots.")
    parser.add_argument("--circ-space", type=int, default=None, choices=[180, 360],
                        help="Optional circular-space override. By default it is inferred from "
                             "the fitted result metadata; a conflicting override raises.")
    parser.add_argument("--optimizer", default="density",
                        choices=["density", "density_legacy", "expectation", "smoothed_exp",
                                 "likelihood", "crps", "balanced_crps", "bias_weighted_crps"],
                        help="Which optimizer's parameters to use.")
    parser.add_argument("--n-subjects", type=int, default=3,
                        help="Number of subjects to show per plot.")
    parser.add_argument("--feat-diffs", nargs="+", type=float, default=None,
                        help="Feature differences to show in data units. "
                             "Auto-derived if not given.")
    args = parser.parse_args()

    results = load_extended_results(args.results_path)
    circ_space = _resolve_plot_circ_space(results, args.circ_space)

    # The stored parameters were produced by a particular surrogate, so use that
    # one rather than today's production artifact; --checkpoint-path overrides and
    # is verified against the run's recorded digest.
    checkpoint = surrogate.checkpoint_for_run(
        args.results_path, explicit=args.checkpoint_path, n_samples=args.n_samples)
    loaded = surrogate.load_surrogate(checkpoint_path=checkpoint)
    if loaded.family == surrogate.FAMILY_WNM:
        prediction_backend = predictor_from_surrogate(loaded)
    else:
        dummy = jnp.asarray(np.random.default_rng(0).uniform(-180, 180, (100, 2)))
        prediction_backend = GridBasedMultiConditionOptimizer(
            str(checkpoint), {"dummy": dummy}, skip_motor_noise=True,
        )

    create_pdf_slice_plots(
        results, prediction_backend, args.output_dir,
        circ_space=circ_space,
        optimizer_names=[args.optimizer],
        n_subjects=args.n_subjects,
        feat_diffs_data=args.feat_diffs,
    )


if __name__ == "__main__":
    main()
