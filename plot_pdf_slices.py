#!/usr/bin/env python3
"""
Plot PDF slices p(mu1_bias | feat_diff) comparing model predictions to empirical data.

Rows = subjects (those with the clearest bias signal / dissociation selected automatically).
Columns = feature difference values.

Run from the repo root:
    PYTHONPATH=.:neural_network_optimization:model_fit_to_data \
        python plot_pdf_slices.py \
        --results-path results/fritsche/extended_fit_results.pkl \
        --checkpoint-path pretrained/model_epoch_1500.pkl \
        --output-dir results/fritsche \
        --circ-space 180
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "neural_network_optimization"))
sys.path.insert(0, str(REPO_ROOT / "model_fit_to_data"))

import jax.numpy as jnp
from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import GridBasedMultiConditionOptimizer
from model_fit_to_data.create_unified_subject_plots import (
    create_pdf_slice_plots,
    load_extended_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Create PDF slice plots from model-fit results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-path", required=True,
                        help="Path to extended_fit_results.pkl.")
    parser.add_argument("--checkpoint-path", default="pretrained/model_epoch_1500.pkl",
                        help="Path to trained NN checkpoint.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output plots.")
    parser.add_argument("--circ-space", type=int, default=360, choices=[180, 360],
                        help="Circular space of the fitted data. Use 180 for axial orientation "
                             "data fitted after doubling into model space; use 360 when data "
                             "already match model space.")
    parser.add_argument("--optimizer", default="density",
                        choices=["density", "expectation", "likelihood", "crps", "balanced_crps"],
                        help="Which optimizer's parameters to use.")
    parser.add_argument("--n-subjects", type=int, default=3,
                        help="Number of subjects to show per plot.")
    parser.add_argument("--feat-diffs", nargs="+", type=float, default=None,
                        help="Feature differences to show in data units. "
                             "Auto-derived if not given.")
    args = parser.parse_args()

    results = load_extended_results(args.results_path)

    dummy = jnp.asarray(np.random.uniform(-180, 180, (100, 2)))
    optimizer = GridBasedMultiConditionOptimizer(
        args.checkpoint_path, {"dummy": dummy}, skip_motor_noise=True,
    )

    create_pdf_slice_plots(
        results, optimizer, args.output_dir,
        circ_space=args.circ_space,
        optimizer_names=[args.optimizer],
        n_subjects=args.n_subjects,
        feat_diffs_data=args.feat_diffs,
    )


if __name__ == "__main__":
    main()
