#!/usr/bin/env python3
"""
Demo: fit the demixing model to Fischer & Whitney (2014b) orientation data.

Downloads the data from the Aozkirli et al. mega-analysis repository,
preprocesses it (cardinal-bias removal, feature-difference and bias columns),
fits the model, and generates plots.

Run from the repo root:
    PYTHONPATH=. /workspaces/.venv/bin/python demo_fischer_whitney.py
"""

import os
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

from shared.run_utils import announce, run  # noqa: E402

DATA_URL = (
    "https://raw.githubusercontent.com/aozkirli/"
    "Large-scale-mega-analysis-on-serial-dependence/main/data/"
    "07_Fischer_Whitney_2014b.csv"
)
PREPARED_CSV = REPO_ROOT / "example_data" / "fischer_whitney_prepared.csv"
RESULTS_BASE = "results/fischer_whitney"

CHECKPOINTS = [
    ("20samples_circular",  "pretrained/model_epoch1500_10ktrain_20samples.pkl"),
    ("100samples_circular", "pretrained/model_epoch1500_10ktrain_100samples.pkl"),
]

PYTHONPATH = ":".join([
    str(REPO_ROOT),
    str(REPO_ROOT / "neural_network_optimization"),
])
ENV = {**os.environ, "PYTHONPATH": PYTHONPATH, "PYTHONUNBUFFERED": "1"}
PYTHON = sys.executable  # use whichever interpreter launched this script


# ── preprocessing ─────────────────────────────────────────────────────────────

def _circ_mean_180(angles: np.ndarray) -> np.ndarray:
    """Circular mean along axis=1 for a 2-D array, in 180° space."""
    return np.degrees(np.angle(np.mean(np.exp(2j * np.radians(angles)), axis=1))) / 2


def _circ_sd_180(angles: np.ndarray) -> np.ndarray:
    """Circular SD along axis=1 for a 2-D array, in 180° space."""
    R = np.abs(np.mean(np.exp(2j * np.radians(angles)), axis=1))
    return np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1 - 1e-9)))) / 2


def _angle_diff_180(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a − b in 180° space, result in (−90, 90]."""
    return (a - b + 90) % 180 - 90


def _remove_cardinal_biases(group: pd.DataFrame, k: int = 40) -> pd.DataFrame:
    """Remove cardinal-orientation biases per subject via two-pass moving window.

    Pass 1 — circular mean & SD over k nearest orientations; flag trials
              where |error − mean| > 3 SD as outliers.
    Pass 2 — circular mean over k nearest non-outlier orientations; subtract
              as the cardinal-bias estimate (circular difference).

    Returns the group with added columns: be_c, is_outlier.
    """
    theta = group["theta"].values.astype(float)
    error = group["error"].values.astype(float)
    n = len(theta)
    k = min(k, n)

    # Pairwise circular distance in 180° space
    diff = np.abs(theta[:, None] - theta[None, :])
    dist = np.minimum(diff, 180 - diff)

    # Pass 1: k-nearest-orientation window → circular mean & SD → outliers
    nn = np.argsort(dist, axis=1)[:, :k]
    mu = _circ_mean_180(error[nn])
    sd = _circ_sd_180(error[nn])
    is_outlier = (np.abs(_angle_diff_180(error, mu)) > 3 * sd).astype(int)

    # Pass 2: same window restricted to non-outliers → bias → circular residual
    valid = np.where(is_outlier == 0)[0]
    k2 = min(k, len(valid))
    nn2 = valid[np.argsort(dist[:, valid], axis=1)[:, :k2]]
    bias = _circ_mean_180(error[nn2])
    be_c = _angle_diff_180(error, bias)

    return group.assign(be_c=be_c, is_outlier=is_outlier)


def prepare_data() -> Path:
    if PREPARED_CSV.exists():
        print(f"Prepared CSV already exists: {PREPARED_CSV}  (delete to re-download)")
        return PREPARED_CSV

    print("Downloading Fischer & Whitney (2014b) data…")
    df = pd.read_csv(DATA_URL, sep=";")
    print(f"  {len(df)} trials, {df['obs'].nunique()} observers")

    # Drop trials without a previous orientation (first trial of each block)
    df = df[df["delta"].notna()].copy()
    df["delta"] = df["delta"].astype(float)

    # Cardinal-bias removal per observer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df = df.groupby("obs", group_keys=False).apply(_remove_cardinal_biases)

    # Signed bias toward previous:  positive = attracted toward N-1
    # delta = previous − current (opposite of shift1 convention used elsewhere)
    df["bias_to_distr_corr"] = np.where(df["delta"] > 0, df["be_c"], -df["be_c"])
    df["abs_td_dist"] = df["delta"].abs()

    out = df.assign(
        expName="Fischer_Whitney_2014b",
        subject=df["obs"],
        condition="orientation",
    )[["expName", "subject", "condition", "abs_td_dist", "bias_to_distr_corr", "is_outlier"]]

    # Add a "combined" pseudo-subject: pool all trials so the model can be fit
    # to the average observer.  Excluded from summary plots by downstream code.
    combined = out.copy()
    combined["subject"] = "combined"
    out = pd.concat([out, combined], ignore_index=True)

    PREPARED_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(PREPARED_CSV, index=False)

    print(f"  abs_td_dist:        [{out['abs_td_dist'].min():.1f}, {out['abs_td_dist'].max():.1f}]")
    print(f"  bias_to_distr_corr: [{out['bias_to_distr_corr'].min():.1f}, {out['bias_to_distr_corr'].max():.1f}]")
    print(f"  outlier rate:       {out['is_outlier'].mean():.1%}")
    print(f"  Saved → {PREPARED_CSV}")
    return PREPARED_CSV


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = prepare_data()

    log_path = REPO_ROOT / RESULTS_BASE / "run.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    with open(log_path, "w") as log:
        for tag, ckpt in CHECKPOINTS:
            out_dir = f"{RESULTS_BASE}_{tag}"

            announce(f"STEP — Fit model [{tag}] → {out_dir}", log)
            run([
                PYTHON, "model_fit_to_data/fit_model_to_data.py",
                "--data-path",       str(csv_path),
                "--checkpoint-path", ckpt,
                "--output-dir",      out_dir,
                "--include-methods", "density", "expectation", "balanced_crps", "bias_weighted_crps",
                "--circ-space",      "180",
            ], cwd=REPO_ROOT, env=ENV, log=log)

            announce(f"STEP — Generate plots [{tag}]", log)
            run([
                PYTHON, "model_fit_to_data/create_unified_subject_plots.py",
                "--results-path",    f"{out_dir}/extended_fit_results.pkl",
                "--checkpoint-path", ckpt,
                "--output-dir",      out_dir,
                "--individual-plots",
                "--summary-plots",
                "--csv-exports",
                "--circ-space",      "180",
            ], cwd=REPO_ROOT, env=ENV, log=log)

    print(f"\nDone.")
    for tag, _ in CHECKPOINTS:
        print(f"  results/fischer_whitney_{tag}/  (fits, plots, CSVs)")
    print(f"  Log: {log_path}")
