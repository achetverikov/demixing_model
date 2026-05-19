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
import pty
import re
import select
import subprocess
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.resolve()
DATA_URL = (
    "https://raw.githubusercontent.com/aozkirli/"
    "Large-scale-mega-analysis-on-serial-dependence/main/data/"
    "07_Fischer_Whitney_2014b.csv"
)
PREPARED_CSV = REPO_ROOT / "example_data" / "fischer_whitney_prepared.csv"
RESULTS_DIR  = "results/fischer_whitney"
CHECKPOINT   = "pretrained/model_epoch_1500.pkl"

PYTHONPATH = ":".join([
    str(REPO_ROOT),
    str(REPO_ROOT / "neural_network_optimization"),
])
ENV = {**os.environ, "PYTHONPATH": PYTHONPATH, "PYTHONUNBUFFERED": "1"}
PYTHON = sys.executable  # use whichever interpreter launched this script

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class LogCleaner:
    """Strip terminal control codes and collapse carriage-return redraws."""

    def __init__(self, log):
        self.log = log
        self.line = []
        self.pending_cr = False

    def write(self, text: str):
        text = ANSI_RE.sub("", text)
        for char in text:
            if self.pending_cr:
                if char == "\n":
                    self._flush_line()
                    self.pending_cr = False
                    continue
                self.line = []
                self.pending_cr = False

            if char == "\r":
                self.pending_cr = True
            elif char == "\n":
                self._flush_line()
            else:
                self.line.append(char)

    def close(self):
        if self.line:
            self._flush_line()
        self.log.flush()

    def _flush_line(self):
        self.log.write("".join(self.line) + "\n")
        self.log.flush()
        self.line = []


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

    PREPARED_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(PREPARED_CSV, index=False)

    print(f"  abs_td_dist:        [{out['abs_td_dist'].min():.1f}, {out['abs_td_dist'].max():.1f}]")
    print(f"  bias_to_distr_corr: [{out['bias_to_distr_corr'].min():.1f}, {out['bias_to_distr_corr'].max():.1f}]")
    print(f"  outlier rate:       {out['is_outlier'].mean():.1%}")
    print(f"  Saved → {PREPARED_CSV}")
    return PREPARED_CSV


# ── subprocess helpers ────────────────────────────────────────────────────────

def announce(title: str, log=None):
    message = f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"
    sys.stdout.write(message)
    sys.stdout.flush()
    if log:
        log.write(message)
        log.flush()


def run(cmd: list, log=None):
    header = f"\n$ {' '.join(str(c) for c in cmd)}\n{'─'*60}\n"
    sys.stdout.write(header)
    sys.stdout.flush()
    if log:
        log.write(header)
        log.flush()

    if log and os.name == "posix" and sys.stdout.isatty():
        run_with_pty(cmd, log)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=ENV,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log:
            log.write(line)
            log.flush()
    proc.wait()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def run_with_pty(cmd: list, log):
    master_fd, slave_fd = pty.openpty()
    cleaner = LogCleaner(log)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=ENV,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd, stderr=subprocess.STDOUT,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = None

        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                cleaner.write(chunk.decode(errors="replace"))

            if proc.poll() is not None:
                # Drain anything left in the pseudo-terminal.
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if master_fd not in ready:
                        break
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    cleaner.write(chunk.decode(errors="replace"))
                break
    finally:
        cleaner.close()
        if slave_fd is not None:
            os.close(slave_fd)
        os.close(master_fd)

    proc.wait()
    if proc is None or proc.returncode != 0:
        sys.exit(proc.returncode if proc is not None else 1)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = prepare_data()

    log_path = REPO_ROOT / RESULTS_DIR / "run.log"
    log_path.parent.mkdir(exist_ok=True, parents=True)

    with open(log_path, "w") as log:
        announce("STEP 1 — Fit model", log)
        run([
            PYTHON, "model_fit_to_data/fit_model_to_data.py",
            "--data-path",       str(csv_path),
            "--checkpoint-path", CHECKPOINT,
            "--output-dir",      RESULTS_DIR,
        #    "--no-skip-motor-noise",
            # "--include-outliers",
            "--include-methods", "density", "expectation", "balanced_crps",
            "--circ-space",      "180",
        ], log=log)

        announce("STEP 2 — Generate plots", log)
        run([
            PYTHON, "model_fit_to_data/create_unified_subject_plots.py",
            "--results-path",    f"{RESULTS_DIR}/extended_fit_results.pkl",
            "--checkpoint-path", CHECKPOINT,
            "--output-dir",      RESULTS_DIR,
            "--individual-plots",
            "--summary-plots",
            "--csv-exports",
            "--circ-space",      "180",
        ], log=log)

    print(f"\nDone.  Results → {RESULTS_DIR}/")
    print(f"       Plots   → {RESULTS_DIR}/summary_plots/")
    print(f"       CSVs    → {RESULTS_DIR}/csv_exports/")
    print(f"       Log     → {log_path}")
