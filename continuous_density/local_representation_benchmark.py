#!/usr/bin/env python3
"""Run resumable local-family ladders and gates on benchmark trajectories."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from continuous_density.benchmark_manifest import assert_operation_allowed


def trajectory_records(reference: Path) -> list[tuple[str, np.ndarray]]:
    blob = np.load(reference, allow_pickle=True)
    design, labels = blob["design"], np.asarray(blob["strata"]).astype(str)
    records = []
    for label in dict.fromkeys(labels):
        rows = np.flatnonzero(labels == label)
        records.append((label, np.asarray(design[rows[0], :3], dtype=float)))
    return records


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--role", choices=("discovery", "selection", "confirmation"),
                        default="selection")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--components", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--wrapped-sizes", type=int, nargs="+", default=[4, 8, 12, 24])
    parser.add_argument("--fourier-sizes", type=int, nargs="+", default=[8, 16, 32, 48])
    parser.add_argument("--spline-knots", type=int, nargs="+", default=[])
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--wrapped-steps", type=int, default=3000)
    parser.add_argument("--wrapped-lr", type=float, default=0.02)
    parser.add_argument("--fourier-maxiter", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    assert_operation_allowed(args.role, "select")
    records = trajectory_records(args.reference)
    if args.labels:
        requested = set(args.labels)
        records = [record for record in records if record[0] in requested]
        missing = requested - {label for label, _ in records}
        if missing:
            raise ValueError(f"unknown trajectory labels: {sorted(missing)}")
    python = sys.executable
    for label, triple in records:
        for component in args.components:
            root = args.out_dir / label / f"component_{component}"
            common = ["--sd-feat1", repr(float(triple[0])),
                      "--sd-feat2", repr(float(triple[1])),
                      "--sd-ident", repr(float(triple[2])),
                      "--component", str(component)]
            wrapped = root / "wrapped"
            fourier = root / "fourier"
            spline = root / "spline"
            _run([python, "-m", "continuous_density.fit_local_wrapped_mixture",
                  str(args.reference), *common, "--components",
                  *map(str, args.wrapped_sizes), "--starts", str(args.starts),
                  "--steps", str(args.wrapped_steps), "--lr", str(args.wrapped_lr),
                  "--split-seed", str(args.split_seed),
                  "--resume", "--out-dir", str(wrapped)])
            _run([python, "-m", "continuous_density.fit_local_fourier",
                  str(args.reference), *common, "--harmonics",
                  *map(str, args.fourier_sizes), "--maxiter", str(args.fourier_maxiter),
                  "--split-seed", str(args.split_seed), "--resume",
                  "--out-dir", str(fourier)])
            metric_paths = [wrapped / "local_wrapped_metrics.csv",
                            fourier / "local_fourier_metrics.csv"]
            if args.spline_knots:
                _run([python, "-m", "continuous_density.fit_local_spline",
                      str(args.reference), *common, "--knots",
                      *map(str, args.spline_knots), "--split-seed", str(args.split_seed),
                      "--out-dir", str(spline)])
                metric_paths.append(spline / "local_spline_metrics.csv")
            _run([python, "-m", "continuous_density.bootstrap_local_metrics",
                  str(args.reference), *map(str, metric_paths), *common,
                  "--split-seed", str(args.split_seed),
                  "--independent-indices", "--bootstrap", str(args.bootstrap),
                  "--quiet",
                  "--out", str(root / "core_equivalence.json")])
            candidates = []
            for size in args.wrapped_sizes:
                candidates.extend(["--candidate", f"w{size}:wrapped_mixture:{size}:"
                                   f"{wrapped / 'local_wrapped_parameters.npz'}"])
            for size in args.fourier_sizes:
                candidates.extend(["--candidate", f"f{size}:maxent_fourier:{size}:"
                                   f"{fourier / 'local_fourier_parameters.npz'}"])
            for size in args.spline_knots:
                candidates.extend(["--candidate", f"s{size}:periodic_spline:{size}:"
                                   f"{spline / 'local_spline_parameters.npz'}"])
            _run([python, "-m", "continuous_density.local_distribution_gate",
                  str(args.reference), *common, *candidates,
                  "--split-seed", str(args.split_seed),
                  "--bootstrap", str(args.bootstrap),
                  "--quiet",
                  "--out", str(root / "distribution_gate.json")])


if __name__ == "__main__":
    main()
