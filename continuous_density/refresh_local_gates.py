#!/usr/bin/env python3
"""Recompute trajectory gates after adding or extending local candidate families."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--sd-feat1", type=float, required=True)
    parser.add_argument("--sd-feat2", type=float, required=True)
    parser.add_argument("--sd-ident", type=float, required=True)
    parser.add_argument("--component", type=int, choices=(1, 2), required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    metrics = sorted(args.case_dir.glob("*/local_*_metrics.csv"))
    if not metrics:
        raise ValueError(f"no local metrics in {args.case_dir}")
    common = ["--sd-feat1", repr(args.sd_feat1), "--sd-feat2", repr(args.sd_feat2),
              "--sd-ident", repr(args.sd_ident), "--component", str(args.component),
              "--split-seed", str(args.split_seed)]
    subprocess.run([sys.executable, "-m", "continuous_density.bootstrap_local_metrics",
        str(args.reference), *map(str, metrics), *common, "--independent-indices",
        "--bootstrap", str(args.bootstrap), "--quiet",
        "--out", str(args.case_dir / "core_equivalence.json")], check=True)
    candidates = []
    prefix = {"wrapped_mixture": "w", "maxent_fourier": "f", "periodic_spline": "s"}
    for path in metrics:
        parameters = path.with_name(path.name.replace("metrics.csv", "parameters.npz"))
        for family, size in (pd.read_csv(path)[["family", "size"]]
                             .drop_duplicates().itertuples(index=False, name=None)):
            candidates.extend(["--candidate",
                f"{prefix[family]}{int(size)}:{family}:{int(size)}:{parameters}"])
    subprocess.run([sys.executable, "-m", "continuous_density.local_distribution_gate",
        str(args.reference), *common, *candidates, "--bootstrap", str(args.bootstrap),
        "--quiet", "--out", str(args.case_dir / "distribution_gate.json")], check=True)


if __name__ == "__main__":
    main()
