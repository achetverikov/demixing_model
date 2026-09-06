#!/usr/bin/env python3
"""Evaluate frozen selection candidates on independent high-N flagship outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("design_manifest", type=Path)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("--components", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    points = json.loads(args.design_manifest.read_text())["points"]
    if args.labels:
        points = [point for point in points if point["trajectory"] in set(args.labels)]
    for point in points:
        triple = point["design"][:3]
        for component in args.components:
            source = args.fit_root / point["trajectory"] / f"component_{component}"
            metrics = sorted(source.glob("*/local_*_metrics.csv"))
            candidates = []
            for path in metrics:
                parameters = path.with_name(path.name.replace("metrics.csv", "parameters.npz"))
                for family, size in (pd.read_csv(path)[["family", "size"]]
                                     .drop_duplicates().itertuples(index=False, name=None)):
                    prefix = {"wrapped_mixture": "w", "maxent_fourier": "f",
                              "periodic_spline": "s"}[family]
                    candidates.extend(["--candidate",
                        f"{prefix}{int(size)}:{family}:{int(size)}:{parameters}"])
            common = ["--sd-feat1", repr(triple[0]), "--sd-feat2", repr(triple[1]),
                      "--sd-ident", repr(triple[2]), "--component", str(component),
                      "--candidate-row", str(point["candidate_row"])]
            out = args.out_dir / point["trajectory"] / f"component_{component}"
            out.mkdir(parents=True, exist_ok=True)
            _run([sys.executable, "-m", "continuous_density.bootstrap_local_metrics",
                  str(args.reference), *map(str, metrics), *common,
                  "--independent-indices", "--bootstrap", str(args.bootstrap), "--quiet",
                  "--out", str(out / "core_equivalence.json")])
            _run([sys.executable, "-m", "continuous_density.local_distribution_gate",
                  str(args.reference), *common, *candidates,
                  "--bootstrap", str(args.bootstrap), "--quiet",
                  "--out", str(out / "distribution_gate.json")])


if __name__ == "__main__":
    main()
