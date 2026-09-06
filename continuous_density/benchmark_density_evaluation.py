#!/usr/bin/env python3
"""Measure local representation parameter count, memory, and evaluation cost."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_density import local_distribution_gate as gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=gate.parse_candidate,
                        required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--outcomes", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = np.linspace(-180, 180, args.outcomes, endpoint=False)[None, :]
    rows = []
    for candidate in args.candidate:
        gate.candidate_logpdf(candidate, samples, row=args.row)
        gate.candidate_mass(candidate, row=args.row)
        starts = []
        for _ in range(args.repeats):
            before = time.perf_counter()
            gate.candidate_logpdf(candidate, samples, row=args.row)
            starts.append(time.perf_counter() - before)
        parameters = {"wrapped_mixture": 3 * candidate.size - 1,
                      "maxent_fourier": 2 * candidate.size,
                      "periodic_spline": candidate.size - 1}[candidate.family]
        rows.append({"label": candidate.label, "family": candidate.family,
            "size": candidate.size, "density_parameters": parameters,
            "bytes_float64": parameters * 8,
            "microseconds_per_outcome": 1e6 * np.median(starts) / args.outcomes})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
