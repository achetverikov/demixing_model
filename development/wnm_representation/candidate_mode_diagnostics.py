#!/usr/bin/env python3
"""Mode count, positions, masses, and switching for fitted local trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from development.wnm_representation import local_distribution_gate as gate
from development.wnm_representation import local_fit_common as common
from development.wnm_representation import mode_diagnostics as modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=gate.parse_candidate,
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for candidate in args.candidate:
        probability = gate.candidate_mass(candidate)
        previous_count = None
        for row, mass in enumerate(probability):
            positions, basin_mass = modes.circular_modes_from_mass(mass, common.GRID)
            count = len(positions)
            rows.append({"label": candidate.label, "family": candidate.family,
                "size": candidate.size, "trajectory_row": row, "mode_count": count,
                "mode_positions_deg": json.dumps(positions.tolist()),
                "mode_masses": json.dumps(basin_mass.tolist()),
                "secondary_mode_mass": float(basin_mass[1] if count > 1 else 0),
                "mode_count_switched": previous_count is not None and count != previous_count})
            previous_count = count
    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Wrote {len(output)} fitted mode rows to {args.out}")


if __name__ == "__main__":
    main()
