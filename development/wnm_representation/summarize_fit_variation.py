#!/usr/bin/env python3
"""Separate fitting-half and optimizer variation across repeated local fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from development.wnm_representation import equivalence


PREDICTIONS = ("pred_density_asymmetry", "pred_mean_bias", "pred_resultant",
               "pred_response_sd", "heldout_nll", "circular_wasserstein_deg")


def compare(primary: pd.DataFrame, repeat: pd.DataFrame) -> list[dict]:
    rows = []
    for key, first in primary.groupby(["family", "size"]):
        second = repeat[(repeat.family == key[0]) & (repeat["size"] == key[1])]
        first, second = (x.sort_values("feat_diff") for x in (first, second))
        if len(first) != len(second):
            continue
        for metric in PREDICTIONS:
            delta = second[metric].to_numpy() - first[metric].to_numpy()
            if metric == "pred_mean_bias":
                delta = equivalence.wrap_deg(delta)
            rows.append({"family": key[0], "size": int(key[1]), "metric": metric,
                         "fitting_half_mean_signed_difference": float(delta.mean()),
                         "fitting_half_rmse": float(np.sqrt(np.mean(delta ** 2))),
                         "fitting_half_max_abs_difference": float(np.max(np.abs(delta)))})
        if "train_nll_start_sd" in first:
            values = np.concatenate([first.train_nll_start_sd, second.train_nll_start_sd])
            rows.append({"family": key[0], "size": int(key[1]),
                         "metric": "optimizer_start_nll_sd",
                         "fitting_half_mean_signed_difference": float(values.mean()),
                         "fitting_half_rmse": float(np.sqrt(np.mean(values ** 2))),
                         "fitting_half_max_abs_difference": float(values.max())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.primary.glob("*/component_*/*/local_*_metrics.csv")):
        relative = path.relative_to(args.primary)
        other = args.repeat / relative
        if not other.exists():
            continue
        case, component = relative.parts[:2]
        for row in compare(pd.read_csv(path), pd.read_csv(other)):
            rows.append({"trajectory": case, "component": int(component.split("_")[-1]),
                         **row})
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} fitting-variation rows to {args.out}")


if __name__ == "__main__":
    main()
