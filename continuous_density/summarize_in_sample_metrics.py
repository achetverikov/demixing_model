#!/usr/bin/env python3
"""Report in-sample and held-out curve errors on every Phase A core metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from continuous_density import equivalence


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, size), group in frame.groupby(["family", "size"]):
        group = group.sort_values("feat_diff")
        for split, prefix in (("fitting", "train_raw_"), ("heldout", "raw_")):
            for metric in ("mean_bias", "resultant", "response_sd", "density_asymmetry"):
                result = equivalence.curve_summary(
                    group[f"pred_{metric}"], group[prefix + metric],
                    circular=metric == "mean_bias")
                rows.append({"family": family, "size": int(size), "split": split,
                             "metric": metric, **result})
        rows.extend([
            {"family": family, "size": int(size), "split": "fitting",
             "metric": "nll", "mean": float(group.train_nll.mean())},
            {"family": family, "size": int(size), "split": "heldout",
             "metric": "nll", "mean": float(group.heldout_nll.mean())},
        ])
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.metrics], ignore_index=True)
    output = summarize(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
