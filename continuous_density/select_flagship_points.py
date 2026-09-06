#!/usr/bin/env python3
"""Freeze high-N flagship points from selection-only local-fit discrepancies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuous_density import equivalence


def candidate_point_scores(group: pd.DataFrame) -> np.ndarray:
    """Largest normalized pointwise error among the frozen Phase A metrics."""
    mean_margin = np.where(group.raw_resultant >= 0.8, 0.25, 0.50)
    mean_error = np.abs(equivalence.wrap_deg(
        group.pred_mean_bias.to_numpy() - group.raw_mean_bias.to_numpy()))
    errors = np.column_stack([
        np.abs(group.pred_density_asymmetry - group.raw_density_asymmetry) / 0.005,
        mean_error / mean_margin,
        np.abs(group.pred_response_sd - group.raw_response_sd) / 0.25,
        group.circular_wasserstein_deg.to_numpy() / 0.50,
    ])
    return np.max(errors, axis=1)


def select_case(case_dir: Path, component: int) -> dict:
    component_dir = case_dir / f"component_{component}"
    paths = [component_dir / "wrapped/local_wrapped_metrics.csv",
             component_dir / "fourier/local_fourier_metrics.csv"]
    frame = pd.concat([pd.read_csv(path) for path in paths if path.exists()],
                      ignore_index=True)
    if frame.empty:
        raise ValueError(f"no fitted candidates in {component_dir}")
    candidates = []
    for (family, size), group in frame.groupby(["family", "size"]):
        group = group.sort_values("feat_diff").reset_index(drop=True)
        score = candidate_point_scores(group)
        candidates.append(pd.DataFrame({
            "feat_diff": group.feat_diff, "family": family, "size": int(size),
            "score": score, "row": np.arange(len(group)),
        }))
    scores = pd.concat(candidates, ignore_index=True)
    best = scores.loc[scores.groupby("feat_diff").score.idxmin()]
    binding = best.loc[best.score.idxmax()]
    return {"trajectory": case_dir.name, "component": component,
            "feat_diff": float(binding.feat_diff),
            "candidate_row": int(binding.row),
            "best_family": str(binding.family), "best_size": int(binding["size"]),
            "normalized_binding_score": float(binding.score)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fit_root", type=Path)
    parser.add_argument("--case", action="append", required=True,
                        help="trajectory:component; repeat for 3--5 flagships")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    blob = np.load(args.reference, allow_pickle=True)
    selected = []
    for spec in args.case:
        name, component_text = spec.rsplit(":", 1)
        record = select_case(args.fit_root / name, int(component_text))
        mask = (np.asarray(blob["strata"]).astype(str) == name
                if len(blob["strata"]) else np.ones(len(blob["design"]), dtype=bool))
        rows = np.flatnonzero(mask & np.isclose(
            blob["design"][:, 3], record["feat_diff"]))
        if len(rows) != 1:
            raise ValueError(f"expected one corpus row for {name} at {record['feat_diff']}")
        record["selection_corpus_row"] = int(rows[0])
        record["design"] = np.asarray(blob["design"][rows[0]], dtype=float).tolist()
        selected.append(record)
    design = np.asarray([record["design"] for record in selected], dtype=np.float32)
    labels = np.asarray([record["trajectory"] for record in selected])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, design=design, strata=labels)
    args.manifest.write_text(json.dumps({"version": 1,
        "selection_reference": str(args.reference), "points": selected}, indent=2) + "\n")
    print(f"Frozen {len(selected)} flagship points in {args.out}")


if __name__ == "__main__":
    main()
