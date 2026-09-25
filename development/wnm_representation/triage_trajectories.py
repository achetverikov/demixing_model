#!/usr/bin/env python3
"""Combine corrected model errors and full-spectrum pilots for benchmark triage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from development.wnm_representation import equivalence


KEYS = ["sd_feat1", "sd_feat2", "sd_ident", "component"]


def spectral_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(KEYS).agg(
        max_required_k_asymmetry=("required_k_asymmetry", "max"),
        unresolved_asymmetry_points=("required_k_asymmetry", lambda x: int((x < 0).sum())),
        max_required_k_fejer_w1=("required_k_fejer_wasserstein", "max"),
        unresolved_fejer_w1_points=(
            "required_k_fejer_wasserstein", lambda x: int((x < 0).sum())),
        max_real_harmonic=("peak_real_abs", "max"),
        max_imag_harmonic=("peak_imag_abs", "max"),
    ).reset_index()


def error_summary(uev: pd.DataFrame, model_label: str) -> pd.DataFrame:
    keys = ["sd_feat1", "sd_feat2", "sd_ident", "which_comp"]
    raw = uev[uev.method == "raw 100k"].set_index(keys + ["dist_feat"])
    model = uev[uev.method == model_label].set_index(keys + ["dist_feat"])
    joined = raw[["mean_bias", "response_sd", "density_asymmetry"]].join(
        model[["mean_bias", "response_sd", "density_asymmetry"]],
        lsuffix="_raw", rsuffix="_pred", how="inner").reset_index()
    rows = []
    for key, group in joined.groupby(keys):
        group = group.sort_values("dist_feat")
        row = dict(zip(KEYS, key))
        for metric, circular in (("mean_bias", True), ("response_sd", False),
                                 ("density_asymmetry", False)):
            summary = equivalence.curve_summary(
                group[f"{metric}_pred"], group[f"{metric}_raw"], circular)
            for name in ("max_abs_error", "mean_signed_error", "amplitude_slope",
                         "longest_sign_run", "longest_run_mean_error"):
                row[f"{metric}_{name}"] = summary[name]
        rows.append(row)
    return pd.DataFrame(rows)


def mode_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize exploratory mode count and switching over each trajectory."""
    rows = []
    for key, group in frame.groupby(KEYS):
        group = group.sort_values("feat_diff")
        count = group.mode_count.to_numpy()
        rows.append({**dict(zip(KEYS, key)), "max_mode_count": int(count.max()),
                     "multimodal_points": int((count > 1).sum()),
                     "mode_count_switches": int((count[1:] != count[:-1]).sum()),
                     "max_secondary_mode_mass": float(group.secondary_mode_mass.max())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", type=Path, nargs="+", required=True)
    parser.add_argument("--uev", type=Path, required=True)
    parser.add_argument("--model-label", default="trajectory density")
    parser.add_argument("--modes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spectrum = pd.concat([pd.read_csv(path) for path in args.spectrum], ignore_index=True)
    table = spectral_summary(spectrum).merge(
        error_summary(pd.read_csv(args.uev), args.model_label), on=KEYS, how="outer")
    if args.modes:
        table = table.merge(mode_summary(pd.read_csv(args.modes)), on=KEYS, how="left")
    table["fails_corrected_model_point_margin"] = (
        (table.density_asymmetry_max_abs_error > 0.005)
        | (table.response_sd_max_abs_error > 0.25)
        | (table.mean_bias_max_abs_error > 0.50))
    table["fails_corrected_model_coherent_margin"] = (
        (table.density_asymmetry_mean_signed_error.abs() > 0.002)
        | (table.response_sd_mean_signed_error.abs() > 0.10)
        | (table.mean_bias_mean_signed_error.abs() > 0.25))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} trajectory rows to {args.out}")


if __name__ == "__main__":
    main()
