#!/usr/bin/env python3
"""Package a research WNM fit into a self-contained production artifact.

The training scripts under ``results/continuous_density_4.1*/scripts`` write a
fit dictionary -- ``variables``, ``selected_step``, ``final_step``,
``validation_nll`` and the path of the run checkpoint it was selected from.  That
is enough to continue research and not enough to run in production: it records no
architecture, so nothing can rebuild the network without the training script, and
no provenance, so nothing downstream can say which observer model or corpus a
prediction came from.

This script reads such a fit, together with the corpus stage that produced it,
and writes the format :func:`continuous_density.wrapped_mixture_model.save_model`
defines: weights, architecture, and the scientific metadata that has to travel
with them.  It changes no weights and retrains nothing.

Usage (from the repo root)::

    python continuous_density/package_wnm_artifact.py \\
        --fit /path/to/wnm_k12_full_n20_alldata-<digest>-best.pkl \\
        --corpus-stage /path/to/results/continuous_density_4.1p \\
        --out pretrained/wnm_k12_20samples.pkl

The packaged file is verified against the research weights before it is written:
both are evaluated on a fixed parameter panel and every mixture parameter must
agree bit for bit.  Packaging is a copy, so anything less than exact agreement
means the architecture was reconstructed wrongly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuous_density import wrapped_mixture_model as wm  # noqa: E402

#: Bumped whenever the artifact's fields or their meaning change.  Consumers
#: compare against it rather than guessing from which keys happen to be present.
ARTIFACT_SCHEMA = "wnm/1"

#: A fixed panel spanning the supported domain: narrow and broad feature noise in
#: both orders, low and high d-prime, and feature differences at both ends.  Used
#: only to prove the packaged artifact reproduces the research weights.
VERIFICATION_PANEL = np.array([
    [10.0, 10.0, 10.0, 2.0],
    [10.0, 120.0, 10.0, 30.0],
    [120.0, 10.0, 10.0, 30.0],
    [200.0, 200.0, 200.0, 90.0],
    [5.0, 200.0, 5.0, 179.0],
    [60.0, 60.0, 30.0, 180.0],
], dtype=np.float32)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_provenance(stage: Path, n_samples: int) -> dict:
    """Read the corpus stage's own manifests; do not restate them from prose.

    Stages differ in where they record the observer sample count and how they
    count trajectories -- a stage that extends an inherited corpus reports only
    the trajectories it added.  Both are read from whichever of the config and
    the manifest actually carries them, and the sample count is checked against
    the request rather than assumed.
    """
    config = json.loads((stage / "config" / "experiment.json").read_text())
    manifests = sorted(stage.glob("artifacts/simulation_manifest*.json"))
    if not manifests:
        raise SystemExit(f"{stage.name}: no simulation manifest under artifacts/")

    def sample_count_of(blob):
        return blob.get("n_samples", config.get("n_samples"))

    matching = [(path, json.loads(path.read_text())) for path in manifests]
    matching = [(path, blob) for path, blob in matching
                if sample_count_of(blob) is not None
                and int(sample_count_of(blob)) == int(n_samples)]
    if not matching:
        raise SystemExit(
            f"{stage.name}: no simulation manifest for n_samples={n_samples} "
            f"(found {[p.name for p in manifests]}; stage config says "
            f"n_samples={config.get('n_samples')})")
    if len(matching) > 1:
        raise SystemExit(
            f"{stage.name}: {len(matching)} simulation manifests claim n_samples={n_samples}: "
            f"{[p.name for p, _ in matching]}")
    manifest_path, simulation = matching[0]
    if not simulation.get("complete"):
        raise SystemExit(f"{stage.name}: {manifest_path.name} is not marked complete")

    # A stage that extends an inherited corpus counts only the cells and
    # trajectories it simulated itself, so name those fields for what they are.
    # Reporting them as corpus totals would understate the training set for any
    # stage built on an inherited corpus.
    return {
        "corpus_stage": stage.name,
        "corpus_manifest": manifest_path.name,
        "corpus_n_cells_simulated_here": simulation.get("n_cells"),
        "corpus_n_trajectories_total": config.get("n_total_trajectories"),
        "corpus_n_trajectories_simulated_here": simulation.get(
            "n_trajectories", simulation.get("n_new_trajectories")),
        "corpus_n_trajectories_inherited": config.get("n_existing_trajectories"),
        "corpus_outcomes_per_cell": simulation.get(
            "outcomes_per_cell",
            (config.get("n_blocks") or 0) * (config.get("block_size") or 0) or None),
        "corpus_density_bins": config.get("density_bins"),
        "corpus_stage_digest": simulation.get("stage_digest"),
        "corpus_stores_raw_outcomes": simulation.get("stores_raw_outcomes"),
        "training_n_components": config.get("n_components"),
        "training_hidden_dims": config.get("hidden_dims"),
        "training_min_scale": config.get("min_scale"),
        "training_n_wraps": config.get("n_wraps"),
    }


def build_meta(fit: dict, fit_path: Path, stage: Path, n_samples: int,
               all_data: bool) -> dict:
    meta = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "family": "wnm",
        "n_samples": int(n_samples),

        # Provenance of the weights themselves.
        "source_fit": fit_path.name,
        "source_checkpoint_digest": file_digest(fit_path),
        "source_run_checkpoint": Path(str(fit.get("checkpoint", ""))).name or None,
        "selected_step": fit.get("selected_step"),
        "final_step": fit.get("final_step"),
        "validation_nll": fit.get("validation_nll"),

        # How the stopping step was chosen.  The production-parity variant trains
        # on every trajectory and selects on an in-sample slice, mirroring the
        # surface NN, which has no validation split at all.  Recorded because the
        # recorded NLL is then an in-sample number and must never be reported as
        # a held-out score.
        "checkpoint_selection": "in_sample_slice" if all_data else "held_out_trajectories",
        "validation_nll_is_in_sample": bool(all_data),

        # Scientific conventions the numbers mean nothing without.
        "period_degrees": wm.PERIOD,
        "density_units": "per model degree",
        "bias_sign_convention": "positive = attraction toward the other item",
        "parameter_order": ["sd_feat1", "sd_feat2", "sd_spat", "feat_diff"],
        "spatial_separation_degrees": 42.0,
        "dprime_relation": "dprime = 42 / sd_spat",
        "component_convention": "predicts component 1; component 2 by swapping sd_feat1/sd_feat2",
        "supported_sd_range": [float(np.exp(wm.SD_LOG_LO)), float(np.exp(wm.SD_LOG_HI))],
        "supported_feat_diff_range": [wm.FEAT_DIFF_LO, wm.FEAT_DIFF_HI],
    }
    meta.update(corpus_provenance(stage, n_samples))
    return meta


def predictions(model, variables, panel):
    dist = model.apply(variables, panel)
    return {key: np.asarray(value) for key, value in dist.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", type=Path, required=True,
                        help="Research fit pickle (variables + selected_step).")
    parser.add_argument("--corpus-stage", type=Path, required=True,
                        help="Stage directory whose corpus trained the fit, e.g. "
                             "results/continuous_density_4.1p.")
    parser.add_argument("--n-samples", type=int, required=True, choices=[20, 100],
                        help="Observer evidence samples per trial the corpus was simulated at.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Destination artifact path.")
    parser.add_argument("--held-out", action="store_true",
                        help="The fit selected its step on held-out trajectories. Omit for the "
                             "production-parity 'alldata' fits, whose selection is in-sample.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing artifact at --out.")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite to replace it")

    with open(args.fit, "rb") as handle:
        fit = pickle.load(handle)
    for key in ("variables", "selected_step"):
        if key not in fit:
            raise SystemExit(f"{args.fit} is missing {key!r}; this is not a research WNM fit")

    config = json.loads((args.corpus_stage / "config" / "experiment.json").read_text())
    model = wm.ConditionalWrappedMixture(
        n_components=config["n_components"],
        hidden_dims=tuple(config["hidden_dims"]),
        min_scale=config["min_scale"])

    # Reconstruction check: the network we just built from the stage config must
    # accept the stored weights unchanged.  A shape mismatch here means the fit
    # and the config disagree about the architecture.
    reference = predictions(model, fit["variables"], VERIFICATION_PANEL)

    meta = build_meta(fit, args.fit, args.corpus_stage, args.n_samples,
                      all_data=not args.held_out)
    wm.save_model(args.out, fit["variables"], model, meta)

    loaded_model, loaded_variables, loaded_meta = wm.load_model(args.out)
    packaged = predictions(loaded_model, loaded_variables, VERIFICATION_PANEL)
    for key, expected in reference.items():
        if not np.array_equal(packaged[key], expected):
            raise SystemExit(
                f"packaged artifact does not reproduce the research weights: {key} differs by "
                f"up to {np.max(np.abs(packaged[key] - expected)):.3e} on the verification "
                "panel. Packaging copies weights, so any difference is a reconstruction bug.")

    print(f"wrote {args.out}")
    print(f"  K={loaded_model.n_components} hidden={tuple(loaded_model.hidden_dims)} "
          f"min_scale={loaded_model.min_scale}")
    print(f"  n_samples={loaded_meta['n_samples']} step={loaded_meta['selected_step']} "
          f"selection={loaded_meta['checkpoint_selection']}")
    print(f"  corpus={loaded_meta['corpus_stage']} "
          f"cells={loaded_meta['corpus_n_cells_simulated_here']} "
          f"trajectories={loaded_meta['corpus_n_trajectories_total']} total, "
          f"{loaded_meta['corpus_n_trajectories_simulated_here']} simulated in this stage")
    print(f"  verified on {len(VERIFICATION_PANEL)} fixed parameter rows, exact match")


if __name__ == "__main__":
    main()
