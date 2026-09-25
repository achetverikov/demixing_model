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

The packaged file is verified before it reaches its destination: it is written to
a temporary path, loaded back through the production loader, and evaluated on a
fixed parameter panel that must reproduce the research weights bit for bit and
that must pass the domain validation the artifact itself advertises.  Only then
is it renamed into place.  Packaging is a copy, so anything less than exact
agreement means the architecture was reconstructed wrongly -- and a failed
packaging must not leave a half-trusted artifact where production will find it.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import wnm as wm  # noqa: E402
from shared.prediction import domain_from_meta, validate_params  # noqa: E402
from shared.hashing import file_sha256  # noqa: E402

#: Bumped whenever the artifact's fields or their meaning change.  Consumers
#: compare against it rather than guessing from which keys happen to be present.
#: wnm/2 replaced the single supported_sd_range + supported_feat_diff_range
#: pair with a per-parameter supported_domain measured from the corpus.
ARTIFACT_SCHEMA = "wnm/2"

#: Parameter names in the order the corpus stores them in ``auxiliary``.
AUXILIARY_ORDER = ("feat_diff", "sd_feat1", "sd_feat2", "dprime")

#: The domain the artifact declares as usable, as opposed to the exact hull the
#: corpus happens to reach.  The two differ by a fraction of a degree at three
#: corners -- the corpus stops at sd_feat1 198.11 and sd_spat 5.0018 -- and the
#: declared bounds round outward to the design's round numbers, accepting that
#: sliver of extrapolation deliberately (user decision, 2026-09-06).  Rounding
#: outward is safe here only because it is a sliver: the overhang is recorded per
#: parameter in the artifact, so it can be checked rather than assumed.
DECLARED_DOMAIN = {
    "sd_feat1": (2.5, 200.0),
    "sd_feat2": (2.5, 200.0),
    "sd_spat": (5.0, 200.0),
    "feat_diff": (0.5, 180.0),
}

#: A fixed panel spanning the supported domain: narrow and broad feature noise in
#: both orders, low and high d-prime, feature differences at both ends, and every
#: corner of the SD box.  It proves the packaged artifact reproduces the research
#: weights, and -- because it is pushed through the production predictor -- that
#: the domain the artifact advertises actually accepts its own boundary.
VERIFICATION_PANEL = np.array([
    [10.0, 10.0, 10.0, 2.0],
    [10.0, 120.0, 10.0, 30.0],
    [120.0, 10.0, 10.0, 30.0],
    [60.0, 60.0, 30.0, 90.0],
    [5.0, 190.0, 5.5, 179.0],
    [2.5, 2.5, 5.0, 0.5],       # the declared low corner, sliver of extrapolation included
    [200.0, 200.0, 200.0, 180.0],  # and the declared high corner
], dtype=np.float32)


def file_digest(path: Path) -> str:
    """Compatibility alias for the canonical streaming file hasher."""
    return file_sha256(path)


def corpus_domain(stage: Path) -> dict:
    """Measure the parameter hull the corpus actually covers.

    The domain an artifact advertises has to come from the data it was trained
    on, not from the featurisation constants. Those constants bracket ``[5, 200]``
    because that is the interval they rescale to ``[-1, 1]``; the corpus reaches
    ``sd_feat`` down to 2.5, so reading the domain off them understated the
    trained region by a factor of two at the low end -- refusing predictions in
    the very region that motivates this surrogate -- while overstating it at
    ``feat_diff = 0``, which the corpus never visits.

    Read from every shard's ``auxiliary`` array, including any inherited block, so
    a stage that extends an earlier corpus reports the whole design rather than
    the part it simulated itself.
    """
    shards = sorted((stage / "artifacts" / "histogram_shards").glob("*.npz"))
    shards += sorted((stage / "artifacts").glob("existing_*_histograms.npz"))
    if not shards:
        raise SystemExit(f"{stage.name}: no corpus shards found under artifacts/")

    low = np.full(4, np.inf)
    high = np.full(4, -np.inf)
    n_cells = 0
    for shard in shards:
        auxiliary = np.load(shard)["auxiliary"]
        low = np.minimum(low, auxiliary.min(axis=0))
        high = np.maximum(high, auxiliary.max(axis=0))
        n_cells += len(auxiliary)

    hull = {name: [float(low[i]), float(high[i])]
            for i, name in enumerate(AUXILIARY_ORDER)}
    # The model takes sd_spat, not d-prime, and the map inverts the interval.
    hull["sd_spat"] = [42.0 / hull["dprime"][1], 42.0 / hull["dprime"][0]]
    hull.pop("dprime")

    # Declared bounds must not meaningfully understate the hull: that would
    # refuse trained parameters, which is how this was wrong in the first place.
    # The tolerance is for representation noise only -- sd_spat's hull top is
    # 42/0.21 = 200.0000062, so declaring 200.0 "loses" six microdegrees.
    UNDERSTATEMENT_TOLERANCE = 1e-3
    # A declared box may round outward past the corpus, but only by a sliver.
    # Without a ceiling, "measured from the corpus" would be a claim the packager
    # does not enforce: a corpus spanning only [50, 60] would still advertise
    # [2.5, 200] and authorise predictions nothing was trained for.
    MAX_OUTWARD_OVERHANG = 5.0
    declared = {key: [float(lo), float(hi)] for key, (lo, hi) in DECLARED_DOMAIN.items()}
    overhang = {}
    for key, (lo, hi) in declared.items():
        hull_lo, hull_hi = hull[key]
        lost = max(lo - hull_lo, hull_hi - hi, 0.0)
        if lost > UNDERSTATEMENT_TOLERANCE:
            raise SystemExit(
                f"declared domain for {key} is [{lo}, {hi}] but the corpus reaches "
                f"[{hull_lo:.6g}, {hull_hi:.6g}]: declaring {lost:.6g} degrees less than was "
                "trained refuses predictions the network can actually make.")
        # Positive means the declaration extends past the corpus at that end.
        overhang[key] = [round(hull_lo - lo, 6), round(hi - hull_hi, 6)]
        widest = max(overhang[key])
        if widest > MAX_OUTWARD_OVERHANG:
            raise SystemExit(
                f"declared domain for {key} is [{lo}, {hi}] but the corpus only reaches "
                f"[{hull_lo:.6g}, {hull_hi:.6g}] -- {widest:.6g} degrees of extrapolation, past "
                f"the {MAX_OUTWARD_OVERHANG:g}-degree ceiling. Either this corpus is not the "
                "one these declared bounds were written for, or the bounds need revisiting; "
                "advertising a domain the corpus never covered authorises predictions nothing "
                "was trained for.")

    return {"supported_domain": declared,
            "corpus_hull": hull,
            "declared_domain_overhang": overhang,
            "supported_domain_note": (
                "supported_domain is the declared usable box; corpus_hull is what the "
                "corpus actually reaches. declared_domain_overhang gives [low, high] "
                "extrapolation accepted at each end, in model degrees."),
            "supported_domain_source": f"{len(shards)} corpus shards, {n_cells} cells",
            "corpus_n_cells_total": n_cells}


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
        # The supported domain is measured from the corpus by corpus_domain()
        # and merged in below. It is deliberately not derived from the
        # featurisation constants, which describe an input rescaling rather than
        # what the network was shown.
    }
    meta.update(corpus_provenance(stage, n_samples))
    meta.update(corpus_domain(stage))
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

    # The two observer models share an architecture, so an n20 fit packaged
    # against the n100 stage would load, run, and be labelled n100 -- weights
    # from one observer wearing the other's provenance. The run checkpoint the
    # fit was selected from lives in the training stage's cache, so require the
    # fit to point back into a cache whose name carries the requested count.
    run_checkpoint = str(fit.get("checkpoint", ""))
    expected_cache = f"wnm_full_n{args.n_samples}"
    if run_checkpoint and expected_cache not in run_checkpoint:
        raise SystemExit(
            f"{args.fit.name} was selected from {run_checkpoint}, which is not an "
            f"n_samples={args.n_samples} training cache ({expected_cache}). Packaging it as "
            f"n_samples={args.n_samples} would attach one observer model's provenance to "
            "another's weights; both counts share an architecture, so nothing downstream "
            "would notice.")
    if not run_checkpoint:
        raise SystemExit(
            f"{args.fit.name} records no run checkpoint, so nothing ties its weights to an "
            f"observer model and --n-samples={args.n_samples} cannot be checked. The two "
            "counts share an architecture, so a mislabelled artifact loads, runs, and is "
            "wrong everywhere silently. Package from a fit that records its source "
            "checkpoint.")

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

    # Write, verify, then rename. A verification failure must not leave a
    # half-trusted artifact at the destination for production to pick up.
    staged = args.out.with_name(args.out.name + ".packaging")
    wm.save_model(staged, fit["variables"], model, meta)
    try:
        loaded_model, loaded_variables, loaded_meta = wm.load_model(staged)
        packaged = predictions(loaded_model, loaded_variables, VERIFICATION_PANEL)
        for key, expected in reference.items():
            if not np.array_equal(packaged[key], expected):
                raise SystemExit(
                    f"packaged artifact does not reproduce the research weights: {key} differs "
                    f"by up to {np.max(np.abs(packaged[key] - expected)):.3e} on the "
                    "verification panel. Packaging copies weights, so any difference is a "
                    "reconstruction bug.")

        # The artifact must accept its own advertised domain, boundary included.
        # Recording a domain the predictor then rejects would refuse legal fits at
        # exactly the extremes the production bounds allow.
        validate_params(VERIFICATION_PANEL, domain=domain_from_meta(loaded_meta),
                        name=f"{args.out.name} verification panel")

        staged.replace(args.out)
    finally:
        staged.unlink(missing_ok=True)

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
