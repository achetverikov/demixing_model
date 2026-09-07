"""Run the closed-loop recovery panel and write its results as an artifact.

`recovery.py` is the library; this is the protocol. It exists because the first
version of the panel was run from an interactive session and only its conclusions
were written down: the truth vector, the seed, the start count and the trial
counts all had to be reverse-engineered afterwards, and the write-up's numbers
could not be regenerated. Every number in RECOVERY_FINDINGS.md is now derived
from the files this writes, per CLAUDE.md section 8.

Three panels, each a separate `--panel` run:

* ``noise_free`` -- fit the model's own smoothed curve at the generating
  parameters. No sampling noise, no KDE, no empirical feature weighting. This is
  the reference point the other two are read against.
* ``empirical`` -- replicates at one trial count, through the real empirical
  target that a subject's data would build.
* ``sample_size`` -- the same case at several trial counts, to see whether the
  error shrinks with data.

Usage::

    JAX_PLATFORMS=cpu python model_fit_to_data/run_recovery_panel.py \
        --panel empirical --out results/continuous_density_4.1q/recovery

Writes ``<out>/<panel>_rows.csv`` (one row per replicate, every field
``RecoveryResult.row()`` carries) and ``<out>/<panel>_summary.json`` (the summary
plus the exact settings the run used, so the run can be repeated).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "model_fit_to_data"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import recovery as R  # noqa: E402
import wnm_scoring as S  # noqa: E402
from continuous_fit import fit_continuous  # noqa: E402
from continuous_optimizer import (  # noqa: E402
    build_bounds, condition_parameter_layout, minimize_continuous)
from density_objective import degenerate_targets  # noqa: E402
from fitting_targets import build_fitting_targets  # noqa: E402
from grid_based_multi_condition_optimizer_jax_loops import (  # noqa: E402
    _compute_curve_losses, bwcrps_energy_score, compute_bwcrps_condition_targets,
    compute_target_bias_curve_core)
from shared import surrogate  # noqa: E402
from shared.config import config  # noqa: E402
from shared.prediction import predictor_from_surrogate  # noqa: E402

#: The panel's one generating configuration. Two conditions with the feature SDs
#: crossed, so a fit that swapped the two would be visible, and a shared spatial
#: SD in the middle of its range.
TRUTH_CONDITIONS = [(15.0, 45.0), (60.0, 20.0)]
TRUTH_SD_SPAT = 25.0
EMP_DENSITY_WEIGHTS_SD = 20.0


def _grids():
    feat_grid = config.create_grid('feat_diff')
    bias_grid = config.create_grid('mu1_bias')
    difference = jnp.abs(bias_grid[:, None] - bias_grid[None, :])
    return feat_grid, jnp.minimum(difference, 360.0 - difference)


def _build_targets(datasets, feat_grid, d_circ):
    return build_fitting_targets(
        {name: jnp.asarray(values) for name, values in datasets.items()},
        feat_diff_grid=feat_grid, d_circ_matrix=d_circ,
        n_mu1_bias=len(config.create_grid('mu1_bias')),
        emp_density_weights_sd=EMP_DENSITY_WEIGHTS_SD, density_bandwidth_rule="sj",
        density_bandwidth_mode="pooled", degenerate_targets=degenerate_targets,
        bwcrps_condition_targets=compute_bwcrps_condition_targets,
        target_bias_curve_core=compute_target_bias_curve_core)


def _case(n_trials):
    return R.RecoveryCase(name=f"n{n_trials}", condition_feature_sds=TRUTH_CONDITIONS,
                          sd_spat=TRUTH_SD_SPAT, n_trials_per_condition=n_trials)


def run_noise_free(predictor, feat_grid, *, objective, n_starts, seed):
    """Fit the model's own curve at the truth, with no target construction at all.

    This isolates the objective and the search: if recovery is not exact here,
    nothing measured against an empirical target can be attributed.
    """
    truth = []
    for sd_feat1, sd_feat2 in TRUTH_CONDITIONS:
        truth.extend([sd_feat1, sd_feat2])
    truth.append(TRUTH_SD_SPAT)

    ideal = jnp.stack([
        S.predicted_asymmetry_curve(predictor, truth[2 * index], truth[2 * index + 1],
                                    truth[-1], feat_grid, EMP_DENSITY_WEIGHTS_SD)
        for index in range(len(TRUTH_CONDITIONS))])

    def objective_fn(parameters):
        total = 0.0
        for index in range(len(TRUTH_CONDITIONS)):
            predicted = S.predicted_asymmetry_curve(
                predictor, parameters[2 * index], parameters[2 * index + 1],
                parameters[-1], feat_grid, EMP_DENSITY_WEIGHTS_SD)
            total = total + _compute_curve_losses(
                predicted[None, :], ideal[index][None, :], loss_type="ccc",
                is_angular=False)[0]
        return total

    bounds_by_axis = surrogate.search_bounds(predictor.domain)
    names = condition_parameter_layout(len(TRUTH_CONDITIONS), fit_motor=False)
    started = time.time()
    fit = minimize_continuous(
        objective_fn,
        build_bounds(len(TRUTH_CONDITIONS), bounds_by_axis["sd_feat"],
                     bounds_by_axis["sd_spat"]),
        names, n_starts=n_starts, seed=seed)
    runtime = time.time() - started

    result = R.RecoveryResult(
        case="noise_free", replicate=0, truth=np.asarray(truth, dtype=np.float64),
        recovered=np.asarray(fit.parameters, dtype=np.float64), names=tuple(names),
        loss_at_truth=float(objective_fn(jnp.asarray(truth))),
        loss_at_fit=float(fit.loss), loss_spread=float(fit.loss_spread),
        n_starts=int(fit.n_starts),
        n_converged=int(sum(start.success for start in fit.starts)),
        at_bound=tuple(fit.at_bound), runtime_seconds=runtime,
        # Only the starts that converged: a failed start's loss is wherever the
        # optimizer stopped, which is not a statement about the landscape.
        start_losses=[float(start.loss) for start in fit.starts if start.success])
    return [result]


def run_replicates(predictor, feat_grid, d_circ, cases, *, objective, n_starts, seed,
                   n_replicates):
    results = []
    for case in cases:
        for replicate in range(n_replicates):
            outcome = R.run_replicate(
                predictor, case, replicate, objective=objective,
                build_targets=lambda datasets: _build_targets(datasets, feat_grid, d_circ),
                fit_continuous=fit_continuous, score_all_conditions=S.score_all_conditions,
                curve_losses=_compute_curve_losses, energy_score=bwcrps_energy_score,
                d_circ_matrix=d_circ, feat_diff_grid=feat_grid,
                emp_density_weights_sd=EMP_DENSITY_WEIGHTS_SD, n_starts=n_starts,
                seed=seed)
            # A progress line, so a long panel is distinguishable from a hung one.
            print(f"{case.name} replicate {replicate}: {outcome.diagnosis}, "
                  f"worst |log ratio| "
                  f"{np.max(np.abs(np.log(outcome.recovered / outcome.truth))):.3f}, "
                  f"{outcome.runtime_seconds:.0f}s", flush=True)
            results.append(outcome)
    return results


def worst_parameter(results):
    """The parameter with the largest absolute log-ratio error, and its value.

    Reported by name. The first write-up quoted a sequence it described as "the
    worst parameter" that in fact tracked one fixed parameter across sample
    sizes; which parameter is worst changes with the sample size, so the name
    has to travel with the number.
    """
    record = {}
    for result in results:
        errors = np.abs(np.log(result.recovered / result.truth))
        index = int(np.argmax(errors))
        record[f"{result.case}_r{result.replicate}"] = {
            "parameter": result.names[index],
            "log_ratio": float(np.log(result.recovered[index] / result.truth[index])),
        }
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", required=True,
                        choices=("noise_free", "empirical", "sample_size"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--objective", default="density")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="a specific WNM artifact. Defaults to the packaged WNM artifact for "
             "--n-samples, not to `production_checkpoint`: production is still the "
             "surface backend until step 7 flips it, and this panel is about the mixture.")
    parser.add_argument("--n-starts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=4000,
                        help="trials per condition for the empirical panel")
    parser.add_argument("--trial-counts", type=int, nargs="+",
                        default=[400, 2000, 10000],
                        help="trials per condition for the sample-size panel")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or surrogate.WNM_DEFAULTS[args.n_samples]
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no WNM artifact at {checkpoint}. Pass --checkpoint, or package one with "
            "continuous_density/package_wnm_artifact.py.")
    predictor = predictor_from_surrogate(surrogate.load_surrogate(checkpoint_path=checkpoint))
    feat_grid, d_circ = _grids()

    if args.panel == "noise_free":
        results = run_noise_free(predictor, feat_grid, objective=args.objective,
                                 n_starts=args.n_starts, seed=args.seed)
    elif args.panel == "empirical":
        results = run_replicates(
            predictor, feat_grid, d_circ, [_case(args.n_trials)],
            objective=args.objective, n_starts=args.n_starts, seed=args.seed,
            n_replicates=args.n_replicates)
    else:
        results = run_replicates(
            predictor, feat_grid, d_circ, [_case(n) for n in args.trial_counts],
            objective=args.objective, n_starts=args.n_starts, seed=args.seed,
            n_replicates=args.n_replicates)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([result.row() for result in results])
    rows_path = args.out / f"{args.panel}_rows.csv"
    rows.to_csv(rows_path, index=False)

    # Summaries are per case: pooling across trial counts would average away the
    # thing the sample-size panel exists to measure.
    by_case = {}
    for case_name in dict.fromkeys(result.case for result in results):
        by_case[case_name] = R.summarise([r for r in results if r.case == case_name])

    summary = {
        "panel": args.panel,
        "settings": {
            "objective": args.objective, "checkpoint": str(checkpoint),
            "checkpoint_sha256": surrogate.file_digest(checkpoint),
            "n_samples": args.n_samples, "n_starts": args.n_starts, "seed": args.seed,
            "n_replicates": args.n_replicates,
            "truth_conditions": TRUTH_CONDITIONS, "truth_sd_spat": TRUTH_SD_SPAT,
            "emp_density_weights_sd": EMP_DENSITY_WEIGHTS_SD,
            "trial_counts": (args.trial_counts if args.panel == "sample_size"
                             else [args.n_trials]),
        },
        "by_case": by_case,
        "worst_parameter": worst_parameter(results),
    }
    summary_path = args.out / f"{args.panel}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"wrote {rows_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
