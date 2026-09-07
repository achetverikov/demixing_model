"""Parameter recovery: does minimising an objective return the parameters that
generated the data?

The forward checks this transition already has -- CCC and MAE of predicted curves
at known parameters -- test the *forward* map. Recovery tests the inverse one, and
they can disagree completely: an objective can reproduce a curve beautifully while
the parameters behind it are unidentified, because several parameter sets produce
nearly the same curve. Only recovery distinguishes "the model fits" from "the
model's parameters mean something".

Three questions the transition plan insists on separating, because a single
number conflates them:

* **Search** -- with the same surrogate, target and data, does the method find a
  good solution reliably, and at what cost?
* **Surrogate** -- with a common search, does the mixture recover better than the
  network?
* **Target/identifiability** -- does minimising this objective recover the
  generating parameters *at all*, even with a good surrogate and search?

This module runs the closed-loop panel: responses drawn from the mixture itself,
fitted back with the mixture. That is deliberately the weaker of the two panels
the plan requires, and it is worth being explicit about what it cannot show. A
model fitting its own samples has no surrogate error, so this panel cannot say
whether the mixture approximates the observer well; it can only expose wiring
faults, gradient faults, and unidentifiability that is intrinsic to the objective.
The main panel generates from the actual simulator and is a separate, expensive
stage.

Recovery failures come in three flavours and the record distinguishes them:

* the search never found the basin (visible as loss at the fit worse than loss at
  the truth);
* the search found a better loss than the truth, so the objective genuinely
  prefers other parameters -- an identifiability finding, not a search failure;
* the search found the same loss at different parameters -- a flat direction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

#: Both losses come out of a float32 objective, so the tie band is expressed in
#: float32 ULPs of the larger loss rather than as an absolute number. 32 ULPs
#: leaves room for the accumulation across conditions and starts that produces
#: the observed ~2e-7 residuals, while staying far below any loss difference
#: that would mean something.
FLOAT32_EPS = float(np.finfo(np.float32).eps)
TIE_TOLERANCE_ULPS = 32.0


@dataclass
class RecoveryCase:
    """One generating configuration, fixed before anything is run."""

    name: str
    #: ``[(sd_feat1, sd_feat2), ...]`` -- one pair per condition.
    condition_feature_sds: Sequence[tuple]
    sd_spat: float
    sd_motor: float = 0.0
    n_trials_per_condition: int = 400
    #: Feature differences the design visits; ``None`` samples them uniformly.
    feat_diff_values: Optional[Sequence[float]] = None

    @property
    def n_conditions(self) -> int:
        return len(self.condition_feature_sds)

    def truth_vector(self, fit_motor: bool = False) -> np.ndarray:
        """The generating parameters in the optimizer's own layout."""
        values: List[float] = []
        for sd_feat1, sd_feat2 in self.condition_feature_sds:
            values.extend([float(sd_feat1), float(sd_feat2)])
        values.append(float(self.sd_spat))
        if fit_motor:
            values.append(float(self.sd_motor))
        return np.asarray(values, dtype=np.float64)


@dataclass
class RecoveryResult:
    """What one replicate produced, including the parts that say it failed."""

    case: str
    replicate: int
    truth: np.ndarray
    recovered: np.ndarray
    names: tuple
    loss_at_truth: float
    loss_at_fit: float
    loss_spread: float
    n_starts: int
    n_converged: int
    at_bound: tuple
    runtime_seconds: float
    start_losses: List[float] = field(default_factory=list)

    @property
    def diagnosis(self) -> str:
        """Which of the three failure modes this replicate shows.

        The distinction matters more than the error magnitude: a search failure
        is fixed by more starts, an identifiability failure is not fixed by
        anything and constrains what the parameter can be said to mean.

        The tie band is float32-relative, not a fixed 1e-9. Both losses are
        computed in float32, where one ULP near a loss of 1 is about 1.2e-7 --
        a hundred times the old 1e-9 threshold. Under that threshold two
        adjacent float32 values on either side of the truth were reported as
        "the search failed" and "the objective prefers other parameters", two
        opposite scientific conclusions drawn from rounding. The noise-free run
        lands at -2.4e-7, which is arithmetic noise and is now reported as the
        tie it is.
        """
        gap = self.loss_at_fit - self.loss_at_truth
        if not np.isfinite(gap):
            raise ValueError(
                f"cannot diagnose case {self.case!r} replicate {self.replicate}: loss at fit "
                f"{self.loss_at_fit!r}, loss at truth {self.loss_at_truth!r}. A non-finite "
                "loss is a failed evaluation, not a tie, and was previously classified as "
                "'loss_tied_with_truth' because every comparison against NaN is false.")
        tolerance = TIE_TOLERANCE_ULPS * FLOAT32_EPS * max(
            1.0, abs(self.loss_at_fit), abs(self.loss_at_truth))
        if gap > tolerance:
            return "search_failed"          # never reached the generating basin
        if gap < -tolerance:
            return "objective_prefers_other_parameters"
        return "loss_tied_with_truth"       # flat direction, or exact recovery

    def errors(self) -> Dict[str, float]:
        """Signed and log-ratio error per parameter.

        Log ratio because these are multiplicative scales: being 5 degrees out at
        200 is not the error that being 5 degrees out at 5 is.
        """
        record: Dict[str, float] = {}
        for name, true_value, fitted in zip(self.names, self.truth, self.recovered):
            record[f"signed_error_{name}"] = float(fitted - true_value)
            record[f"log_ratio_{name}"] = float(np.log(fitted / true_value))
        return record

    def row(self) -> Dict[str, object]:
        """Flat record, one per replicate."""
        record: Dict[str, object] = {
            "case": self.case, "replicate": self.replicate,
            "loss_at_truth": self.loss_at_truth, "loss_at_fit": self.loss_at_fit,
            "loss_gap": self.loss_at_fit - self.loss_at_truth,
            "diagnosis": self.diagnosis, "loss_spread": self.loss_spread,
            "n_starts": self.n_starts, "n_converged": self.n_converged,
            "at_bound": ",".join(self.at_bound), "runtime_seconds": self.runtime_seconds,
            # The start-level evidence the diagnosis rests on. "search_failed"
            # versus "the objective prefers other parameters" is a claim about
            # whether the search covered the space, and a row that reports only
            # the winning loss cannot be audited on that point afterwards.
            "start_losses": ",".join(f"{value:.9g}" for value in self.start_losses),
            "best_start_loss": (min(self.start_losses) if self.start_losses else float("nan")),
            "worst_start_loss": (max(self.start_losses) if self.start_losses else float("nan")),
        }
        for name, true_value, fitted in zip(self.names, self.truth, self.recovered):
            record[f"true_{name}"] = float(true_value)
            record[f"fit_{name}"] = float(fitted)
        record.update(self.errors())
        return record


def sample_mixture_responses(predictor, sd_feat1, sd_feat2, sd_spat, feat_diff,
                             rng, sd_motor: float = 0.0) -> np.ndarray:
    """Draw one bias per trial from the mixture at each trial's feature difference.

    Sampling from the model that will be fitted is what makes this the *closed
    loop* panel: it removes surrogate error entirely, so anything that fails here
    is wiring, gradients, or the objective itself.
    """
    feat_diff = np.asarray(feat_diff, dtype=np.float64)
    rows = jnp.stack([
        jnp.full(feat_diff.shape, float(sd_feat1), jnp.float32),
        jnp.full(feat_diff.shape, float(sd_feat2), jnp.float32),
        jnp.full(feat_diff.shape, float(sd_spat), jnp.float32),
        jnp.asarray(feat_diff, jnp.float32)], axis=-1)

    sd_motor = float(sd_motor)
    if not np.isfinite(sd_motor) or sd_motor < 0:
        raise ValueError(
            f"sd_motor must be finite and non-negative, got {sd_motor!r}: motor noise enters "
            "as a variance, so -30 draws exactly the samples +30 does while the case record "
            "says -30.")
    # A concrete zero reaches `distribution` as zero, which it skips entirely.
    # Collapsing it to None here would instead have meant "use the predictor's
    # own SD", so a motor-carrying predictor would have added noise to a case
    # that asked for none.
    distribution = predictor.distribution(rows, validate=False, sd_motor=sd_motor)
    weights = np.asarray(jnp.exp(distribution["log_pi"]), dtype=np.float64)
    means = np.asarray(distribution["mu"], dtype=np.float64)
    sigmas = np.asarray(distribution["sigma"], dtype=np.float64)

    # One component per trial, then a normal about its mean, then wrapped. The
    # wrap is what makes it a *wrapped* normal rather than a truncated one.
    cumulative = np.cumsum(weights, axis=-1)
    picks = (rng.random((len(feat_diff), 1)) < cumulative).argmax(axis=-1)
    rows_index = np.arange(len(feat_diff))
    draws = rng.normal(means[rows_index, picks], sigmas[rows_index, picks])
    return ((draws + 180.0) % 360.0) - 180.0


def generate_case_data(predictor, case: RecoveryCase, rng) -> Dict[str, np.ndarray]:
    """``{condition: (n_trials, 2)}`` of ``[feat_diff, bias]`` in model degrees."""
    datasets: Dict[str, np.ndarray] = {}
    for index, (sd_feat1, sd_feat2) in enumerate(case.condition_feature_sds):
        if case.feat_diff_values is None:
            feat_diff = rng.uniform(2.0, 178.0, case.n_trials_per_condition)
        else:
            feat_diff = rng.choice(np.asarray(case.feat_diff_values, dtype=np.float64),
                                   case.n_trials_per_condition)
        bias = sample_mixture_responses(predictor, sd_feat1, sd_feat2, case.sd_spat,
                                        feat_diff, rng, case.sd_motor)
        datasets[f"c{index}"] = np.stack([feat_diff, bias], axis=-1).astype(np.float32)
    return datasets


def run_replicate(predictor, case: RecoveryCase, replicate: int, *, objective: str,
                  build_targets, fit_continuous, score_all_conditions,
                  curve_losses, energy_score, d_circ_matrix, feat_diff_grid,
                  emp_density_weights_sd: float, n_starts: int, seed: int,
                  fit_motor: bool = False) -> RecoveryResult:
    """Generate, fit without revealing the truth, and score both.

    The objective is evaluated *at the generating parameters* as well as at the
    fit. Without that, a failure cannot be attributed: a large parameter error
    with a worse-than-truth loss is a search failure, the same error with a
    better-than-truth loss is the objective preferring other parameters, and
    those call for opposite responses.
    """
    rng = np.random.default_rng([seed, replicate])
    datasets = generate_case_data(predictor, case, rng)
    targets = build_targets(datasets)

    trials = [(jnp.asarray(values[:, 0]), jnp.asarray(values[:, 1]))
              for values in datasets.values()]

    # The motor SD the data was generated at has to reach the fit, or the loop is
    # not closed: responses drawn with motor noise would be fitted and scored by a
    # model without it, and every parameter error and diagnosis below would be
    # measuring that mismatch instead of recovery. Held fixed when it is not
    # searched; `fit_continuous` refuses to do both.
    fixed_motor = 0.0 if fit_motor else float(case.sd_motor)

    started = time.time()
    fit = fit_continuous(
        predictor, targets, list(datasets), objective=objective,
        curve_losses=curve_losses, energy_score=energy_score,
        d_circ_matrix=d_circ_matrix, feat_diff_grid=feat_diff_grid,
        emp_density_weights_sd=emp_density_weights_sd, condition_trials=trials,
        sd_motor=fixed_motor, fit_motor=fit_motor, n_starts=n_starts, seed=seed,
        verbosity=0)
    runtime = time.time() - started

    # Truth is scored through the same motor-carrying predictor the fit used.
    # Scoring it without the motor noise would make the comparison that the whole
    # diagnosis rests on a comparison between two different models.
    scoring_predictor = (predictor.with_motor_noise(fixed_motor) if fixed_motor
                         else predictor)

    truth = case.truth_vector(fit_motor=fit_motor)
    loss_at_truth = float(score_all_conditions(
        objective, scoring_predictor, targets, jnp.asarray(truth), curve_losses=curve_losses,
        energy_score=energy_score, d_circ_matrix=d_circ_matrix,
        feat_diff_grid=feat_diff_grid, emp_density_weights_sd=emp_density_weights_sd,
        condition_trials=trials, fit_motor=fit_motor))

    recovered = np.array(
        [fit["condition_results"][name][key]
         for name in datasets for key in ("sd_feat1", "sd_feat2")]
        + [fit["shared_params"]["sd_spat"]]
        + ([fit["shared_params"]["sd_motor"]] if fit_motor else []))

    names = tuple(
        [f"{key}_c{index}" for index in range(case.n_conditions)
         for key in ("sd_feat1", "sd_feat2")]
        + ["sd_spat"] + (["sd_motor"] if fit_motor else []))

    return RecoveryResult(
        case=case.name, replicate=replicate, truth=truth, recovered=recovered,
        names=names, loss_at_truth=loss_at_truth, loss_at_fit=float(fit["best_loss"]),
        loss_spread=float(fit["loss_spread"]), n_starts=int(fit["n_starts"]),
        n_converged=int(fit["n_converged"]), at_bound=tuple(fit["at_bound"]),
        runtime_seconds=runtime, start_losses=list(fit["start_losses"]))


def summarise(results: Sequence[RecoveryResult]) -> Dict[str, object]:
    """Per-parameter bias and RMSE, and the failure mix.

    Correlation between fitted and true parameters is deliberately not reported:
    it is high whenever the design spans a wide range, regardless of whether any
    individual estimate is any good.
    """
    if not results:
        raise ValueError("no replicates to summarise")

    names = results[0].names
    summary: Dict[str, object] = {
        "n_replicates": len(results),
        "diagnoses": {diagnosis: sum(r.diagnosis == diagnosis for r in results)
                      for diagnosis in ("search_failed",
                                        "objective_prefers_other_parameters",
                                        "loss_tied_with_truth")},
        "n_with_boundary_hits": sum(1 for r in results if r.at_bound),
        "median_runtime_seconds": float(np.median([r.runtime_seconds for r in results])),
    }
    for index, name in enumerate(names):
        log_ratios = np.array([np.log(r.recovered[index] / r.truth[index])
                               for r in results])
        summary[f"{name}_median_log_ratio"] = float(np.median(log_ratios))
        summary[f"{name}_rmse_log_ratio"] = float(np.sqrt(np.mean(log_ratios ** 2)))
    return summary


def parameter_family(name: str) -> str:
    """``sd_feat1_c3`` -> ``sd_feat``. The family is what a range panel pools over."""
    if name.startswith("sd_feat"):
        return "sd_feat"
    if name.startswith("sd_spat"):
        return "sd_spat"
    if name.startswith("sd_motor"):
        return "sd_motor"
    return name


def range_summary(results: Sequence[RecoveryResult],
                  drop_railed: bool = True) -> Dict[str, object]:
    """True-versus-recovered agreement for a panel that spans the parameter range.

    `summarise` deliberately omits correlation, and for a fixed generating vector
    that is right: with every replicate at the same truth, correlation measures
    the scatter of the estimate against a constant and means nothing. This
    function is for the opposite design -- truths drawn across the whole range --
    where correlation is exactly the question: does the recovered parameter track
    the one that generated the data, over the range the model will be used on?

    Reported per parameter family, on the log scale, because these are
    multiplicative scales and a correlation dominated by the two-decade spread of
    the design is not the same claim as one that survives on log residuals.

    Three numbers are given together on purpose, because correlation alone is
    flattering:

    * ``pearson_r_log`` -- how well the recovered value tracks the true one.
      Inflated by the width of the design: sample a wider range and it rises
      without the estimator improving.
    * ``slope_log`` -- the regression of recovered on true, both logged. 1.0 is
      faithful; below 1 is compression toward the middle of the range, which a
      high correlation hides completely.
    * ``rmse_log_ratio`` -- the actual error, in units the design width cannot
      inflate.

    Args:
        drop_railed: exclude parameters whose fit sits on a search bound. A
            railed value is the bound's position, not an estimate, so including
            it measures where the bounds are. The count is reported either way.
    """
    if not results:
        raise ValueError("no replicates to summarise")

    from scipy import stats

    pairs: Dict[str, List[tuple]] = {}
    railed: Dict[str, int] = {}
    for result in results:
        bound_names = {name.split("@")[0] for name in result.at_bound}
        for name, true_value, fitted in zip(result.names, result.truth, result.recovered):
            family = parameter_family(name)
            pairs.setdefault(family, [])
            railed.setdefault(family, 0)
            if name in bound_names:
                railed[family] += 1
                if drop_railed:
                    continue
            pairs[family].append((float(true_value), float(fitted)))

    summary: Dict[str, object] = {
        "n_replicates": len(results),
        "drop_railed": bool(drop_railed),
        "diagnoses": {diagnosis: sum(r.diagnosis == diagnosis for r in results)
                      for diagnosis in ("search_failed",
                                        "objective_prefers_other_parameters",
                                        "loss_tied_with_truth")},
    }
    for family, observations in pairs.items():
        record: Dict[str, object] = {
            "n": len(observations),
            "n_railed": railed[family],
            "railed_fraction": railed[family] / max(1, railed[family] + len(observations)),
        }
        if len(observations) >= 3:
            true_values = np.array([pair[0] for pair in observations])
            fitted = np.array([pair[1] for pair in observations])
            log_true, log_fit = np.log(true_values), np.log(fitted)
            regression = stats.linregress(log_true, log_fit)
            record.update({
                "pearson_r_log": float(stats.pearsonr(log_true, log_fit)[0]),
                "spearman_r": float(stats.spearmanr(true_values, fitted)[0]),
                "pearson_r_raw": float(stats.pearsonr(true_values, fitted)[0]),
                "slope_log": float(regression.slope),
                "intercept_log": float(regression.intercept),
                "median_log_ratio": float(np.median(log_fit - log_true)),
                "rmse_log_ratio": float(np.sqrt(np.mean((log_fit - log_true) ** 2))),
                "true_range": [float(true_values.min()), float(true_values.max())],
            })
        summary[family] = record
    return summary
