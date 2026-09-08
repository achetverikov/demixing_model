"""Search strategies for the single-condition WNM likelihood benchmark.

All strategies consume the same log-parameter likelihood evaluator and return
natural-unit parameters.  This keeps optimizer comparisons about the search,
not about subtly different likelihood implementations.
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from model_fit_to_data.continuous_optimizer import dispersed_starts
from model_fit_to_data.grid_based_multi_condition_optimizer_jax_loops import centered_grid
from model_fit_to_data.run_fingerprint import effective_feat_step_schedule
from model_fit_to_data.wnm_scoring import trial_log_density


@dataclass(frozen=True)
class Candidate:
    """One optimizer start or stage winner, rescored by the common evaluator."""

    start_index: int
    start: np.ndarray
    start_loss: float
    parameters: np.ndarray
    reported_loss: float
    loss: float
    success: bool
    status: str
    n_iterations: int
    n_evaluations: int
    elapsed_seconds: float
    at_bound: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    """A search result with enough detail to compare accuracy and cost."""

    method: str
    parameters: np.ndarray
    loss: float
    candidates: tuple[Candidate, ...]
    elapsed_seconds: float
    n_evaluations: int
    settings: dict


def polish_search(evaluator, trials, bounds, source: SearchResult, **lbfgsb_kwargs):
    """Polish a search winner with one SciPy L-BFGS-B start and retain both traces."""
    polished = scipy_lbfgsb(
        evaluator, trials, bounds, starts=np.asarray([source.parameters]),
        **lbfgsb_kwargs)
    winner = polished if polished.loss < source.loss else source
    return SearchResult(
        method=f"{source.method}+scipy-lbfgsb",
        parameters=winner.parameters, loss=winner.loss,
        candidates=source.candidates + polished.candidates,
        elapsed_seconds=source.elapsed_seconds + polished.elapsed_seconds,
        n_evaluations=source.n_evaluations + polished.n_evaluations,
        settings={"source": source.settings, "polish": polished.settings})


class LikelihoodEvaluator:
    """Reusable JIT-compiled WNM point-likelihood and gradient.

    Trials are explicit dynamic arguments.  Reusing one instance therefore
    reuses the compiled executable for every dataset of the same trial count;
    the previous continuous fitter captured each dataset in a new closure.
    """

    def __init__(self, predictor):
        self.predictor = predictor

        def loss(log_parameters, feature_difference, bias):
            parameters = jnp.exp(log_parameters)
            return -jnp.sum(trial_log_density(
                predictor, parameters[0], parameters[1], parameters[2],
                feature_difference, bias))

        self.loss_fn = loss
        self.value = jax.jit(loss)
        self.value_and_grad = jax.jit(jax.value_and_grad(loss))
        self.batch_value = jax.jit(jax.vmap(loss, in_axes=(0, None, None)))

    @staticmethod
    def trials(values):
        values = np.asarray(values, dtype=np.float32)
        return jnp.asarray(values[:, 0]), jnp.asarray(values[:, 1])

    def score(self, parameters, trials) -> float:
        feature_difference, bias = self.trials(trials)
        return float(self.value(jnp.log(jnp.asarray(parameters, jnp.float32)),
                                feature_difference, bias))

    def warm_scalar(self, trials, parameters) -> float:
        """Compile and execute the scalar value/gradient path once."""
        feature_difference, bias = self.trials(trials)
        started = time.perf_counter()
        result = self.value_and_grad(
            jnp.log(jnp.asarray(parameters, jnp.float32)), feature_difference, bias)
        jax.block_until_ready(result)
        return time.perf_counter() - started

    def warm_batch(self, trials, parameters, batch_size) -> float:
        """Compile and execute the fixed-size batched value path once."""
        feature_difference, bias = self.trials(trials)
        row = jnp.log(jnp.asarray(parameters, jnp.float32))
        rows = jnp.broadcast_to(row, (batch_size, len(row)))
        started = time.perf_counter()
        jax.block_until_ready(self.batch_value(rows, feature_difference, bias))
        return time.perf_counter() - started


def _log_bounds(bounds):
    bounds = np.asarray(bounds, dtype=np.float64)
    return np.log(bounds[:, 0]), np.log(bounds[:, 1])


def _bound_hits(parameters, bounds, rtol=1e-6):
    hits = []
    for index, (value, (low, high)) in enumerate(zip(parameters, bounds)):
        tolerance = rtol * (high - low)
        if value - low <= tolerance:
            hits.append(f"p{index}@low")
        elif high - value <= tolerance:
            hits.append(f"p{index}@high")
    return tuple(hits)


def scipy_lbfgsb(evaluator: LikelihoodEvaluator, trials, bounds, *, n_starts=8,
                  seed=0, workers=1, starts=None, max_iterations=500,
                  tolerance=1e-9, gradient_tolerance=1e-6) -> SearchResult:
    """SciPy L-BFGS-B with a JAX value/gradient, optionally concurrent by start."""
    if starts is None:
        starts = dispersed_starts(bounds, n_starts, seed)
    else:
        starts = np.asarray(starts, dtype=np.float64)
        n_starts = len(starts)
    feature_difference, bias = evaluator.trials(trials)
    lower, upper = _log_bounds(bounds)

    def run(item):
        index, start = item
        start_loss = evaluator.score(start, trials)

        def objective(log_parameters):
            value, gradient = evaluator.value_and_grad(
                jnp.asarray(log_parameters, jnp.float32), feature_difference, bias)
            value = float(value)
            gradient = np.asarray(gradient, dtype=np.float64)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                return np.inf, np.zeros_like(gradient)
            return value, gradient

        started = time.perf_counter()
        fit = minimize(
            objective, np.log(start), jac=True, method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": max_iterations, "ftol": tolerance,
                     "gtol": gradient_tolerance})
        parameters = np.exp(fit.x)
        loss = evaluator.score(parameters, trials)
        return Candidate(
            start_index=index, start=np.asarray(start), start_loss=start_loss,
            parameters=parameters, reported_loss=float(fit.fun), loss=loss,
            success=bool(fit.success), status=str(fit.message),
            n_iterations=int(fit.nit), n_evaluations=int(fit.nfev),
            elapsed_seconds=time.perf_counter() - started,
            at_bound=_bound_hits(parameters, bounds))

    started = time.perf_counter()
    indexed = list(enumerate(starts))
    if workers == 1:
        candidates = [run(item) for item in indexed]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            candidates = list(pool.map(run, indexed))
    elapsed = time.perf_counter() - started
    successful = [candidate for candidate in candidates if candidate.success]
    if not successful:
        statuses = sorted({candidate.status for candidate in candidates})
        raise RuntimeError(
            f"none of {n_starts} L-BFGS-B starts converged "
            f"(statuses: {statuses})")
    best = min(successful, key=lambda candidate: candidate.loss)
    return SearchResult(
        method="scipy-lbfgsb" if workers == 1 else "scipy-lbfgsb-threaded",
        parameters=best.parameters, loss=best.loss, candidates=tuple(candidates),
        elapsed_seconds=elapsed,
        n_evaluations=sum(candidate.n_evaluations for candidate in candidates),
        settings={"n_starts": n_starts, "seed": seed, "workers": workers,
                  "max_iterations": max_iterations, "tolerance": tolerance,
                  "gradient_tolerance": gradient_tolerance})


def hierarchical_search(evaluator: LikelihoodEvaluator, trials, bounds, *,
                        points_per_axis=9, n_stages=5, batch_size=512,
                        coordinates="log") -> SearchResult:
    """Batched global lattice followed by local three-dimensional zooms."""
    if points_per_axis < 3 or points_per_axis % 2 == 0:
        raise ValueError("points_per_axis must be an odd integer of at least 3")
    bounds = np.asarray(bounds, dtype=np.float64)
    if coordinates == "log":
        hard_low, hard_high = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    elif coordinates == "linear":
        hard_low, hard_high = bounds[:, 0], bounds[:, 1]
    else:
        raise ValueError("coordinates must be 'log' or 'linear'")
    low, high = hard_low.copy(), hard_high.copy()
    feature_difference, bias = evaluator.trials(trials)
    candidates = []
    total_evaluations = 0
    started = time.perf_counter()

    for stage in range(n_stages):
        stage_started = time.perf_counter()
        axes = [np.linspace(lo, hi, points_per_axis) for lo, hi in zip(low, high)]
        coordinate_grid = np.asarray(list(product(*axes)), dtype=np.float64)
        natural_grid = (np.exp(coordinate_grid) if coordinates == "log"
                        else coordinate_grid)
        log_grid = np.log(natural_grid).astype(np.float32)
        losses = []
        for offset in range(0, len(log_grid), batch_size):
            batch = log_grid[offset:offset + batch_size]
            real_size = len(batch)
            if real_size < batch_size:
                batch = np.concatenate([
                    batch, np.repeat(batch[-1:], batch_size - real_size, axis=0)])
            scored = evaluator.batch_value(
                jnp.asarray(batch), feature_difference, bias)
            losses.append(np.asarray(scored)[:real_size])
        losses = np.concatenate(losses)
        best_index = int(np.argmin(losses))
        best_coordinate = coordinate_grid[best_index]
        best_parameters = natural_grid[best_index]
        best_loss = evaluator.score(best_parameters, trials)
        total_evaluations += len(log_grid) + 1
        candidates.append(Candidate(
            start_index=stage, start=natural_grid[len(natural_grid) // 2],
            start_loss=float(losses[len(natural_grid) // 2]),
            parameters=best_parameters, reported_loss=float(losses[best_index]),
            loss=best_loss, success=True,
            status=f"stage-{stage + 1}", n_iterations=1,
            n_evaluations=len(log_grid) + 1,
            elapsed_seconds=time.perf_counter() - stage_started,
            at_bound=_bound_hits(best_parameters, bounds)))

        step = (high - low) / (points_per_axis - 1)
        low = np.maximum(hard_low, best_coordinate - step)
        high = np.minimum(hard_high, best_coordinate + step)

    best = min(candidates, key=lambda candidate: candidate.loss)
    return SearchResult(
        method=f"hierarchical-{coordinates}", parameters=best.parameters,
        loss=best.loss, candidates=tuple(candidates),
        elapsed_seconds=time.perf_counter() - started,
        n_evaluations=total_evaluations,
        settings={"points_per_axis": points_per_axis, "n_stages": n_stages,
                  "batch_size": batch_size, "coordinates": coordinates})


def _batched_grid_losses(evaluator, log_grid, feature_difference, bias, batch_size):
    losses = []
    for offset in range(0, len(log_grid), batch_size):
        batch = log_grid[offset:offset + batch_size]
        real_size = len(batch)
        if real_size < batch_size:
            batch = np.concatenate([
                batch, np.repeat(batch[-1:], batch_size - real_size, axis=0)])
        scored = evaluator.batch_value(jnp.asarray(batch), feature_difference, bias)
        losses.append(np.asarray(scored)[:real_size])
    return np.concatenate(losses)


def surface_production_hierarchical_search(
        evaluator: LikelihoodEvaluator, trials, bounds, *, shared_grid_size=40,
        feat_grid_size=20, min_grid_step=1.0, zoom_factor=0.5,
        batch_size=512) -> SearchResult:
    """The production surface hierarchy, evaluated with the WNM likelihood.

    The surface fitter searches the shared spatial parameter on an outer linear
    grid.  At every spatial point it independently refines the two feature SDs
    through the production feature-step schedule, then halves the spatial range
    around the winning point until its spacing reaches one degree.  Only the
    absent motor-noise axis is omitted here.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    feat_low, feat_high = bounds[0]
    if not np.array_equal(bounds[0], bounds[1]):
        raise ValueError("surface hierarchy requires equal feature-SD bounds")
    spat_hard_low, spat_hard_high = bounds[2]
    feat_steps = effective_feat_step_schedule(
        feat_grid_size, feat_low, feat_high)
    feature_difference, bias = evaluator.trials(trials)
    center = (feat_low + feat_high) / 2.0
    spat_low, spat_high = spat_hard_low, spat_hard_high
    candidates = []
    total_evaluations = 0
    started = time.perf_counter()

    while True:
        stage_started = time.perf_counter()
        spatial = np.linspace(spat_low, spat_high, shared_grid_size)
        best_features = np.full((shared_grid_size, 2), center, dtype=np.float64)
        best_losses = np.full(shared_grid_size, np.inf)

        for step in feat_steps:
            make_axes = jax.vmap(
                lambda value: centered_grid(
                    value, feat_grid_size, step, feat_low, feat_high))
            feat1 = np.asarray(make_axes(jnp.asarray(best_features[:, 0])))
            feat2 = np.asarray(make_axes(jnp.asarray(best_features[:, 1])))
            grid1 = np.broadcast_to(
                feat1[:, :, None], (shared_grid_size, feat_grid_size, feat_grid_size))
            grid2 = np.broadcast_to(
                feat2[:, None, :], (shared_grid_size, feat_grid_size, feat_grid_size))
            grid3 = np.broadcast_to(spatial[:, None, None], grid1.shape)
            parameter_grid = np.stack([grid1, grid2, grid3], axis=-1).reshape(-1, 3)
            losses = _batched_grid_losses(
                evaluator, np.log(parameter_grid).astype(np.float32),
                feature_difference, bias, batch_size)
            losses = losses.reshape(shared_grid_size, feat_grid_size ** 2)
            indices = np.argmin(losses, axis=1)
            stage_features = parameter_grid.reshape(
                shared_grid_size, feat_grid_size ** 2, 3)[
                    np.arange(shared_grid_size), indices, :2]
            stage_losses = losses[np.arange(shared_grid_size), indices]
            improved = stage_losses < best_losses
            best_features[improved] = stage_features[improved]
            best_losses[improved] = stage_losses[improved]
            total_evaluations += parameter_grid.shape[0]

        best_index = int(np.argmin(best_losses))
        parameters = np.asarray([
            *best_features[best_index], spatial[best_index]])
        loss = evaluator.score(parameters, trials)
        start = np.asarray([center, center, (spat_low + spat_high) / 2.0])
        start_loss = evaluator.score(start, trials)
        total_evaluations += 2
        candidates.append(Candidate(
            start_index=len(candidates),
            start=start, start_loss=start_loss,
            parameters=parameters, reported_loss=float(best_losses[best_index]),
            loss=loss, success=True, status=f"stage-{len(candidates) + 1}",
            n_iterations=len(feat_steps),
            n_evaluations=shared_grid_size * feat_grid_size ** 2 * len(feat_steps) + 2,
            elapsed_seconds=time.perf_counter() - stage_started,
            at_bound=_bound_hits(parameters, bounds)))

        spatial_step = (spat_high - spat_low) / (shared_grid_size - 1)
        if spatial_step <= min_grid_step:
            break
        half_range = (spat_high - spat_low) * zoom_factor / 2.0
        spat_low = max(spat_hard_low, parameters[2] - half_range)
        spat_high = min(spat_hard_high, parameters[2] + half_range)

    best = min(candidates, key=lambda candidate: candidate.loss)
    return SearchResult(
        method="surface-production-hierarchical",
        parameters=best.parameters, loss=best.loss,
        candidates=tuple(candidates),
        elapsed_seconds=time.perf_counter() - started,
        n_evaluations=total_evaluations,
        settings={"shared_grid_size": shared_grid_size,
                  "feat_grid_size": feat_grid_size,
                  "feature_step_schedule": feat_steps,
                  "min_grid_step": min_grid_step,
                  "zoom_factor": zoom_factor,
                  "batch_size": batch_size,
                  "coordinates": "linear",
                  "motor_noise": "fixed-zero"})


def _wnm_unpack(log_parameters, _name=None):
    values = np.exp(np.asarray(log_parameters, dtype=np.float64))
    return dict(zip(("sd_feat1", "sd_feat2", "sd_spat"), values))


def _bbz_trace_result(evaluator, trials, bounds, trace, method, elapsed, settings):
    candidates = []
    for record in trace:
        start = np.asarray([record[f"init_{name}"] for name in
                            ("sd_feat1", "sd_feat2", "sd_spat")])
        has_result = all(f"res_{name}" in record for name in
                         ("sd_feat1", "sd_feat2", "sd_spat"))
        parameters = (np.asarray([record[f"res_{name}"] for name in
                                  ("sd_feat1", "sd_feat2", "sd_spat")])
                      if has_result else start)
        loss = evaluator.score(parameters, trials)
        candidates.append(Candidate(
            start_index=int(record["start_idx"]), start=start,
            start_loss=float(record["start_obj"]), parameters=parameters,
            reported_loss=float(record["result_obj"]) if has_result else np.inf,
            loss=loss, success=bool(has_result and np.isfinite(loss)),
            status=str(record["status"]), n_iterations=int(record.get("n_iter", 0)),
            n_evaluations=int(record.get("nfev", 0)),
            elapsed_seconds=float(record["elapsed_sec"]),
            at_bound=_bound_hits(parameters, bounds)))
    successful = [candidate for candidate in candidates if candidate.success]
    if not successful:
        raise RuntimeError(f"{method} returned no finite candidate")
    best = min(successful, key=lambda candidate: candidate.loss)
    return SearchResult(
        method=method, parameters=best.parameters, loss=best.loss,
        candidates=tuple(candidates), elapsed_seconds=elapsed,
        n_evaluations=sum(candidate.n_evaluations for candidate in candidates),
        settings=settings)


def bbz_pybads_search(evaluator, trials, bounds, implementation, *, n_starts=8,
                      seed=0):
    """Call BBZ's production PyBADS multistart wrapper on WNM log-SDs."""
    starts = np.log(dispersed_starts(bounds, n_starts, seed))
    lower, upper = _log_bounds(bounds)
    feature_difference, bias = evaluator.trials(trials)

    def value(log_parameters):
        return float(evaluator.value(
            jnp.asarray(log_parameters, jnp.float32), feature_difference, bias))

    trace = []
    started = time.perf_counter()
    implementation._pybads_multistart(
        value, starts, list(zip(lower, upper)), n_starts,
        trace=trace, name="wnm-likelihood", _unpack=_wnm_unpack)
    elapsed = time.perf_counter() - started
    return _bbz_trace_result(
        evaluator, trials, bounds, trace, "bbz-pybads", elapsed,
        {"n_starts": n_starts, "seed": seed, "coordinates": "log",
         "implementation": "observer_models._pybads_multistart",
         "optimizer_defaults": "pybads"})


def bbz_jax_bads_search(evaluator, trials, bounds, implementation, *, n_starts=8,
                        seed=0):
    """Call BBZ's production JAX-BADS entry point on WNM log-SDs."""
    if seed != 0:
        raise ValueError("BBZ JAX-BADS fixes its optimizer PRNG seed at zero")
    starts = np.log(dispersed_starts(bounds, n_starts, seed))
    lower, upper = _log_bounds(bounds)
    feature_difference, bias = evaluator.trials(trials)
    trace = []
    started = time.perf_counter()
    implementation.bads_jax_multistart(
        evaluator.loss_fn, (feature_difference, bias), starts,
        list(zip(lower, upper)), n_starts, trace=trace,
        name="wnm-likelihood", _unpack=_wnm_unpack)
    elapsed = time.perf_counter() - started
    defaults = {key: value for key, value in implementation._DEFAULTS.items()
                if np.asarray(value).ndim == 0}
    settings = {
        "n_starts": n_starts, "seed": seed, "coordinates": "log",
        "implementation": "bads_jax.bads_jax_multistart",
        "budget": 300 * (len(bounds) + 2),
        **{key: float(value) for key, value in defaults.items()},
    }
    return _bbz_trace_result(
        evaluator, trials, bounds, trace, "bbz-jax-bads", elapsed, settings)


class JaxoptLbfgsb:
    """Fully device-resident JAXopt L-BFGS-B, vmapped across starts."""

    def __init__(self, evaluator: LikelihoodEvaluator, bounds, *, n_starts=8,
                 max_iterations=500, tolerance=1e-4, max_linesearch=50):
        import jaxopt

        self.evaluator = evaluator
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.n_starts = n_starts
        self.solver = jaxopt.LBFGSB(
            fun=evaluator.loss_fn, maxiter=max_iterations, tol=tolerance,
            maxls=max_linesearch, stop_if_linesearch_fails=True, jit=True)
        lower, upper = _log_bounds(bounds)
        self.jax_bounds = (jnp.asarray(lower, jnp.float32),
                           jnp.asarray(upper, jnp.float32))

        def run_one(start, feature_difference, bias):
            return self.solver.run(
                start, self.jax_bounds, feature_difference, bias)

        self.run_batch = jax.jit(jax.vmap(run_one, in_axes=(0, None, None)))

    def compile(self, trials, *, seed=0, starts=None) -> float:
        """Compile the fixed-shape batched solve without timing an optimization."""
        if starts is None:
            starts = dispersed_starts(self.bounds, self.n_starts, seed)
        feature_difference, bias = self.evaluator.trials(trials)
        started = time.perf_counter()
        self.run_batch.lower(
            jnp.log(jnp.asarray(starts, jnp.float32)),
            feature_difference, bias).compile()
        return time.perf_counter() - started

    def run(self, trials, *, seed=0, starts=None) -> SearchResult:
        if starts is None:
            starts = dispersed_starts(self.bounds, self.n_starts, seed)
        else:
            starts = np.asarray(starts, dtype=np.float64)
        feature_difference, bias = self.evaluator.trials(trials)
        started = time.perf_counter()
        output = self.run_batch(
            jnp.log(jnp.asarray(starts, jnp.float32)), feature_difference, bias)
        jax.block_until_ready(output)
        elapsed = time.perf_counter() - started
        parameters = np.exp(np.asarray(output.params))
        losses = np.asarray(self.evaluator.batch_value(
            jnp.log(jnp.asarray(parameters, jnp.float32)),
            feature_difference, bias))
        failed = np.asarray(output.state.failed_linesearch)
        errors = np.asarray(output.state.error)
        iterations = np.asarray(output.state.iter_num)
        evaluations = np.asarray(output.state.num_fun_eval)
        start_losses = np.asarray(self.evaluator.batch_value(
            jnp.log(jnp.asarray(starts, jnp.float32)), feature_difference, bias))
        candidates = tuple(Candidate(
            start_index=index, start=np.asarray(starts[index]),
            start_loss=float(start_losses[index]),
            parameters=np.asarray(parameters[index]),
            reported_loss=float(np.asarray(output.state.value)[index]),
            loss=float(losses[index]),
            success=bool(not failed[index]
                         and errors[index] <= self.solver.tol
                         and np.isfinite(losses[index])),
            status=(f"error={errors[index]:.6g}; "
                    f"failed_linesearch={bool(failed[index])}"),
            n_iterations=int(iterations[index]),
            n_evaluations=int(evaluations[index]),
            elapsed_seconds=elapsed / len(starts),
            at_bound=_bound_hits(parameters[index], self.bounds))
            for index in range(len(starts)))
        finite = [candidate for candidate in candidates if np.isfinite(candidate.loss)]
        if not finite:
            raise RuntimeError("every JAXopt start returned a non-finite loss")
        best = min(finite, key=lambda candidate: candidate.loss)
        return SearchResult(
            method="jaxopt-lbfgsb", parameters=best.parameters, loss=best.loss,
            candidates=candidates, elapsed_seconds=elapsed,
            n_evaluations=int(evaluations.sum()),
            settings={"n_starts": len(starts), "seed": seed,
                      "max_iterations": self.solver.maxiter,
                      "tolerance": self.solver.tol,
                      "max_linesearch": self.solver.maxls})


class JaxBads:
    """BBZ's device-resident BADS with trial arrays kept as dynamic JIT inputs.

    ``implementation`` is the imported BBZ ``bads_jax`` module.  It is injected
    because BBZ is a sibling repository rather than a DM dependency.  Unlike
    BBZ's public wrapper, this adapter closes only the predictor and BADS
    settings; datasets of the same shape therefore reuse the compiled solve.
    """

    def __init__(self, evaluator: LikelihoodEvaluator, bounds, implementation, *,
                 n_starts=8, budget=None, **overrides):
        self.evaluator = evaluator
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.implementation = implementation
        self.n_starts = n_starts
        lower, upper = _log_bounds(bounds)
        self.plausible_lower = lower + 0.2 * (upper - lower)
        self.plausible_upper = lower + 0.8 * (upper - lower)
        self.span = self.plausible_upper - self.plausible_lower
        if budget is None:
            budget = 300 * (len(bounds) + 2)
        self.config = implementation._make_config(len(bounds), budget, **overrides)

        plausible_lower = jnp.asarray(self.plausible_lower, jnp.float32)
        span = jnp.asarray(self.span, jnp.float32)

        def run_batch(z0, key, feature_difference, bias):
            def objective(z):
                return evaluator.loss_fn(
                    plausible_lower + z * span, feature_difference, bias)

            return implementation._run_bads(
                objective, z0, key, len(bounds), self.config)

        self.run_batch = jax.jit(run_batch)

    def _normalized_starts(self, starts):
        log_starts = np.clip(np.log(starts), self.plausible_lower,
                             self.plausible_upper)
        return log_starts, (log_starts - self.plausible_lower) / self.span

    def compile(self, trials, *, seed=0, starts=None) -> float:
        """Compile the fixed-shape batched solve without timing an optimization."""
        if starts is None:
            starts = dispersed_starts(self.bounds, self.n_starts, seed)
        _, z0 = self._normalized_starts(starts)
        feature_difference, bias = self.evaluator.trials(trials)
        started = time.perf_counter()
        self.run_batch.lower(
            jnp.asarray(z0, jnp.float32), jax.random.PRNGKey(seed),
            feature_difference, bias).compile()
        return time.perf_counter() - started

    def run(self, trials, *, seed=0, starts=None) -> SearchResult:
        if starts is None:
            starts = dispersed_starts(self.bounds, self.n_starts, seed)
        else:
            starts = np.asarray(starts, dtype=np.float64)
        log_starts, z0 = self._normalized_starts(starts)
        feature_difference, bias = self.evaluator.trials(trials)
        key = jax.random.PRNGKey(seed)
        started = time.perf_counter()
        z_best, reported, evaluations, iterations = jax.block_until_ready(
            self.run_batch(jnp.asarray(z0, jnp.float32), key,
                           feature_difference, bias))
        elapsed = time.perf_counter() - started

        parameters = np.exp(
            self.plausible_lower + np.asarray(z_best) * self.span)
        losses = np.asarray(self.evaluator.batch_value(
            jnp.log(jnp.asarray(parameters, jnp.float32)),
            feature_difference, bias))
        start_losses = np.asarray(self.evaluator.batch_value(
            jnp.asarray(log_starts, jnp.float32), feature_difference, bias))
        reported = np.asarray(reported)
        evaluations = np.asarray(evaluations)
        iterations = int(iterations)
        candidates = tuple(Candidate(
            start_index=index, start=np.exp(log_starts[index]),
            start_loss=float(start_losses[index]),
            parameters=np.asarray(parameters[index]),
            reported_loss=float(reported[index]), loss=float(losses[index]),
            success=bool(np.isfinite(losses[index])
                         and losses[index] <= start_losses[index] + 1e-6),
            status=f"completed {iterations} batched iterations",
            n_iterations=iterations, n_evaluations=int(evaluations[index]),
            elapsed_seconds=elapsed / len(starts),
            at_bound=_bound_hits(parameters[index], self.bounds))
            for index in range(len(starts)))
        finite = [candidate for candidate in candidates
                  if np.isfinite(candidate.loss)]
        if not finite:
            raise RuntimeError("every JAX-BADS start returned a non-finite loss")
        best = min(finite, key=lambda candidate: candidate.loss)
        serializable = {
            key: value for key, value in self.config.items()
            if np.asarray(value).ndim == 0
        }
        return SearchResult(
            method="jax-bads", parameters=best.parameters, loss=best.loss,
            candidates=candidates, elapsed_seconds=elapsed,
            n_evaluations=int(evaluations.sum()),
            settings={"n_starts": len(starts), "seed": seed,
                      **{key: float(value) for key, value in serializable.items()}})
