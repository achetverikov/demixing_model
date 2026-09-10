"""Bounded gradient search over continuous noise parameters.

The hierarchical and exhaustive backends both search a lattice: they evaluate a
grid, keep the best point, and refine around it. That is the right shape for a
surrogate whose output is a stored surface, and it is the only shape available
when the objective cannot be differentiated. The wrapped-normal mixture changes
that -- its density, moments and signed-arc asymmetry are closed forms, and the
production CCC loss is a smooth function of them -- so the whole path from
parameters to loss has a gradient.

The recovery panel selected this as the WNM production search: 64 deterministic
starts evaluated with the pinned batched JAX L-BFGS-B port as two sequential
batches of 32. The surface backend retains its hierarchical lattice search.

Design notes that are not free choices:

* **Log parameterisation.** Noise SDs are multiplicative quantities spanning
  2.5 to 200 degrees. Optimising them linearly makes a step that is reasonable
  near 200 meaningless near 2.5.
* **Real bounds, not a squashing function.** A sigmoid reparameterisation would
  make every solution interior and hide the boundary hits that say a fit railed.
  Railing is information about the model or the data and must stay visible.
* **Every start is recorded.** The spread of final losses across starts is what
  distinguishes "this objective has one basin" from "this search got lucky", and
  it is the quantity the search comparison needs. It is not small here: a
  three-condition fixture gave a spread of 1.03 across six converged starts. The
  start budget is therefore a setting with scientific consequences. The frozen
  production value is 64 and is part of the run fingerprint.

**Precision.** The repo runs JAX in its default float32 everywhere, and the
surface backend's deployed numbers were produced that way, so this module does
*not* enable x64 -- doing so globally would change those numbers, and doing so
locally is not something JAX supports. Values and gradients are therefore
computed in float32. That sets a floor on the
achievable tolerance: float32 carries about seven decimal digits, so a
convergence test tighter than roughly 1e-8 relative is testing arithmetic noise
and will report a converged run whose last few digits are meaningless. The
defaults below are chosen against that floor rather than copied from a textbook,
and the benchmark records ``loss_spread`` so a search that is merely stalling on
noise is distinguishable from one that agrees across starts.

The recovery panel selected float32 with ``highest`` matmul precision. Do not
enable x64 here: it is a global JAX flag, so it would also change the surface
backend's arithmetic and invalidate its parity fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax_lbfgsb import BatchedLbfgsb, CONVERGED_FTOL, CONVERGED_PGTOL


OPTIMIZER_VERSION = "jax-lbfgsb@0350da1"
DEFAULT_N_STARTS = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_DTYPE = "float32"
DEFAULT_MATMUL_PRECISION = "highest"


@dataclass(frozen=True)
class StartOutcome:
    """What one multistart run did, kept whether or not it won."""

    start: np.ndarray
    solution: np.ndarray
    loss: float
    success: bool
    status: str
    n_iterations: int
    n_evaluations: int
    at_bound: tuple


@dataclass
class ContinuousFit:
    """The result of a bounded multistart gradient search.

    Attributes:
        parameters: best solution, in natural units (degrees).
        loss: its objective value.
        starts: every start's outcome, in the order they were run.
        n_starts: how many were run.
        loss_spread: max minus min final loss over the starts that converged. A
            wide spread means the search, not the objective, decided the answer.
        starts: includes the starts that failed; only converged ones can win.
        at_bound: which coordinates of the winner sit on a bound.
        names: parameter names, positionally matching ``parameters``.
    """

    parameters: np.ndarray
    loss: float
    starts: list
    n_starts: int
    loss_spread: float
    at_bound: tuple
    names: tuple
    settings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Flat record for a results table or a run fingerprint."""
        record = {f"fit_{name}": float(value)
                  for name, value in zip(self.names, self.parameters)}
        record.update(
            loss=float(self.loss),
            n_starts=int(self.n_starts),
            loss_spread=float(self.loss_spread),
            n_converged=int(sum(s.success for s in self.starts)),
            at_bound=",".join(self.at_bound) if self.at_bound else "",
            total_evaluations=int(sum(s.n_evaluations for s in self.starts)),
        )
        return record


def dispersed_starts(bounds: Sequence[tuple], n_starts: int, seed: int) -> np.ndarray:
    """Deterministic, dispersed starting points, spread in log space.

    A scrambled stratified sample rather than a uniform draw: with only a handful
    of starts, an unstratified sample leaves whole regions of the box unvisited
    often enough to matter, and "the gradient search missed the basin" then
    reports the sampler rather than the search.

    The seed is part of the run's identity; the same seed must give the same
    starts, or two runs of the "same" fit are not comparable.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}")
    rng = np.random.default_rng(seed)
    log_bounds = np.log(np.asarray(bounds, dtype=np.float64))
    n_dims = len(bounds)

    # Latin hypercube: one sample per stratum per dimension, permuted independently.
    strata = (rng.permutation(n_starts) for _ in range(n_dims))
    unit = np.stack([(order + rng.random(n_starts)) / n_starts for order in strata], axis=-1)
    return np.exp(log_bounds[:, 0] + unit * (log_bounds[:, 1] - log_bounds[:, 0]))


def _bound_report(solution: np.ndarray, bounds: np.ndarray, names: Sequence[str],
                  rtol: float = 1e-6) -> tuple:
    """Which coordinates are sitting on a bound, and which end."""
    hits = []
    for value, (low, high), name in zip(solution, bounds, names):
        span = high - low
        if value - low <= rtol * span:
            hits.append(f"{name}@low")
        elif high - value <= rtol * span:
            hits.append(f"{name}@high")
    return tuple(hits)


def minimize_continuous(objective: Callable, bounds: Sequence[tuple],
                        names: Sequence[str], n_starts: int = DEFAULT_N_STARTS,
                        seed: int = 0, batch_size: int = DEFAULT_BATCH_SIZE,
                        dtype: str = DEFAULT_DTYPE,
                        matmul_precision: str = DEFAULT_MATMUL_PRECISION,
                        max_iterations: int = 500, tolerance: float = 1e-9,
                        gradient_tolerance: float = 1e-6,
                        jit: bool = True) -> ContinuousFit:
    """Minimise ``objective`` over ``bounds`` with bounded L-BFGS-B in log space.

    Args:
        objective: JAX-traceable, taking a ``(n_params,)`` array in natural units
            and returning a scalar loss. It must be differentiable; if it indexes
            a grid or takes an argmin, this is the wrong search for it.
        bounds: ``(low, high)`` per parameter, in natural units. All must be
            strictly positive, since the search works in log space.
        names: parameter names, for boundary reporting.
        n_starts: deterministic multistart count; production uses 64.
        seed: makes the starts reproducible; recorded in the result.
        batch_size: starts per sequential accelerator batch; production uses 32.
        dtype: optimizer and objective array dtype; production uses float32.
        matmul_precision: JAX matmul precision policy; production uses highest.
        max_iterations: per start.
        tolerance: L-BFGS-B ``ftol``, the relative reduction in the objective
            below which a start stops. Floored by float32 arithmetic; see the
            module docstring.
        gradient_tolerance: L-BFGS-B ``gtol``, on the projected gradient.
        jit: retained for API compatibility. ``BatchedLbfgsb`` always uses JIT.

    Returns:
        :class:`ContinuousFit`, carrying every start's outcome, not just the best.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be (n_params, 2), got {bounds.shape}")
    if len(names) != len(bounds):
        raise ValueError(f"{len(names)} names for {len(bounds)} bounds")
    if np.any(bounds[:, 0] <= 0):
        raise ValueError(
            "bounds must be strictly positive: the search optimises log SDs, because a "
            "step that is reasonable near 200 degrees is meaningless near 2.5.")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        bad = [names[i] for i in np.flatnonzero(bounds[:, 0] >= bounds[:, 1])]
        raise ValueError(f"empty bounds for {bad}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if dtype != DEFAULT_DTYPE:
        raise ValueError(f"production continuous search requires dtype='float32', got {dtype!r}")
    if matmul_precision != DEFAULT_MATMUL_PRECISION:
        raise ValueError(
            "production continuous search requires matmul_precision='highest'; default GPU "
            "TF32 changed the likelihood basin in validation")
    if not jit:
        raise ValueError("BatchedLbfgsb is a JIT optimizer; jit=False is not supported")

    def in_log(log_params):
        return objective(jnp.exp(log_params))

    array_dtype = jnp.float32
    solver = BatchedLbfgsb(
        in_log,
        jnp.asarray(np.log(bounds[:, 0]), dtype=array_dtype),
        jnp.asarray(np.log(bounds[:, 1]), dtype=array_dtype),
        maxiter=max_iterations, ftol=tolerance, gtol=gradient_tolerance)
    starts = dispersed_starts(bounds, n_starts, seed)
    outcomes = []
    with jax.default_matmul_precision(matmul_precision):
        for first in range(0, n_starts, batch_size):
            batch_starts = starts[first:first + batch_size]
            result = solver.run(jnp.asarray(np.log(batch_starts), dtype=array_dtype))
            natural = jnp.clip(jnp.exp(result.x),
                               jnp.asarray(bounds[:, 0], dtype=array_dtype),
                               jnp.asarray(bounds[:, 1], dtype=array_dtype))
            solutions = np.asarray(jax.device_get(natural), dtype=np.float64)
            # Natural-unit clipping is needed because exp(log(high)) can round a
            # float32 endpoint just outside its declared scientific bound. Keep
            # every stored loss tied to the stored, clipped parameters.
            losses = np.asarray(jax.device_get(jax.vmap(objective)(natural)),
                                dtype=np.float64)
            statuses = np.asarray(jax.device_get(result.status), dtype=np.int32)
            iterations = np.asarray(jax.device_get(result.iterations), dtype=np.int32)
            evaluations = np.asarray(jax.device_get(result.evaluations), dtype=np.int32)
            messages = result.messages()
            for index, start in enumerate(batch_starts):
                solution = solutions[index]
                outcomes.append(StartOutcome(
                    start=start, solution=solution, loss=float(losses[index]),
                    success=int(statuses[index]) in (
                        int(CONVERGED_PGTOL), int(CONVERGED_FTOL)),
                    status=messages[index], n_iterations=int(iterations[index]),
                    n_evaluations=int(evaluations[index]),
                    at_bound=_bound_report(solution, bounds, names)))

    finite = [o for o in outcomes if np.isfinite(o.loss)]
    if not finite:
        raise RuntimeError(
            f"all {n_starts} starts returned a non-finite loss; the objective is not "
            "evaluable anywhere in these bounds, which is a problem with the objective "
            "or the data rather than with the search.")

    # Keep convergence as a diagnostic, but use the port's validated canonical
    # selection rule: minimum finite rescored loss, stable by start order. A
    # float32 line search can report abnormal termination at a numerically good
    # endpoint, so excluding it would not reproduce the selected search.
    successful = [o for o in finite if o.success]
    if not successful:
        raise RuntimeError(
            f"none of {n_starts} starts converged (statuses: "
            f"{sorted({o.status for o in outcomes})}). Every remaining loss is wherever "
            "its start halted, not a minimum, so there is no fit to report. Raise "
            "max_iterations, loosen the tolerances, or check the objective.")

    best = min(finite, key=lambda o: o.loss)
    converged = [o.loss for o in successful]
    spread = float(max(converged) - min(converged)) if len(converged) > 1 else 0.0

    return ContinuousFit(
        parameters=best.solution, loss=best.loss, starts=outcomes, n_starts=n_starts,
        loss_spread=spread, at_bound=best.at_bound, names=tuple(names),
        settings={"n_starts": n_starts, "seed": seed, "max_iterations": max_iterations,
                  "tolerance": tolerance, "gradient_tolerance": gradient_tolerance,
                  "batch_size": batch_size, "dtype": dtype,
                  "matmul_precision": matmul_precision,
                  "parameterisation": "log", "method": "BatchedLbfgsb",
                  "optimizer_version": OPTIMIZER_VERSION,
                  # Recorded because the bounds come from the loaded artifact's
                  # domain: two runs with the same seed and settings but
                  # different artifacts search different boxes, and a railed
                  # parameter means nothing without the bound it railed against.
                  "bounds": [[float(low), float(high)] for low, high in bounds],
                  "bound_names": list(names)})


def condition_parameter_layout(n_conditions: int, fit_motor: bool) -> tuple:
    """Names of the searched parameters, in order.

    Two feature SDs per condition, one shared spatial SD, and one shared motor SD
    when it is free. A fixed zero motor SD is the separate no-motor case, not a
    log parameter with a bound at zero -- log space has no zero.
    """
    if n_conditions < 1:
        raise ValueError(f"need at least one condition, got {n_conditions}")
    names = []
    for index in range(n_conditions):
        names.extend([f"sd_feat1_c{index}", f"sd_feat2_c{index}"])
    names.append("sd_spat")
    if fit_motor:
        names.append("sd_motor")
    return tuple(names)


def build_bounds(n_conditions: int, sd_feat_bounds: tuple, sd_spat_bounds: tuple,
                 motor_bounds: Optional[tuple] = None) -> np.ndarray:
    """Bounds matching :func:`condition_parameter_layout`.

    ``sd_feat_bounds`` and ``sd_spat_bounds`` come from the surrogate, via
    ``shared.surrogate.search_bounds``: the two families were trained on
    different domains, and a search must not propose parameters its own
    surrogate never saw.
    """
    bounds = []
    for _ in range(n_conditions):
        bounds.extend([sd_feat_bounds, sd_feat_bounds])
    bounds.append(sd_spat_bounds)
    if motor_bounds is not None:
        bounds.append(motor_bounds)
    return np.asarray(bounds, dtype=np.float64)
