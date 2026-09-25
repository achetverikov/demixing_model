#!/usr/bin/env python3
"""Fit one experimental condition directly through the conditional density.

The trial likelihood is evaluated at each trial's own continuous ``feat_diff``::

    log L(theta) = sum_t log q(b_t | sd_feat1, sd_feat2, sd_ident, feat_diff_t)

No feat_diff grid, no bias binning and no surface interpolation is involved.
Motor noise, when enabled, is added analytically by widening the mixture
components (variances add), not by convolving a density grid.

The three SDs are optimised through a bounded sigmoid map onto the trained
``[5, 200]`` domain (``design.SD_BOUNDS``), so the fit can never query the
density outside the region the model ever saw — outside it the emulator is an
extrapolation and its likelihood is meaningless.

This is a demonstration on top of a validated density model, not a replacement
for the production fitting pipeline.

Usage (from the repo root).  The prepared example datasets carry ``abs_td_dist``
(the target-distractor feature distance) and ``bias_to_distr_corr`` (the bias),
which are this script's defaults::

    PYTHONPATH=. python -m development.wnm_transition.fit_demo \
        --model $DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl \
        --data example_data/fischer_whitney_prepared.csv \
        --condition-col condition --condition combined

Add ``--checkpoint pretrained/model_epoch1500_10ktrain_100samples.pkl`` to fit
the production surface network independently to the same trials.  This is a
likelihood comparison between each model's own optimum, not an evaluation of
one model at the other model's parameters.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from surrogate_training.wnm import data as data_mod
from surface_computation import wnm_design as design_mod
from shared import wnm as wm

#: Fitted in a bounded sigmoid parameterisation of the trained SD domain.
PARAM_NAMES = ('sd_feat1', 'sd_feat2', 'sd_ident')
SD_LO, SD_HI = design_mod.SD_BOUNDS


def sd_from_theta(theta3):
    """Map unconstrained ``theta`` to SDs in ``[SD_LO, SD_HI]`` via a sigmoid."""
    return SD_LO + (SD_HI - SD_LO) * jax.nn.sigmoid(theta3)


def make_objective(model, variables, feat_diff, bias, fit_motor_noise: bool,
                   n_wraps: int = 4):
    feat_diff = jnp.asarray(feat_diff, dtype=jnp.float32)
    bias = jnp.asarray(bias, dtype=jnp.float32)

    def neg_loglik(theta):
        sd = sd_from_theta(theta[:3])
        params = jnp.stack([jnp.broadcast_to(sd[0], feat_diff.shape),
                            jnp.broadcast_to(sd[1], feat_diff.shape),
                            jnp.broadcast_to(sd[2], feat_diff.shape),
                            feat_diff], axis=-1)
        dist = model.apply(variables, params)
        if fit_motor_noise:
            dist = wm.add_motor_noise(dist, jnp.exp(theta[3]))
        return -jnp.sum(wm.mixture_logpdf(bias, dist, n_wraps))

    return jax.jit(neg_loglik)


def fit(model, variables, feat_diff, bias, fit_motor_noise: bool = False,
        n_restarts: int = 8, steps: int = 800, lr: float = 0.05, seed: int = 0,
        n_wraps: int = 4):
    """Multi-start Adam in the bounded parameterisation; returns the best restart.

    Returns ``(nll, sd_values, theta)`` where ``sd_values`` are the SDs (plus
    ``sd_motor`` when fitted) and ``theta`` is the raw optimiser state of the best
    restart (used for timing).
    """
    objective = make_objective(model, variables, feat_diff, bias,
                               fit_motor_noise, n_wraps)
    grad_fn = jax.jit(jax.value_and_grad(objective))
    rng = np.random.default_rng(seed)
    n_theta = 4 if fit_motor_noise else 3

    best = (np.inf, None, None)
    for _ in range(n_restarts):
        theta = jnp.asarray(rng.uniform(-2.0, 2.0, n_theta), dtype=jnp.float32)
        if fit_motor_noise:  # motor noise starts near a few degrees, in log space
            theta = theta.at[3].set(float(np.log(rng.uniform(2.0, 20.0))))
        tx = optax.adam(lr)
        opt_state = tx.init(theta)
        for _ in range(steps):
            loss, grads = grad_fn(theta)
            updates, opt_state = tx.update(grads, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
        loss = float(objective(theta))
        if loss < best[0]:
            sd = np.asarray(sd_from_theta(theta[:3]))
            values = np.concatenate([sd, [float(np.exp(theta[3]))]]) if fit_motor_noise else sd
            best = (loss, values, np.asarray(theta))
    return best


def production_condition_nll(checkpoint: Path, sd_values, feat_diff, bias):
    """Total NLL of a condition under the production surface NN.

    Reads the checkpoint at the fitted ``(sd_feat1, sd_feat2, sd_ident)`` and each
    trial's own feat_diff (linear feat interpolation, re-normalised over mu1,
    circularly-linear in bias), so it is directly comparable with the density
    model's NLL on the same trials.
    """
    from development.wnm_transition import compare_existing_model as cmp
    predictor = cmp.production_predictor(checkpoint)
    return production_predictor_condition_nll(predictor, sd_values, feat_diff, bias)


def production_predictor_condition_nll(predictor, sd_values, feat_diff, bias):
    """Per-trial NLL using one predicted surface for the whole condition."""
    from development.wnm_transition import compare_existing_model as cmp
    sd = np.asarray(sd_values, dtype=np.float64)[:3]
    surface = predictor(np.asarray([[sd[0], sd[1], sd[2], 0.0]]))[0]
    prof = cmp.interpolated_profile_single(surface, np.asarray(feat_diff))
    logp = cmp._score_interp(prof)(np.asarray(bias)[:, None])[:, 0]
    finite = np.isfinite(bias)
    if not np.any(finite):
        return np.inf
    return float(-np.sum(np.where(finite, logp, 0.0)))


def fit_production(predictor, feat_diff, bias, n_restarts: int = 8,
                   seed: int = 0, maxiter: int = 300):
    """Independently fit the production surface NN by bounded numerical search."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    starts = [np.full(3, (SD_LO + SD_HI) / 2.0)]
    starts.extend(rng.uniform(SD_LO, SD_HI, (max(0, n_restarts - 1), 3)))
    objective = lambda sd: production_predictor_condition_nll(
        predictor, sd, feat_diff, bias)
    best = None
    bounds = [(SD_LO, SD_HI)] * 3
    for start in starts:
        result = minimize(objective, start, method='Powell', bounds=bounds,
                          options={'maxiter': maxiter, 'xtol': 1e-3, 'ftol': 1e-5})
        if best is None or result.fun < best.fun:
            best = result
    return float(best.fun), np.asarray(best.x), best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path,
                   help='CSV with feat_diff and bias columns')
    p.add_argument('--feat-diff-col', default='abs_td_dist',
                   help="feature-distance column (example data: 'abs_td_dist')")
    p.add_argument('--bias-col', default='bias_to_distr_corr',
                   help="bias column (example data: 'bias_to_distr_corr')")
    p.add_argument('--condition-col', default=None)
    p.add_argument('--condition', default=None)
    p.add_argument('--exclude-outside-domain', action='store_true',
                   help='explicitly omit trials with feat_diff outside the trained domain')
    p.add_argument('--fit-motor-noise', action='store_true')
    p.add_argument('--n-restarts', type=int, default=8)
    p.add_argument('--checkpoint', type=Path, default=None,
                   help='optional production surface NN to fit independently')
    p.add_argument('--production-maxiter', type=int, default=300)
    args = p.parse_args()

    model, variables, meta = wm.load_model(args.model)
    if args.checkpoint is not None:
        from development.wnm_transition.compare_existing_model import checkpoint_n_samples
        model_n = data_mod.model_n_samples(meta)
        checkpoint_n = checkpoint_n_samples(args.checkpoint)
        if model_n is not None and checkpoint_n is not None and model_n != checkpoint_n:
            raise ValueError(f'density model targets n_samples={model_n}, but checkpoint '
                             f'name indicates n_samples={checkpoint_n}')
    df = pd.read_csv(args.data)
    if args.condition_col and args.condition:
        df = df[df[args.condition_col].astype(str) == args.condition]
    df = df.dropna(subset=[args.feat_diff_col, args.bias_col])
    if df.empty:
        raise ValueError('no finite trials remain after filtering')
    feat_diff = df[args.feat_diff_col].to_numpy(np.float32)
    outside = ((feat_diff < design_mod.FEAT_DIFF_BOUNDS[0])
               | (feat_diff > design_mod.FEAT_DIFF_BOUNDS[1]))
    if np.any(outside) and not args.exclude_outside_domain:
        raise ValueError(f'feat_diff must lie within trained domain '
                         f'{design_mod.FEAT_DIFF_BOUNDS}; {outside.sum()} trials are outside '
                         '(pass --exclude-outside-domain to omit them explicitly)')
    if np.any(outside):
        print(f'excluding {outside.sum()} trials outside trained feat_diff domain '
              f'{design_mod.FEAT_DIFF_BOUNDS}')
        df = df.loc[~outside].copy()
        feat_diff = df[args.feat_diff_col].to_numpy(np.float32)
    bias = df[args.bias_col].to_numpy(np.float32)
    print(f"{len(df)} trials; feat_diff in [{feat_diff.min():.1f}, "
          f"{feat_diff.max():.1f}] (continuous, not binned; trained domain "
          f"{design_mod.FEAT_DIFF_BOUNDS})")

    t0 = time.time()
    nll, values, theta = fit(model, variables, feat_diff, bias,
                             fit_motor_noise=args.fit_motor_noise,
                             n_restarts=args.n_restarts, n_wraps=meta.get('n_wraps', 4))
    names = PARAM_NAMES + (('sd_motor',) if args.fit_motor_noise else ())
    print(f"\nfit in {time.time() - t0:.1f}s   negative log likelihood {nll:.3f}")
    for name, value in zip(names, values):
        print(f"  {name:10s} {value:8.3f}")

    if args.checkpoint is not None:
        from development.wnm_transition.compare_existing_model import production_predictor
        if args.fit_motor_noise:
            print('\nNOTE: production fit has no additional motor-noise parameter; '
                  'its optimum is not parameter-matched to the four-parameter density fit.')
        t_prod = time.time()
        prod_nll, prod_values, _ = fit_production(
            production_predictor(args.checkpoint), feat_diff, bias,
            n_restarts=args.n_restarts, seed=0, maxiter=args.production_maxiter)
        print(f"\nindependent production fit in {time.time() - t_prod:.1f}s   "
              f"negative log likelihood {prod_nll:.3f}")
        for name, value in zip(PARAM_NAMES, prod_values):
            print(f"  {name:10s} {value:8.3f}")
        print(f"NLL difference (density - production): {nll - prod_nll:.3f}; "
              'lower is better')

    # Timing figures requested by the task's performance report.
    objective = make_objective(model, variables, feat_diff, bias,
                               args.fit_motor_noise, meta.get('n_wraps', 4))
    objective(jnp.asarray(theta)).block_until_ready()
    t0 = time.time()
    for _ in range(100):
        objective(jnp.asarray(theta)).block_until_ready()
    print(f"\n{len(df)}-trial likelihood: {(time.time() - t0) * 10:.3f} ms per evaluation")
    print(f"checkpoint size: {args.model.stat().st_size / 1e3:.0f} kB")


if __name__ == '__main__':
    main()
