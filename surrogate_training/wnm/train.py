#!/usr/bin/env python3
"""Train the conditional wrapped-normal mixture on raw EM bias samples.

The objective is the negative log likelihood of individual simulated biases.
No histogram, KDE, bias grid or likelihood surface is constructed anywhere in
the loss.

Usage (from the repo root)::

    PYTHONPATH=. python -m surrogate_training.wnm.train \
        --source sim_samples_10k_100samples_circular_em_fullcov_free_weights \
        --corpus-files 800 --corpus-sims 400 --components 8 \
        --out $DEMIXING_ARTIFACT_ROOT/continuous_density/wnmix_k8.pkl

``--source`` accepts either a corpus directory name under
``$DEMIXING_ARTIFACT_ROOT`` (grid parameters, free of new simulation cost) or an
``.npz`` written by ``generate_training_data.py`` (continuous parameters).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from surrogate_training.wnm import data as data_mod
from shared import wnm as wm


def make_loss(model, n_wraps: int, moment_weight=0., moment_huber_delta=.1,
              cvar_weight=0., cvar_fraction=.1):
    """Raw NLL with optional grouped moment calibration and tail emphasis."""
    @jax.jit
    def loss_fn(variables, x, b, w):
        dist = model.apply(variables, x)
        if b.ndim == 1:
            logp = wm.mixture_logpdf(b, dist, n_wraps)
            return -jnp.sum(w * logp) / jnp.maximum(jnp.sum(w), 1.0)

        logp = wm.mixture_logpdf_samples(b, dist, n_wraps)
        count = jnp.maximum(jnp.sum(w, axis=1), 1.)
        group_nll = -jnp.sum(w * logp, axis=1) / count
        mean_nll = jnp.mean(group_nll)

        samples = jnp.exp(1j * jnp.radians(b))
        empirical = jnp.sum(w * samples, axis=1) / count
        predicted = wm.circular_moment(dist)
        error = jnp.stack([jnp.real(predicted - empirical),
                           jnp.imag(predicted - empirical)], axis=-1)
        absolute = jnp.abs(error)
        huber = jnp.where(absolute <= moment_huber_delta,
                          .5 * error ** 2,
                          moment_huber_delta * (absolute - .5 * moment_huber_delta))
        moment_loss = jnp.mean(jnp.sum(huber, axis=-1))

        n_tail = max(1, int(group_nll.shape[0] * cvar_fraction))
        tail_nll = jnp.mean(jax.lax.top_k(group_nll, n_tail)[0])
        return (mean_nll + moment_weight * moment_loss
                + cvar_weight * (tail_nll - mean_nll))

    return loss_fn


def evaluate_store(loss_fn, variables, store, rng, n_batches: int, batch_size: int):
    """Mean NLL over random batches of a held-out store."""
    total = 0.0
    for _ in range(n_batches):
        x, b, w = store.batch(rng, batch_size)
        total += float(loss_fn(variables, x, b, w))
    return total / n_batches


def mixed_batch(primary, augmentation, augmentation_fraction, rng, batch_size):
    """Sample a fixed augmentation fraction while retaining global replay."""
    if augmentation is None:
        return primary.batch(rng, batch_size)
    n_aug = int(round(batch_size * augmentation_fraction))
    parts = [primary.batch(rng, batch_size - n_aug),
             augmentation.batch(rng, n_aug)]
    order = rng.permutation(batch_size)
    return tuple(np.concatenate([a, b], axis=0)[order]
                 for a, b in zip(*parts))


def mixed_grouped_batch(primary, augmentation, augmentation_fraction, rng,
                        n_groups, outcomes_per_group):
    """Grouped equivalent of :func:`mixed_batch`."""
    if augmentation is None:
        return primary.grouped_batch(rng, n_groups, outcomes_per_group)
    n_aug = int(round(n_groups * augmentation_fraction))
    parts = [primary.grouped_batch(rng, n_groups - n_aug, outcomes_per_group),
             augmentation.grouped_batch(rng, n_aug, outcomes_per_group)]
    order = rng.permutation(n_groups)
    return tuple(np.concatenate([a, b], axis=0)[order]
                 for a, b in zip(*parts))


def evaluate_mixed(loss_fn, variables, primary, augmentation,
                   augmentation_fraction, rng, n_batches, batch_size):
    total = 0.
    for _ in range(n_batches):
        x, b, w = mixed_batch(primary, augmentation, augmentation_fraction,
                              rng, batch_size)
        total += float(loss_fn(variables, x, b, w))
    return total / n_batches


def trajectory_validation_metrics(model, variables, store, labels, n_wraps=4,
                                  min_resultant=.2):
    """Raw-NLL and maximum moment errors on fixed labeled trajectories."""
    from continuous_density import evaluate as evaluate_mod

    labels = np.asarray(labels)
    if labels.shape != (store.n_rows,):
        raise ValueError('trajectory labels must match reference rows')
    group_nll, mean_errors, sd_errors, identified_mean, identified_sd = [], [], [], [], []
    group_names = []
    for component in (0, 1):
        params = (store.design if component == 0 else
                  np.asarray(wm.mirror_params(store.design)))
        bias = store.bias[:, :, component]
        nll, _ = evaluate_mod.model_case_nll(
            model, variables, params, bias, n_wraps=n_wraps,
            case_block=16, chunk=min(2000, bias.shape[1]))
        empirical = evaluate_mod.empirical_moments(bias)
        dist = model.apply(variables, jnp.asarray(params))
        pred_mean, _ = wm.mean_and_resultant(dist)
        pred_sd = wm.circular_sd(dist)
        mean_error = np.abs(np.asarray(wm.wrap_deg(
            np.asarray(pred_mean) - empirical['mean_bias'])))
        sd_error = np.abs(np.asarray(pred_sd) - empirical['circ_sd'])
        for label in np.unique(labels):
            keep = labels == label
            identified = keep & (empirical['resultant'] >= min_resultant)
            group_nll.append(float(np.nanmean(nll[keep])))
            mean_errors.append(float(np.nanmax(mean_error[keep])))
            sd_errors.append(float(np.nanmax(sd_error[keep])))
            identified_mean.append(float(np.nanmax(mean_error[identified]))
                                   if np.any(identified) else np.nan)
            identified_sd.append(float(np.nanmax(sd_error[identified]))
                                 if np.any(identified) else np.nan)
            group_names.append(f'{label}:component{component + 1}')

    worst_nll = int(np.nanargmax(group_nll))
    worst_mean = int(np.nanargmax(mean_errors))
    worst_sd = int(np.nanargmax(sd_errors))
    worst_identified_mean = int(np.nanargmax(identified_mean))
    worst_identified_sd = int(np.nanargmax(identified_sd))
    return {
        'trajectory_nll': float(np.nanmean(group_nll)),
        'worst_trajectory_nll': float(group_nll[worst_nll]),
        'worst_trajectory_nll_group': group_names[worst_nll],
        'max_mean_error': float(mean_errors[worst_mean]),
        'max_mean_error_group': group_names[worst_mean],
        'max_sd_error': float(sd_errors[worst_sd]),
        'max_sd_error_group': group_names[worst_sd],
        'moment_min_resultant': min_resultant,
        'max_mean_error_identified': float(identified_mean[worst_identified_mean]),
        'max_mean_error_identified_group': group_names[worst_identified_mean],
        'max_sd_error_identified': float(identified_sd[worst_identified_sd]),
        'max_sd_error_identified_group': group_names[worst_identified_sd],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True,
                   help='corpus directory name/path, or an .npz training file')
    p.add_argument('--augmentation', default=None,
                   help='optional raw NPZ sampled alongside --source')
    p.add_argument('--augmentation-fraction', type=float, default=.5)
    p.add_argument('--init-model', type=Path, default=None,
                   help='fine-tune an existing compatible density checkpoint')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--components', type=int, default=8, help='mixture components K')
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 128, 128])
    p.add_argument('--min-scale', type=float, default=0.25)
    p.add_argument('--n-wraps', type=int, default=4)
    p.add_argument('--grouped-outcomes', type=int, default=1,
                   help='raw outcomes sharing each sampled parameter/component')
    p.add_argument('--moment-weight', type=float, default=0.)
    p.add_argument('--moment-huber-delta', type=float, default=.1)
    p.add_argument('--cvar-weight', type=float, default=0.)
    p.add_argument('--cvar-fraction', type=float, default=.1)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=20)
    p.add_argument('--corpus-files', type=int, default=800)
    p.add_argument('--corpus-sims', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--selection-reference', type=Path, default=None,
                   help='modest raw trajectory NPZ used during checkpoint selection')
    p.add_argument('--selection-every', type=int, default=2000,
                   help='steps between trajectory-reference evaluations')
    p.add_argument('--checkpoint-metric', choices=['val-nll', 'trajectory-nll'],
                   default='val-nll',
                   help='metric minimized when retaining the best checkpoint')
    p.add_argument('--moment-min-resultant', type=float, default=.2,
                   help='minimum empirical resultant for interpretable max moment errors')
    args = p.parse_args()
    if not 0. <= args.augmentation_fraction <= 1.:
        p.error('--augmentation-fraction must lie in [0,1]')
    if args.checkpoint_metric == 'trajectory-nll' and not args.selection_reference:
        p.error('--checkpoint-metric trajectory-nll requires --selection-reference')
    if args.selection_every < 1 or args.selection_every % args.eval_every:
        p.error('--selection-every must be a positive multiple of --eval-every')
    if not 0 <= args.moment_min_resultant <= 1:
        p.error('--moment-min-resultant must lie in [0,1]')
    if args.grouped_outcomes < 1 or args.batch_size % args.grouped_outcomes:
        p.error('--grouped-outcomes must be positive and divide --batch-size')
    if args.moment_weight < 0 or args.cvar_weight < 0:
        p.error('--moment-weight and --cvar-weight cannot be negative')
    if args.moment_huber_delta <= 0 or not 0 < args.cvar_fraction <= 1:
        p.error('--moment-huber-delta must be positive and --cvar-fraction in (0,1]')
    if ((args.moment_weight or args.cvar_weight) and args.grouped_outcomes == 1):
        p.error('moment/CVaR losses require --grouped-outcomes greater than 1')

    print(f"Loading {args.source} ...", flush=True)
    store, source_meta = data_mod.load_source(
        args.source, corpus_files=args.corpus_files, corpus_sims=args.corpus_sims,
        seed=args.seed, progress=True)
    source_strata = source_meta.pop('strata', None)
    if source_strata:
        source_meta['n_trajectory_labels'] = len(set(source_strata))
    train_store, val_store = data_mod.split_by_params(
        store, args.val_fraction, seed=args.seed)
    print(f"{store.n_rows} parameter rows, {store.n_observations:,} finite outcomes; "
          f"held out {val_store.n_rows} rows by parameter triple")

    augmentation_train = augmentation_val = None
    augmentation_meta = None
    if args.augmentation:
        augmentation, augmentation_meta = data_mod.load_source(args.augmentation)
        augmentation_strata = augmentation_meta.pop('strata', None)
        if augmentation_strata:
            augmentation_meta['n_trajectory_labels'] = len(set(augmentation_strata))
        augmentation_train, augmentation_val = data_mod.split_by_params(
            augmentation, args.val_fraction, seed=args.seed)
        data_mod.require_matching_n_samples(
            {'source_meta': source_meta}, augmentation_meta)
        print(f"augmentation: {augmentation.n_rows} parameter rows, "
              f"{augmentation.n_observations:,} finite outcomes; sampling "
              f"{args.augmentation_fraction:.0%} per batch")

    selection_store = selection_labels = selection_meta = None
    if args.selection_reference:
        selection_store, selection_meta = data_mod.load_source(args.selection_reference)
        selection_labels = selection_meta.pop('strata', None)
        if not selection_labels:
            raise ValueError('--selection-reference must contain trajectory labels')
        selection_meta['n_trajectory_labels'] = len(set(selection_labels))
        data_mod.require_matching_n_samples({'source_meta': source_meta}, selection_meta)
        print(f'selection reference: {selection_store.n_rows} rows x '
              f'{selection_store.bias.shape[1]} simulations')

    if args.init_model:
        model, variables, init_meta = wm.load_model(args.init_model)
        if model.n_components != args.components:
            raise ValueError(f'--components={args.components} does not match initialized '
                             f'model K={model.n_components}')
        data_mod.require_matching_n_samples(init_meta, source_meta)
        print(f'Fine-tuning {args.init_model}')
    else:
        model = wm.ConditionalWrappedMixture(
            n_components=args.components, hidden_dims=tuple(args.hidden),
            min_scale=args.min_scale)
    rng = np.random.default_rng(args.seed)
    if not args.init_model:
        x0, _, _ = train_store.batch(rng, 8)
        variables = model.init(jax.random.PRNGKey(args.seed), x0)
    n_params = sum(x.size for x in jax.tree.leaves(variables))
    print(f"K={args.components}, {n_params:,} network parameters")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.05, peak_value=args.lr, warmup_steps=args.warmup,
        decay_steps=args.steps, end_value=args.lr * 0.02)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = tx.init(variables)

    loss_fn = make_loss(model, args.n_wraps, args.moment_weight,
                        args.moment_huber_delta, args.cvar_weight,
                        args.cvar_fraction)

    @jax.jit
    def update(variables, opt_state, x, b, w):
        loss, grads = jax.value_and_grad(loss_fn)(variables, x, b, w)
        updates, opt_state = tx.update(grads, opt_state, variables)
        return optax.apply_updates(variables, updates), opt_state, loss

    history = []
    best = (np.inf, variables)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.grouped_outcomes == 1:
            x, b, w = mixed_batch(train_store, augmentation_train,
                                  args.augmentation_fraction, rng, args.batch_size)
        else:
            x, b, w = mixed_grouped_batch(
                train_store, augmentation_train, args.augmentation_fraction, rng,
                args.batch_size // args.grouped_outcomes, args.grouped_outcomes)
        variables, opt_state, loss = update(variables, opt_state, x, b, w)
        if step % args.eval_every == 0 or step == args.steps:
            val_rng = np.random.default_rng(args.seed + 1)
            val_nll = evaluate_mixed(
                loss_fn, variables, val_store, augmentation_val,
                args.augmentation_fraction, val_rng,
                args.eval_batches, args.batch_size)
            history.append({'step': step, 'train_nll': float(loss),
                            'val_nll': val_nll})
            trajectory_metrics = None
            if (selection_store is not None and
                    (step % args.selection_every == 0 or step == args.steps)):
                trajectory_metrics = trajectory_validation_metrics(
                    model, variables, selection_store, selection_labels, args.n_wraps,
                    args.moment_min_resultant)
                history[-1].update(trajectory_metrics)
            score = (val_nll if args.checkpoint_metric == 'val-nll' else
                     (trajectory_metrics['trajectory_nll']
                      if trajectory_metrics else np.inf))
            if score < best[0]:
                best = (score, jax.tree.map(np.array, variables))
            detail = ''
            if trajectory_metrics:
                detail = (f"  trajectory {trajectory_metrics['trajectory_nll']:.4f}"
                          f"  max-mean[R] "
                          f"{trajectory_metrics['max_mean_error_identified']:.3f}"
                          f"  max-sd[R] {trajectory_metrics['max_sd_error_identified']:.3f}")
            print(f"step {step:6d}  train {float(loss):8.4f}  val {val_nll:8.4f}"
                  f"{detail}  ({time.time() - t0:.0f}s)", flush=True)

    meta = dict(vars(args) | {'out': str(args.out), 'source_meta': source_meta,
                              'augmentation_meta': augmentation_meta,
                              'selection_meta': selection_meta,
                              'best_checkpoint_score': best[0], 'history': history,
                              'n_network_params': n_params,
                              'train_seconds': time.time() - t0})
    meta = json.loads(json.dumps(meta, default=str))
    wm.save_model(args.out, best[1], model, meta)
    print(f"Best {args.checkpoint_metric} {best[0]:.4f}; saved {args.out} "
          f"({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == '__main__':
    main()
