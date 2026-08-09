#!/usr/bin/env python3
"""Fit free per-dissimilarity mixtures on one raw UEV trajectory.

This separates conditional-network smoothing from mixture-family capacity. Each
dissimilarity gets an independent K-component wrapped-normal mixture, initialized
from the conditional model and fitted by raw-sample NLL on half the simulations.
All reported likelihoods and moments use the untouched other half.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import optax
import pandas as pd

from continuous_density import compare_existing_model as comparison
from continuous_density import data as data_mod
from continuous_density import design as design_mod
from continuous_density import evaluate
from continuous_density import wrapped_mixture_model as wm


def _inverse_softplus(x):
    """Stable inverse of softplus for positive ``x``."""
    return x + jnp.log(-jnp.expm1(-x))


def _dist(theta, min_scale):
    return {'log_pi': jax.nn.log_softmax(theta['logits'], axis=-1),
            'mu': wm.wrap_deg(theta['mu']),
            'sigma': jax.nn.softplus(theta['raw_scale']) + min_scale}


def fit_free_mixtures(initial, train_samples, min_scale=.25, steps=1500,
                      batch_size=128, lr=.02, seed=0, n_wraps=4):
    """Jointly optimize independent mixture parameters for each trajectory row."""
    theta = {'logits': initial['log_pi'], 'mu': initial['mu'],
             'raw_scale': _inverse_softplus(
                 jnp.maximum(initial['sigma'] - min_scale, 1e-4))}
    samples = jnp.asarray(train_samples)
    tx = optax.chain(optax.clip_by_global_norm(5.), optax.adam(lr))
    state = tx.init(theta)

    @jax.jit
    def step(theta, state, key):
        index = jax.random.randint(key, (samples.shape[0], batch_size),
                                   0, samples.shape[1])
        batch = jnp.take_along_axis(samples, index, axis=1)

        def loss_fn(value):
            logp = wm.mixture_logpdf_samples(batch, _dist(value, min_scale), n_wraps)
            return -jnp.mean(logp)

        loss, grad = jax.value_and_grad(loss_fn)(theta)
        updates, state = tx.update(grad, state, theta)
        return optax.apply_updates(theta, updates), state, loss

    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    for iteration in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        theta, state, loss = step(theta, state, subkey)
        if iteration % 250 == 0 or iteration == steps:
            print(f'step {iteration:4d}: minibatch NLL {float(loss):.5f} '
                  f'({time.time() - t0:.1f}s)', flush=True)
    return jax.tree.map(np.asarray, _dist(theta, min_scale))


def fine_tune_network(model, variables, params, train_samples, steps=3000,
                      batch_size=4096, lr=3e-4, seed=1, n_wraps=4):
    """Locally fine-tune the existing smooth conditional network."""
    params = jnp.asarray(params)
    samples = jnp.asarray(train_samples)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.adam(lr))
    state = tx.init(variables)

    @jax.jit
    def step(variables, state, key):
        row_key, sample_key = jax.random.split(key)
        rows = jax.random.randint(row_key, (batch_size,), 0, samples.shape[0])
        columns = jax.random.randint(sample_key, (batch_size,), 0, samples.shape[1])
        x = params[rows]
        bias = samples[rows, columns]

        def loss_fn(value):
            return -jnp.mean(wm.mixture_logpdf(
                bias, model.apply(value, x), n_wraps))

        loss, grad = jax.value_and_grad(loss_fn)(variables)
        updates, state = tx.update(grad, state, variables)
        return optax.apply_updates(variables, updates), state, loss

    key = jax.random.PRNGKey(seed)
    t0 = time.time()
    for iteration in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        variables, state, loss = step(variables, state, subkey)
        if iteration % 500 == 0 or iteration == steps:
            print(f'local network step {iteration:4d}: minibatch NLL '
                  f'{float(loss):.5f} ({time.time() - t0:.1f}s)', flush=True)
    return jax.tree.map(np.asarray, variables)


def _dist_nll(dist, samples, n_wraps=4, chunk=5000):
    total = np.zeros(samples.shape[0])
    count = np.zeros(samples.shape[0])
    for start in range(0, samples.shape[1], chunk):
        block = samples[:, start:start + chunk]
        finite = np.isfinite(block)
        logp = np.asarray(wm.mixture_logpdf_samples(
            jnp.asarray(np.where(finite, block, 0.)), dist, n_wraps))
        total += np.where(finite, logp, 0.).sum(axis=1)
        count += finite.sum(axis=1)
    return -total / count


def _production_nll(profiles, samples, chunk=5000):
    return comparison._case_nll_from_scorer(
        comparison._score_interp(profiles), samples, chunk=chunk)


def _metric_error(pred, raw, circular=False):
    return evaluate.wrap_error(pred - raw) if circular else pred - raw


def _dist_asymmetry(dist, n_wraps=4):
    grid = comparison.mu1_grid_np()
    log_density = np.asarray(wm.mixture_logpdf_grid(
        jnp.asarray(grid), dist, n_wraps)).T
    return evaluate.density_asymmetry(log_density)


def plot_diagnostic(frame, out, title):
    methods = [('raw held-out 50k', '#111111', '-'),
               ('conditional density', '#2563eb', '--'),
               ('production NN', '#ea580c', ':'),
               ('local fine-tune', '#a21caf', (0, (5, 1))),
               ('free K=12', '#15803d', '-.')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex='col',
                             gridspec_kw={'height_ratios': [2, 1]})
    for col, (metric, label, circular) in enumerate([
            ('mean_bias', 'Mean bias, °', True),
            ('density_asymmetry', 'P(bias > 0) − P(bias < 0)', False),
            ('response_sd', 'Circular response SD, °', False)]):
        for method, color, ls in methods:
            axes[0, col].plot(frame.dist_feat, frame[f'{metric}_{method}'],
                              color=color, ls=ls, lw=1.8, label=method)
            if method != 'raw 100k':
                err = _metric_error(frame[f'{metric}_{method}'],
                                    frame[f'{metric}_raw held-out 50k'], circular)
                axes[1, col].plot(frame.dist_feat, err, color=color, ls=ls, lw=1.5)
        axes[0, col].set_ylabel(label)
        unit = ', °' if metric != 'density_asymmetry' else ''
        axes[1, col].set(xlabel='Feature dissimilarity, °',
                         ylabel=f'prediction − raw{unit}')
        axes[1, col].axhline(0, color='#888888', lw=.8)
        for ax in axes[:, col]:
            ax.grid(True, color='#e5e7eb', lw=.6)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--sd-feat1', type=float, default=30.)
    p.add_argument('--sd-feat2', type=float, default=60.)
    p.add_argument('--spat-dprime', type=float, default=.5)
    p.add_argument('--component', type=int, choices=(1, 2), default=2)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.02)
    p.add_argument('--local-steps', type=int, default=3000)
    p.add_argument('--local-batch-size', type=int, default=4096)
    p.add_argument('--local-lr', type=float, default=3e-4)
    args = p.parse_args()

    model, variables, meta = wm.load_model(args.model)
    store, ref_meta = data_mod.load_npz(args.reference)
    data_mod.require_matching_n_samples(meta, ref_meta)
    sd_ident = design_mod.UEV_SPAT_DIFF / args.spat_dprime
    mask = (np.isclose(store.design[:, 0], args.sd_feat1)
            & np.isclose(store.design[:, 1], args.sd_feat2)
            & np.isclose(store.design[:, 2], sd_ident))
    order = np.flatnonzero(mask)[np.argsort(store.design[mask, 3])]
    if len(order) != 90:
        raise ValueError(f'expected 90 rows on the target trajectory, found {len(order)}')
    params = store.design[order]
    if args.component == 2:
        params = np.asarray(wm.mirror_params(params))
    samples = store.bias[order, :, args.component - 1]
    split = samples.shape[1] // 2
    train_samples, test_samples = samples[:, :split], samples[:, split:]

    initial = model.apply(variables, jnp.asarray(params))
    free = fit_free_mixtures(initial, train_samples,
                             min_scale=model.min_scale, steps=args.steps,
                             batch_size=args.batch_size, lr=args.lr,
                             n_wraps=meta.get('n_wraps', 4))
    conditional = jax.tree.map(np.asarray, initial)
    local_variables = fine_tune_network(
        model, variables, params, train_samples, steps=args.local_steps,
        batch_size=args.local_batch_size, lr=args.local_lr,
        n_wraps=meta.get('n_wraps', 4))
    local = jax.tree.map(np.asarray,
                         model.apply(local_variables, jnp.asarray(params)))

    predictor = comparison.production_predictor(args.checkpoint)
    surface = predictor(params[:1])[0]
    production_profiles = comparison.interpolated_profile_single(
        surface, params[:, 3])
    grid = comparison.mu1_grid_np()
    prod_density = np.exp(production_profiles)
    prod_moment = ((prod_density * np.exp(1j * np.radians(grid))[None, :]).sum(axis=1)
                   * comparison.mu1_cell_width())

    raw = evaluate.empirical_moments(test_samples)
    frame = pd.DataFrame({'dist_feat': store.design[order, 3],
                          'mean_bias_raw held-out 50k': raw['mean_bias'],
                          'response_sd_raw held-out 50k': raw['circ_sd']})
    frame['density_asymmetry_raw held-out 50k'] = (
        evaluate.empirical_density_asymmetry(test_samples))
    for name, dist in [('conditional density', conditional),
                       ('local fine-tune', local), ('free K=12', free)]:
        frame[f'mean_bias_{name}'] = np.asarray(wm.mean_and_resultant(dist)[0])
        frame[f'response_sd_{name}'] = np.asarray(wm.circular_sd(dist))
        frame[f'density_asymmetry_{name}'] = _dist_asymmetry(
            dist, meta.get('n_wraps', 4))
    frame['mean_bias_production NN'] = np.degrees(np.angle(prod_moment))
    frame['response_sd_production NN'] = np.degrees(np.sqrt(
        -2. * np.log(np.clip(np.abs(prod_moment), 1e-12, 1.))))
    frame['density_asymmetry_production NN'] = evaluate.density_asymmetry(
        production_profiles.T)
    frame['nll_conditional_density'] = _dist_nll(conditional, test_samples,
                                                  meta.get('n_wraps', 4))
    frame['nll_free_K12'] = _dist_nll(free, test_samples, meta.get('n_wraps', 4))
    frame['nll_local_fine_tune'] = _dist_nll(
        local, test_samples, meta.get('n_wraps', 4))
    frame['nll_production_NN'] = _production_nll(production_profiles, test_samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / 'local_capacity_diagnostic.csv', index=False)
    np.savez(args.out_dir / 'local_free_k12.npz', **free)
    wm.save_model(args.out_dir / 'local_finetuned_network.pkl', local_variables,
                  model, {**meta, 'diagnostic_local_fine_tune': True,
                          'local_steps': args.local_steps,
                          'local_batch_size': args.local_batch_size,
                          'local_lr': args.local_lr})
    title = (f'Local capacity: σ=({args.sd_feat1:g}°, {args.sd_feat2:g}°), '
             f'spatial d′={args.spat_dprime:g}, component {args.component}')
    plot_diagnostic(frame, args.out_dir / 'local_capacity_diagnostic.png', title)
    for metric, circular in [('mean_bias', True), ('density_asymmetry', False),
                             ('response_sd', False)]:
        print(f'\n{metric} maximum absolute error:')
        for method in ['conditional density', 'production NN',
                       'local fine-tune', 'free K=12']:
            err = _metric_error(frame[f'{metric}_{method}'],
                                frame[f'{metric}_raw held-out 50k'], circular)
            unit = '°' if metric != 'density_asymmetry' else ''
            print(f'  {method:20s} {np.max(np.abs(err)):.4f}{unit}')
    print('\nheld-out raw NLL averaged across trajectory:')
    for column in ['nll_conditional_density', 'nll_production_NN',
                   'nll_local_fine_tune', 'nll_free_K12']:
        print(f'  {column:24s} {frame[column].mean():.6f}')
    print(f'Wrote {args.out_dir}')


if __name__ == '__main__':
    main()
