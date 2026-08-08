#!/usr/bin/env python3
"""Train the conditional wrapped-normal mixture on raw EM bias samples.

The objective is the negative log likelihood of individual simulated biases.
No histogram, KDE, bias grid or likelihood surface is constructed anywhere in
the loss.

Usage (from the repo root)::

    PYTHONPATH=. python continuous_density/train.py \
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

from continuous_density import data as data_mod
from continuous_density import wrapped_mixture_model as wm


def make_loss(model, n_wraps: int):
    """Masked mean NLL; weight 0 drops the rare non-finite EM outcome."""
    @jax.jit
    def loss_fn(variables, x, b, w):
        dist = model.apply(variables, x)
        logp = wm.mixture_logpdf(b, dist, n_wraps)
        return -jnp.sum(w * logp) / jnp.maximum(jnp.sum(w), 1.0)

    return loss_fn


def evaluate_store(loss_fn, variables, store, rng, n_batches: int, batch_size: int):
    """Mean NLL over random batches of a held-out store."""
    total = 0.0
    for _ in range(n_batches):
        x, b, w = store.batch(rng, batch_size)
        total += float(loss_fn(variables, x, b, w))
    return total / n_batches


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True,
                   help='corpus directory name/path, or an .npz training file')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--components', type=int, default=8, help='mixture components K')
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 128, 128])
    p.add_argument('--min-scale', type=float, default=0.25)
    p.add_argument('--n-wraps', type=int, default=4)
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
    args = p.parse_args()

    print(f"Loading {args.source} ...", flush=True)
    store, source_meta = data_mod.load_source(
        args.source, corpus_files=args.corpus_files, corpus_sims=args.corpus_sims,
        seed=args.seed, progress=True)
    train_store, val_store = data_mod.split_by_params(
        store, args.val_fraction, seed=args.seed)
    print(f"{store.n_rows} parameter rows, {store.n_observations:,} finite outcomes; "
          f"held out {val_store.n_rows} rows by parameter triple")

    model = wm.ConditionalWrappedMixture(
        n_components=args.components, hidden_dims=tuple(args.hidden),
        min_scale=args.min_scale)
    rng = np.random.default_rng(args.seed)
    x0, b0, w0 = train_store.batch(rng, 8)
    variables = model.init(jax.random.PRNGKey(args.seed), x0)
    n_params = sum(x.size for x in jax.tree.leaves(variables))
    print(f"K={args.components}, {n_params:,} network parameters")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.lr * 0.05, peak_value=args.lr, warmup_steps=args.warmup,
        decay_steps=args.steps, end_value=args.lr * 0.02)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = tx.init(variables)

    loss_fn = make_loss(model, args.n_wraps)

    @jax.jit
    def update(variables, opt_state, x, b, w):
        loss, grads = jax.value_and_grad(loss_fn)(variables, x, b, w)
        updates, opt_state = tx.update(grads, opt_state, variables)
        return optax.apply_updates(variables, updates), opt_state, loss

    history = []
    best = (np.inf, variables)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, b, w = train_store.batch(rng, args.batch_size)
        variables, opt_state, loss = update(variables, opt_state, x, b, w)
        if step % args.eval_every == 0 or step == args.steps:
            val_rng = np.random.default_rng(args.seed + 1)
            val_nll = evaluate_store(loss_fn, variables, val_store, val_rng,
                                     args.eval_batches, args.batch_size)
            history.append({'step': step, 'train_nll': float(loss),
                            'val_nll': val_nll})
            if val_nll < best[0]:
                best = (val_nll, jax.tree.map(np.array, variables))
            print(f"step {step:6d}  train {float(loss):8.4f}  val {val_nll:8.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = dict(vars(args) | {'out': str(args.out), 'source_meta': source_meta,
                              'best_val_nll': best[0], 'history': history,
                              'n_network_params': n_params,
                              'train_seconds': time.time() - t0})
    meta = json.loads(json.dumps(meta, default=str))
    wm.save_model(args.out, best[1], model, meta)
    print(f"Best held-out NLL {best[0]:.4f}; saved {args.out} "
          f"({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == '__main__':
    main()
