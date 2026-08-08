#!/usr/bin/env python3
"""Reproducible inference benchmark for a trained conditional-density model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from continuous_density import wrapped_mixture_model as wm


def _time_ms(fn, repeats: int) -> float:
    fn().block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        fn().block_until_ready()
    return 1000.0 * (time.perf_counter() - start) / repeats


def benchmark(model, variables, n_trials: int = 1000, repeats: int = 100,
              n_wraps: int = 4) -> dict:
    """Time one conditional prediction and one ``n_trials`` likelihood."""
    one_x = jnp.asarray([[35., 70., 45., 43.25]], dtype=jnp.float32)
    feat = jnp.linspace(2., 180., n_trials)
    trial_x = jnp.column_stack([jnp.full(n_trials, 35.),
                                jnp.full(n_trials, 70.),
                                jnp.full(n_trials, 45.), feat])
    bias = 20. * jnp.sin(jnp.radians(feat))

    predict = jax.jit(lambda: model.apply(variables, one_x)['log_pi'])

    @jax.jit
    def likelihood():
        dist = model.apply(variables, trial_x)
        return -jnp.sum(wm.mixture_logpdf(bias, dist, n_wraps))

    return {
        'device': str(jax.devices()[0]),
        'n_trials': int(n_trials),
        'repeats': int(repeats),
        'one_condition_prediction_ms': _time_ms(predict, repeats),
        'trial_likelihood_ms': _time_ms(likelihood, repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--trials', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    model, variables, meta = wm.load_model(args.model)
    result = benchmark(model, variables, args.trials, args.repeats,
                       meta.get('n_wraps', 4))
    result['checkpoint_bytes'] = args.model.stat().st_size
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
