"""Estimator/run names shared by simulation writers and cloud readers; no JAX."""
import os
from pathlib import Path

ESTIMATOR_ID = 'wrapped_likelihood_em_v1'


def qualify_name(name, namespace):
    suffix = '_'+namespace
    return name if name.endswith(suffix) else name+suffix


def qualify_path(path, namespace):
    path = Path(path)
    return path.with_name(qualify_name(path.name, namespace))


def qualify_prefix(prefix, namespace):
    prefix = prefix.strip('/')
    return prefix if prefix.split('/')[-1] == namespace else '/'.join(filter(None, [prefix, namespace]))


def count_suffix(value):
    return f'{value//1000}k' if value >= 1000 and value%1000 == 0 else str(value)


def grid_namespace(n_simulations, n_samples, algorithm='EM', diagonal_covariance=True,
                   fix_weights=False, wrap_feat=True):
    geometry = 'circular' if wrap_feat else 'linear'
    covariance = 'diagcov' if diagonal_covariance else 'fullcov'
    weights = 'fix_weights' if fix_weights else 'free_weights'
    return qualify_name(f'sim_samples_{count_suffix(n_simulations)}_{n_samples}samples_'
                        f'{geometry}_{algorithm.lower()}_{covariance}_{weights}', ESTIMATOR_ID)


def add_reader_arguments(parser, counts=True):
    parser.add_argument('--run-namespace', default=os.getenv('SIMULATION_NAMESPACE'),
        help='exact writer run namespace; default derives it from simulation/sample counts')
    parser.add_argument('--legacy-estimator', action='store_true',
        help='inspect the exact supplied prefix/registry of a historical run')
    if counts:
        parser.add_argument('--n-simulations', type=int, default=int(os.getenv('N_SIMULATIONS', '10000')))
        parser.add_argument('--n-samples', type=int, default=int(os.getenv('N_SAMPLES', '100')))


def reader_namespace(args):
    return None if args.legacy_estimator else (args.run_namespace or grid_namespace(args.n_simulations, args.n_samples))


def resolve_reader(args, config=None):
    namespace = reader_namespace(args)
    if namespace:
        if config is not None:
            config.prefix = qualify_prefix(config.prefix, namespace)
        if getattr(args, 'registry', None):
            args.registry = qualify_name(args.registry, namespace)
    return namespace
