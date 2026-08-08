#!/usr/bin/env python3
"""Select diagnostic mu1 cases and score NN checkpoints against reference surfaces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "neural_network_optimization"))

from mirror_aware_model import MirrorAwareMu1Predictor, infer_native_mu1_rows
from shared.mu1_axis import (GRID_CONVENTION, guard_surface_mu1_axis,
                            mu1_grid_np, mu1_size)


SCENARIOS = ('peaked', 'flat', 'seam', 'asymmetric', 'multimodal')
METRICS = ('kl', 'energy', 'moment', 'asymmetry', 'hellinger', 'probability_ise')
SURFACE_NAME = re.compile(
    r'averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl')
MU1_GRID = mu1_grid_np()
POSITIVE = (MU1_GRID > 0) & ~np.isclose(np.abs(MU1_GRID), 180.0)
NEGATIVE = (MU1_GRID < 0) & ~np.isclose(np.abs(MU1_GRID), 180.0)


def _iter_surface_records(folder, stride=1, wanted=None):
    """Stream individual or bundled surfaces without materializing a full grid."""
    wanted = None if wanted is None else set(wanted)

    def selected(name):
        digest = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
        return digest % stride == 0

    def wanted_name(name):
        if wanted is None:
            return True
        match = SURFACE_NAME.fullmatch(name)
        return match is not None and tuple(map(float, match.groups())) in wanted

    folder = Path(folder)
    bundles = sorted(folder.glob('surface_bundle_*.pkl.gz'))
    if bundles:
        for bundle_path in bundles:
            if not selected(bundle_path.name):
                continue
            with gzip.open(bundle_path, 'rb') as handle:
                bundle = pickle.load(handle)
            for filename, raw in bundle['surfaces'].items():
                if not wanted_name(filename):
                    continue
                record = pickle.loads(raw)
                guard_surface_mu1_axis(record['surface'], source=filename)
                yield record
        return

    for surface_path in sorted(folder.glob('averaged_sf1_*_sf2_*_sp_*.pkl')):
        if not selected(surface_path.name) or not wanted_name(surface_path.name):
            continue
        with surface_path.open('rb') as handle:
            record = pickle.load(handle)
        guard_surface_mu1_axis(record['surface'], source=str(surface_path))
        yield record


def _probabilities(log_density):
    values = np.asarray(log_density, dtype=np.float64)
    values -= values.max(axis=-2, keepdims=True)
    probabilities = np.exp(values)
    return probabilities / probabilities.sum(axis=-2, keepdims=True)


def _surface_scores(surface):
    probabilities = np.stack([
        _probabilities(surface.mu1_comp1_surface),
        _probabilities(surface.mu1_comp2_surface),
    ])
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    peak = np.max(probabilities, axis=1)
    seam_mass = probabilities[:, np.r_[0:5, mu1_size() - 5:mu1_size()], :].sum(axis=1)
    signed = (probabilities[:, POSITIVE, :].sum(axis=1) -
              probabilities[:, NEGATIVE, :].sum(axis=1))

    left = np.roll(probabilities, 1, axis=1)
    right = np.roll(probabilities, -1, axis=1)
    relative_height = probabilities / np.maximum(
        probabilities.max(axis=1, keepdims=True), 1e-300)
    modes = ((probabilities > left) & (probabilities >= right) &
             (relative_height >= 0.15) &
             (probabilities >= 1.5 / mu1_size())).sum(axis=1)

    return {
        'peaked': float(peak.mean()),
        'flat': float((entropy / np.log(mu1_size())).mean()),
        'seam': float(np.quantile(seam_mass, 0.95)),
        'asymmetric': float(np.abs(signed).mean()),
        'multimodal': float((modes >= 2).mean()),
        'normalized_entropy': float((entropy / np.log(mu1_size())).mean()),
        'mean_peak_probability': float(peak.mean()),
    }


def select_scenarios(args):
    if args.candidate_stride < 1:
        raise ValueError('--candidate-stride must be at least one')
    candidates = []
    for record in _iter_surface_records(
            Path(args.surfaces_folder).resolve(), args.candidate_stride):
        params = record['parameters']
        scores = _surface_scores(record['surface'])
        candidates.append({
            'sd_feat1': float(params['sd_feat1']),
            'sd_feat2': float(params['sd_feat2']),
            'sd_spat': float(params['sd_spat']),
            **scores,
        })

    selected = []
    used = set()
    for scenario in SCENARIOS:
        ranked = sorted(candidates, key=lambda row: row[scenario], reverse=True)
        scenario_rows = []
        for row in ranked:
            key = (row['sd_feat1'], row['sd_feat2'], row['sd_spat'])
            if key in used:
                continue
            point = np.asarray(key)
            if any(np.linalg.norm(point - np.asarray((chosen['sd_feat1'],
                                                       chosen['sd_feat2'],
                                                       chosen['sd_spat']))) <
                   args.min_parameter_distance for chosen in scenario_rows):
                continue
            selected.append({'scenario': scenario, 'selection_score': row[scenario], **row})
            scenario_rows.append(row)
            used.add(key)
            if len(scenario_rows) == args.per_scenario:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['scenario', 'sd_feat1', 'sd_feat2', 'sd_spat', 'selection_score',
                  'normalized_entropy', 'mean_peak_probability', *SCENARIOS]
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} unique surfaces into {output}")


def _read_manifest(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = tuple(float(row[name]) for name in ('sd_feat1', 'sd_feat2', 'sd_spat'))
        result[key] = row['scenario']
    return result


def _surface_map(folder, wanted=None):
    result = {}
    for record in _iter_surface_records(Path(folder).resolve(), wanted=wanted):
        params = record['parameters']
        key = tuple(float(params[name]) for name in ('sd_feat1', 'sd_feat2', 'sd_spat'))
        result[key] = record['surface']
    return result


def _checkpoint_epoch(path):
    match = re.search(r'epoch_(\d+)', Path(path).name)
    if match is None:
        raise ValueError(f"Cannot read epoch from checkpoint name {path}")
    return int(match.group(1))


def _latest_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    checkpoints = list(path.glob('model_epoch_*.pkl'))
    if not checkpoints:
        raise FileNotFoundError(f"No model_epoch_*.pkl checkpoint in {path}")
    return max(checkpoints, key=lambda item: int(re.search(r'epoch_(\d+)', item.name).group(1)))


def _load_predictor(checkpoint):
    with checkpoint.open('rb') as handle:
        data = pickle.load(handle)
    if data.get('grid_convention') != GRID_CONVENTION:
        raise ValueError(f"{checkpoint} is not a {GRID_CONVENTION} checkpoint")
    if int(data.get('mu1_bias_grid_size', -1)) != mu1_size():
        raise ValueError(f"{checkpoint} has the wrong mu1 row count")
    model = MirrorAwareMu1Predictor(
        native_mu1_rows=infer_native_mu1_rows(data['params']))
    return lambda inputs: np.asarray(model.apply(data['params'], jnp.asarray(inputs)))


def _column_metrics(prediction, target):
    p = _probabilities(prediction)
    q = _probabilities(target)
    grid = np.deg2rad(MU1_GRID)
    degree_grid = MU1_GRID
    difference = np.abs(degree_grid[:, None] - degree_grid[None, :])
    distance = np.minimum(difference, 360.0 - difference)

    kl = np.sum(q * (np.log(np.maximum(q, 1e-300)) -
                     np.log(np.maximum(p, 1e-300))), axis=0)
    cross = np.einsum('nf,nm,mf->f', p, distance, q)
    pred_self = np.einsum('nf,nm,mf->f', p, distance, p)
    target_self = np.einsum('nf,nm,mf->f', q, distance, q)
    delta = p - q
    moment = (np.sum(delta * np.cos(grid)[:, None], axis=0) ** 2 +
              np.sum(delta * np.sin(grid)[:, None], axis=0) ** 2) / 4.0
    signed_delta = (delta[POSITIVE].sum(axis=0) - delta[NEGATIVE].sum(axis=0))
    return {
        'kl': kl,
        'energy': (2 * cross - pred_self - target_self) / 360.0,
        'moment': moment,
        'asymmetry': signed_delta ** 2 / 4.0,
        'hellinger': 1.0 - np.sum(np.sqrt(p * q), axis=0),
        'probability_ise': np.sum(delta ** 2, axis=0),
    }


def _evaluation_rows(profile, predictor, truth, scenarios):
    rows = []
    for params, scenario in scenarios.items():
        if params not in truth:
            raise KeyError(f"Reference surface missing manifest parameters {params}")
        sf1, sf2, sp = params
        surface = truth[params]
        orientations = [(1, [sf1, sf2, sp], surface.mu1_comp1_surface)]
        if sf1 != sf2:
            orientations.append((2, [sf2, sf1, sp], surface.mu1_comp2_surface))
        predictions = predictor(np.asarray([item[1] for item in orientations]))
        for prediction, (component, _, target) in zip(predictions, orientations):
            metrics = _column_metrics(prediction, target)
            for feat_index, feat_diff in enumerate(np.asarray(surface.feat_diff_grid)):
                row = {
                    'profile': profile, 'scenario': scenario,
                    'sd_feat1': sf1, 'sd_feat2': sf2, 'sd_spat': sp,
                    'component': component, 'feat_diff': float(feat_diff),
                }
                row.update({name: float(values[feat_index])
                            for name, values in metrics.items()})
                rows.append(row)
    return rows


def _write_evaluation(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['profile', 'scenario', 'sd_feat1', 'sd_feat2', 'sd_spat',
              'component', 'feat_diff', *METRICS]
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['profile'], row['scenario'])].append(row)
    summary = output.with_name(output.stem + '_summary.csv')
    with summary.open('w', newline='') as handle:
        fields = ['profile', 'scenario', 'n_columns',
                  *[f'{metric}_mean' for metric in METRICS],
                  *[f'{metric}_median' for metric in METRICS]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (profile, scenario), group in sorted(grouped.items()):
            row = {'profile': profile, 'scenario': scenario, 'n_columns': len(group)}
            row.update({f'{metric}_mean': float(np.mean([item[metric] for item in group]))
                        for metric in METRICS})
            row.update({f'{metric}_median': float(np.median([item[metric] for item in group]))
                        for metric in METRICS})
            writer.writerow(row)
    print(f"Wrote {len(rows)} feature-dissimilarity rows to {output}")
    print(f"Wrote stratified summary to {summary}")


def _write_trajectory(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ['epoch', 'target', 'scenario', 'sd_feat1', 'sd_feat2', 'sd_spat',
              'component', 'feat_diff', *METRICS]
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        feat_diff = row['feat_diff']
        band = '2-60' if feat_diff <= 60 else '62-120' if feat_diff <= 120 else '122-180'
        grouped[(row['epoch'], row['target'], row['scenario'], band)].append(row)
    summary = output.with_name(output.stem + '_summary.csv')
    with summary.open('w', newline='') as handle:
        summary_fields = ['epoch', 'target', 'scenario', 'feat_diff_band', 'n_columns',
                          *[f'{metric}_mean' for metric in METRICS]]
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key, group in sorted(grouped.items()):
            epoch, target, scenario, band = key
            row = {'epoch': epoch, 'target': target, 'scenario': scenario,
                   'feat_diff_band': band, 'n_columns': len(group)}
            row.update({f'{metric}_mean': float(np.mean([item[metric] for item in group]))
                        for metric in METRICS})
            writer.writerow(row)
    print(f"Wrote {len(rows)} trajectory rows to {output}")
    print(f"Wrote trajectory summary to {summary}")


def evaluate(args):
    scenarios = _read_manifest(args.manifest)
    truth = _surface_map(args.truth_folder)
    rows = []
    for specification in args.checkpoint:
        try:
            profile, path = specification.split('=', 1)
        except ValueError as exc:
            raise ValueError('--checkpoint must be PROFILE=FILE_OR_DIRECTORY') from exc
        checkpoint = _latest_checkpoint(path)
        print(f"Evaluating {profile}: {checkpoint}")
        rows.extend(_evaluation_rows(profile, _load_predictor(checkpoint), truth, scenarios))

    if args.reference_repeat:
        repeat = _surface_map(args.reference_repeat)
        for params, scenario in scenarios.items():
            if params not in repeat:
                raise KeyError(f"Repeat reference missing {params}")
            sf1, sf2, sp = params
            targets = truth[params]
            predictions = repeat[params]
            pairs = [(1, predictions.mu1_comp1_surface, targets.mu1_comp1_surface)]
            if sf1 != sf2:
                pairs.append((2, predictions.mu1_comp2_surface, targets.mu1_comp2_surface))
            for component, prediction, target in pairs:
                metrics = _column_metrics(prediction, target)
                for index, feat_diff in enumerate(np.asarray(targets.feat_diff_grid)):
                    row = {'profile': 'reference_repeat', 'scenario': scenario,
                           'sd_feat1': sf1, 'sd_feat2': sf2, 'sd_spat': sp,
                           'component': component, 'feat_diff': float(feat_diff)}
                    row.update({name: float(value[index]) for name, value in metrics.items()})
                    rows.append(row)
    _write_evaluation(rows, args.output)


def evaluate_trajectory(args):
    scenarios = _read_manifest(args.manifest)
    targets = {}
    for specification in args.target:
        try:
            label, folder = specification.split('=', 1)
        except ValueError as exc:
            raise ValueError('--target must be LABEL=FOLDER') from exc
        targets[label] = _surface_map(folder, wanted=scenarios)
        missing = set(scenarios) - set(targets[label])
        if missing:
            raise KeyError(f"Target {label!r} is missing manifest parameters {sorted(missing)}")

    checkpoint_dir = Path(args.checkpoints).resolve()
    checkpoints = sorted(
        checkpoint_dir.glob('model_epoch_*.pkl'), key=_checkpoint_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No trajectory checkpoints in {checkpoint_dir}")

    rows = []
    for checkpoint in checkpoints:
        epoch = _checkpoint_epoch(checkpoint)
        print(f"Evaluating trajectory epoch {epoch}")
        predictor = _load_predictor(checkpoint)
        for label, surfaces in targets.items():
            evaluated = _evaluation_rows(str(epoch), predictor, surfaces, scenarios)
            for row in evaluated:
                row['epoch'] = epoch
                row['target'] = label
                del row['profile']
            rows.extend(evaluated)
    _write_trajectory(rows, args.output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    select = subparsers.add_parser('select', help='Select diverse cases from existing surfaces')
    select.add_argument('--surfaces-folder', required=True)
    select.add_argument('--per-scenario', type=int, default=3)
    select.add_argument('--candidate-stride', type=int, default=8,
                        help='Score a deterministic 1/N subset of the full grid (default: 8)')
    select.add_argument('--min-parameter-distance', type=float, default=30.0,
                        help='Minimum Euclidean spacing within a scenario in parameter degrees')
    select.add_argument('--output', required=True)
    select.set_defaults(func=select_scenarios)

    score = subparsers.add_parser('evaluate', help='Score checkpoints against reference surfaces')
    score.add_argument('--truth-folder', required=True)
    score.add_argument('--reference-repeat', default=None,
                       help='Optional independent high-resolution folder to quantify reference noise')
    score.add_argument('--manifest', required=True)
    score.add_argument('--checkpoint', action='append', required=True,
                       help='PROFILE=checkpoint file or directory; repeat for each ablation')
    score.add_argument('--output', required=True)
    score.set_defaults(func=evaluate)

    trajectory = subparsers.add_parser(
        'trajectory', help='Score every saved checkpoint against several targets')
    trajectory.add_argument('--manifest', required=True)
    trajectory.add_argument('--target', action='append', required=True,
                            help='LABEL=FOLDER; repeat for training and references')
    trajectory.add_argument('--checkpoints', required=True)
    trajectory.add_argument('--output', required=True)
    trajectory.set_defaults(func=evaluate_trajectory)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = parse_args()
    parsed.func(parsed)
