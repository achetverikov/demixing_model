#!/usr/bin/env python3
"""
Grid Status Monitor
===================

Monitor likelihood surface grid computation across machines and grid levels.
Provides progress tracking, machine performance, timing estimates, and
actionable recommendations.

Usage (run from repo root with PYTHONPATH set):

    # One-shot status report
    PYTHONPATH=. python surface_computation/grid_status_monitor.py

    # Continuous refresh every 30 s
    PYTHONPATH=. python surface_computation/grid_status_monitor.py --monitor --refresh 30

    # Show active-chunk details
    PYTHONPATH=. python surface_computation/grid_status_monitor.py --details

    # Compact one-liner
    PYTHONPATH=. python surface_computation/grid_status_monitor.py --quiet

    # Target a specific samples folder (overrides auto-detection)
    PYTHONPATH=. python surface_computation/grid_status_monitor.py \\
        --samples-dir results/sim_samples_10k_100samples_circular_em_diagcov_free_weights

    # Export status to JSON
    PYTHONPATH=. python surface_computation/grid_status_monitor.py --export

The script auto-detects the active samples folder from the same parameters
that simulated_samples_grid.py uses (n-simulations, n-samples, algorithm,
--diagonal-covariance, --fix-weights) so the defaults always stay in sync.
Pass --samples-dir to override.
"""

import json
import os
import re
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Ensure repo root is importable when called as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.utils import resolve_results_path
from shared.config import Config
import jax_fit_functions as jf
import simulated_samples_grid as _ssg
from simulated_samples_grid import (
    get_grid_status,
    get_grid_level_info,
    load_progress_state,
    build_samples_folder_name,
)


def _resolve_samples_dir(args) -> Path:
    """Return the samples output directory, honouring --samples-dir if given."""
    if args.samples_dir:
        return Path(args.samples_dir)
    geometry = 'circular' if jf.wrap_1st else 'linear'
    folder_name = build_samples_folder_name(
        n_simulations=args.n_simulations,
        n_samples=args.n_samples,
        algorithm=args.algorithm,
        diagonal_covariance=args.diagonal_covariance,
        fix_weights=args.fix_weights,
    )
    return resolve_results_path(folder_name, args.results_dir)


def _completed_surface_ids_from_bundles(bundle_dir: Optional[Path]) -> set:
    """Read local chunk bundle manifests and return completed surface IDs."""
    completed = set()
    if not bundle_dir or not bundle_dir.exists():
        return completed
    for manifest_file in bundle_dir.glob("surface_bundle_*.manifest.json"):
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
            completed.update(manifest.get('surface_ids', []))
        except (OSError, json.JSONDecodeError):
            continue
    return completed


def _pipeline_level_status(averaged_surfaces_dir: Path,
                           bundle_dir: Optional[Path] = None) -> Dict:
    """Progress dict (same shape as get_grid_status output) for pipeline mode.

    Counts individual averaged surface files plus any chunk bundle manifests.
    The expected total comes from the canonical averaged-surface grid.
    """
    cfg = Config()
    step = cfg.param_step
    vals = list(range(int(cfg.param_range_low), int(cfg.param_range_high) + int(step), int(step)))
    n_vals = len(vals)
    expected = n_vals * n_vals * (n_vals + 1) // 2

    completed_ids = _completed_surface_ids_from_bundles(bundle_dir)
    completed = 0
    if averaged_surfaces_dir.exists():
        completed = sum(1 for _ in averaged_surfaces_dir.glob("averaged_sf1_*.pkl"))
    completed = max(completed, len(completed_ids))

    missing = max(0, expected - completed)
    rate = completed / expected * 100 if expected > 0 else 0.0
    return {
        'completed_count': completed,
        'expected_total': expected,
        'missing_count': missing,
        'completion_rate': rate,
    }


def get_comprehensive_status(output_dir: Path,
                              averaged_surfaces_dir: Optional[Path] = None,
                              bundle_dir: Optional[Path] = None) -> Dict:
    """Collect progress, machine performance, timing, and recommendations."""
    _ssg.config.samples_folder = output_dir
    pipeline_mode = averaged_surfaces_dir is not None

    if pipeline_mode:
        level1_status = _pipeline_level_status(averaged_surfaces_dir, bundle_dir=bundle_dir)
        level2_status = {'completed_count': 0, 'expected_total': 0,
                         'missing_count': 0, 'completion_rate': 0.0}
    else:
        level1_status = get_grid_status(grid_level=1)
        level2_status = get_grid_status(grid_level=2)
    progress = load_progress_state(output_dir)

    active_work = _analyze_active_work(output_dir)
    machine_performance = _analyze_machine_performance(output_dir)
    scan_dir = bundle_dir if pipeline_mode and bundle_dir else (averaged_surfaces_dir if pipeline_mode else output_dir)
    scan_pattern = "surface_bundle_*.manifest.json" if pipeline_mode and bundle_dir else ("averaged_sf1_*.pkl" if pipeline_mode else None)
    time_analysis = _analyze_computation_timing(
        scan_dir, level1_status, level2_status, file_pattern=scan_pattern
    )

    return {
        'timestamp': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'pipeline_mode': pipeline_mode,
        'averaged_surfaces_dir': str(averaged_surfaces_dir) if pipeline_mode else None,
        'bundle_dir': str(bundle_dir) if bundle_dir else None,
        'level1': level1_status,
        'level2': level2_status,
        'overall_progress': progress,
        'active_work': active_work,
        'machine_performance': machine_performance,
        'time_analysis': time_analysis,
    }


def _analyze_active_work(output_dir: Path) -> Dict:
    """Scan lock files to find currently active chunks and machines."""
    active_work = {
        'level1_chunks': [],
        'level2_chunks': [],
        'total_active_machines': set(),
        'work_distribution': {},
    }

    for lock_file in output_dir.glob("computing_L*_chunk_*.lock"):
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)

            chunk_id = lock_file.stem.replace('computing_', '')
            machine_id = lock_data.get('machine_id', 'unknown')
            chunk_info = {
                'chunk_id': chunk_id,
                'machine_id': machine_id,
                'start_time': lock_data.get('start_time', 0),
                'running_hours': (time.time() - lock_data.get('start_time', time.time())) / 3600,
                'timestamp': lock_data.get('timestamp', 'unknown'),
            }

            if chunk_id.startswith('L1_'):
                active_work['level1_chunks'].append(chunk_info)
            elif chunk_id.startswith('L2_'):
                active_work['level2_chunks'].append(chunk_info)

            active_work['total_active_machines'].add(machine_id)
            dist = active_work['work_distribution'].setdefault(machine_id, {'level1': 0, 'level2': 0})
            if chunk_id.startswith('L1_'):
                dist['level1'] += 1
            elif chunk_id.startswith('L2_'):
                dist['level2'] += 1

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            continue

    active_work['total_active_machines'] = list(active_work['total_active_machines'])
    active_work['total_active_chunks'] = (
        len(active_work['level1_chunks']) + len(active_work['level2_chunks'])
    )
    return active_work


def _analyze_machine_performance(output_dir: Path) -> Dict:
    """Parse per-machine progress_summary_*.json files for performance stats."""
    machine_stats = {}

    for summary_file in output_dir.glob("progress_summary_*.json"):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            machine_id = summary.get('machine_id', 'unknown')
            session_stats = summary.get('session_stats', {})
            surfaces = session_stats.get('samples_computed', session_stats.get('surfaces_computed', 0))
            compute_time = session_stats.get('total_computation_time', 0)

            machine_stats[machine_id] = {
                'last_update': summary.get('timestamp', 'unknown'),
                'total_surfaces_computed': surfaces,
                'total_chunks_processed': session_stats.get('chunks_processed', 0),
                'total_computation_time_hours': compute_time / 3600,
                'completion_rate': summary.get('completion_rate', 0),
                'surfaces_per_hour': surfaces / (compute_time / 3600) if compute_time > 0 else None,
            }
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            continue

    return machine_stats


def _analyze_computation_timing(
    output_dir: Path, level1_status: Dict, level2_status: Dict,
    file_pattern: Optional[str] = None,
) -> Dict:
    """Estimate throughput and completion time from recent file activity.

    file_pattern: glob pattern to match output files (default: sample files).
    """
    timing_analysis: Dict = {
        'surfaces_per_hour': 0,
        'estimated_completion': {},
        'recent_activity': {},
        'bottlenecks': [],
    }

    if not output_dir or not output_dir.exists():
        return timing_analysis

    cutoff_time = time.time() - 3600
    recent_count = 0
    if file_pattern:
        # Pipeline mode: glob for averaged surface files
        try:
            for entry in output_dir.glob(file_pattern):
                try:
                    if entry.stat().st_mtime > cutoff_time:
                        recent_count += 1
                except OSError:
                    continue
        except OSError:
            pass
    else:
        with os.scandir(output_dir) as entries:
            for entry in entries:
                if re.match(r"(surface|samples)_sf1_.*\.pkl", entry.name):
                    try:
                        if entry.stat().st_mtime > cutoff_time:
                            recent_count += 1
                    except OSError:
                        continue

    timing_analysis['recent_activity']['active'] = recent_count > 0
    if recent_count > 0:
        timing_analysis['surfaces_per_hour'] = recent_count
        timing_analysis['recent_activity']['last_hour_surfaces'] = recent_count

        rate = recent_count
        l1_rem = level1_status['missing_count']
        l2_rem = level2_status['missing_count']
        l1_h = l1_rem / rate
        l2_h = l2_rem / rate

        timing_analysis['estimated_completion'] = {
            'level1': {
                'remaining_surfaces': l1_rem,
                'estimated_hours': l1_h,
                'estimated_completion_time': (datetime.now() + timedelta(hours=l1_h)).isoformat(),
            },
            'level2': {
                'remaining_surfaces': l2_rem,
                'estimated_hours': l2_h,
                'estimated_completion_time': (
                    datetime.now() + timedelta(hours=l1_h + l2_h)
                ).isoformat(),
            },
            'total_project': {
                'total_remaining': l1_rem + l2_rem,
                'estimated_hours': l1_h + l2_h,
                'estimated_completion_time': (
                    datetime.now() + timedelta(hours=l1_h + l2_h)
                ).isoformat(),
            },
        }

    return timing_analysis




def print_comprehensive_status(
    output_dir: Path,
    show_details: bool = True,
    averaged_surfaces_dir: Optional[Path] = None,
    bundle_dir: Optional[Path] = None,
) -> None:
    """Print a formatted status report to stdout."""
    status = get_comprehensive_status(
        output_dir, averaged_surfaces_dir=averaged_surfaces_dir, bundle_dir=bundle_dir
    )

    print("=" * 70)
    print("GRID COMPUTATION STATUS")
    print("=" * 70)
    print(f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if status.get('pipeline_mode'):
        print(f"Mode   : pipeline")
        print(f"Surfaces: {status['averaged_surfaces_dir']}")
        if status.get('bundle_dir'):
            print(f"Bundles : {status['bundle_dir']}")
    else:
        print(f"Folder : {status['output_dir']}")

    level1 = status['level1']
    level2 = status['level2']
    pipeline_mode = status.get('pipeline_mode', False)

    print(f"\nPROGRESS")
    print("-" * 40)
    bar = "█" * int(level1['completion_rate'] // 5) + "░" * (20 - int(level1['completion_rate'] // 5))
    l1_label = "Averaged surfaces" if pipeline_mode else "Level 1 (coarse) "
    print(f"  {l1_label}: {level1['completed_count']:>6,}/{level1['expected_total']:,} "
          f"({level1['completion_rate']:.1f}%)  [{bar}]  {level1['missing_count']:,} remaining")
    if not pipeline_mode:
        bar2 = "█" * int(level2['completion_rate'] // 5) + "░" * (20 - int(level2['completion_rate'] // 5))
        print(f"  Level 2 (fine)  : {level2['completed_count']:>6,}/{level2['expected_total']:,} "
              f"({level2['completion_rate']:.1f}%)  [{bar2}]  {level2['missing_count']:,} remaining")
        total_done = level1['completed_count'] + level2['completed_count']
        total_exp = level1['expected_total'] + level2['expected_total']
        if total_exp > 0:
            print(f"  Total           : {total_done:>6,}/{total_exp:,} "
                  f"({total_done / total_exp * 100:.1f}%)")

    active = status['active_work']
    print(f"\nACTIVE WORK")
    print("-" * 40)
    print(f"  Machines : {len(active['total_active_machines'])} "
          f"— {', '.join(active['total_active_machines']) or 'none'}")
    print(f"  Chunks   : {active['total_active_chunks']} "
          f"(L1: {len(active['level1_chunks'])}, L2: {len(active['level2_chunks'])})")

    if show_details and active['total_active_chunks'] > 0:
        print(f"\nACTIVE CHUNKS")
        print("-" * 40)
        for chunk in active['level1_chunks'] + active['level2_chunks']:
            level_tag = "L1" if chunk['chunk_id'].startswith('L1_') else "L2"
            print(f"  [{level_tag}] {chunk['chunk_id']}  machine={chunk['machine_id']}"
                  f"  running={chunk['running_hours']:.1f}h")

    machine_perf = status['machine_performance']
    active_machines = set(active['total_active_machines'])
    print(f"\nMACHINE RATES")
    print("-" * 40)
    total_rate = 0.0
    all_machines = active_machines | set(machine_perf.keys())
    for mid in sorted(all_machines):
        if mid in machine_perf:
            s = machine_perf[mid]
            rate = s['surfaces_per_hour']
            rate_str = f"{rate:,.0f} surf/h" if rate else "n/a"
            total_rate += rate or 0.0
            active_tag = " [active]" if mid in active_machines else ""
            print(f"  {mid}{active_tag}: {s['total_surfaces_computed']:,} surfaces, "
                  f"{s['total_computation_time_hours']:.1f}h compute, {rate_str}")
        else:
            print(f"  {mid} [active, no summary yet]")
    if len(all_machines) > 1 and total_rate > 0:
        print(f"  Total rate: {total_rate:,.0f} surf/h")
    if total_rate > 0:
        l1_rem = level1['missing_count']
        l2_rem = level2['missing_count']
        if l1_rem > 0:
            l1_h = l1_rem / total_rate
            l1_eta = datetime.now() + timedelta(hours=l1_h)
            print(f"  Est. L1 done:   {l1_eta.strftime('%Y-%m-%d %H:%M')} "
                  f"({l1_h:.1f}h, {l1_rem:,} remaining)")
        else:
            print(f"  Est. L1 done:   complete")
        total_rem = l1_rem + l2_rem
        if total_rem > 0:
            total_h = total_rem / total_rate
            total_eta = datetime.now() + timedelta(hours=total_h)
            print(f"  Est. L1+L2 done:{total_eta.strftime('%Y-%m-%d %H:%M')} "
                  f"({total_h:.1f}h, {total_rem:,} remaining)")
        else:
            print(f"  Est. L1+L2 done: complete")
    if not all_machines:
        print("  No machine data yet.")

    print("=" * 70)


def monitor_status_loop(output_dir: Path, refresh_seconds: int = 60,
                        show_details: bool = False,
                        averaged_surfaces_dir: Optional[Path] = None,
                        bundle_dir: Optional[Path] = None) -> None:
    """Refresh status on a fixed interval until Ctrl-C."""
    print(f"Continuous monitoring, refresh every {refresh_seconds}s. Ctrl-C to stop.")
    try:
        while True:
            print("\033[2J\033[H", end="")
            print_comprehensive_status(output_dir, show_details=show_details,
                                       averaged_surfaces_dir=averaged_surfaces_dir,
                                       bundle_dir=bundle_dir)
            print(f"\nRefreshing in {refresh_seconds}s...")
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def export_status_json(output_dir: Path, output_file: Optional[str] = None,
                       averaged_surfaces_dir: Optional[Path] = None,
                       bundle_dir: Optional[Path] = None) -> str:
    """Write current status dict to a JSON file and return its path."""
    status = get_comprehensive_status(
        output_dir, averaged_surfaces_dir=averaged_surfaces_dir, bundle_dir=bundle_dir
    )
    if output_file is None:
        output_file = f"grid_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"Status exported to: {output_file}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor likelihood surface grid computation progress.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--samples-dir', type=str, default=None,
                        help='Explicit path to samples output directory (overrides auto-detection)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base results directory (default: results)')
    parser.add_argument('--averaged-surfaces-dir', type=str, default=None,
                        help='Path to averaged surfaces directory (enables pipeline mode tracking)')
    parser.add_argument('--bundle-dir', type=str, default=None,
                        help='Path to local surface bundle directory for pipeline bundle mode')
    # Parameters mirroring simulated_samples_grid.py for auto folder detection
    parser.add_argument('--n-simulations', type=int, default=10000)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--algorithm', type=str.upper, default='EM', choices=['EM', 'VBEM', 'VBEM_MIX'])
    parser.add_argument('--diagonal-covariance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fix-weights', action='store_true', default=False)
    # Display / mode options
    parser.add_argument('--monitor', action='store_true',
                        help='Continuous monitoring mode')
    parser.add_argument('--refresh', type=int, default=60,
                        help='Refresh interval in seconds (default: 60)')
    parser.add_argument('--details', '--detailed', action='store_true',
                        help='Show per-chunk details')
    parser.add_argument('--export', action='store_true',
                        help='Export status to a JSON file and exit')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename for --export (default: auto-timestamped)')
    parser.add_argument('--quiet', action='store_true',
                        help='Print a compact one-line summary and exit')

    args = parser.parse_args()

    output_dir = _resolve_samples_dir(args)
    avg_dir = Path(args.averaged_surfaces_dir) if args.averaged_surfaces_dir else None
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else None

    if not output_dir.exists():
        print(f"Samples directory not found: {output_dir}")
        sys.exit(1)

    if args.export:
        export_status_json(output_dir, args.output, averaged_surfaces_dir=avg_dir,
                           bundle_dir=bundle_dir)
        return

    if args.monitor or args.refresh != 60:
        monitor_status_loop(output_dir, refresh_seconds=args.refresh,
                            show_details=args.details, averaged_surfaces_dir=avg_dir,
                            bundle_dir=bundle_dir)
        return

    if args.quiet:
        status = get_comprehensive_status(output_dir, averaged_surfaces_dir=avg_dir,
                                          bundle_dir=bundle_dir)
        l1 = status['level1']
        l2 = status['level2']
        if status.get('pipeline_mode'):
            pct = l1['completion_rate']
            print(f"Pipeline: {l1['completed_count']}/{l1['expected_total']} ({pct:.1f}%)  "
                  f"Active: {len(status['active_work']['total_active_machines'])} machines")
        else:
            total_done = l1['completed_count'] + l2['completed_count']
            total_exp  = l1['expected_total'] + l2['expected_total']
            pct = total_done / total_exp * 100 if total_exp else 0.0
            print(f"Overall: {total_done}/{total_exp} ({pct:.1f}%)  "
                  f"L1: {l1['completion_rate']:.1f}%  L2: {l2['completion_rate']:.1f}%  "
                  f"Active: {len(status['active_work']['total_active_machines'])} machines")
        return

    print_comprehensive_status(output_dir, show_details=args.details,
                               averaged_surfaces_dir=avg_dir, bundle_dir=bundle_dir)

if __name__ == "__main__":
    main()
