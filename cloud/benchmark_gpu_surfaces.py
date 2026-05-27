#!/usr/bin/env python3
"""Benchmark actual averaged-surface compute time on one GPU.

The benchmark uses the real pipeline compute path, but disables Redis and object
storage while timing so the numbers reflect sampling/fitting/KDE work rather
than coordination or upload overhead.
"""

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))


def _collect_hardware_info() -> Dict:
    """Collect GPU and system hardware info for anomaly diagnosis."""
    info: Dict = {}

    if shutil.which("nvidia-smi"):
        fields = (
            "name,index,memory.total,memory.free,driver_version,"
            "compute_cap,temperature.gpu,power.draw,power.max_watts,"
            "clocks.gr,clocks.mem,ecc.mode.current"
        )
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
            keys = [k.strip() for k in fields.split(",")]
            gpus = []
            for line in out.strip().splitlines():
                vals = [v.strip() for v in line.split(",")]
                gpus.append(dict(zip(keys, vals)))
            info["nvidia_smi"] = gpus
        except Exception as exc:
            info["nvidia_smi_error"] = str(exc)

    # CPU / memory from /proc (Linux)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        model_names = list(dict.fromkeys(
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ))
        cpu_count = cpuinfo.count("processor\t:")
        info["cpu_model"] = model_names[0] if model_names else "unknown"
        info["cpu_count"] = cpu_count
    except Exception:
        pass

    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                info["ram_total_kb"] = int(line.split()[1])
                break
    except Exception:
        pass

    return info


@contextmanager
def without_object_store_env():
    keys = [
        "S3_BUCKET",
        "OBJECT_STORE_BUCKET",
        "S3_PREFIX",
        "OBJECT_STORE_PREFIX",
    ]
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    frac = position - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def _group_params(group: List[Tuple]) -> Dict[str, float]:
    _, _, _, sf1, sf2, sp, _ = group[0]
    return {
        "sd_feat1": float(min(sf1, sf2)),
        "sd_feat2": float(max(sf1, sf2)),
        "sd_spat": float(sp),
    }


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _upload_outputs(paths: Iterable[Path], prefix: str) -> List[str]:
    from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore

    old_prefix = os.environ.get("S3_PREFIX")
    os.environ["S3_PREFIX"] = prefix.strip("/")
    try:
        config = ObjectStoreConfig.from_env()
        if config is None:
            return []
        store = SurfaceObjectStore(config)
        return [store.upload_file(path) for path in paths]
    finally:
        if old_prefix is None:
            os.environ.pop("S3_PREFIX", None)
        else:
            os.environ["S3_PREFIX"] = old_prefix


def run_condition(args, n_samples: int) -> Tuple[Dict, List[Dict], List[Path]]:
    import jax
    import jax.numpy as jnp
    from surface_computation.lock_backend import FileLockBackend
    from surface_computation.simulated_samples_grid import ChunkedGridComputer
    from shared.utils import get_git_commit

    condition = f"{args.n_simulations}x{n_samples}"
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{_sanitize(args.gpu_name or 'unknown_gpu')}_{condition}_{args.seed}"
    )
    out_dir = Path(args.output_dir) / run_id
    surfaces_dir = out_dir / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    with without_object_store_env():
        computer = ChunkedGridComputer(
            machine_id=args.machine_id,
            grid_level=args.grid_level,
            save_full_results=False,
            save_csv=False,
            diagonal_covariance=args.diagonal_covariance,
            fix_weights=args.fix_weights,
            algorithm=args.algorithm,
            lock_backend=FileLockBackend(out_dir / "locks"),
            lock_ttl=1800,
            pipeline=True,
            averaged_surfaces_dir=surfaces_dir,
            completion_registry=None,
            bundle_output=False,
            save_samples=False,
            bias_bandwidth=args.bias_bandwidth,
            feat_bandwidth=args.feat_bandwidth,
        )

    eligible_indices = [
        idx for idx, group in enumerate(computer.param_groups)
        if len(group) >= 2
    ]
    if len(eligible_indices) < args.surface_count + 1:
        raise RuntimeError("Not enough parameter groups for benchmark sample")

    rng = random.Random(args.seed)
    selected_indices = rng.sample(eligible_indices, args.surface_count + 1)
    warmup_index = selected_indices[0]
    timed_indices = selected_indices[1:]
    feat_diff_vals = computer.config.create_grid("feat_diff") if hasattr(computer, "config") else None
    if feat_diff_vals is None:
        from surface_computation.simulated_samples_grid import config
        feat_diff_vals = config.create_grid("feat_diff")
    feat_diff_vals = feat_diff_vals[feat_diff_vals > 0]

    samples_params = {
        "feat_diff_step": 2,
        "mu1_bias_step": 2,
        "mu2_bias_step": 6,
        "feat_diff_range": (4, 180),
        "mu1_bias_range": (-180, 180),
        "mu2_bias_range": (-498, 498),
        "n_simulations": args.n_simulations,
        "n_samples": n_samples,
        "random_seed": args.random_seed,
    }

    print(f"Benchmark condition {condition}: warmup group {warmup_index}")
    warmup_start = time.perf_counter()
    warmup_status = computer._process_group_pipeline(
        computer.param_groups[warmup_index], samples_params, feat_diff_vals
    )
    warmup_seconds = time.perf_counter() - warmup_start
    if warmup_status != "saved":
        raise RuntimeError(f"Warmup failed with status {warmup_status}")

    rows: List[Dict] = []
    for i, group_index in enumerate(timed_indices, start=1):
        group = computer.param_groups[group_index]
        params = _group_params(group)
        print(f"Timed {i}/{len(timed_indices)} group {group_index}: {params}")
        started = time.perf_counter()
        status = computer._process_group_pipeline(group, samples_params, feat_diff_vals)
        seconds = time.perf_counter() - started
        rows.append({
            "run_id": run_id,
            "condition": condition,
            "gpu_name": args.gpu_name,
            "offer_id": args.offer_id,
            "price_per_hour": args.price_per_hour,
            "n_simulations": args.n_simulations,
            "n_samples": n_samples,
            "surface_index": i,
            "group_index": group_index,
            **params,
            "timed_seconds": seconds,
            "status": status,
        })
        if status != "saved":
            raise RuntimeError(f"Timed surface {i} failed with status {status}")

    times = [row["timed_seconds"] for row in rows]
    mean_seconds = statistics.mean(times)
    median_seconds = statistics.median(times)
    surfaces_per_hour = 3600 / mean_seconds if mean_seconds > 0 else float("nan")
    cost_per_1000 = (
        args.price_per_hour / surfaces_per_hour * 1000
        if args.price_per_hour is not None and surfaces_per_hour > 0
        else None
    )

    summary = {
        "run_id": run_id,
        "condition": condition,
        "gpu_name": args.gpu_name,
        "offer_id": args.offer_id,
        "price_per_hour": args.price_per_hour,
        "machine_id": args.machine_id,
        "n_simulations": args.n_simulations,
        "n_samples": n_samples,
        "surface_count": len(rows),
        "seed": args.seed,
        "warmup_group_index": warmup_index,
        "warmup_surface": _group_params(computer.param_groups[warmup_index]),
        "warmup_seconds": warmup_seconds,
        "warmup_status": warmup_status,
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "p90_seconds": _percentile(times, 0.90),
        "min_seconds": min(times),
        "max_seconds": max(times),
        "surfaces_per_hour": surfaces_per_hour,
        "cost_per_1000_surfaces": cost_per_1000,
        "estimated_l1_cost": cost_per_1000 * 4.2 if cost_per_1000 is not None else None,
        "estimated_full_l2_cost": cost_per_1000 * 32.8 if cost_per_1000 is not None else None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "jax_devices": [str(device) for device in jax.devices()],
        "diagonal_covariance": args.diagonal_covariance,
        "fix_weights": args.fix_weights,
        "algorithm": args.algorithm,
        "hardware": _collect_hardware_info(),
    }

    csv_path = out_dir / f"{run_id}.csv"
    json_path = out_dir / f"{run_id}.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(json.dumps(summary, indent=2))
    return summary, rows, [csv_path, json_path]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark GPU surface compute time.")
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--n-samples", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--surface-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260527,
                        help="Seed for selecting benchmark surface groups")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Simulation random seed")
    parser.add_argument("--grid-level", type=int, default=1, choices=[1, 2])
    parser.add_argument("--machine-id", default=os.getenv("HOSTNAME", "benchmark"))
    parser.add_argument("--gpu-name", default=os.getenv("BENCHMARK_GPU_NAME", "unknown"))
    parser.add_argument("--offer-id", default=os.getenv("BENCHMARK_OFFER_ID"))
    parser.add_argument("--price-per-hour", type=float,
                        default=float(os.getenv("BENCHMARK_PRICE_PER_HOUR", "nan")))
    parser.add_argument("--output-dir", default="results/gpu_benchmarks")
    parser.add_argument("--upload-prefix", default=os.getenv(
        "BENCHMARK_S3_PREFIX", "demixing/benchmarks/gpu_surface_timing"
    ))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--diagonal-covariance", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--fix-weights", action="store_true", default=False)
    parser.add_argument("--algorithm", type=str.upper,
                        choices=["EM", "VBEM", "VBEM_MIX"], default="EM")
    parser.add_argument("--bias-bandwidth", type=float, default=0.075)
    parser.add_argument("--feat-bandwidth", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_summaries = []
    upload_paths = []
    for n_samples in args.n_samples:
        summary, _, paths = run_condition(args, n_samples)
        all_summaries.append(summary)
        upload_paths.extend(paths)

    aggregate_path = Path(args.output_dir) / (
        f"gpu_benchmark_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps({"summaries": all_summaries}, indent=2))
    upload_paths.append(aggregate_path)

    if not args.no_upload:
        uploaded = _upload_outputs(upload_paths, args.upload_prefix)
        print("Uploaded benchmark outputs:")
        for key in uploaded:
            print(f"  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
