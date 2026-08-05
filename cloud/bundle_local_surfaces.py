#!/usr/bin/env python3
"""Bundle a local directory of averaged surfaces into chunk bundles.

The grid workers in ``surface_computation/simulated_samples_grid.py`` write
surfaces as ``surface_bundle_<chunk_id>.pkl.gz`` plus a ``.manifest.json``
sidecar, but that writer is wired into the worker's param groups and completion
registry, so it cannot repack a directory that already holds loose
``averaged_sf1_*_sf2_*_sp_*.pkl`` files.  This script produces byte-identical
payloads in that same format from such a directory, so both distribution sets
can be shipped (and consumed) the same way.

Consumers of the output:
  - ``neural_network_optimization/mirror_aware_training.py`` (auto-detects bundles)
  - ``shared.utils.ensure_averaged_surface_file`` (materialises one surface on demand)
  - ``cloud/unpack_surface_bundles.py`` (expands back to loose ``.pkl``)

Example:
    python cloud/bundle_local_surfaces.py \\
        --input-dir results/averaged_surfaces_10k_100samples_circular \\
        --output-dir results/averaged_surfaces_10k_100samples_circular_bundles \\
        --n-simulations 10000 --n-samples 100 --random-seed 42
"""

import argparse
import gzip
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

SURFACE_RE = re.compile(r"^averaged_sf1_([\d.]+)_sf2_([\d.]+)_sp_([\d.]+)\.pkl$")


def _parse(name: str) -> Tuple[str, str, str]:
    m = SURFACE_RE.match(name)
    if m is None:
        raise ValueError(f"Not an averaged-surface filename: {name}")
    return m.group(1), m.group(2), m.group(3)


def collect_surfaces(input_dir: Path) -> List[Path]:
    """Return surface files ordered by (sf1, sf2, sp) so chunking is deterministic."""
    files = [p for p in input_dir.glob("averaged_sf1_*_sf2_*_sp_*.pkl")
             if SURFACE_RE.match(p.name)]
    return sorted(files, key=lambda p: tuple(float(v) for v in _parse(p.name)))


def write_bundle(paths: List[Path], chunk_id: str, output_dir: Path,
                 params: dict, source_dir: Path, compresslevel: int,
                 overwrite: bool) -> Tuple[Path, dict, str]:
    bundle_path = output_dir / f"surface_bundle_{chunk_id}.pkl.gz"
    manifest_path = output_dir / f"surface_bundle_{chunk_id}.manifest.json"

    surfaces = {p.name: p.read_bytes() for p in paths}
    surface_ids = ["|".join(_parse(p.name)) for p in paths]
    manifest = {
        "chunk_id": chunk_id,
        "machine_id": "local-repack",
        "surface_count": len(surface_ids),
        "surface_ids": surface_ids,
        "surface_files": sorted(surfaces),
        "parameters": params,
        "created_at": datetime.now().isoformat(),
        # Provenance for surfaces that were repacked rather than freshly computed:
        # the grid worker records the commit that produced them, which this script
        # cannot know, so `parameters.git_commit` is null and these fields say why.
        "repacked_from": str(source_dir),
        "repacked_by": "cloud/bundle_local_surfaces.py",
    }

    if bundle_path.exists() and not overwrite:
        return bundle_path, manifest, "skipped"

    tmp_bundle = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    with gzip.open(tmp_bundle, "wb", compresslevel=compresslevel) as f:
        pickle.dump({"manifest": manifest, "surfaces": surfaces},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_bundle.replace(bundle_path)

    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_manifest.replace(manifest_path)
    return bundle_path, manifest, "written"


def verify_bundle(bundle_path: Path, source_dir: Path) -> int:
    """Re-read a bundle and byte-compare every surface against its source file."""
    with gzip.open(bundle_path, "rb") as f:
        payload = pickle.load(f)
    checked = 0
    for filename, content in payload["surfaces"].items():
        source = source_dir / filename
        if not source.exists():
            raise AssertionError(f"{bundle_path.name}: source missing for {filename}")
        if source.read_bytes() != content:
            raise AssertionError(f"{bundle_path.name}: payload differs for {filename}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repack loose averaged surfaces into chunk bundles.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory holding loose averaged_sf1_*_sf2_*_sp_*.pkl files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write bundles and manifests into")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Surfaces per bundle (default 50, matching the grid workers)")
    parser.add_argument("--chunk-prefix", default="repack",
                        help="Chunk-id prefix; bundles are named "
                             "surface_bundle_<prefix>_chunk_<start>_<end>.pkl.gz")
    parser.add_argument("--compresslevel", type=int, default=9,
                        help="gzip level (default 9, matching the grid workers)")
    parser.add_argument("--n-simulations", type=int, required=True,
                        help="Simulations per surface, recorded in the manifest")
    parser.add_argument("--n-samples", type=int, required=True,
                        help="Observer evidence samples per item, recorded in the manifest")
    parser.add_argument("--random-seed", type=int, required=True,
                        help="Seed the surfaces were generated with, recorded in the manifest")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rewrite bundles that already exist")
    parser.add_argument("--verify", action="store_true",
                        help="Re-read each bundle and byte-compare against the sources")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N surfaces (for a quick trial run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the chunk plan without writing anything")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    files = collect_surfaces(args.input_dir)
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        print(f"No averaged surfaces found in {args.input_dir}", file=sys.stderr)
        return 1

    n_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    print(f"Surfaces        : {len(files)}")
    print(f"Chunk size      : {args.chunk_size}")
    print(f"Bundles to write: {n_chunks}")
    if args.dry_run:
        for i in range(0, len(files), args.chunk_size):
            end = min(i + args.chunk_size, len(files))
            print(f"  surface_bundle_{args.chunk_prefix}_chunk_{i:05d}_{end:05d}.pkl.gz  "
                  f"{files[i].name} .. {files[end - 1].name}")
        return 0

    params = {
        "n_simulations": args.n_simulations,
        "n_samples": args.n_samples,
        "random_seed": args.random_seed,
        "git_commit": None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = verified = 0
    total_bytes = 0
    for idx, i in enumerate(range(0, len(files), args.chunk_size), start=1):
        chunk = files[i:i + args.chunk_size]
        end = i + len(chunk)
        chunk_id = f"{args.chunk_prefix}_chunk_{i:05d}_{end:05d}"
        bundle_path, _, status = write_bundle(
            chunk, chunk_id, args.output_dir, params, args.input_dir,
            args.compresslevel, args.overwrite)
        if status == "written":
            written += 1
            total_bytes += bundle_path.stat().st_size
        else:
            skipped += 1
        if args.verify:
            verified += verify_bundle(bundle_path, args.input_dir)
        if idx % 25 == 0 or idx == n_chunks:
            print(f"  [{idx}/{n_chunks}] {status} {bundle_path.name}", flush=True)

    print("Bundling complete")
    print(f"  Bundles written  : {written}")
    print(f"  Bundles skipped  : {skipped}")
    print(f"  Bytes written    : {total_bytes} ({total_bytes / 1e9:.2f} GB)")
    if args.verify:
        print(f"  Surfaces verified: {verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
