#!/usr/bin/env python3
"""Pack surface bundles into release tarballs with checksums.

GitHub caps a single release asset at 2 GB, and both averaged-surface sets are
larger than that, so each set ships as several tarballs.  Splitting happens at
bundle boundaries rather than with ``split(1)``: every tarball unpacks on its own
into a directory the loaders can read, so a partial or resumed download still
yields usable surfaces instead of an inert fragment.

Produces, in the output directory:
  <name>.part01.tar .. <name>.partNN.tar   each below --max-bytes
  SHA256SUMS                               checksums for all parts
  <name>.index.json                        part -> bundle mapping and totals

Example:
    python cloud/package_surface_release.py \\
        --input-dir results/averaged_surfaces_10k_100samples_circular_bundles \\
        --output-dir release/surfaces_100samples \\
        --name averaged_surfaces_10k_100samples_circular
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import List

GIB = 1024 ** 3


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def plan_parts(bundles: List[Path], max_bytes: int) -> List[List[Path]]:
    """Greedily group bundles (with their manifests) into parts under max_bytes."""
    parts: List[List[Path]] = []
    current: List[Path] = []
    current_size = 0
    for bundle in bundles:
        manifest = bundle.with_name(bundle.name.removesuffix(".pkl.gz") + ".manifest.json")
        size = bundle.stat().st_size + (manifest.stat().st_size if manifest.exists() else 0)
        if size > max_bytes:
            raise ValueError(
                f"{bundle.name} is {size} bytes, larger than --max-bytes {max_bytes}; "
                "re-bundle with a smaller --chunk-size")
        if current and current_size + size > max_bytes:
            parts.append(current)
            current, current_size = [], 0
        current.append(bundle)
        current_size += size
    if current:
        parts.append(current)
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack surface bundles into release tarballs.")
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory of surface_bundle_*.pkl.gz files and manifests")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write tarball parts, SHA256SUMS and the index")
    parser.add_argument("--name", required=True,
                        help="Base name for the parts, e.g. the surface-set directory name")
    parser.add_argument("--max-bytes", type=int, default=int(1.8 * GIB),
                        help="Maximum size per part (default 1.8 GiB, under GitHub's 2 GB cap)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the part plan without writing tarballs")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 2

    bundles = sorted(args.input_dir.glob("surface_bundle_*.pkl.gz"))
    if not bundles:
        print(f"No surface_bundle_*.pkl.gz in {args.input_dir}", file=sys.stderr)
        return 1

    missing = [b.name for b in bundles
               if not b.with_name(b.name.removesuffix(".pkl.gz") + ".manifest.json").exists()]
    if missing:
        # Manifests are what let consumers index a set without decompressing it,
        # so shipping a bundle without one silently degrades every reader.
        print(f"Refusing to package: {len(missing)} bundles have no manifest "
              f"(first: {missing[0]})", file=sys.stderr)
        return 1

    parts = plan_parts(bundles, args.max_bytes)
    total_bytes = sum(b.stat().st_size for b in bundles)
    print(f"Bundles     : {len(bundles)}")
    print(f"Total size  : {total_bytes / 1e9:.2f} GB")
    print(f"Parts       : {len(parts)} (max {args.max_bytes / 1e9:.2f} GB each)")
    for i, part in enumerate(parts, start=1):
        size = sum(b.stat().st_size for b in part)
        print(f"  part{i:02d}: {len(part):4d} bundles  {size / 1e9:.2f} GB")
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {"name": args.name, "parts": [], "bundle_count": len(bundles),
             "total_bundle_bytes": total_bytes}
    checksums = []

    for i, part in enumerate(parts, start=1):
        part_path = args.output_dir / f"{args.name}.part{i:02d}.tar"
        tmp = part_path.with_suffix(part_path.suffix + ".tmp")
        # Bundles are already gzipped; an outer compression layer would cost time
        # for nothing, so the tarball is a plain container.
        with tarfile.open(tmp, "w") as tar:
            for bundle in part:
                manifest = bundle.with_name(
                    bundle.name.removesuffix(".pkl.gz") + ".manifest.json")
                tar.add(bundle, arcname=f"{args.name}/{bundle.name}")
                tar.add(manifest, arcname=f"{args.name}/{manifest.name}")
        tmp.replace(part_path)

        digest = sha256(part_path)
        checksums.append((digest, part_path.name))
        index["parts"].append({
            "part": i,
            "file": part_path.name,
            "bytes": part_path.stat().st_size,
            "bundle_count": len(part),
            "sha256": digest,
            "first_bundle": part[0].name,
            "last_bundle": part[-1].name,
        })
        print(f"  wrote {part_path.name}  {part_path.stat().st_size / 1e9:.2f} GB", flush=True)

    with open(args.output_dir / "SHA256SUMS", "w") as f:
        for digest, name in checksums:
            f.write(f"{digest}  {name}\n")
    with open(args.output_dir / f"{args.name}.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print("Packaging complete")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Verify with: cd {args.output_dir} && sha256sum -c SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
