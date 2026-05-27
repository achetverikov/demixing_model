#!/usr/bin/env python3
"""Unpack chunk surface bundles into the standard averaged-surface directory."""

import argparse
import gzip
import json
import pickle
from pathlib import Path


def _load_bundle(bundle_path: Path) -> dict:
    with gzip.open(bundle_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "surfaces" not in data:
        raise ValueError(f"Invalid bundle format: {bundle_path}")
    if not isinstance(data["surfaces"], dict):
        raise ValueError(f"Invalid surfaces payload in {bundle_path}")
    return data


def _write_atomic(path: Path, content: bytes, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        return "skipped"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(content)
    tmp.replace(path)
    return "written"


def unpack_bundles(bundle_dir: Path, output_dir: Path, overwrite: bool = False,
                   require_manifests: bool = False) -> dict:
    bundle_paths = sorted(bundle_dir.glob("surface_bundle_*.pkl.gz"))
    if not bundle_paths:
        raise FileNotFoundError(f"No surface_bundle_*.pkl.gz files found in {bundle_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "bundles": 0,
        "surfaces_seen": 0,
        "surfaces_written": 0,
        "surfaces_skipped": 0,
        "manifest_mismatches": [],
        "errors": [],
    }

    for bundle_path in bundle_paths:
        stats["bundles"] += 1
        try:
            data = _load_bundle(bundle_path)
            surfaces = data["surfaces"]

            manifest_path = bundle_path.with_name(
                bundle_path.name.removesuffix(".pkl.gz") + ".manifest.json"
            )
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                expected_files = set(manifest.get("surface_files", []))
                actual_files = set(surfaces)
                if expected_files and expected_files != actual_files:
                    stats["manifest_mismatches"].append(bundle_path.name)
            elif require_manifests:
                raise FileNotFoundError(f"Missing manifest for {bundle_path.name}")

            for filename, content in surfaces.items():
                if not filename.startswith("averaged_sf1_") or not filename.endswith(".pkl"):
                    raise ValueError(f"Unexpected surface filename in {bundle_path.name}: {filename}")
                stats["surfaces_seen"] += 1
                result = _write_atomic(output_dir / filename, content, overwrite=overwrite)
                if result == "written":
                    stats["surfaces_written"] += 1
                else:
                    stats["surfaces_skipped"] += 1
        except Exception as e:
            stats["errors"].append({"bundle": bundle_path.name, "error": str(e)})

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Unpack surface bundle files.")
    parser.add_argument("--bundle-dir", required=True, type=Path,
                        help="Directory containing surface_bundle_*.pkl.gz files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory where averaged_sf1_*.pkl files will be written")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing averaged surface files")
    parser.add_argument("--require-manifests", action="store_true",
                        help="Fail bundles that do not have a local manifest JSON")
    args = parser.parse_args()

    stats = unpack_bundles(
        args.bundle_dir,
        args.output_dir,
        overwrite=args.overwrite,
        require_manifests=args.require_manifests,
    )

    print("Unpack complete")
    print(f"  Bundles          : {stats['bundles']}")
    print(f"  Surfaces seen    : {stats['surfaces_seen']}")
    print(f"  Surfaces written : {stats['surfaces_written']}")
    print(f"  Surfaces skipped : {stats['surfaces_skipped']}")
    if stats["manifest_mismatches"]:
        print(f"  Manifest mismatch: {len(stats['manifest_mismatches'])}")
    if stats["errors"]:
        print(f"  Errors           : {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"    {err['bundle']}: {err['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
