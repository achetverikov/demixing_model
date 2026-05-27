#!/usr/bin/env python3
"""Download surface bundle objects from S3-compatible storage."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync remote surface bundles locally.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Local directory for downloaded bundle and manifest files")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing local files")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded without writing files")
    args = parser.parse_args()

    config = ObjectStoreConfig.from_env()
    if config is None:
        print("Missing S3_BUCKET or OBJECT_STORE_BUCKET in the environment.", file=sys.stderr)
        return 2

    store = SurfaceObjectStore(config)
    names = store.list_bundle_object_names()
    if not names:
        print("No remote surface_bundle_*.pkl.gz or manifest objects found.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0
    bytes_downloaded = 0

    for name in names:
        target = args.output_dir / name
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.dry_run:
            print(f"Would download {name} -> {target}")
            continue
        tmp = target.with_suffix(target.suffix + ".tmp")
        store.download_file(name, tmp)
        tmp.replace(target)
        downloaded += 1
        bytes_downloaded += target.stat().st_size

    print("Bundle sync complete")
    print(f"  Remote objects    : {len(names)}")
    print(f"  Downloaded        : {downloaded}")
    print(f"  Skipped existing  : {skipped}")
    print(f"  Bytes downloaded  : {bytes_downloaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
