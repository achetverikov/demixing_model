#!/usr/bin/env python3
"""List or delete all objects under the configured object-store prefix."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or delete objects under S3_PREFIX.")
    parser.add_argument("--yes", action="store_true",
                        help="Actually delete objects. Without this, only a dry-run is shown.")
    parser.add_argument("--max-delete", type=int, default=None,
                        help="Optional safety cap on number of objects to delete")
    args = parser.parse_args()

    config = ObjectStoreConfig.from_env()
    if config is None:
        print("Missing S3_BUCKET or OBJECT_STORE_BUCKET in the environment.", file=sys.stderr)
        return 2
    if not config.prefix:
        print("Refusing to operate on bucket root. Set S3_PREFIX/OBJECT_STORE_PREFIX.", file=sys.stderr)
        return 2

    store = SurfaceObjectStore(config)
    objects = store.list_objects()
    total_bytes = sum(obj["size"] for obj in objects)

    print("Remote Prefix Cleanup")
    print(f"  Bucket : {config.bucket}")
    print(f"  Prefix : {config.prefix}")
    print(f"  Objects: {len(objects):,}")
    print(f"  Bytes  : {total_bytes:,}")

    for obj in objects[:20]:
        print(f"  {obj['key']} ({obj['size']} bytes)")
    if len(objects) > 20:
        print(f"  ... {len(objects) - 20:,} more")

    if not args.yes:
        print("Dry run only. Pass --yes to delete these objects.")
        return 0

    if args.max_delete is not None and len(objects) > args.max_delete:
        print(
            f"Refusing to delete {len(objects):,} objects because --max-delete={args.max_delete}.",
            file=sys.stderr,
        )
        return 1

    for obj in objects:
        store.delete_key(obj["key"])
    print(f"Deleted {len(objects):,} objects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
