#!/usr/bin/env python3
"""Reconcile a Redis completed-surface registry from remote bundle manifests."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

from lock_backend import load_env, make_lock_backend
from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="Reconcile Redis completion registry from object storage.")
    parser.add_argument("--registry", default=os.getenv("COMPLETION_REGISTRY"),
                        help="Redis completion registry name")
    parser.add_argument("--replace", action="store_true",
                        help="Clear registry first, then rebuild exactly from storage")
    parser.add_argument("--clear-extra", action="store_true",
                        help="Remove Redis IDs that are not present in storage manifests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without mutating Redis")
    args = parser.parse_args()

    if not args.registry:
        print("Missing --registry or COMPLETION_REGISTRY.", file=sys.stderr)
        return 2

    config = ObjectStoreConfig.from_env()
    if config is None:
        print("Missing S3_BUCKET or OBJECT_STORE_BUCKET in the environment.", file=sys.stderr)
        return 2

    store = SurfaceObjectStore(config)
    storage_ids = store.list_completed_surface_ids()

    backend = make_lock_backend("redis")
    redis_ids = backend.completed_members(args.registry)

    missing_in_redis = storage_ids - redis_ids
    extra_in_redis = redis_ids - storage_ids

    print("Redis Reconciliation")
    print(f"  Registry        : {args.registry}")
    print(f"  Storage surfaces: {len(storage_ids):,}")
    print(f"  Redis surfaces  : {len(redis_ids):,}")
    print(f"  Add to Redis    : {len(missing_in_redis):,}")
    print(f"  Extra in Redis  : {len(extra_in_redis):,}")

    if args.dry_run:
        print("Dry run: no Redis changes made.")
        return 0

    if args.replace:
        backend.clear_completed(args.registry, confirm=True)
        redis_ids = set()
        missing_in_redis = storage_ids
        extra_in_redis = set()

    for surface_id in sorted(missing_in_redis):
        backend.mark_completed(args.registry, surface_id)

    removed = 0
    if args.clear_extra:
        for surface_id in sorted(extra_in_redis):
            backend.remove_completed(args.registry, surface_id)
            removed += 1

    final_count = len(backend.completed_members(args.registry))
    print(f"  Added           : {len(missing_in_redis):,}")
    print(f"  Removed         : {removed:,}")
    print(f"  Final Redis     : {final_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
