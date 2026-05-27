#!/usr/bin/env python3
"""Report remote bundle and Redis completion status for cloud runs."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

from lock_backend import load_env, make_lock_backend
from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


def _expected_counts() -> dict:
    # L1: 10..200 step 10 -> 20 values. L2: 5..200 step 5 -> 40 values.
    def canonical_count(n_values: int) -> int:
        return n_values * n_values * (n_values + 1) // 2

    return {
        "l1": canonical_count(20),
        "l2_full": canonical_count(40),
    }


def _redis_completed(registry: str) -> set:
    backend = make_lock_backend("redis")
    return backend.completed_members(registry)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show cloud bundle/Redis status.")
    parser.add_argument("--registry", default=os.getenv("COMPLETION_REGISTRY"),
                        help="Redis completion registry name")
    parser.add_argument("--no-redis", action="store_true",
                        help="Skip Redis status even if REDIS_HOST is configured")
    args = parser.parse_args()

    load_env()
    config = ObjectStoreConfig.from_env()
    if config is None:
        print("Missing S3_BUCKET or OBJECT_STORE_BUCKET in the environment.", file=sys.stderr)
        return 2

    store = SurfaceObjectStore(config)
    objects = store.list_bundle_objects()
    manifest_count = sum(1 for obj in objects if obj["name"].endswith(".manifest.json"))
    bundle_count = sum(1 for obj in objects if obj["name"].endswith(".pkl.gz"))
    total_bytes = sum(obj["size"] for obj in objects)
    remote_completed = store.list_completed_surface_ids()

    expected = _expected_counts()

    print("Cloud Surface Status")
    print(f"  Bucket          : {config.bucket}")
    print(f"  Prefix          : {config.prefix or '(root)'}")
    print(f"  Bundle objects  : {bundle_count}")
    print(f"  Manifest objects: {manifest_count}")
    print(f"  Remote bytes    : {total_bytes:,}")
    print(f"  Remote surfaces : {len(remote_completed):,}")
    print(f"  L1 progress     : {len(remote_completed):,}/{expected['l1']:,} ({len(remote_completed) / expected['l1'] * 100:.1f}%)")
    print(f"  L2 full progress: {len(remote_completed):,}/{expected['l2_full']:,} ({len(remote_completed) / expected['l2_full'] * 100:.1f}%)")

    if args.no_redis:
        return 0

    if not os.getenv("REDIS_HOST"):
        print("  Redis           : not configured")
        return 0

    if not args.registry:
        print("  Redis           : configured, but no --registry or COMPLETION_REGISTRY")
        return 0

    try:
        redis_completed = _redis_completed(args.registry)
    except Exception as e:
        print(f"  Redis           : error reading registry {args.registry}: {e}")
        return 1

    missing_in_redis = remote_completed - redis_completed
    extra_in_redis = redis_completed - remote_completed
    print(f"  Redis registry  : {args.registry}")
    print(f"  Redis surfaces  : {len(redis_completed):,}")
    print(f"  Missing in Redis: {len(missing_in_redis):,}")
    print(f"  Extra in Redis  : {len(extra_in_redis):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
