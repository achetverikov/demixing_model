#!/usr/bin/env python3
"""Smoke test S3-compatible storage used by Vast workers.

Creates one tiny fake averaged-surface object, verifies head/list behavior,
then deletes it unless --keep is passed.
"""

import argparse
import pickle
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "surface_computation"))

from surface_computation.object_store import ObjectStoreConfig, SurfaceObjectStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test object-store connectivity.")
    parser.add_argument("--keep", action="store_true", help="Keep the uploaded smoke object")
    args = parser.parse_args()

    config = ObjectStoreConfig.from_env()
    if config is None:
        print("Missing S3_BUCKET or OBJECT_STORE_BUCKET in the environment.", file=sys.stderr)
        return 2

    store = SurfaceObjectStore(config)
    filename = "averaged_sf1_999.0_sf2_999.0_sp_999.0.pkl"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        with open(path, "wb") as f:
            pickle.dump({"smoke": True}, f)

        key = store.upload_surface(path)
        print(f"Uploaded s3://{config.bucket}/{key}")

    if not store.object_exists(filename):
        print("Uploaded object was not found by head_object.", file=sys.stderr)
        return 1
    print("Verified head_object")

    completed = store.list_completed_surface_ids()
    expected_id = "999.0|999.0|999.0"
    if expected_id not in completed:
        print("Uploaded object was not found by list_objects_v2.", file=sys.stderr)
        return 1
    print("Verified list_objects_v2")

    if not args.keep:
        store.delete_surface(filename)
        print("Deleted smoke object")

    print("Object-store smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
