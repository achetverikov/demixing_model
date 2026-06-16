#!/usr/bin/env python3
"""Restore a redis_backup_*.ndjson produced by redis_backup.py.

Recreates every key byte-for-byte via RESTORE, preserving type and TTL. By
default it targets the REDIS_* host in .env; point --host/--port/etc. elsewhere
to restore into a fresh instance.

Usage:
    python cloud/redis_restore.py cloud/redis_backups/redis_backup_<ts>.ndjson
    python cloud/redis_restore.py <file> --replace        # overwrite existing keys
    python cloud/redis_restore.py <file> --host newhost --port 6380 --password ...
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "surface_computation"))

from lock_backend import load_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore a Redis ndjson backup.")
    ap.add_argument("backup", help="path to redis_backup_*.ndjson")
    ap.add_argument("--replace", action="store_true", help="overwrite keys that already exist")
    ap.add_argument("--host"); ap.add_argument("--port", type=int)
    ap.add_argument("--username"); ap.add_argument("--password")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import redis
    load_env()
    r = redis.Redis(
        host=args.host or os.getenv("REDIS_HOST"),
        port=args.port or int(os.getenv("REDIS_PORT", 6379)),
        password=args.password or os.getenv("REDIS_PASSWORD"),
        username=args.username or os.getenv("REDIS_USERNAME", "default"),
        decode_responses=False,
        socket_connect_timeout=10,
    )
    if not r.ping():
        raise SystemExit("Redis did not respond to PING")

    restored = skipped = 0
    for line in Path(args.backup).read_text().splitlines():
        rec = json.loads(line)
        if rec.get("_manifest"):
            print(f"restoring {rec['key_count']} keys from backup {rec['created_utc']} "
                  f"(was {rec.get('source_host')}) -> {r.connection_pool.connection_kwargs.get('host')}")
            continue
        if rec["dump_b64"] is None:
            skipped += 1
            continue
        key = rec["key"].encode("utf-8", "surrogateescape")
        payload = base64.b64decode(rec["dump_b64"])
        ttl = rec["pttl_ms"]
        ttl = ttl if ttl and ttl > 0 else 0
        if args.dry_run:
            restored += 1
            continue
        try:
            r.restore(key, ttl, payload, replace=args.replace)
            restored += 1
        except redis.exceptions.ResponseError as e:
            if "BUSYKEY" in str(e):
                skipped += 1
            else:
                raise
    verb = "would restore" if args.dry_run else "restored"
    print(f"{verb} {restored} keys, skipped {skipped} (existing/empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
