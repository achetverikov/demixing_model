"""Lock backends for chunk coordination.

Two implementations share the same interface:
  RedisLockBackend  — distributed, works across machines, no filesystem coupling
  FileLockBackend   — local, works without Redis, uses .lock files on a shared drive

Both expose acquire / release / refresh and a held() context manager that
automatically heartbeats the lock and releases it on exit.

Usage:
    backend = make_lock_backend('redis')   # reads REDIS_* from env
    backend = make_lock_backend('file', lock_dir=Path('results/samples'))
    backend = make_lock_backend('auto')    # redis if env configured, else file

    with backend.held('chunk_id', 'PC1', ttl_seconds=1800) as acquired:
        if not acquired:
            continue   # someone else has it
        do_work()
"""

import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class LockBackend(ABC):
    @abstractmethod
    def acquire(self, key: str, owner: str, ttl_seconds: int) -> bool: ...

    @abstractmethod
    def release(self, key: str, owner: str) -> None: ...

    @abstractmethod
    def refresh(self, key: str, owner: str, ttl_seconds: int) -> bool: ...

    @abstractmethod
    def active_locks(self) -> Dict[str, dict]: ...

    def completed_members(self, registry: str) -> set:
        """Return completed item IDs for backends that support a durable registry."""
        return set()

    def mark_completed(self, registry: str, item_id: str) -> None:
        """Mark one completed item. File locks intentionally make this a no-op."""
        return None

    def remove_completed(self, registry: str, item_id: str) -> None:
        """Remove one completed item. File locks intentionally make this a no-op."""
        return None

    def clear_completed(self, registry: str, confirm: bool = False) -> int:
        """Clear a completed registry for backends that support one."""
        if not confirm:
            raise RuntimeError("Pass confirm=True to clear a completed registry")
        return 0

    @contextmanager
    def held(self, key: str, owner: str, ttl_seconds: int = 1800):
        """Acquire lock, heartbeat every ttl/3 seconds, release on exit.

        Yields True if the lock was acquired, False otherwise.
        On False the caller should skip this chunk (someone else owns it).
        """
        if not self.acquire(key, owner, ttl_seconds):
            yield False
            return

        with self.maintain(key, owner, ttl_seconds):
            yield True

    @contextmanager
    def maintain(self, key: str, owner: str, ttl_seconds: int = 1800):
        """Heartbeat and finally release a lock that the caller already acquired."""

        stop = threading.Event()

        def _heartbeat():
            interval = max(ttl_seconds / 3, 0.1)
            while not stop.wait(interval):
                if not self.refresh(key, owner, ttl_seconds):
                    return

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=1)
            self.release(key, owner)


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class RedisLockBackend(LockBackend):
    # Atomically delete only if we still own the key
    _LUA_RELEASE = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    end
    return 0
    """
    # Atomically refresh TTL only if we still own the key
    _LUA_REFRESH = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    end
    return 0
    """

    def __init__(self, host: str, port: int, password: str,
                 username: str = 'default',
                 prefix: str = 'demixing:lock:',
                 completed_prefix: str = 'demixing:completed:',
                 connect_timeout: int = 5):
        import redis as redis_lib
        self._r = redis_lib.Redis(
            host=host, port=port, password=password, username=username,
            decode_responses=True, socket_connect_timeout=connect_timeout,
        )
        self._prefix = prefix
        self._completed_prefix = completed_prefix
        self._r.ping()  # fail fast if misconfigured

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _completed_key(self, registry: str) -> str:
        return f"{self._completed_prefix}{registry}"

    def acquire(self, key: str, owner: str, ttl_seconds: int = 1800) -> bool:
        return bool(self._r.set(self._k(key), owner, nx=True, ex=int(ttl_seconds)))

    def release(self, key: str, owner: str) -> None:
        self._r.eval(self._LUA_RELEASE, 1, self._k(key), owner)

    def refresh(self, key: str, owner: str, ttl_seconds: int = 1800) -> bool:
        return bool(self._r.eval(
            self._LUA_REFRESH, 1, self._k(key), owner, int(ttl_seconds)
        ))

    def active_locks(self) -> Dict[str, dict]:
        result = {}
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor, match=f"{self._prefix}*", count=200)
            for k in keys:
                owner = self._r.get(k)
                ttl = self._r.ttl(k)
                result[k.removeprefix(self._prefix)] = {
                    'owner': owner,
                    'ttl_remaining_seconds': ttl,
                }
            if cursor == 0:
                break
        return result

    def completed_members(self, registry: str) -> set:
        return set(self._r.smembers(self._completed_key(registry)))

    def mark_completed(self, registry: str, item_id: str) -> None:
        self._r.sadd(self._completed_key(registry), item_id)

    def mark_completed_batch(self, registry: str, item_ids) -> int:
        """Add multiple completed items in a single round-trip. Returns number added."""
        items = list(item_ids)
        if not items:
            return 0
        return int(self._r.sadd(self._completed_key(registry), *items))

    def remove_completed(self, registry: str, item_id: str) -> None:
        self._r.srem(self._completed_key(registry), item_id)

    def clear_completed(self, registry: str, confirm: bool = False) -> int:
        if not confirm:
            raise RuntimeError("Pass confirm=True to clear a completed registry")
        return int(self._r.delete(self._completed_key(registry)))

    def clear_all(self, confirm: bool = False) -> int:
        """Delete all locks under this prefix. Pass confirm=True to proceed."""
        if not confirm:
            raise RuntimeError("Pass confirm=True to clear all locks")
        keys = []
        cursor = 0
        while True:
            cursor, batch = self._r.scan(cursor, match=f"{self._prefix}*", count=200)
            keys.extend(batch)
            if cursor == 0:
                break
        if keys:
            return self._r.delete(*keys)
        return 0


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------

class FileLockBackend(LockBackend):
    """File-based lock compatible with OneDrive-synced shared folders.

    Lock files contain an owner plus a unique acquisition token. The file mtime
    is the heartbeat, so a long-lived but actively refreshed lock does not become
    stale, and an expired process cannot refresh or release a replacement lock.
    """

    def __init__(self, lock_dir: Path, stale_multiplier: float = 1.0):
        self._dir = Path(lock_dir)
        # A lock is considered stale after ttl_seconds * stale_multiplier
        self._stale_mult = stale_multiplier
        self._tokens = {}

    def _lf(self, key: str) -> Path:
        return self._dir / f"computing_{key}.lock"

    def acquire(self, key: str, owner: str, ttl_seconds: int = 10800) -> bool:
        lf = self._lf(key)
        self._dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        try:
            with open(lf, 'x') as f:
                json.dump({
                    'owner': owner,
                    'token': token,
                    'start_time': time.time(),
                    'timestamp': datetime.now().isoformat(),
                }, f)
                f.flush()
                os.fsync(f.fileno())
            self._tokens[(key, owner)] = token
            return True
        except FileExistsError:
            return self._reclaim_if_stale(lf, key, owner, ttl_seconds)

    def release(self, key: str, owner: str) -> None:
        lf = self._lf(key)
        token = self._tokens.get((key, owner))
        if token is None:
            return
        try:
            with open(lf) as f:
                data = json.load(f)
            if data.get('owner') != owner or data.get('token') != token:
                return
            before = lf.stat()
            with open(lf) as f:
                data = json.load(f)
                if data.get('owner') != owner or data.get('token') != token:
                    return
            after = lf.stat()
            if self._same_file_state(before, after):
                lf.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        finally:
            self._tokens.pop((key, owner), None)

    def refresh(self, key: str, owner: str, ttl_seconds: int = 10800) -> bool:
        lf = self._lf(key)
        token = self._tokens.get((key, owner))
        if token is None:
            return False
        try:
            with open(lf) as f:
                data = json.load(f)
            if data.get('owner') != owner or data.get('token') != token:
                return False
            os.utime(lf, None)
            return True
        except (OSError, json.JSONDecodeError):
            return False

    @staticmethod
    def _same_file_state(left, right) -> bool:
        return (
            left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns
        ) == (
            right.st_dev, right.st_ino, right.st_size, right.st_mtime_ns
        )

    def _reclaim_if_stale(self, lf: Path, key: str, owner: str,
                           ttl_seconds: int) -> bool:
        try:
            before = lf.stat()
            age = time.time() - before.st_mtime
            if age <= ttl_seconds * self._stale_mult:
                return False
            try:
                with open(lf) as f:
                    json.load(f)
            except json.JSONDecodeError:
                pass
            after = lf.stat()
            if not self._same_file_state(before, after):
                return False
            lf.unlink(missing_ok=True)
            return self.acquire(key, owner, ttl_seconds)
        except (FileNotFoundError, OSError):
            return self.acquire(key, owner, ttl_seconds)
        return False

    def active_locks(self) -> Dict[str, dict]:
        result = {}
        for lf in self._dir.glob("computing_*.lock"):
            try:
                with open(lf) as f:
                    data = json.load(f)
                key = lf.stem.removeprefix('computing_')
                result[key] = {
                    'owner': data.get('owner'),
                    'age_seconds': time.time() - lf.stat().st_mtime,
                    'held_since_seconds': time.time() - data.get('start_time', 0),
                    'ttl_remaining_seconds': None,
                }
            except (json.JSONDecodeError, OSError):
                continue
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_lock_backend(backend: str,
                      lock_dir: Optional[Path] = None,
                      **redis_kwargs) -> LockBackend:
    """Return a LockBackend instance.

    backend: 'redis' | 'file' | 'auto'
      auto — uses redis if REDIS_HOST is set in env, else file.

    For redis, kwargs / env vars (REDIS_HOST, REDIS_PORT, REDIS_PASSWORD,
    REDIS_USERNAME) are used.  For file, lock_dir is required.
    """
    if backend == 'auto':
        backend = 'redis' if (os.getenv('REDIS_HOST') or redis_kwargs.get('host')) else 'file'

    if backend == 'redis':
        host = redis_kwargs.get('host') or os.getenv('REDIS_HOST', '')
        port = int(redis_kwargs.get('port') or os.getenv('REDIS_PORT', 6379))
        password = redis_kwargs.get('password') or os.getenv('REDIS_PASSWORD', '')
        username = redis_kwargs.get('username') or os.getenv('REDIS_USERNAME', 'default')
        prefix = redis_kwargs.get('prefix') or os.getenv('REDIS_LOCK_PREFIX', 'demixing:lock:')
        completed_prefix = (
            redis_kwargs.get('completed_prefix')
            or os.getenv('REDIS_COMPLETED_PREFIX', 'demixing:completed:')
        )
        if not host:
            raise ValueError("Redis host not configured (set REDIS_HOST or pass host=)")
        return RedisLockBackend(
            host=host, port=port, password=password, username=username,
            prefix=prefix, completed_prefix=completed_prefix,
        )

    if backend == 'file':
        if lock_dir is None:
            raise ValueError("lock_dir is required for FileLockBackend")
        return FileLockBackend(Path(lock_dir))

    raise ValueError(f"Unknown lock backend {backend!r}; choose 'redis', 'file', or 'auto'")


def load_env(path: str = '.env') -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no-op if missing)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, _, v = line.partition('=')
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass
