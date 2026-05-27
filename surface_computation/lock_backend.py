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

    @contextmanager
    def held(self, key: str, owner: str, ttl_seconds: int = 1800):
        """Acquire lock, heartbeat every ttl/3 seconds, release on exit.

        Yields True if the lock was acquired, False otherwise.
        On False the caller should skip this chunk (someone else owns it).
        """
        if not self.acquire(key, owner, ttl_seconds):
            yield False
            return

        stop = threading.Event()

        def _heartbeat():
            interval = max(ttl_seconds / 3, 10)
            while not stop.wait(interval):
                self.refresh(key, owner, ttl_seconds)

        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        try:
            yield True
        finally:
            stop.set()
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
                 connect_timeout: int = 5):
        import redis as redis_lib
        self._r = redis_lib.Redis(
            host=host, port=port, password=password, username=username,
            decode_responses=True, socket_connect_timeout=connect_timeout,
        )
        self._prefix = prefix
        self._r.ping()  # fail fast if misconfigured

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

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

    Lock files are JSON: {owner, start_time, timestamp}.
    TTL is enforced by comparing start_time against the current clock;
    refresh() touches the file mtime as a heartbeat.
    """

    def __init__(self, lock_dir: Path, stale_multiplier: float = 1.0):
        self._dir = Path(lock_dir)
        # A lock is considered stale after ttl_seconds * stale_multiplier
        self._stale_mult = stale_multiplier

    def _lf(self, key: str) -> Path:
        return self._dir / f"computing_{key}.lock"

    def acquire(self, key: str, owner: str, ttl_seconds: int = 10800) -> bool:
        lf = self._lf(key)
        try:
            with open(lf, 'x') as f:
                json.dump({
                    'owner': owner,
                    'start_time': time.time(),
                    'timestamp': datetime.now().isoformat(),
                }, f)
            return True
        except FileExistsError:
            return self._reclaim_if_stale(lf, key, owner, ttl_seconds)

    def release(self, key: str, owner: str) -> None:
        lf = self._lf(key)
        try:
            lf.unlink()
        except FileNotFoundError:
            pass

    def refresh(self, key: str, owner: str, ttl_seconds: int = 10800) -> bool:
        lf = self._lf(key)
        try:
            lf.touch()
            return True
        except OSError:
            return False

    def _reclaim_if_stale(self, lf: Path, key: str, owner: str,
                           ttl_seconds: int) -> bool:
        try:
            with open(lf) as f:
                data = json.load(f)
            age = time.time() - data.get('start_time', 0)
            existing_owner = data.get('owner', '')
            is_stale = (
                age > ttl_seconds * self._stale_mult
                or existing_owner == owner  # reclaim own crashed lock
            )
            if is_stale:
                lf.unlink(missing_ok=True)
                return self.acquire(key, owner, ttl_seconds)
        except (json.JSONDecodeError, OSError):
            lf.unlink(missing_ok=True)
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
                    'age_seconds': time.time() - data.get('start_time', 0),
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
        if not host:
            raise ValueError("Redis host not configured (set REDIS_HOST or pass host=)")
        return RedisLockBackend(host=host, port=port, password=password, username=username)

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
