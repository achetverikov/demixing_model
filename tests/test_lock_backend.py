import json
import os
import time

import pytest

from surface_computation.lock_backend import FileLockBackend


def _replace_owner(lock_file, owner):
    lock_file.write_text(json.dumps({
        "owner": owner,
        "start_time": time.time(),
        "timestamp": "test",
    }))


def test_stale_owner_cannot_refresh_or_release_replacement(tmp_path):
    backend = FileLockBackend(tmp_path)
    assert backend.acquire("chunk", "old", ttl_seconds=30)
    lock_file = backend._lf("chunk")
    _replace_owner(lock_file, "new")

    assert not backend.refresh("chunk", "old", ttl_seconds=30)
    backend.release("chunk", "old")

    assert lock_file.exists()
    assert json.loads(lock_file.read_text())["owner"] == "new"


def test_refresh_prevents_stale_reclamation(tmp_path):
    backend = FileLockBackend(tmp_path)
    assert backend.acquire("chunk", "active", ttl_seconds=2)
    lock_file = backend._lf("chunk")
    old = time.time() - 10
    os.utime(lock_file, (old, old))

    assert backend.refresh("chunk", "active", ttl_seconds=2)
    assert not backend.acquire("chunk", "other", ttl_seconds=2)


def test_crashed_instance_cannot_delete_same_owner_replacement(tmp_path):
    crashed = FileLockBackend(tmp_path)
    replacement = FileLockBackend(tmp_path)
    assert crashed.acquire("chunk", "machine", ttl_seconds=2)
    lock_file = crashed._lf("chunk")
    old = time.time() - 10
    os.utime(lock_file, (old, old))
    assert replacement.acquire("chunk", "machine", ttl_seconds=2)

    assert not crashed.refresh("chunk", "machine", ttl_seconds=2)
    crashed.release("chunk", "machine")

    assert lock_file.exists()
    assert json.loads(lock_file.read_text())["token"] == replacement._tokens[("chunk", "machine")]


def test_stale_lock_is_reclaimed(tmp_path):
    backend = FileLockBackend(tmp_path)
    assert backend.acquire("chunk", "old", ttl_seconds=2)
    lock_file = backend._lf("chunk")
    old = time.time() - 10
    os.utime(lock_file, (old, old))

    assert backend.acquire("chunk", "new", ttl_seconds=2)
    assert json.loads(lock_file.read_text())["owner"] == "new"


def test_maintain_releases_after_exception(tmp_path):
    backend = FileLockBackend(tmp_path)
    assert backend.acquire("chunk", "owner", ttl_seconds=30)

    with pytest.raises(RuntimeError):
        with backend.maintain("chunk", "owner", ttl_seconds=30):
            raise RuntimeError("boom")

    assert not backend._lf("chunk").exists()


def test_maintain_heartbeats_past_ttl(tmp_path):
    backend = FileLockBackend(tmp_path)
    contender = FileLockBackend(tmp_path)
    assert backend.acquire("chunk", "owner", ttl_seconds=0.3)

    with backend.maintain("chunk", "owner", ttl_seconds=0.3):
        time.sleep(0.5)
        assert not contender.acquire("chunk", "other", ttl_seconds=0.3)
