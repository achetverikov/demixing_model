"""Small hashing helpers shared by artifact and run-identity code."""
from __future__ import annotations

import hashlib
import os


def file_sha256(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed to keep memory use bounded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
