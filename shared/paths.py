"""Filesystem path helpers used by runtime and command-line entry points."""
from pathlib import Path

def resolve_results_path(path: str, results_dir: str = "results") -> Path:
    """Place relative outputs under results_dir while preserving folder name."""
    candidate = Path(path)
    if candidate.is_absolute() or not results_dir:
        return candidate
    if candidate.parts and candidate.parts[0] == results_dir:
        return candidate
    return Path(results_dir) / candidate

def resolve_input_path(path: str, results_dir: str = "results") -> Path:
    """Resolve inputs, checking results_dir fallback for relative paths."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    resolved = resolve_results_path(path, results_dir)
    return resolved if resolved.exists() else candidate
