#!/usr/bin/env python3
"""Create and validate discovery/selection/locked-confirmation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROLES = ("discovery", "selection", "confirmation")


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_trajectory_keys(design: np.ndarray) -> set[tuple[float, float, float]]:
    """Mirror-invariant SD triples, excluding feature difference."""
    x = np.asarray(design, dtype=np.float64)
    low = np.minimum(x[:, 0], x[:, 1])
    high = np.maximum(x[:, 0], x[:, 1])
    return set(map(tuple, np.round(np.column_stack([low, high, x[:, 2]]), 8)))


def corpus_record(path: Path, role: str, precision_increment: bool = False) -> dict:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    blob = np.load(path, allow_pickle=True)
    design, bias = blob["design"], blob["bias"]
    return {"path": str(path), "role": role, "sha256": file_sha256(path),
            "precision_increment": precision_increment,
            "n_rows": int(len(design)), "n_outcomes": int(bias.shape[1]),
            "n_components": int(bias.shape[2]),
            "trajectory_keys": [list(key) for key in sorted(canonical_trajectory_keys(design))]}


def validate_manifest(manifest: dict) -> None:
    records = manifest.get("corpora", [])
    roles = {record["role"] for record in records}
    if not roles.issubset(ROLES):
        raise ValueError("unknown benchmark role")
    by_role = {role: set() for role in ROLES}
    for record in records:
        keys = {tuple(key) for key in record["trajectory_keys"]}
        if by_role[record["role"]] & keys and not record.get("precision_increment", False):
            raise ValueError(f"duplicate trajectory in {record['role']} corpora")
        by_role[record["role"]].update(keys)
    overlap = by_role["selection"] & by_role["confirmation"]
    if overlap:
        raise ValueError(f"selection/confirmation trajectory overlap: {sorted(overlap)[:3]}")


def assert_operation_allowed(role: str, operation: str) -> None:
    """Prevent confirmation data from entering fitting or adaptive refinement."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role}")
    forbidden = {"fit", "select", "adapt", "debug"}
    if role == "confirmation" and operation in forbidden:
        raise PermissionError(f"{operation} is forbidden on locked confirmation data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path, action="append", default=[])
    parser.add_argument("--selection-increment", type=Path, action="append", default=[],
                        help="prespecified higher-N points nested in selection trajectories")
    parser.add_argument("--confirmation", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for role in ROLES:
        records.extend(corpus_record(path, role) for path in getattr(args, role))
    records.extend(corpus_record(path, "selection", precision_increment=True)
                   for path in args.selection_increment)
    manifest = {"version": 1, "confirmation_locked": bool(args.confirmation),
                "corpora": records}
    validate_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(records)} corpus records to {args.out}")


if __name__ == "__main__":
    main()
