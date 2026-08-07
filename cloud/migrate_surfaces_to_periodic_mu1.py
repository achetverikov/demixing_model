#!/usr/bin/env python3
"""Migrate stored surfaces from the legacy 181-row mu1 axis to the periodic 180.

The mu1_bias axis is circular, and the legacy representation gave one physical
angle (±180°) two rows.  The raw simulated surfaces are **bit-identical** at
those two rows, so the duplicate is pure redundancy in the representation and
can be dropped without resimulating — but it has to be dropped *on disk*,
because every surface pickle also carries its own ``mu1_bias_grid``, which no
config change can reach.  Loaders therefore carry a guard, not a shim: anything
still on the old axis is an error pointing here.

What this does, per surface:

* skip anything already at 180 rows (shape is the migrated/not marker, so the
  script is idempotent and safe to re-run);
* **verify** the two endpoint rows are identical before touching anything, and
  abort loudly otherwise — a difference would mean the data does not satisfy the
  identity this migration assumes;
* drop the last row from both mu1 surfaces **and** trim the embedded
  ``mu1_bias_grid`` in lockstep (``Surface.__post_init__`` validates the arrays
  against that grid, so trimming one without the other raises);
* rewrite atomically.

It handles both storage layouts: loose ``.pkl`` files and the ``.pkl.gz``
bundles that the released artifact tarballs ship.  Bundles are rewritten in
place with the same chunk ids and member names, so their ``.manifest.json``
sidecars — which carry only surface ids, counts, and parameters — stay valid.
After migrating a released set, re-pack it with ``cloud/package_surface_release.py``
so the ``SHA256SUMS`` match; the old tarballs are the backup.

Usage:
    python cloud/migrate_surfaces_to_periodic_mu1.py <dir> [<dir> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import io
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from shared.mu1_axis import (LEGACY_MU1_GRID_SIZE, is_legacy_mu1_length,
                             mu1_size, trim_legacy_grid)
from shared.utils import SurfaceUnpickler


class MigrationError(RuntimeError):
    """Raised when a surface cannot be migrated safely."""


def _as_array(x):
    return np.asarray(x)


def migrate_surface_obj(surface, source: str = "<surface>"):
    """Return (migrated_surface, changed).  Raises if the endpoints disagree."""
    grid = _as_array(surface.mu1_bias_grid)
    rows = _as_array(surface.mu1_comp1_surface).shape[0]

    if grid.size == mu1_size() and rows == mu1_size():
        return surface, False
    if not (is_legacy_mu1_length(grid.size) and is_legacy_mu1_length(rows)):
        raise MigrationError(
            f"{source}: mu1 axis has {grid.size} grid points / {rows} surface "
            f"rows; expected {mu1_size()} (migrated) or {LEGACY_MU1_GRID_SIZE} "
            f"(legacy). Refusing to guess.")

    # The two endpoint rows are the same angle. Verify before trimming: if they
    # differ, this file is not what the migration assumes and dropping a row
    # would destroy information.
    updates = {"mu1_bias_grid": trim_legacy_grid(grid)}
    for field in ("mu1_comp1_surface", "mu1_comp2_surface"):
        arr = _as_array(getattr(surface, field))
        max_diff = float(np.max(np.abs(arr[0] - arr[-1])))
        if max_diff != 0.0:
            raise MigrationError(
                f"{source}: {field} rows for -180 and +180 differ by "
                f"{max_diff:g}, but they are the same angle. Aborting.")
        updates[field] = arr[:-1]

    # dataclasses.replace re-runs __post_init__, which cross-checks the trimmed
    # arrays against the trimmed grid.
    return dataclasses.replace(surface, **updates), True


def _migrate_payload(payload, source: str):
    """Migrate the Surface inside a loaded pickle payload. Returns (payload, changed)."""
    if isinstance(payload, dict) and "surface" in payload:
        migrated, changed = migrate_surface_obj(payload["surface"], source)
        if changed:
            payload = dict(payload)
            payload["surface"] = migrated
        return payload, changed
    if hasattr(payload, "mu1_bias_grid"):
        return migrate_surface_obj(payload, source)
    raise MigrationError(f"{source}: not a surface pickle.")


def migrate_loose_file(path: Path, dry_run: bool = False) -> bool:
    with open(path, "rb") as f:
        payload = SurfaceUnpickler(f).load()
    payload, changed = _migrate_payload(payload, str(path))
    if changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(path)
    return changed


def migrate_bundle(path: Path, dry_run: bool = False) -> int:
    """Migrate every member of a surface bundle. Returns how many changed."""
    with gzip.open(path, "rb") as f:
        bundle = pickle.load(f)
    surfaces = bundle["surfaces"]

    n_changed = 0
    for name, raw in list(surfaces.items()):
        payload = SurfaceUnpickler(io.BytesIO(raw)).load()
        payload, changed = _migrate_payload(payload, f"{path}::{name}")
        if changed:
            surfaces[name] = pickle.dumps(payload)
            n_changed += 1

    if n_changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb") as f:
            pickle.dump(bundle, f)
        tmp.replace(path)
    return n_changed


#: Surface directories also hold fit results, caches, and other pickles. Match
#: surfaces by their filename convention rather than trying to load everything —
#: but *report* what was skipped, since a silently shorter file list is exactly
#: the failure mode this migration exists to eliminate.
SURFACE_GLOBS = ("surface_sf1_*.pkl", "averaged_sf1_*.pkl")


def migrate_directory(directory: Path, dry_run: bool = False) -> tuple[int, int]:
    """Migrate every loose surface and bundle under ``directory`` (recursively)."""
    n_files = n_changed = 0

    surfaces = sorted({p for glob in SURFACE_GLOBS for p in directory.rglob(glob)})
    skipped = [p for p in directory.rglob("*.pkl")
               if p not in set(surfaces) and not p.name.endswith(".tmp")]
    if skipped:
        print(f"  (not surface files, left alone: {len(skipped)} .pkl — "
              f"e.g. {skipped[0].name})")

    for path in surfaces:
        if path.name.endswith(".tmp"):
            continue
        n_files += 1
        if migrate_loose_file(path, dry_run):
            n_changed += 1
            print(f"  trimmed {path}")

    for path in sorted(directory.rglob("*.pkl.gz")):
        n_files += 1
        changed = migrate_bundle(path, dry_run)
        if changed:
            n_changed += 1
            print(f"  trimmed {changed} member(s) in {path}")

    return n_files, n_changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directories", nargs="+", type=Path,
                        help="Surface directories (loose .pkl and/or .pkl.gz bundles).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    args = parser.parse_args(argv)

    total_files = total_changed = 0
    for directory in args.directories:
        if not directory.exists():
            print(f"!! {directory} does not exist", file=sys.stderr)
            return 2
        print(f"Scanning {directory} ...")
        n_files, n_changed = migrate_directory(directory, args.dry_run)
        print(f"  {n_files} file(s) scanned, {n_changed} migrated"
              f"{' (dry run)' if args.dry_run else ''}")
        total_files += n_files
        total_changed += n_changed

    print(f"Done: {total_changed}/{total_files} file(s) migrated to the "
          f"{mu1_size()}-point periodic mu1 axis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
