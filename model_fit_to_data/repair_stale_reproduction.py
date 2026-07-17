#!/usr/bin/env python3
"""Strip fit-result entries that failed postprocess_fitted_likelihoods.py's
stored-vs-rescored reproduction gate, so a subsequent fit_model_to_data.py
--resume run refits just those (subject, condition, optimizer) combinations.

Reads the trial_loglik_checks.csv written by postprocess_fitted_likelihoods.py
(it writes that file before raising on a gate failure), finds rows where
within_tolerance is False, and deletes the matching {optimizer}_* keys from
each affected condition entry in extended_fit_results.pkl. The pkl is backed
up before any change.
"""
import argparse
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


def sanitize(value):
    return re.sub(r"[^\w]", "_", str(value)).strip("_")


def canonical_key(key):
    parts = str(key).split("#")
    if len(parts) < 3:
        return str(key)
    return f"{sanitize(parts[0])}#{sanitize(parts[1])}#{sanitize('#'.join(parts[2:]))}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", required=True, help="extended_fit_results.pkl")
    parser.add_argument(
        "--checks-csv", required=True,
        help="trial_loglik_checks.csv from postprocess_fitted_likelihoods.py",
    )
    args = parser.parse_args()

    checks = pd.read_csv(args.checks_csv)
    if "within_tolerance" not in checks.columns:
        print(f"No within_tolerance column in {args.checks_csv}; nothing to repair.", file=sys.stderr)
        sys.exit(1)

    failed = checks[~checks["within_tolerance"]]
    if failed.empty:
        print("No failing rows found; nothing to repair.")
        sys.exit(1)

    path = Path(args.results_path)
    with path.open("rb") as f:
        results = pickle.load(f)

    raw_by_canonical: dict[str, list[str]] = {}
    for k in results:
        raw_by_canonical.setdefault(canonical_key(k), []).append(k)

    removed = 0
    unmatched = []
    for _, row in failed.iterrows():
        ck = f"{sanitize(row['subject'])}#{sanitize(row['experiment'])}#{sanitize(row['condition'])}"
        raws = raw_by_canonical.get(ck)
        opt = row["optimizer"]
        if not raws:
            unmatched.append((ck, opt))
            continue
        for raw in raws:
            entry = results[raw]
            stale_keys = [k for k in entry if k.startswith(f"{opt}_")]
            if not stale_keys:
                continue
            for k in stale_keys:
                del entry[k]
            removed += 1
            print(
                f"Stripped {len(stale_keys)} '{opt}_*' keys from {raw} "
                f"(abs_diff={row['abs_diff']:.6g} > tolerance {row['floor_tolerance']:.6g})"
            )

    if unmatched:
        print(f"Warning: {len(unmatched)} failing rows had no matching pkl entry: {unmatched}", file=sys.stderr)

    if removed == 0:
        print("No matching entries found to strip; nothing changed.")
        sys.exit(1)

    backup = path.with_name(f"{path.name}.pre-repair-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    print(f"Backed up {path} -> {backup}")

    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"Saved {path} ({removed} entries repaired)")


if __name__ == "__main__":
    main()
