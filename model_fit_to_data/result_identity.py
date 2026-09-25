"""Canonical identity for persisted fit-result keys.

Result keys are part of the on-disk contract. Fitting, plotting, migration, and
repair must map labels to exactly the same key or a valid result can become
unreachable, or two distinct labels can silently collide after sanitization.
"""
from __future__ import annotations

import re
from typing import Dict


def sanitize_result_key_part(value) -> str:
    """Replace non-word characters with underscores and trim edge underscores."""
    return re.sub(r"[^\w]", "_", str(value)).strip("_")


def canonical_condition_key(key) -> str:
    """Canonicalize a legacy subject#experiment#condition result key."""
    parts = str(key).split("#")
    if len(parts) < 3:
        return str(key)
    subject, experiment = parts[0], parts[1]
    condition = "#".join(parts[2:])
    return (
        f"{sanitize_result_key_part(subject)}#"
        f"{sanitize_result_key_part(experiment)}#"
        f"{sanitize_result_key_part(condition)}"
    )


def make_condition_key(subject, experiment, condition) -> str:
    """Build a canonical condition key from its three logical labels."""
    return "#".join(
        sanitize_result_key_part(value)
        for value in (subject, experiment, condition)
    )


def make_experiment_key(subject, experiment) -> str:
    """Build a canonical subject-by-experiment key."""
    return "#".join(
        sanitize_result_key_part(value)
        for value in (subject, experiment)
    )


def canonicalize_result_keys(results: Dict) -> Dict:
    """Canonicalize persisted result keys and reject lossy-key collisions."""
    canonical = {}
    source_keys = {}
    for key, entry in results.items():
        ckey = canonical_condition_key(key)
        if ckey in canonical and source_keys[ckey] != str(key):
            raise ValueError(
                "Result-key collision after sanitization: "
                f"{source_keys[ckey]!r} and {str(key)!r} both map to {ckey!r}"
            )
        canonical[ckey] = entry
        source_keys[ckey] = str(key)
    return canonical


def validate_group_key_uniqueness(df, exp_col: str, subject_col: str,
                                  condition_col: str) -> None:
    """Fail before fitting when distinct labels collapse to the same key."""
    seen_subjects = {}
    seen_conditions = {}
    label_cols = [subject_col, exp_col, condition_col]
    for values in df[label_cols].drop_duplicates().itertuples(index=False, name=None):
        raw = tuple(str(value) for value in values)
        for source, sanitized, seen in (
            (raw[:2], make_experiment_key(*raw[:2]), seen_subjects),
            (raw, make_condition_key(*raw), seen_conditions),
        ):
            previous = seen.get(sanitized)
            if previous is not None and previous != source:
                raise ValueError(
                    "Distinct labels collide after result-key sanitization: "
                    f"{previous!r} and {source!r} both map to {sanitized!r}"
                )
            seen[sanitized] = source
