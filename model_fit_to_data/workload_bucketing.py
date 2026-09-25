"""Dataset-independent compile-shape bucketing for exact-coordinate WNM fits.

This small helper lives in demixing_model because ordinary CSV fitting must not
require the sibling contextual_biases_database checkout. Compiled bundles use the
same observed-range policy upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable


POLICY_VERSION = "observed-range-v1"


@dataclass(frozen=True)
class BucketAssignment:
    group_id: str
    static_signature: tuple
    coordinate_count: int
    capacity: int
    padding_count: int
    cohort: int
    policy_version: str = POLICY_VERSION


def plan_workload_buckets(groups: Iterable[tuple[str, Hashable, int]],
                          max_span: int = 64) -> tuple[BucketAssignment, ...]:
    """Cluster observed coordinate counts while keeping static JAX shapes stable."""
    if max_span < 0:
        raise ValueError(f"max_span must be non-negative, got {max_span}")

    by_signature: dict[tuple, list[tuple[str, int]]] = {}
    seen = set()
    for group_id, signature, count in groups:
        group_id = str(group_id)
        if group_id in seen:
            raise ValueError(f"duplicate group_id {group_id!r}")
        seen.add(group_id)
        if count < 1:
            raise ValueError(f"coordinate count must be positive for {group_id!r}")
        signature = tuple(signature) if isinstance(signature, (list, tuple)) else (signature,)
        by_signature.setdefault(signature, []).append((group_id, int(count)))

    assignments = []
    for signature in sorted(by_signature, key=repr):
        rows = sorted(by_signature[signature], key=lambda row: (row[1], row[0]))
        cohorts: list[list[tuple[str, int]]] = []
        for row in rows:
            if not cohorts or row[1] - cohorts[-1][0][1] > max_span:
                cohorts.append([row])
            else:
                cohorts[-1].append(row)
        for cohort_index, cohort in enumerate(cohorts):
            capacity = max(count for _, count in cohort)
            assignments.extend(
                BucketAssignment(
                    group_id=group_id,
                    static_signature=signature,
                    coordinate_count=count,
                    capacity=capacity,
                    padding_count=capacity - count,
                    cohort=cohort_index,
                )
                for group_id, count in cohort
            )

    return tuple(sorted(assignments, key=lambda assignment: assignment.group_id))
