# JAX L-BFGS-B port validation snapshot

This directory is a compact, versioned snapshot of the validation evidence
supporting the 2026-09-10 selection of 64-start batched JAX L-BFGS-B with
float32 arrays and `highest` matmul precision for the four retained WNM
objectives.

The files were copied without modification from
`$DEMIXING_ARTIFACT_ROOT/continuous_density_4.1q/recovery/single_condition_n100/`.
`JAX_LBFGSB_PORT_COMPARISON_FINDINGS.md` describes the protocol, results, and
artifact inventory. Run manifests retain the paths recorded on the generating
machine as provenance; those paths are not expected to exist in another
checkout.

Included:

- every run manifest and run-level summary from the `jax_lbfgsb_port_*`
  validation directories;
- the device/precision diagnostic records;
- the assembled comparison tables, scaling summaries, multistart summaries,
  and run logs; and
- the matched-KDE and matched-smoothed development comparison tables used in
  the final cross-objective decision.

Excluded:

- `tasks/` records from the full recovery runs. Those per-dataset/per-start
  JSON files total about 65 MB uncompressed and are redundant with the included
  run summaries for the model-selection claims. They remain in the external
  artifact root and are identified by the included manifests.

This snapshot is validation evidence, not a runtime dependency. Production
code must not resolve or load files from this directory while fitting data.
