# WNM transition handoff

Status: 2026-09-11. R1--R6 are complete. The production WNM optimizer, full
CSH2026 fit, matched surface refit, bounds, fingerprints, and direct WNM reports
are done. Do not reopen the optimizer selection: production is the pinned
64-start `BatchedLbfgsb` configuration.

## Next

1. In `bias_model_comparison`, make `bbz_exports_complete` require BBZ
   predictive contract version 3; it currently accepts the stale version-2
   CSH2026 exports.
2. Refit BBZ `density` and `mse_smoothed` for both `csh2026` and
   `csh2026_separate`, preserving versioned old fits, then regenerate curves,
   predictive metrics, and likelihood exports.
3. Rebuild the BMC artifact and CSH2026 reports and compare them with
   `$DEMIXING_ARTIFACT_ROOT/bmc_report_backups/pre_dm_version_20260911/`.
4. Run a small stratified actual-GMM check over high-, medium-, and low-gap
   CSH2026 cases, summarize fidelity by dissimilarity band, and make the WNM
   promotion decision.

The extreme S15 smoothed-expectation check is under
`$DEMIXING_ARTIFACT_ROOT/csh2026_20samples_wnm/forward_truth_check_s15_color_hv_1_high_low/`.
WNM matches actual GMM truth at both tested parameter vectors; the surface NN
does not at its fitted vector. This is one case, not an average-performance
result.

Do not commit the 335 MB `bias_model_comparison/analysis/outputs - backup/`
archive or generated raw simulation files. Run only one JAX/GPU job at a time.
