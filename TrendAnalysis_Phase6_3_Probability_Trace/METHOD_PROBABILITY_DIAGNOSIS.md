# Method A/B probability diagnosis from Phase 6.2

## What source review establishes

Both Phase 5.2 probability builders compute an empirical weighted class share:

`probability(class) = weighted support mass in class / total weighted support mass`.

This has no positive lower bound. If no positive-weight supporting observation falls in a class, its probability is exactly zero. This is not necessarily an arithmetic defect; it is a known failure mode of an unsmoothed empirical estimator when the support set is small or class-sparse.

- **Method A:** the frozen audit reports support sample count 12 and effective sample size 12 throughout the selected outputs. The unit-weight support is small, so one observation changes the class share by about 8.33 percentage points. If zero observations fall in a class, its empirical share is zero. The audit measured exact zeros for the actual class in 68 of 187 available Method A forecasts (36.36%).
- **Method B:** the audit reports mean sample count about 35.65 but mean effective sample size about 14.29. Unequal weights reduce the amount of independent effective support. Actual-class exact zeros were much rarer (1 of 150 available forecasts), but the output remains an empirical weighted share and can be overconfident.
- **Combined model:** actual-class exact zeros appeared in 17 of 208 folds (8.17%). The raw combined mean top-label confidence was 62.41% while accuracy was 25.96%; raw UP probability averaged 45.19% while UP was observed in 28.85% of folds. These are model-skill concerns, not just a display problem.

## Important limitation

The already-frozen snapshot does not retain the builder-level counts needed to associate every historical zero with a specific zero source count. The mechanism is confirmed by source code, and the aggregate zero pattern is observed, but the exact per-fold support-count reconstruction is not claimed. The Phase 6.3 patch propagates those fields in future exports and validates their consistency.

## What is deliberately not changed

No smoothing constant is deployed here. The historical Phase 6.2 joint smoothing sweep selected parameters after this period had already been inspected. It is useful sensitivity analysis, but it is not an untouched final evaluation. A production probability change must be evaluated as a separately versioned candidate using chronological outcomes that have matured strictly before each prediction and then tested on a genuinely new evaluation window.

## Current research direction

1. Restore source traceability and fail-closed contract checks.
2. Prototype a prior-aware or Dirichlet-smoothed probability estimator separately from the current Phase 5.2 estimator.
3. Compare the candidate to the unchanged stored baseline using maturity-gated proper scores, class calibration, and confidence-ranking/selective-coverage diagnostics.
4. Do not declare success from accuracy alone or from the already-inspected historic holdout.
