# Phase 6.3: probability source-trace propagation

## Purpose

The Phase 6.2 zero-probability audit found that builder-level trace fields were absent from all 208 frozen folds. Source review identifies the drop point: `build_method_a_probability()` and `build_method_b_probability()` calculate `class_counts`, `weighted_class_counts`, and `probability_basis`, plus Method B diagnostics `exact_condition_count`, `weight_concentration`, and `stability_score`, but `PredictionEngine` adapts each builder result into `MethodPrediction`. Before this patch, `MethodPrediction` omitted those fields from both its dataclass and `as_dict()`, so `PredictionFoldResult` could not preserve them in the snapshot.

This patch propagates the exact builder metadata through `MethodPrediction`, raises the new fold recording contract to version 2, and validates the metadata at export. The loader remains able to validate existing recording-contract version 1 snapshots.

## Important scope boundary

This is an **auditability and contract fix**, not a probability-model change. It does not add smoothing, alter any probability vector, change relationship selection, change thresholds, or change decisions. The historic Phase 5.8 snapshot remains frozen and immutable. The current snapshot cannot be repaired retrospectively because the discarded source metadata was not retained in its fold records.

## Apply

1. Extract the ZIP into the repository root, keeping the `TrendAnalysis_Phase6_3_Probability_Trace` folder.
2. Review `apply_phase6_3_probability_trace.py` and the patch notes.
3. From the repository root run:

   ```bash
   py TrendAnalysis_Phase6_3_Probability_Trace/apply_phase6_3_probability_trace.py
   ```

   The script verifies exact source anchors, prepares edits before writing, syntax-checks modified Python, and backs up six existing source/test files. It refuses to proceed if any expected source anchor does not match. It also creates `tests/test_phase6_3_probability_trace.py`.

## Verify after applying

```bash
py -m pytest -q \
  tests/test_phase5_8_fold_recording_contract.py \
  tests/test_phase5_8_snapshot_export_preflight.py \
  tests/test_phase6_2_method_probability_audit.py \
  tests/test_phase5_snapshot.py \
  tests/test_method_a_prediction.py \
  tests/test_method_b_prediction.py \
  tests/test_method_a.py \
  tests/test_method_b.py \
  tests/test_phase6_3_probability_trace.py
```

Then inspect:

```bash
git diff --check
git diff -- analysis/prediction.py analysis/real_prediction_validation.py analysis/phase5_snapshot.py tests/test_phase5_8_fold_recording_contract.py tests/test_phase5_8_snapshot_export_preflight.py tests/test_phase6_2_method_probability_audit.py tests/test_phase6_3_probability_trace.py
```

Do **not** run Phase 5.8 just to check this patch. The included tests are the first gate. A later Phase 5.8 export, when deliberately approved and its runtime budget accepted, would be needed to populate traces in a new snapshot. Do not overwrite `artifacts/phase5_8/RELIANCE_Nifty_50_6M_1M`.

## Why smoothing is not bundled

The existing Phase 6.2 smoothing sweep was chosen after the historical period had already been inspected. That makes its selected values post-hoc sensitivity results rather than deployment-grade settings. Changing the estimator now would also change probabilities, baseline-relative decisions, combination weights, and potentially trends. First restore traceability; then run the revised estimator as a separately versioned candidate and judge it on genuinely new, maturity-eligible outcomes.
