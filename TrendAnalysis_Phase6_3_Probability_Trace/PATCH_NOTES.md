# Patch notes: Phase 6.3 probability source trace

- Carry `class_counts`, `weighted_class_counts`, and `probability_basis` from each Phase 5.2 probability builder into `MethodPrediction` and its serialized output.
- Preserve Method B `exact_condition_count`, `weight_concentration`, and `stability_score` so effective support and weight concentration can be reviewed.
- Upgrade the fold recording contract from v1 to v2 for newly exported snapshots.
- Enforce complete, non-negative class traces, sample-count reconciliation, and weighted-count-to-probability-vector reconstruction for v2.
- Keep contract-v1 validation available for already-frozen artifacts.
- Add tests for both Method A/B wrapper propagation, v2 source-trace validation, snapshot-export preflight, and the Phase 6.2 audit round-trip fixture.
- Do not modify probability values, model calibration, selection thresholds, or historical artifacts.
