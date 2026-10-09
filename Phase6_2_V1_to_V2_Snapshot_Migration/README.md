# Phase 6.2: Create v2 snapshot from the existing v1 snapshot

This is a standalone snapshot migration tool, not a patch installer. It never modifies v1 and never reruns Phase 5.8.

## Direct migration when the current v1 already includes exact Method A/B vectors

Run from the TrendAnalysis repository root:

```bash
py /path/to/promote_phase5_snapshot_v1_to_v2.py artifacts/phase5_8/RELIANCE_Nifty_50_6M_1M_v1
```

The destination defaults to `artifacts/phase5_8/RELIANCE_Nifty_50_6M_1M_v2`.

## If the fold rows do not include those vectors

The tool refuses to invent them. If the separate Method A-only and Method B-only snapshots already exist, pass their snapshot directories:

```bash
py /path/to/promote_phase5_snapshot_v1_to_v2.py \
  artifacts/phase5_8/RELIANCE_Nifty_50_6M_1M_v1 \
  --method-a-snapshot PATH_TO_METHOD_A_ONLY_SNAPSHOT \
  --method-b-snapshot PATH_TO_METHOD_B_ONLY_SNAPSHOT
```

For these sidecars, each fold's `probabilities_pct` must be the actual individual method's probability vector, not the combined probability vector. The tool aligns rows by `prediction_date`, checks `actual_class` where present, verifies all source folds are covered, checks hashes and probability distributions, and writes a new `_v2` directory.

If no exact method vectors exist in the v1 or method-only snapshots, an exact v2 audit snapshot cannot be generated from those files. A trend label or combined probability vector is not enough to reconstruct the individual method distributions.

## What it preserves / changes

- Copies the existing snapshot artifacts without editing the source v1 directory.
- Writes the exact method probability maps into each v2 fold.
- Changes the manifest schema to `"2"` and refreshes artifact and directory checksums.
- Records where the method vectors came from and states that predictions were not recalculated.
- Does not use DuckDB/OLAP or the 250-fold prediction loop.

## After it succeeds

```bash
SNAPSHOT="artifacts/phase5_8/RELIANCE_Nifty_50_6M_1M_v2"
py phase6_2_method_probability_audit.py "$SNAPSHOT"
py phase6_2_probability_surface_audit.py "$SNAPSHOT"
py phase6_2_diagnostics_from_snapshot.py "$SNAPSHOT"
py phase6_from_snapshot.py "$SNAPSHOT"
py phase6_selective_from_snapshot.py "$SNAPSHOT"
py phase5_8_decision_layer_audit.py "$SNAPSHOT"
```
