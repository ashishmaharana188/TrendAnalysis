from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.historical_fold_audit import (
    audit_prediction_fold,
    print_historical_fold_audit,
)
from analysis.phase5_snapshot import (
    default_snapshot_path,
    export_phase5_snapshot,
)
from analysis.real_olap_validation import RealOLAPValidationConfig
from analysis.real_prediction_validation import (
    print_real_olap_prediction_report,
    validate_real_olap_predictions,
)


def main() -> None:
    config = RealOLAPValidationConfig(
        ticker="RELIANCE",
        benchmark="Nifty_50",
        analysis_timeframe="6M",
        holding_period_months=1.0,
        entry_mode="next_trading_day",
        max_folds=250,
        step_trading_days=5,
        min_training_observations=12,
        hardened_validation=True,
        performance_cache=True,
        progress_logging=True,
        progress_every=10,
        adaptive_relationship_search=False,
        relationship_progress_every_candidates=1000,
        relationship_candidate_batch_size=512,
    )

    # Resolve the canonical destination before any expensive work. A completed
    # run must never be repeated only to discover that its artifact already exists.
    snapshot_root = os.getenv("PHASE5_SNAPSHOT_ROOT", "artifacts/phase5_8")
    snapshot_path = default_snapshot_path(config, root=snapshot_root)
    if snapshot_path.exists():
        raise FileExistsError(
            f"Canonical Phase 5.8 artifact already exists: {snapshot_path}. "
            "Inspect/archive it before intentionally starting another full run."
        )

    result = validate_real_olap_predictions(config)
    print_real_olap_prediction_report(result)

    assert result.candidate_predictions > 0
    assert result.evaluated_predictions > 0
    assert result.latest_validated_prediction_date is not None
    assert result.latest_market_date is not None
    assert result.skipped_provenance_failed == 0

    aug24 = next(
        (
            fold for fold in result.prediction_folds
            if fold.prediction_date == date(2026, 8, 24)
        ),
        None,
    )
    if aug24 is not None:
        assert aug24.observed_trend in {"UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"}
        if aug24.observed_trend == "SIDEWAYS":
            assert not aug24.trade_eligible
            assert aug24.trade_reason == "SIDEWAYS_CURRENT_TREND"

    eligible_folds = [fold for fold in result.prediction_folds if fold.trade_eligible]
    if eligible_folds:
        audit_date = eligible_folds[-1].prediction_date
        audit = audit_prediction_fold(config, audit_date, result)
        print_historical_fold_audit(audit)
        assert audit.trade_eligible
        assert audit.observed_trend in {"UP", "DOWN"}
        assert audit.predicted_trend == audit.observed_trend
        assert audit.entry_after_prediction
        assert audit.exit_after_entry
        assert audit.one_month_window
        assert audit.return_reconciled
    else:
        print("PHASE 5.8 TRADE GATE: no historically trade-eligible folds were found.")

    # One canonical run artifact. No automatically incremented v1/v2/v3 directories.
    # The exporter validates every fold's recording contract before creating files.
    exported = export_phase5_snapshot(
        result,
        config,
        snapshot_path,
        repository_root=REPO_ROOT,
    )

    method_a_outputs = sum(fold.method_a_output is not None for fold in result.prediction_folds)
    method_b_outputs = sum(fold.method_b_output is not None for fold in result.prediction_folds)
    method_a_vectors = sum(
        fold.method_a_output is not None and bool(fold.method_a_probabilities_pct)
        for fold in result.prediction_folds
    )
    method_b_vectors = sum(
        fold.method_b_output is not None and bool(fold.method_b_probabilities_pct)
        for fold in result.prediction_folds
    )
    method_a_usable_vectors = sum(
        sum(float(value) for value in fold.method_a_probabilities_pct.values()) > 0.0
        for fold in result.prediction_folds
        if fold.method_a_output is not None
    )
    method_b_usable_vectors = sum(
        sum(float(value) for value in fold.method_b_probabilities_pct.values()) > 0.0
        for fold in result.prediction_folds
        if fold.method_b_output is not None
    )
    selection_records = sum(bool(fold.selection_metadata) for fold in result.prediction_folds)

    print(f"PHASE 5.8 CANONICAL RUN ARTIFACT: {exported}")
    print(f"Fold records: {len(result.prediction_folds)}")
    print(f"Method A output objects recorded: {method_a_outputs}/{len(result.prediction_folds)}")
    print(f"Method B output objects recorded: {method_b_outputs}/{len(result.prediction_folds)}")
    print(f"Method A exact probability vectors recorded: {method_a_vectors}/{len(result.prediction_folds)}")
    print(f"Method B exact probability vectors recorded: {method_b_vectors}/{len(result.prediction_folds)}")
    print(f"Method A non-zero probability distributions: {method_a_usable_vectors}/{len(result.prediction_folds)}")
    print(f"Method B non-zero probability distributions: {method_b_usable_vectors}/{len(result.prediction_folds)}")
    print(f"Per-fold selection metadata recorded: {selection_records}/{len(result.prediction_folds)}")
    print("Snapshot status: FROZEN")
    print("Phase 6+ reads this artifact; it does not rerun OLAP prediction folds.")
    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION: PASS")
    print("PHASE 5.8 TRADE GATE VALIDATION: PASS")
    print("No model optimisation is applied here; Phase 5 remains diagnostic and Phase 6 covers calibration/OOS trading performance.")


if __name__ == "__main__":
    main()
