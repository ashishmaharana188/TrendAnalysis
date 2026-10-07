from datetime import date

from analysis.historical_fold_audit import (
    audit_prediction_fold,
    print_historical_fold_audit,
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

    result = validate_real_olap_predictions(config)
    print_real_olap_prediction_report(result)

    assert result.candidate_predictions > 0
    assert result.evaluated_predictions > 0
    assert result.latest_validated_prediction_date is not None
    assert result.latest_market_date is not None
    assert result.skipped_provenance_failed == 0

    # 24-Aug-2026 remains a useful historical-state diagnostic, but it is not
    # assumed to be a trade entry. In particular, a sideways observed trend
    # must never become a trade merely because the model predicts UP/DOWN.
    aug24 = next(
        (fold for fold in result.prediction_folds if fold.prediction_date == date(2026, 8, 24)),
        None,
    )
    if aug24 is not None:
        assert aug24.observed_trend in {"UP", "SIDEWAYS", "DOWN", "NO_CLEAR_TREND"}
        if aug24.observed_trend == "SIDEWAYS":
            assert not aug24.trade_eligible
            assert aug24.trade_reason == "SIDEWAYS_CURRENT_TREND"

    # Audit the latest genuinely trade-eligible historical fold, if one exists.
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

    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION: PASS")
    print("PHASE 5.8 TRADE GATE VALIDATION: PASS")
    print("No model optimisation is applied here; Phase 5 remains diagnostic and Phase 6 covers calibration/OOS trading performance.")


if __name__ == "__main__":
    main()
