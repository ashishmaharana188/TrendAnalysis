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
        progress_every=5,
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

    print("PHASE 5.8 REAL OLAP PREDICTION VALIDATION: PASS")
    print("IMPORTANT: This is diagnostic Phase 5 validation; Phase 6 calibration/OOS performance is still required.")


if __name__ == "__main__":
    main()
