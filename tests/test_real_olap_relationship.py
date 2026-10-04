from __future__ import annotations

from analysis.real_olap_validation import (
    RealOLAPValidationConfig,
    validate_real_olap_relationships,
)


def main() -> None:
    config = RealOLAPValidationConfig(
        ticker="RELIANCE",
        benchmark="Nifty_50",
        analysis_timeframe="6M",
        holding_period_months=1.0,
        entry_mode="next_trading_day",
        max_folds=60,
        step_trading_days=20,
        min_training_observations=12,
    )

    result = validate_real_olap_relationships(config)

    assert result.panel_rows >= config.min_training_observations
    assert result.walk_forward_folds > 0
    assert result.leakage_violations == 0
    assert result.state_future_violations == 0
    assert result.outcome_temporal_violations == 0
    assert result.latest_market_date is not None
    assert result.first_prediction_date is not None

    print("PHASE 4.9 REAL OLAP RELATIONSHIP VALIDATION: PASS")
    print("Ticker:", result.ticker)
    print("Benchmark:", result.benchmark)
    print("Panel rows:", result.panel_rows)
    print("Walk-forward folds:", result.walk_forward_folds)
    print("Method A hit rate:", round(result.method_a_hit_rate_pct, 2))
    print("Method B hit rate:", round(result.method_b_hit_rate_pct, 2))
    print("Combined hit rate:", round(result.combined_hit_rate_pct, 2))
    print("Leakage violations:", result.leakage_violations)
    print("State future violations:", result.state_future_violations)
    print("Outcome temporal violations:", result.outcome_temporal_violations)
    print("Latest market date:", result.latest_market_date)


if __name__ == "__main__":
    main()
